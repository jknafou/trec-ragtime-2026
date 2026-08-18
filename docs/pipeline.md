# Pipeline reference

One pipeline serves all three RAGTIME tasks. The offline half turns the raw multilingual
collection into three parallel English/native renderings of the same passages and one index per
rendering. The online half turns a report request into a nugget bank, answers each nugget with a
small agentic RAG loop, audits coverage, and projects the result into the Task 1 report, the Task 2
ranked list and the Task 3 nugget bank.

```
download -> chunk -> merge -> translate -> reconcile -> len_max -> packing -> vectorize -> assemble
                                                                        |
request -> decompose (seed bank) -> k parallel RAG loops -> coverage audit -> saturation
                                                                        |
                                             citation scoring -> select and serialize -> T1 / T2 / T3
```

Every stage writes into the artifact tree under `ragtime.common.Layout`, atomically (temp file,
rename, `_SUCCESS` marker), so the tree is the checkpoint and a relaunch recomputes only what is
missing.

## Corpus download

**In:** the RAGTIME2 dataset repository on the Hugging Face Hub.
**Out:** `corpus/<family>/<chunker-hash>/corpus-preprocess/raw/`.

`preprocess.download` fetches the raw gzipped JSONL files rather than a re-encoded dataset, checks
every file's sha256 against the recorded LFS blob hash, and only then promotes the temporary
directory into place. The reader (`preprocess.corpus`) takes each file's language from its name;
the four collection languages are Chinese, English, Russian and Spanish.

## Chunking

**In:** the raw documents.
**Out:** `documents.parquet` (`document_id, lang, text`) and `sentences.parquet`
(`sentence_id, document_id, sentence_index, lang, start, end, paragraph_index, token_count`).

Documents are split into paragraphs on blank-line and structural boundaries, with pure boilerplate
lines dropped by a versioned rule set. The document text is normalised to NFC once, and each
paragraph is segmented into sentences by a SaT segmenter running on its ONNX backend. Sentences are
stored as byte offsets into the document text, never as copies, and every offset is verified to
reproduce the segmenter's own segment. Sentence ids are minted here, before any translation exists,
which is what makes the ids shared across the three renderings by construction. Token counts come
from the pinned BGE-M3 tokenizer. Chunking emits no passages.

## Translation-unit merge

**In:** the two spine tables.
**Out:** a merge map, one row per non-English sentence.

A sentence too short to be a usable machine-translation unit is grouped with a structural
neighbour: a paragraph-opening sentence merges forward, anything else merges backward, subject to a
token cap recomputed on the fused text and never crossing a paragraph boundary except for
singletons. English emits no rows, since it passes through unchanged. This stage is CPU only and
the map is purely additive; nothing in the spine is rewritten.

## Machine translation

**In:** the merge map (high tier) or the reconciled sentence inventory (low tier).
**Out:** `translations_raw/<rendering>/...`, one row per translation unit, markers unscrubbed.

Two arms produce the two English renderings. The high tier, `omt`, translates merge units with
NLLB-200-3.3B served through CTranslate2; the constituents of a unit are joined with a marker
character so that per-sentence English can be recovered afterwards. The low tier, `omt_opus`,
translates one sentence per call with the three bilingual OPUS-MT Marian checkpoints, so
correspondence is one-to-one by construction and no marker is involved. English sentences are
handled by a separate identity pass that loads no model. Batching is decided only by the hashed
translation configuration and rows are written in canonical document order, so concurrency never
reaches the output. Splitting, fusion and marker scrubbing are left to the next stage,
which makes a change of policy a short CPU pass instead of a re-translation.

## Reconciliation

**In:** the raw translation rows.
**Out:** `final/<hash>/sentences.parquet`, `final/<hash>/translations/<rendering>.parquet`,
`remap.parquet`.

This stage answers the one question translation leaves open: did the unit marker survive? A unit
that comes back with exactly the expected number of non-empty segments is split back, and each
constituent keeps its own id, span and English text. Anything else is fused: the constituents become
a single sentence spanning the whole unit, a first-class sentence with no tombstone and no nullable
text. Each document is then renumbered densely. The raw translation rows, not the merge map, are the
ground truth, and they are required to partition each document. English identity rows are verified
against the source span but not stored, since an English passage renders from its native span in
every rendering.

## Length sidecar and packing

**In:** the final sentence inventory and the three translation tables.
**Out:** a per-sentence length sidecar, then `passages/<hash>/passages.parquet`.

The sidecar records each final sentence's token length in all three renderings plus their maximum.
Packing groups sentences into passages against that maximum rather than against native length, so
the passage boundaries are identical in all three renderings and the same passage id addresses the
same content whichever text is read. A sentence that exceeds the budget on its own becomes its own
passage and is counted. The passage table carries ids, language and token counts, and no text.

## Passage store

**In:** the documents, the final inventory, the translations and one packing.
**Out:** an LMDB environment keyed by passage id.

`common.passage_store` composes each passage's text on demand: the native rendering is a single
slice of the document text, the translated renderings are the joined sentence translations. All
three renderings for a passage are bulk-loaded into one environment, so choosing which rendering to
read is a lookup rather than a reload. This is the artifact behind `passage_lang`, the reading knob.
It has one writer, so it is built as a single shard.

## Index build

**In:** the packed passages.
**Out:** vector blocks, then one FAISS, one Seismic and one PLAID index per part, plus a manifest.

The build is split by resource class. Vectorization (GPU) encodes a fixed block of passages, in
table order, with the three encoders: BGE-M3 for the dense leg, MILCO for the learned-sparse leg,
and PLAID-X for the late-interaction leg. Batch membership derives from a table that has no
rendering column, so it is identical across renderings. Assembly (CPU) reads a part's blocks in
order and writes the three engines over one ordinal space, mapping ordinals back to passage ids
through the part's own id map. The manifest is published only after the assembled cells are checked
for completeness, disjointness and exact agreement with the packed passage id set. English is
indexed once and shared by all three renderings, which the path scheme expresses directly.

## Retrieval

**In:** one or more English query strings.
**Out:** a ranked list of `(passage_id, score)`. Never text.

The query is encoded once per leg by three separate checkpoints, not by one model with three
outputs: BGE-M3 for the dense leg, MILCO for the learned-sparse leg, PLAID-X for the
late-interaction leg. Each leg is then searched in each of the four language cells, which makes
twelve pools per query, every one of them taken to `max(top_k, reranker depth)`. Within a cell the
parts are merged on raw score, which is sound because the parts are the same encoder over disjoint
passages and none of the engines normalises against anything index-global. Across legs and
languages the pools are fused with reciprocal rank fusion at `retrieval.rrf.k`; twelve pools of a
hundred fuse to several hundred distinct passages, of which only the top
`retrieval.reranker.depth` are rescored by the Qwen3 cross-encoder, reading them in the rendering
that was searched. Everything below that cut keeps its fusion order and is then truncated at
`top_k`, so most of the fused pool never reaches the cross-encoder. The score that comes back is
the reranker's log-probability of "yes", which is negative, and larger is better.

Retrieval returns ids and scores only: the rendering handed to the model is the
caller's decision, and the two translation knobs stay independent because of it. Text is a separate
by-id fetch, and the cited document id is always derived from the passage id, so a citation resolves
to the original document whatever was searched or read.

## Retrieval service

**In:** query requests on a filesystem queue.
**Out:** ranked ids and scores, plus per-leg timings.

Opening a whole rendering is expensive, so one long-lived process holds it: the three legs and the
reranker resident, and a query dispatched to one replica and finished there. How the stack is laid
out over the cards is a launch decision, not a property of the code. The runs behind this
repository used one replica whose late-interaction leg was sharded one language cell per card,
with a reranker instance on each of six cards; a smaller node instead runs a complete stack per
card. Language cells are searched in forked processes, because the sparse engine holds the
interpreter lock for the duration of a search, so threads there buy nothing.

The transport is a directory of request and reply files rather than HTTP, which lets a one-core
client drive a multi-GPU service across a shared filesystem; where the two ends share no
filesystem, the same client speaks to a forwarded HTTP port instead. The service publishes a
descriptor carrying its rendering, its index hash and a heartbeat; clients resolve a live service
by rendering, and refuse a reply whose rendering does not match the one the config asked for.
Because retrieval returns no text, one service covers all three reading renderings; only changing
the searched index costs a reload.

What the cross-encoder rescores is the fused head, in every run and on both transports.

## Decompose

**In:** the report request (title, problem statement, background) and, from round 1 on, the
evidence the loops gathered.
**Out:** the nugget bank for that round.

One function grows the bank, parameterized by round. Round 0 reads only the report request: it
fixes a breadth band from the request's own report limit, drafts a set of single-sentence
English questions in one constrained generation, and revises them once with a self-critique pass
that deduplicates, fills obvious gaps and assigns weights. Because it sees no passages, the
round-0 bank is identical across the arms of a family, and it is published once per (report
request, seed) so that every arm reads the same one. Rounds 1 and later are the coverage audit.
Every nugget carries a question, a weight, an aggregator type of `AND` or `OR`, and an
accumulating list of answers and retrieved passages.

## RAG loop

**In:** one nugget, as `{nugget_id, question}` and nothing else.
**Out:** committed claims with their answers, the passages seen, and per-passage best scores.

The loop is one growing conversation with four possible actions: search, submit a claim, submit an
answer, or abstain. Which actions are legal on a turn is enforced twice, once by narrowing the
action enum in the constrained decoding schema and once in code: search disappears when the search
budget is spent, and abstaining only becomes legal after a minimum search effort and at least one
failed answer attempt. A claim carries a verbatim span, the passage id it came from, the short
answer and the report sentence. Committing checks that the passage is one the model was actually
shown and that the span occurs, character for character after NFC normalisation, in the exact text
it was shown; a near miss gets an actionable hint and a bounded same-turn retry that does not
consume a turn. A claim that cannot be grounded is dropped with its reason recorded. Submitting an
answer succeeds only if a claim is already committed. The loop always returns a result: timeouts and
malformed generations become terminal records rather than exceptions, so one bad loop cannot cancel
its siblings.

The loops of a round run concurrently as coroutines under a semaphore whose width comes from the
serving instance's own reported concurrency, not from the scheduler. The report request text is
not passed in, which keeps a loop answering its question rather than the whole report request.

## Coverage audit and saturation

**In:** the round's loop results, with passage text rendered through the run's reading knob.
**Out:** the grown bank, and the decision to stop.

Evidence is folded onto the bank first, append-only and idempotent. One constrained generation then
labels each open nugget's coverage as full, partial or none, proposes new nuggets for the gaps it
sees, and proposes prunes. The judge is not trusted unconditionally: a nugget with no committed
answer is not closed however it was labelled, and a nugget that has answers is not pruned. New
nuggets pass a topicality gate before being minted, and the bank is reweighted and deduplicated with
only open nuggets eligible. Stopping is not the judge's decision either: the round loop tracks how
many genuinely new nuggets each round produced and stops when that novelty stays at or below a
threshold for a configured streak. A bounded sweep round then answers any nugget minted by the final
audit that was never fanned out.

## Citation scoring

**In:** the final bank and the per-loop records.
**Out:** `citation_scores/scores.jsonl`, one row per (nugget, answer, document).

Each cited answer gets one importance judgement against its nugget's question, mapped onto a numeric
scale and broadcast to the documents that support it. That judgement sees the nugget's question and
the claim sentence and **no passage at all**: it asks how much of the question the claim answers,
not whether the cited text supports the claim. The final score is the product of those two
judgements, nugget importance and claim importance, so either factor can veto. This stage ranks, it
never gates: nothing here can drop a claim or a citation, and it fills only the values of reference
entries whose keys the loop already fixed.

## Select and serialize

**In:** one finished cell (all topics of one arm at one seed).
**Out:** the deliverables named by the config's `outputs` block.

The terminal projection is deterministic and runs per topic. The bank is assembled from the last
complete round and enriched from the loop records, then deduplicated once, before any projection:
near-duplicate questions are merged by embedding similarity through connected components, and inside
`OR` nuggets near-duplicate answers are merged by an embedding candidate confirmed by a model call.
A cap on nuggets and on answers per nugget is applied once and feeds both Task 1 and Task 3.

**Task 1** is a coverage-first budgeted greedy selection: one sentence for each distinct answered
nugget in weight order, then a second pass filling the residual character budget by weight times
score. A sentence is admitted only if it carries at least one citation, an exact repeat merges its
citations into the first copy instead of being emitted twice, and the budget is measured with NFKC
character length, which is the track's rule and is not the NFC rule used for span
commitment.

**Task 2** is built from the committed claims: every document supporting a claim is a retrieved
document, its value is the persisted claim importance times nugget importance, a document cited more
than once accumulates, and remaining ties are broken by the reranker score before the ranking is
emitted. Ties are separated in the score column rather than left for the evaluator to reorder.

**Task 3** emits the capped bank, answered and unanswered nuggets alike, with the aggregator type and
sorted reference keys.

## Submission validation

**In:** the emitted files.
**Out:** a pass, or a refusal that publishes nothing.

Task 1 and Task 3 are checked by the organisers' run validator, pinned at one commit, against a
normalised copy of the topics file; a crash is distinguished from a verdict, and "the validator did
not run" is a third state that also refuses. The Task 2 run file is re-read from the bytes just
written and checked column by column: six fields, the literal `Q0`, document ids that resolve, ranks
equal to their position, contiguous topics, no more rows per topic than declared, and strictly
descending scores. A manifest fragment is written per run, and a reduce step over the fragments
produces the family manifest with the assessment priority order.
