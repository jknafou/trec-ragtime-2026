"""The decompose prompt surface: templates plus the decoding knobs, in one place.

This module owns how decompose addresses the one shared LLM: the prompt text for each of
decompose's calls, and :func:`decoding_kwargs`, the read of the ``decomposition.decoding`` block
every one of those calls is issued under. Both are the prompt-side surface of the same
fairness-shared config block, whose hash is the recorded prompt hash, so they live together and are
read from ``cfg`` rather than typed as module defaults.

It builds no client and issues no call: :func:`decoding_kwargs` returns plain kwargs the caller
hands to ``ClientBundle.llm.generate``. There is no wrapper around
``serving.llm.guided_json`` here, since a second emission path would defeat the single-client rule.

Prompt-shape facts the templates below encode literally:

* ``problem_statement`` is the coverage spine and comes first, labelled primary; ``background`` is
  persona and scope and comes second, labelled context. The decomposer never decomposes
  ``background`` alone.
* ``title``, the request's short label, is rendered between them; :func:`_title_section` explains
  why that position rather than above the spine. It enters decompose only, because the RAG loop is
  request-blind by construction, which is what makes a loop's behaviour a function of its nugget
  question alone.
* Round 0 is facet enumeration from an empty bank, not an update or audit framing.
* The ``k`` band is a breadth hint, stated as such; the model is told a wider or narrower answer is
  acceptable when the request warrants it, because nothing downstream clamps its output.
* The self-critique pass is scoped to dedup, coverage and weight, with no open-ended "reconsider
  your reasoning" channel.
* Every schema in use carries a leading ``rationale`` field and the prompts tell the model to use
  it, which keeps the grammar constraining only the output slot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ragtime.serving.llm import decoding_kwargs as _serving_decoding_kwargs

if TYPE_CHECKING:
    from .exemplars import Exemplar

__all__ = [
    "AUDIT_SYSTEM",
    "SEED_SYSTEM",
    "decoding_kwargs",
    "dedup_confirm_prompt",
    "gap_audit_prompt",
    "on_topic_prompt",
    "seed_prompt",
    "self_critique_prompt",
]

SEED_SYSTEM = (
    "You decompose an English report request into atomic question-nuggets. "
    "A nugget is ONE single-sentence English question about ONE single fact, "
    "answerable from a document collection. You never answer the questions and you "
    "never invent facts. Always fill the leading 'rationale' field with your reasoning "
    "before the structured output."
)


#: The system prompt for rounds 1 and later. A second system message, not a second model and not a
#: second decoding block: the audit is the same operation at a different round, so it shares
#: ``decoding_kwargs`` and the same ``clients.llm`` singleton. It differs from :data:`SEED_SYSTEM`
#: only where the role differs, in that the auditor is shown evidence and judges coverage over it.
AUDIT_SYSTEM = (
    "You audit the coverage of an English report request that is being answered from a "
    "document collection. You are given the question-nuggets asked so far, what was actually "
    "found for each, and the evidence the search returned. You judge which nuggets are now "
    "fully answered, you propose nuggets for facets that are still missing, and you flag "
    "nuggets that are off topic or redundant. A nugget is ONE single-sentence English question "
    "about ONE single fact. You never answer the questions and you never invent facts. Always "
    "fill the leading 'rationale' field with your reasoning before the structured output."
)


def decoding_kwargs(cfg: Any) -> dict[str, Any]:
    """Return the ``decomposition.decoding`` kwargs for ``GenCtx``.

    A delegation to :func:`ragtime.serving.llm.decoding_kwargs`, not a second reader. Which block
    a stage decodes from is a per-stage choice and stays here; how a block becomes generation
    kwargs is one implementation, in the module that owns ``GenCtx``'s shape.
    """
    return _serving_decoding_kwargs(cfg.blocks.get("decomposition", {}).get("decoding", {}))


def _title_section(title: str) -> str | None:
    """Return the title section, or ``None`` when the topic has none.

    It is rendered after the problem statement and before the background: neither a headline above
    the spine nor context beside the background, but a subordinate disambiguator attached to the
    spine, so that section order agrees with the section labels rather than fighting them.

    Four reasons for that position:

    1. A short field is more attractive to decompose than a long one, so the headline position
       costs most. Real titles are two to four word subject labels, where every content word reads
       as a facet, while the real facets live in a several-hundred-word problem statement. Putting
       a maximally decomposable, maximally salient string above the field labelled primary invites
       exactly the drift this module already forbids for ``background``.
    2. The spine's primacy is a contract, not a layout preference: the problem statement comes
       first, labelled primary, and a title above it would contradict that.
    3. A thin problem statement must stay the thin-seed case. ``grow_nuggets`` already defines
       degenerate behaviour for a short ``problem_statement``. If the title were the headline, a
       thin statement plus a rich title would silently become "decompose the title": a different
       algorithm reached by data rather than by config, and one no run record would show.
    4. A field that can be absent cannot be the request's headline. Some topics files carry no
       ``title`` key, so this section must be omittable, and it is omitted entirely rather than
       rendered blank, on the same rule ``background`` follows: never a labelled-but-blank field
       to hallucinate into.

    The anti-drift clause sits inside this section rather than being folded into the existing
    "Do not decompose the background alone" sentence, which keeps the guard adjacent to the field
    it governs and makes the section strictly additive: with no title, every prompt here renders
    exactly the bytes it rendered before.
    """
    if not title or not title.strip():
        return None
    return (
        "TITLE (the request's own short label — it names the SUBJECT of the problem "
        "statement above; it is NOT itself the information need. Use it to disambiguate "
        "what the request is about, never as the thing to decompose: a facet the problem "
        f"statement demands is required even if the title does not mention it):\n{title}"
    )


def _exemplar_block(exemplars: Sequence[Exemplar]) -> str:
    """Render the few-shot block. It carries no title, even though requests have one.

    The exemplar sets are frozen and versioned, because the ``decomposition`` block's hash is the
    recorded prompt hash and is shared byte-identically across a run family. Giving the exemplars
    titles would either edit a frozen set in place or require a new set plus a
    ``decomposition.few_shot.exemplar_set`` change in every config, which is a fairness-family
    decision rather than a prompt edit.

    It is also unnecessary: the exemplars demonstrate granularity, count and format rather than
    request structure, and already omit the breadth hint, the requirements block and every other
    section of the live prompt.
    """
    if not exemplars:
        return ""
    parts = ["Worked examples of the required granularity and format:"]
    for i, ex in enumerate(exemplars, start=1):
        questions = "\n".join(f"    - {q}" for q in ex.nuggets)
        parts.append(
            f"  Example {i}\n"
            f"  PROBLEM STATEMENT (primary): {ex.problem_statement}\n"
            f"  BACKGROUND (context): {ex.background}\n"
            f"  Question-nuggets:\n{questions}"
        )
    return "\n\n".join(parts)


def seed_prompt(
    problem_statement: str,
    background: str,
    k_band: tuple[int, int],
    exemplars: Sequence[Exemplar] = (),
    *,
    title: str = "",
) -> str:
    """Render the round-0 facet-enumeration prompt.

    ``k_band`` is the resolved ``(k_min, k_max)`` from
    :func:`ragtime.pipeline.decompose.kband.limit_to_k`, not the config's table of triples, and is
    rendered as an explicit breadth hint.

    ``background`` may be empty, either through ``decomposition.background: false`` or because the
    topic has none; the section is then omitted entirely rather than rendered with an empty body,
    so the model is never shown a labelled-but-blank field to hallucinate into. ``title`` follows
    the same omit-when-empty rule and is keyword-only, so the positional signature is unchanged.
    """
    k_min, k_max = k_band
    sections = [
        "TASK: enumerate the distinct aspects a good report must cover.",
        (
            "The nugget bank is currently EMPTY — you are seeding it from the request "
            "alone. You have NO retrieved documents and must not pretend to any: "
            "enumerate the facets the request itself demands, not facts you believe."
        ),
        f"PROBLEM STATEMENT (primary — this is the coverage spine):\n{problem_statement}",
    ]
    if (title_section := _title_section(title)) is not None:
        sections.append(title_section)
    if background and background.strip():
        sections.append(
            "BACKGROUND (context/persona — it biases which facets matter, it is NOT "
            f"itself the information need):\n{background}"
        )
    sections.append(
        "Do not decompose the background alone; the problem statement is the spine and "
        "every required facet of it must be covered even if the background does not "
        "emphasise it."
    )
    sections.append(
        f"BREADTH HINT: aim for at least {k_min} and around {k_max} question-nuggets. "
        "This is a hint derived from the request's length budget, not a quota — emit "
        "more if the request genuinely spans more facets, fewer if it is narrow. Never "
        "pad with filler questions to reach the number."
    )
    sections.append(
        "REQUIREMENTS for every nugget:\n"
        "  - one single-sentence English question, ENDING IN '?' AND NOTHING AFTER IT;\n"
        "  - about exactly ONE fact (split compound questions);\n"
        "  - answerable from documents, not from opinion;\n"
        "  - distinct from every other nugget you emit.\n"
        # With no thinking channel the model will use the `question` field as one, and that text
        # would be published verbatim as a Task-3 nugget. `grow_nuggets.is_question` rejects it,
        # but a rejected emission is a wasted generation, so the prompt says so plainly.
        "Put ALL reasoning -- merging, splitting, second thoughts, corrections -- in 'rationale'. "
        "A 'question' containing your deliberation is DISCARDED, not cleaned up.\n"
        "Leave 'nugget_id' null — ids are minted downstream."
    )
    block = _exemplar_block(exemplars)
    if block:
        sections.append(block)
    return "\n\n".join(sections)


def self_critique_prompt(
    problem_statement: str,
    background: str,
    questions: Sequence[str],
    k_band: tuple[int, int],
    *,
    title: str = "",
) -> str:
    """Render the single self-critique pass: dedup, coverage and weight, nothing else.

    ``title`` is shown here for the same reason ``problem_statement`` is: this pass judges whether
    the request demands a facet no nugget covers, and how central each facet is to the request, so
    it must see the same request the draft call saw. A title visible to the draft and invisible to
    its critic would make the two calls disagree about what the request is, and the disagreement
    would surface as coverage edits nobody asked for.

    The scope is stated as a closed list. The design allows a single self-critique pass
    scoped to dedup and coverage rather than open-ended reasoning correction; the ``weight``
    judgement rides along in the same response, since this is the pass that already re-examines
    every surviving nugget.
    """
    k_min, k_max = k_band
    listing = "\n".join(f"  {i}. {q}" for i, q in enumerate(questions, start=1))
    sections = [
        (
            "TASK: revise a drafted set of question-nuggets. Do EXACTLY three things "
            "and nothing else:"
        ),
        (
            "  (1) MERGE near-duplicates — if two nuggets ask the same question in "
            "different words, keep the clearer one and drop the other.\n"
            "  (2) COVERAGE — if the request demands a facet no nugget covers, add one "
            "nugget for it.\n"
            "  (3) WEIGHT — give every surviving nugget a 'weight' in [0.0, 1.0] for how "
            "central that facet is to the request (1.0 = the report is wrong without it, "
            "0.0 = peripheral colour)."
        ),
        (
            "Do NOT re-argue, re-word for style, or second-guess anything else about the "
            "draft. A nugget you neither merged nor added must come back unchanged."
        ),
        f"PROBLEM STATEMENT (primary — this is the coverage spine):\n{problem_statement}",
    ]
    if (title_section := _title_section(title)) is not None:
        sections.append(title_section)
    if background and background.strip():
        sections.append(f"BACKGROUND (context/persona):\n{background}")
    sections.append(f"DRAFTED NUGGETS:\n{listing}")
    sections.append(
        f"For reference, the request's breadth hint was {k_min}-{k_max} nuggets; it is a "
        "hint, so do not add or drop nuggets merely to hit it."
    )
    sections.append(
        "Return the FULL revised set (kept + added, merged ones omitted), each with its "
        "'question' and its 'weight'. Leave 'nugget_id' null."
    )
    return "\n\n".join(sections)


def gap_audit_prompt(
    problem_statement: str,
    background: str,
    tagged: Sequence[Mapping[str, Any]],
    evidence: Sequence[tuple[str, str]] = (),
    *,
    round: int,
    title: str = "",
) -> str:
    """Render the audit prompt for rounds 1 and later: mark-covered, gap-detect and prune, in one call.

    ``tagged`` is the current bank, each entry a mapping with ``nugget_id``, ``question``,
    ``aggregator_type``, ``status`` and the short answers the loops committed for it. It is a plain
    mapping rather than a record type so this module keeps importing nothing from
    :mod:`~ragtime.pipeline.decompose.coverage_audit`, which imports this one. ``evidence`` is
    ``[(passage_id, text)]`` from the passages the loops already retrieved this round; the auditor
    issues no queries of its own.

    Three prompt facts this template encodes literally:

    * coverage is judged over the committed answers, never over free text, on the AutoNuggetizer
      three-way ``full``/``partial``/``none`` scale, and only ``full`` closes a nugget;
    * the AND/OR aggregation boundary lives here, because ``common.Nugget`` carries no member list
      the grammar could constrain per member: an ``OR`` nugget is ``full`` as soon as one answer
      answers it, an ``AND`` nugget only when every enumerated member is present;
    * new nuggets come from two sources, evidence-driven (a retrieved passage mentions a
      facet nobody asked about) and request-driven (the problem statement demands a facet retrieval
      never surfaced). The second is why a bank can grow past what the corpus happened to return,
      and why ``trigger_passage_id`` is nullable rather than required.

    ``title`` is shown for the third bullet's sake: a request-driven gap is by definition a facet
    the request demands and retrieval never surfaced, so the auditor must see the whole request. The
    seed saw the title, and an auditor that did not would judge request coverage against a strictly
    smaller request than the one the bank was seeded from.
    """
    listing = "\n".join(_tagged_line(n) for n in tagged) or "  (the bank is empty)"
    sections = [
        (
            f"TASK: audit the coverage of a report request after round {round - 1} of "
            "evidence gathering. Do EXACTLY three things and nothing else:"
        ),
        (
            "  (1) COVERAGE — for every nugget below, judge how well the ANSWERS listed under "
            "it cover its question: 'full' (the question is now answered), 'partial' (some of "
            "it is answered) or 'none'. Judge ONLY the answers shown; a nugget with no answers "
            "is 'none'. An 'OR' nugget is 'full' as soon as ONE answer answers it. An 'AND' "
            "nugget is 'full' only when EVERY member of its enumeration is present among the "
            "answers — if any member is missing it is 'partial', not 'full'.\n"
            "  (2) ADD — propose a nugget for each facet still missing. Two sources, both "
            "wanted: a facet the EVIDENCE mentions that no nugget asks about (set "
            "'trigger_passage_id' to that passage's id), and a facet the PROBLEM STATEMENT "
            "demands that nothing has surfaced yet (set 'trigger_passage_id' to null). Propose "
            "nothing if nothing is missing.\n"
            "  (3) PRUNE — list the nugget_ids of nuggets that are OFF TOPIC or REDUNDANT with "
            "another nugget. Do NOT prune a nugget merely because it is unanswered: an "
            "unanswered on-topic question is a valid deliverable."
        ),
        (
            "Be biased toward another round: proposing a facet that turns out to be covered "
            "costs one question, while calling a partly-answered nugget 'full' ends the search "
            "for it permanently."
        ),
        f"PROBLEM STATEMENT (primary — this is the coverage spine):\n{problem_statement}",
    ]
    if (title_section := _title_section(title)) is not None:
        sections.append(title_section)
    if background and background.strip():
        sections.append(f"BACKGROUND (context/persona):\n{background}")
    sections.append(f"CURRENT NUGGET BANK (with what was found for each):\n{listing}")
    if evidence:
        rendered = "\n\n".join(f"  [{pid}]\n  {text}" for pid, text in evidence)
        sections.append(
            "EVIDENCE RETRIEVED THIS ROUND (the only evidence there is — you cannot search):\n"
            f"{rendered}"
        )
    sections.append(
        "REQUIREMENTS for every nugget you ADD:\n"
        "  - one single-sentence English question, ENDING IN '?' AND NOTHING AFTER IT;\n"
        "  - about exactly ONE fact (split compound questions);\n"
        "  - a genuine facet of the PROBLEM STATEMENT, not of the documents;\n"
        "  - distinct from every nugget already in the bank;\n"
        "  - 'weight' in [0.0, 1.0] for how central that facet is to the request.\n"
        "Put ALL reasoning -- merging, splitting, second thoughts, corrections -- in "
        "'rationale'. A 'question' containing your deliberation is DISCARDED, not cleaned up."
    )
    return "\n\n".join(sections)


def _tagged_line(nugget: Mapping[str, Any]) -> str:
    """One bank entry as the auditor sees it: id, aggregator, question, answers so far."""
    answers = tuple(nugget.get("answers", ()) or ())
    body = "\n".join(f"      - {a}" for a in answers) if answers else "      (no answers yet)"
    return (
        f"  {nugget.get('nugget_id', '?')} [{nugget.get('aggregator_type', 'OR')}] "
        f"{nugget.get('question', '')}\n    ANSWERS FOUND:\n{body}"
    )


def on_topic_prompt(
    question: str,
    problem_statement: str,
    passages: Any | None = None,
    *,
    title: str = "",
) -> str:
    """Render the grounding-gate prompt: is ``question`` a genuine facet of the request?

    This is an admission gate over an already-generated nugget, kept separate from the call that
    generated it so an accepted and a rejected nugget each carry their own logged rationale. It
    judges subject only: never a drop for lack of evidence, and never a dedup judgement.

    ``passages`` is the later-round evidence context and is ``None`` at round 0; when present, its
    rendered text is offered as context for judging topicality only, never as evidence the
    question must be supported by.

    ``title`` is threaded here even though ``background`` is not, because the two sit on different
    axes. ``background`` is orthogonal to the subject and adds no admission signal, only the risk
    of rejecting a required facet the persona happens not to emphasise. ``title`` is on the same
    axis as the spine: it names the subject, and the gate's own instruction is to answer false if
    the nugget drifts to an adjacent subject, which is precisely what a subject label resolves.
    Generator and gate symmetry settles it: from round 1 the auditor proposes gaps against a
    request that includes the title, and a gate judging those proposals against a request that does
    not would produce rejections that are artifacts of the plumbing rather than of topicality.

    The trade-off is that a title is a lossy summary and a rejection here is unrecoverable, since
    the facet is never minted, never fanned and never reaches Task 3. The section text therefore
    tells the judge explicitly that the title does not narrow the problem statement, and reverting
    is a single keyword argument at ``grow_nuggets._audit_round``'s ``on_topic`` call site.
    """
    sections = [
        (
            "TASK: decide whether a candidate question-nugget is ON TOPIC for a report "
            "request — i.e. whether answering it is a genuine facet of the information "
            "need below."
        ),
        f"PROBLEM STATEMENT (the information need):\n{problem_statement}",
    ]
    if (title_section := _title_section(title)) is not None:
        sections.append(title_section)
    sections.append(f"CANDIDATE NUGGET:\n{question}")
    if passages:
        sections.append(
            "RETRIEVED CONTEXT (for judging topicality only — the nugget does NOT have "
            f"to be answered by it):\n{passages}"
        )
    sections.append(
        "Answer true only if a report answering the problem statement would be expected "
        "to cover this question. Answer false if it drifts to an adjacent subject, is "
        "about the source documents rather than the request, or is unanswerable in "
        "principle. Put your reasoning in 'rationale'."
    )
    return "\n\n".join(sections)


def dedup_confirm_prompt(kept: str, candidate: str) -> str:
    """Render the dedup gate prompt: are two nuggets the same question?

    The embedding cosine only proposes the pair, as a generous recall pre-filter; this
    call is the arbiter, and it is asked for the conjunction the ``dedup`` schema encodes,
    paraphrase match and entity match, so a merely topically close pair is not merged away.
    """
    return "\n\n".join(
        [
            (
                "TASK: decide whether two question-nuggets are DUPLICATES — i.e. any "
                "correct answer to one is a correct answer to the other."
            ),
            f"NUGGET A (already in the bank):\n{kept}",
            f"NUGGET B (candidate):\n{candidate}",
            (
                "Set 'paraphrase_match' true only if they ask the same thing in different "
                "words; set 'entity_match' true only if they are about the same entities, "
                "quantities and time frame. Set 'duplicate' true only if BOTH hold — a "
                "merge is irreversible, so when in doubt keep them separate. Explain "
                "briefly in 'reason'."
            ),
        ]
    )
