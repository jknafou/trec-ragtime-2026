"""Machine translation over the sentence spine: the raw half of both translated renderings.

Two ``orchestration.saturate.StageAdapter`` implementations over the corpus tables ``chunk``
wrote, run after ``merge``:

1. :class:`OmtIdentityAdapter` (``translate_omt_identity``), CPU only. English is over a
   third of the corpus's sentences and needs no MT, so it gets its own substage on the CPU
   template: filtering it inside the GPU stage would make every English-heavy shard claim a
   GPU, pull the model off shared storage and translate nothing, since shards are
   document-ordered and whole shards are English. It writes identity rows (text unchanged,
   ``model_id="identity"``) and imports no model at all.
2. :class:`OmtTranslateAdapter` (``translate_omt``), the non-English sentences through
   ``serving.mt``.

The same two adapters produce both translated renderings, and which one they produce is a
property of the hashed ``translation`` block rather than of a separate code path. With
``translate_units: merged`` they translate the merge map's ``§``-joined units with
``facebook/nllb-200-3.3B`` through CTranslate2 and write ``omt``, the high tier. With
``translate_units: sentences`` they translate one sentence per call with the bilingual
OPUS-MT checkpoints and write ``omt_opus``, the low tier, where correspondence is one to one
by construction and no marker is involved. The low tier runs against an already-reconciled
inventory named by ``RAGTIME_INVENTORY_DIR``, because the arm that reconciled the corpus
resolves to a different ``translation`` hash and this arm therefore cannot compute the path.

In the high tier the unit of work is the merge unit, and so is the unit of storage. ``merge``
decided which consecutive sentences MT must see together; this stage joins them with ``§``
(:mod:`ragtime.preprocess.merge_join`), translates the joined string once, and stores that
English string once, as a single row keyed by the unit's first constituent's ``sentence_id``
with the markers unscrubbed. It does not split the result back and does not fuse: both are
reconciliation's job, since that is the only stage that can renumber the sentence inventory
the decision implies. Storing raw makes a change of fusion policy a short CPU re-pass over
``translations_raw`` instead of a re-translation of the corpus.

Four properties are load-bearing:

- Sentence text is ``documents.text[start:end]``, sliced from the one stored copy. Nothing
  is re-segmented and no raw source file is read. A merge unit's source text is the
  concatenation of its constituent slices joined by the marker; the constituents are
  adjacent in the document, so no separator is fabricated and CJK units carry no invented
  spaces.
- Translate-once is a property of the input rather than a predicate. A sentence appears
  exactly once in ``sentences.parquet`` and exactly once in the merge map, so there is no
  ``seen`` set and no passage id anywhere here.
- Work units are contiguous, document-atomic row ranges over the work table, read via
  ``spine.iter_row_range``, so nothing is materialised: Parquet's footer is already the
  index. The merge map is co-ordered with the sentence spine, which lets the GPU arm shard
  the map directly and read a contiguous range of it rather than building a per-worker index
  over tens of millions of keys.
- Bytes do not depend on hardware. Bucket composition comes only from the hashed,
  family-shared ``translation.config`` (``bucket_token_budget`` and
  ``max_sentences_per_bucket``); the per-shape safety knob is passed to CTranslate2 and never
  enters composition. The sort key is total (``(len(tokens), sentence_id)``) and rows are
  written in canonical ``(document_id, sentence_index)`` order regardless of bucket
  completion order, so concurrency never leaks into the file.

The output path and the queue name carry both the ``translation`` hash and the ``merge`` hash
(``Layout.translations_raw_path``). Keying by the translation block alone would let a re-run
under a different merge map find a stale ``_SUCCESS`` and reuse translations produced by the
previous grouping; unit boundaries are as much an input to the MT call as the beam size is.

The MT client comes from the ``serving`` singleton registry, since a stage never instantiates
a model. This module needs three things from it: ``tokenize(text, src_lang,
max_input_length=, stats=) -> Sequence[str]``, ``translate_batch(items, *, target_lang,
beam_size, len_ratio_a, len_ratio_b, max_decoding_cap, max_input_length,
no_repeat_ngram_size, tier, stats) -> dict[id, str]``, and ``assert_marker_atomic(marker,
probe_lang)``. :class:`MtInput` is the item shape, structurally identical to
``serving.mt.MtSentence``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping  # Runtime import: per-direction knobs isinstance it
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragtime.common import Layout, Statistics, get_logger
from ragtime.common.io import concat_parquet, is_done, iter_parquet_batches, write_parquet_stream
from ragtime.common.schemas import translation_raw_arrow_schema
from ragtime.config import all_hashes
from ragtime.orchestration import saturate
from ragtime.orchestration.run_identity import run_family
from ragtime.serving.batching import Batcher, Tier

from . import spine
from .merge_join import MARKER, contains_marker, join_constituents, split_translation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator, Sequence

    from ragtime.config import RunConfig

__all__ = [
    "OPUS_VARIANT",
    "UNITS_MERGED",
    "UNITS_SENTENCES",
    "VARIANT",
    "CorruptMergeMapError",
    "MtInput",
    "OmtIdentityAdapter",
    "OmtTranslateAdapter",
    "TranslationUnit",
    "UnitMember",
    "buckets",
    "build_sentence_units",
    "build_units",
    "flores_code_for",
    "inventory_hash_of_node",
    "unit_row",
]

_log = get_logger("preprocess.translate_omt")

_DEFAULT_BASE = "runs"  # real runs pass an explicit base

#: The rendering these adapters produce. Knob 2 ("which rendering the LLM reads").
VARIANT = "omt"
#: The rendering the same adapters produce when pointed at an already-reconciled inventory
#: (``RAGTIME_INVENTORY_DIR``): the low tier, OPUS-MT, per sentence, no merge. It names the
#: raw artifact's directory, so the two arms' raw output can never share a path even when
#: their ``translation`` hashes are the two components a reader compares.
OPUS_VARIANT = "omt_opus"
#: ``model_id`` of an English pass-through row (never an MT model).
IDENTITY_MODEL_ID = "identity"

#: The manifest field a reconciled node uses to declare its own sentence-inventory key.
_MANIFEST_INVENTORY_KEY = "inventory_hash"

_HASH_DIRLEN = 12

# Semantic defaults, used only when the fairness-shared `translation.config` omits the
# key. Every one of these changes the artifact, so a config that sets them folds them into
# the hash, which is why `no_repeat_ngram_size` lives here rather than as a client default.

_DEFAULT_TARGET_LANG = "eng_Latn"
_DEFAULT_BEAM_SIZE = 4
_DEFAULT_BUCKET_TOKEN_BUDGET = 16384
_DEFAULT_MAX_SENTENCES_PER_BUCKET = 512
_DEFAULT_LEN_RATIO_B = 10.0
_DEFAULT_MAX_DECODING_CAP = 512
_DEFAULT_MAX_INPUT_LENGTH = 512  # NLLB's cap; CT2's own default is 1024
_DEFAULT_NO_REPEAT_NGRAM = 4
_DEFAULT_MARKER_PROBE_LANG = "spa_Latn"

#: ``translation.config.translate_units``, the granularity of one MT call. Semantic and
#: hashed, and explicit rather than inferred from the model name.
#:
#: - ``merged``: the merge map's units, where sub-threshold sentences are ``§``-joined and
#:   translated as one string, so per-sentence English is recovered by splitting the marker
#:   back (and, when it did not survive, by fusing). This is the NLLB arm.
#: - ``sentences``: one sentence span in, one translation out, keyed by ``sentence_id``.
#:   One to one by construction: no marker, no split-back, no fusion, no merge map read.
#:
#: The switch lives here rather than in the ``merge`` block: ``merge`` is its own
#: hash node and the completed NLLB corpus is keyed by it, so editing it to say "this arm
#: does not merge" would move ``merge_hash`` and orphan the finished translation.
UNITS_MERGED = "merged"
UNITS_SENTENCES = "sentences"
_UNIT_MODES = frozenset({UNITS_MERGED, UNITS_SENTENCES})
_DEFAULT_TRANSLATE_UNITS = UNITS_MERGED  # the shipped NLLB arm's behaviour

# Execution shape defaults (non-shared `execution` block).
_DEFAULT_TRANSLATE_SHARDS = 1000
_DEFAULT_IDENTITY_SHARDS = 200

# The merge-map columns the GPU arm projects; the map is its work table.
_MAP_COLUMNS = (
    "sentence_id",
    "document_id",
    "sentence_index",
    "lang",
    "start",
    "end",
    "merge_unit_id",
    "merge_constituent_index",
    "merge_constituent_count",
)
# The sentence columns the CPU identity arm projects.
_SENTENCE_COLUMNS = ("sentence_id", "document_id", "sentence_index", "lang", "start", "end")


class CorruptMergeMapError(RuntimeError):
    """The merge map contradicts itself or the spine.

    The map is a second artifact describing the first one, so it can disagree with it. Each
    checked condition corresponds to a way a merged unit would otherwise be silently wrong
    (a citation resolving to text that is not the sentence's): a truncated map would build a
    smaller unit than the rule chose, a unit spanning two documents would fuse unrelated
    text, and disordered indices would make reconciliation attribute the wrong split segment
    to a sentence.
    """


# --------------------------------------------------------------------------- #
# Language tagging (FLORES-200), incl. the zh Hans/Hant script-ratio decision.
# --------------------------------------------------------------------------- #
# Coarse, deterministic Han-script markers: high-frequency characters that exist in only one
# of the two scripts. A unit is Hant only if it carries strictly more traditional-only than
# simplified-only markers; ties and marker-free text stay Hans (the corpus's dominant
# script). A small frozen table rather than a dependency: it must be
# byte-stable across machines and years.
_HANT_MARKERS = frozenset(
    "國說這個對時會們學實現發經過進開關電幾萬樣東車書長門問間與為灣產業內從應該議認讓點無麼兒頭買賣錢場總統華語藝術體傳縣專爾樂園顯陳鄉節環"
)
_HANS_MARKERS = frozenset(
    "国说这个对时会们学实现发经过进开关电几万样东车书长门问间与为湾产业内从应该议认让点无么儿头买卖钱场总统华语艺术体传县专尔乐园显陈乡节环"
)
_FLORES = {"en": "eng_Latn", "es": "spa_Latn", "ru": "rus_Cyrl"}


def _model_id_for(model_id: str | Mapping[str, str], src_lang: str) -> str:
    """This row's model identity, per direction when the arm is several bilingual checkpoints.

    A KeyError here is the right failure: stamping some other direction's identity on the
    row would make the corpus claim a model that never saw the sentence.
    """
    if isinstance(model_id, Mapping):
        return str(model_id[src_lang])
    return str(model_id)


def flores_code_for(lang: str, text: str) -> str:
    """FLORES-200 code for a corpus language tag, resolving ``zh`` by script ratio.

    ``en``/``es``/``ru`` map directly; ``zh`` becomes ``zho_Hant`` when the text carries more
    traditional-only than simplified-only markers, else ``zho_Hans``. Resolved on the text
    that is actually translated, the joined unit, so a merged zh unit decides Hans or Hant on
    all of its constituents' evidence rather than the first one's alone.
    """
    if lang == "zh":
        hant = sum(1 for ch in text if ch in _HANT_MARKERS)
        hans = sum(1 for ch in text if ch in _HANS_MARKERS)
        return "zho_Hant" if hant > hans else "zho_Hans"
    code = _FLORES.get(lang)
    if code is None:
        raise ValueError(f"no FLORES-200 code for corpus language {lang!r}")
    return code


# --------------------------------------------------------------------------- #
# Units: merge-map rows plus document text into one `§`-joined MT input.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class UnitMember:
    """One constituent sentence of a translation unit, sliced from its document."""

    sentence_id: str
    document_id: str
    sentence_index: int
    lang: str
    text: str


@dataclass(frozen=True, slots=True)
class TranslationUnit:
    """One MT input: a lone sentence, or a ``§``-joined run of merged constituents."""

    unit_id: str  # the first constituent's sentence_id, always a real spine id
    members: tuple[UnitMember, ...]
    text: str  # what actually goes to MT
    src_lang: str

    @property
    def merged(self) -> bool:
        return len(self.members) > 1


@dataclass(frozen=True, slots=True)
class MtInput:
    """One pre-tokenized MT call item, structurally ``serving.mt.MtSentence``.

    ``sentence_id`` is the unit's id, its first constituent's sentence id: the client keys
    its returned dict by this field, and a unit is what a call translates.
    """

    sentence_id: str
    src_lang: str
    text: str
    tokens: Sequence[str]


def _validate_unit(members: Sequence[Mapping[str, Any]]) -> None:
    """Raise on any way the merge map can contradict itself."""
    first = members[0]
    unit = first["merge_unit_id"]
    counts = {int(m["merge_constituent_count"]) for m in members}
    if len(counts) != 1:
        raise CorruptMergeMapError(
            f"{unit}: members declare different merge_constituent_count values "
            f"{sorted(counts)}; the map contradicts itself"
        )
    declared = counts.pop()
    if declared != len(members):
        raise CorruptMergeMapError(
            f"{unit}: merge_constituent_count={declared} but {len(members)} of its "
            f"sentences are present ({[m['sentence_id'] for m in members]}); a truncated "
            "or over-wide map would build a unit the merge rule never chose"
        )
    if any(m["document_id"] != first["document_id"] for m in members):
        raise CorruptMergeMapError(
            f"{unit}: spans documents {sorted({m['document_id'] for m in members})}; a "
            "merge unit is a contiguous run of one document's sentences"
        )
    if [int(m["merge_constituent_index"]) for m in members] != list(range(len(members))):
        raise CorruptMergeMapError(
            f"{unit}: constituent indices "
            f"{[int(m['merge_constituent_index']) for m in members]} are not "
            f"0..{len(members) - 1} in spine order; reconciliation's split-back would "
            "attribute the wrong segment to a sentence"
        )
    if unit != first["sentence_id"]:
        raise CorruptMergeMapError(
            f"{unit}: unit is not named after its first constituent "
            f"{first['sentence_id']!r}; the row key would not be a real sentence id"
        )


def _member(row: Mapping[str, Any], doc_text: str) -> UnitMember:
    return UnitMember(
        sentence_id=row["sentence_id"],
        document_id=row["document_id"],
        sentence_index=int(row["sentence_index"]),
        lang=row["lang"],
        text=doc_text[int(row["start"]) : int(row["end"])],
    )


def _unit_of(members: Sequence[UnitMember], marker: str) -> TranslationUnit:
    """Build one unit from its constituents, ``§``-joined when there is more than one."""
    texts = [m.text for m in members]
    text = join_constituents(texts, marker) if len(texts) > 1 else texts[0]
    return TranslationUnit(
        unit_id=members[0].sentence_id,
        members=tuple(members),
        text=text,
        src_lang=flores_code_for(members[0].lang, text),
    )


def build_sentence_units(
    rows: Sequence[Mapping[str, Any]],
    doc_text: str,
    *,
    stats: Any = None,
) -> list[TranslationUnit]:
    """One unit per non-English sentence: the 1:1 arm (``translate_units: sentences``).

    No merge map, no marker, no grouping: the unit is the sentence span, so its id is a
    real spine id and ``merge_constituent_count`` is 1 for every row it produces. That
    makes the split-back and the fusion fallback unreachable for this arm rather than
    merely unused, so the correspondence is one to one by construction.

    English is skipped here exactly as it is in the merged arm: it is the CPU identity
    substage's pass-through and must never reach a GPU.
    """
    units: list[TranslationUnit] = []
    for row in rows:
        if row["lang"] == "en":
            continue
        member = _member(row, doc_text)
        if not member.text:
            # A zero-length span cannot be translated and would come back empty; the spine
            # should never contain one, so say so rather than emit a blank row.
            raise CorruptMergeMapError(
                f"{member.sentence_id}: empty sentence span [{row['start']}, {row['end']}) "
                "; the spine and documents.parquet are not from the same corpus build"
            )
        units.append(_unit_of([member], MARKER))
    if stats is not None and units:
        stats.emit("translate.sentence_units", float(len(units)), lang=units[0].members[0].lang)
    return units


def build_units(
    rows: Sequence[Mapping[str, Any]],
    doc_text: str,
    *,
    marker: str = MARKER,
    stats: Any = None,
) -> list[TranslationUnit]:
    """Group one document's merge-map rows into MT units, in order.

    A run of consecutive rows forms one unit when they carry the same ``merge_unit_id``.
    Every run is validated by :func:`_validate_unit`, since the map is a separate artifact
    and is checked rather than trusted, including a one-member run: a map that lost two of a
    unit's three rows would otherwise translate the survivor alone and never say so.

    One deliberate refusal: a constituent whose source already contains the marker declines
    the merge, and its whole run is broken into lone sentences (counted as
    ``translate.merge_declined_marker_in_source``). Escaping was rejected because an escape
    sequence's own translation has never been measured, whereas translating the constituents
    alone is exactly the pre-merge behaviour.
    """
    units: list[TranslationUnit] = []
    run: list[Mapping[str, Any]] = []
    seen_units: set[str] = set()

    def flush() -> None:
        if not run:
            return
        _validate_unit(run)
        members = [_member(r, doc_text) for r in run]
        if len(members) > 1 and any(contains_marker(m.text, marker) for m in members):
            if stats is not None:
                stats.emit("translate.merge_declined_marker_in_source", lang=members[0].lang)
            units.extend(_unit_of([m], marker) for m in members)
        else:
            units.append(_unit_of(members, marker))
        run.clear()

    for row in rows:
        if run and run[-1]["merge_unit_id"] != row["merge_unit_id"]:
            flush()
        if not run:
            uid = str(row["merge_unit_id"])
            if uid in seen_units:
                raise CorruptMergeMapError(
                    f"{uid}: reappears after its constituents ended at "
                    f"{row['sentence_id']}; a merge unit is a contiguous run of one "
                    "document's sentences, so a second run means the map is corrupt"
                )
            seen_units.add(uid)
        run.append(row)
    flush()

    if stats is not None:
        for unit in units:
            lang = unit.members[0].lang
            stats.emit("translate.merge_units", lang=lang)
            if unit.merged:
                stats.emit("translate.merge_units_merged", lang=lang)
    return units


def unit_row(
    unit: TranslationUnit,
    out_text: str,
    *,
    model_id: str,
    model_config_hash: str,
    merge_hash: str,
    created_at: str,
    marker: str = MARKER,
    stats: Any = None,
) -> dict[str, Any]:
    """Turn one unit's MT output into the single row it stores, markers intact.

    No split, no fuse, no scrub. The unit is what MT translated, so the unit is what gets
    stored: one row under the first constituent's real ``sentence_id``, carrying
    ``merge_constituent_count`` so reconciliation can call
    ``merge_join.split_translation(text_raw, n)`` from the row alone.

    ``split_translation`` is called here purely as a probe, so the marker-survival rate
    stays measurable on counters. It changes no stored byte; the decision it foreshadows
    belongs to reconciliation.
    """
    if unit.merged and stats is not None:
        segments = split_translation(out_text, len(unit.members), marker)
        stats.emit(
            "translate.merge_split_ok" if segments else "translate.merge_split_failed",
            lang=unit.members[0].lang,
        )
    return {
        "sentence_id": unit.unit_id,
        "document_id": unit.members[0].document_id,
        "variant": VARIANT,
        "text_raw": out_text,
        "source_lang": unit.src_lang,
        "merge_unit_id": unit.unit_id if unit.merged else None,
        "merge_constituent_count": len(unit.members),
        "model_id": model_id,
        "model_config_hash": model_config_hash,
        "merge_hash": merge_hash,
        "created_at": created_at,
    }


def buckets(items: Sequence[MtInput], tier: Tier) -> list[list[MtInput]]:
    """Length-bucket pre-tokenized items, by length only and never by language.

    Two steps, so the composition is reproducible rather than merely stable: items are put in
    the total order ``(len(tokens), sentence_id)``, then handed to the shared
    ``serving.batching.Batcher`` (whose own sort is stable, hence order-preserving on ties).
    Shuffling the input therefore cannot change a single bucket.

    ``tier`` carries the two semantic bounds from the hashed ``translation.config`` (token
    budget and sentence cap). The per-shape safety knob does not appear here: if
    it did, the same shard would produce different bytes on an A100 than on an H100 and the
    fairness invariant would break silently.
    """
    ordered = sorted(items, key=lambda it: (len(it.tokens), it.sentence_id))
    return Batcher(length_of=lambda it: len(it.tokens)).batch(ordered, tier)


# --------------------------------------------------------------------------- #
# Config reads (semantic knobs from the fairness-shared `translation` block).
# --------------------------------------------------------------------------- #
def _tcfg(cfg: RunConfig) -> dict[str, Any]:
    return dict(cfg.blocks.get("translation", {}).get("config", {}))


def _translation_hash(cfg: RunConfig) -> str:
    """The semantic translation hash; keys the queue, the artifact path and every row."""
    return all_hashes(cfg)["translation"]


def _merge_hash(cfg: RunConfig) -> str:
    """The semantic merge hash: the other half of this artifact's identity."""
    return all_hashes(cfg)["merge"]


def inventory_hash_of_node(inventory: Path) -> str:
    """The sentence-inventory key a reconciled ``final/<recon12>/`` node declares for itself.

    Read from the node's ``manifest.json`` (``preprocess.reconcile._write_manifest`` writes
    it), never inferred from the directory name and never computed from this arm's config.
    Both shortcuts are wrong for the same reason: the directory is named by ``recon12``,
    which hashes the whole ``reconcile`` block including the storage policy, while what this
    arm's output depends on is only which sentences exist. The two happen to coincide today,
    and a rule that is right by coincidence is the kind that stops being right silently.


    A node with no ``inventory_hash`` field predates the inventory key. It raises rather than
    falling back, because the fallback would be a guess at the key of a multi-GB GPU artifact:
    re-run ``reconcile``, a short CPU pass, so the node declares its own key.
    """
    manifest = inventory / "manifest.json"
    try:
        raw = manifest.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise FileNotFoundError(
            f"{manifest}: the low-tier arm keys its raw output by the inventory hash the "
            "reconciled node declares; without the manifest that key is unknowable"
        ) from exc
    # write_jsonl: one record per line, and this manifest is a single record.
    record = json.loads(raw.splitlines()[0])
    value = str(record.get(_MANIFEST_INVENTORY_KEY) or "")
    if not value:
        raise KeyError(
            f"{manifest} carries no {_MANIFEST_INVENTORY_KEY!r}; it predates the "
            "inventory/node key split. Re-run `reconcile` against this config "
            "so the node declares its own inventory key; do not substitute "
            "the directory name, which is `recon12` and names more than this arm depends on"
        )
    return value


def _now() -> str:
    """UTC ISO-8601 provenance stamp (seconds resolution)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_payload(shard: Path) -> dict[str, Any]:
    return json.loads(shard.read_text(encoding="utf-8").strip())


# Canonical row order is the work table's own row order, which is already
# ``(document order, sentence_index)``: the tables are written in corpus-file document order
# and densely indexed within a document, and this stage never re-orders documents. So both
# arms emit in the order they read, and concurrency cannot leak into the file (the MT rows
# are built from the ordered unit list, never from the completion order of the buckets).
#
# Not a sort by document_id: that string is UUID-shaped, so a lexicographic
# sort would re-order documents away from the stored order and break the property the merge
# hook depends on, namely every document being a contiguous row range in the concatenated
# artifact, in the same order as the sentence spine.




# --------------------------------------------------------------------------- #
# The saturate.StageAdapter pair.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _Shard:
    """One claimable work unit: a document-atomic row range over this arm's work table."""

    name: str
    payload: dict = field(default_factory=dict)


@dataclass(slots=True)
class _IdentityCtx:
    """The CPU substage's context; holds no model."""

    documents: Path
    sentences: Path
    model_config_hash: str
    merge_hash: str


@dataclass(slots=True)
class _TranslateCtx:
    """The GPU substage's resident context: one MT client and the semantic knobs."""

    mt: Any
    documents: Path
    merge_map: Path
    #: The semantic model identity stamped on each row. A plain string for a multilingual
    #: arm; a ``{source_lang: identity}`` mapping for a per-direction one, where the row's
    #: ``model_id`` must name the checkpoint that produced that row's English.
    model_id: str | Mapping[str, str]
    model_config_hash: str
    merge_hash: str
    target_lang: str = _DEFAULT_TARGET_LANG
    beam_size: int = _DEFAULT_BEAM_SIZE
    len_ratio_a: dict[str, float] = field(default_factory=dict)
    len_ratio_b: float = _DEFAULT_LEN_RATIO_B
    max_decoding_cap: int = _DEFAULT_MAX_DECODING_CAP
    max_input_length: int = _DEFAULT_MAX_INPUT_LENGTH
    #: Repetition guard, hashed as ``translation.config.no_repeat_ngram_size`` rather than
    #: left as a client-side default: it materially changes output text, so leaving it in a
    #: function signature would stop the config file being the complete run record.
    no_repeat_ngram_size: int = _DEFAULT_NO_REPEAT_NGRAM
    #: The merged-unit join marker. It changes the MT input and hence the bytes, so
    #: it is read from the fairness-shared ``translation.config``; ``§`` is the shipped one.
    marker: str = MARKER
    tier: Tier = field(default_factory=Tier)  # semantic bucket composition bounds
    # CT2's token-denominated `max_batch_size` clamp. Set to the semantic bucket budget, so
    # the guard is armed (an oversized bucket gets split rather than OOMing) but never
    # actually re-splits a conforming bucket.
    ct2_max_batch_tokens: int = _DEFAULT_BUCKET_TOKEN_BUDGET


class _OmtAdapterBase:
    """Shared path/shard/validate/merge machinery for the two OMT substages.

    They differ only in their work table (the English arm shards ``sentences.parquet``, the
    MT arm shards the merge map, which is co-ordered with it) and in the template they run
    on. Both write the same ``translations_raw`` schema under the same variant + hash node,
    into separate ``part``s so two substages never write one file.
    """

    stage_prefix = "translate_omt"
    template = "workqueue_worker_cpu.sbatch"
    part = "data"
    shards_key = "translate_shards"
    default_shards = _DEFAULT_TRANSLATE_SHARDS

    def __init__(
        self, cfg: RunConfig, *, base: str | os.PathLike[str] | None = None, stats: Any = None
    ) -> None:
        self.base = str(base) if base is not None else _DEFAULT_BASE
        self.model_config_hash = _translation_hash(cfg)
        self.merge_hash = _merge_hash(cfg)
        # Optional: translate an already-reconciled inventory instead of the spine.
        #
        # The spine is not what gets read at query time: passages are packed from
        # `final/<recon12>/sentences.parquet`, whose rows are the sentences that actually
        # get used. A fused final sentence is still a contiguous span (its constituents
        # were adjacent), so it translates as one slice of `documents.text` exactly like
        # any other row. Pointing the arm here means the unit translated is the unit used,
        # with no post-hoc join, no re-keying and no fusion handling anywhere.
        #
        # It is an execution path, not a semantic knob: it selects which existing artifact
        # to read and where to write, and changes no hashed value. It therefore comes from
        # the environment alongside WQ_DIR / PREPROCESS_ROLE (the same class of per-launch
        # wiring) rather than from the fairness-globbed config, and it cannot be derived
        # from this config's own hashes, because the target node is keyed by a different
        # translation hash (the arm that reconciled it), which is why it must be
        # named explicitly rather than computed.
        inv = os.environ.get("RAGTIME_INVENTORY_DIR", "").strip()
        self.inventory: Path | None = Path(inv) if inv else None
        # Hash-suffixed stage name: an edit to either semantic block resolves to an entirely
        # new queue subtree, so a stale `done/` can never be reused.
        self.stage = (
            f"{self.stage_prefix}_{self.model_config_hash[:_HASH_DIRLEN]}"
            f"_{self.merge_hash[:_HASH_DIRLEN]}"
        )
        if self.inventory is not None:
            # A distinct queue subtree: the same shard name over a different work table is
            # different work, and reusing the spine run's `done/` would silently skip it.
            self.stage = f"{self.stage}_inv{self.inventory.name[:_HASH_DIRLEN]}"
        self.stats: Statistics = stats if stats is not None else Statistics()
        # work() to validate() hand-off, same process and back to back in run_worker: the
        # expected row-id set for a shard.
        self._expected: dict[str, frozenset[str]] = {}

    # -- paths ---------------------------------------------------------- #
    def _layout(self, cfg: RunConfig) -> Layout:
        return Layout(
            run_dir=self.base,
            base=self.base,
            family=run_family(cfg),
            chunker_hash=all_hashes(cfg)["chunker"],
        )

    def documents_path(self, cfg: RunConfig) -> Path:
        return self._layout(cfg).documents_path()

    def sentences_path(self, cfg: RunConfig) -> Path:
        return self._layout(cfg).sentences_path()

    def merge_map_path(self, cfg: RunConfig) -> Path:
        """The merge map this arm's units come from (``Layout``, never hand-built)."""
        return self._layout(cfg).merge_map_path(self.merge_hash)

    def out_path(self, cfg: RunConfig) -> Path:
        if self.inventory is not None:
            # Outside the final node, keyed exactly like the NLLB counterpart: the same
            # `translations_raw/<variant>/<recipe12>-<input12>/` scheme, where the second
            # component names the work table this arm consumed. For NLLB that is the merge
            # map (`merge_hash`); for this arm it is the reconciled inventory, so it is
            # `inventory_hash`, read from the node's own manifest, because this arm's
            # config resolves to a different `translation` hash and therefore cannot compute
            # the value (the same reason `RAGTIME_INVENTORY_DIR` names the node explicitly).
            #
            # Outside the node on purpose: its NLLB counterpart sits outside `final/` and
            # survives any reconcile-side change, while a path inside the node would be
            # orphaned the moment `recon12` moved, which would make an otherwise-cheap
            # reconcile edit expensive. `inventory_hash` (not `recon12`) is the key because
            # that is what the bytes actually depend on: a storage-policy edit changes the
            # node but not one span.
            return self._layout(cfg).translations_raw_path(
                OPUS_VARIANT,
                self.model_config_hash,
                inventory_hash_of_node(self.inventory),
                part=self.part,
            )
        return self._layout(cfg).translations_raw_path(
            VARIANT, self.model_config_hash, self.merge_hash, part=self.part
        )

    def work_table(self, cfg: RunConfig) -> Path:
        """The table this arm shards (overridden by the MT arm)."""
        if self.inventory is not None:
            return self.inventory / "sentences.parquet"
        return self.sentences_path(cfg)

    # -- seed ----------------------------------------------------------- #
    def n_shards(self, cfg: RunConfig) -> int:
        return max(
            1,
            int(cfg.blocks.get("execution", {}).get(self.shards_key, self.default_shards)),
        )

    def shards(self, cfg: RunConfig) -> Iterator[_Shard]:
        """Contiguous, document-atomic row ranges over this arm's work table.

        No slice files are written, unlike the chunk stage's pre-split: the payload is the row
        range plus the matching ``documents.parquet`` range, and ``work`` reads exactly those
        row groups through the footer index.
        """
        table = self.work_table(cfg)
        ranges, boundary_docs = spine.plan_shards(table, n_shards=self.n_shards(cfg))
        aligned = spine.align_documents(self.documents_path(cfg), ranges, boundary_docs)
        _log.info(
            "preprocess.translate_omt.shards",
            stage=self.stage,
            table=str(table),
            shards=len(aligned),
        )
        for r in aligned:
            yield _Shard(name=r.name, payload=r.payload())

    # -- work helpers --------------------------------------------------- #
    def _write(self, shard: Path, rows: list[dict[str, Any]]) -> Path:
        out = saturate.shard_out_path(shard)
        write_parquet_stream(out, rows, schema=translation_raw_arrow_schema())
        self._expected[str(out)] = frozenset(r["sentence_id"] for r in rows)
        return out

    # -- validate / merge ----------------------------------------------- #
    def validate(self, path: Path) -> bool:
        """The shard's ``_SUCCESS`` plus "the file round-trips exactly the rows work built".

        It does not check size: a shard of an all-English document range legitimately emits
        zero MT rows (and vice-versa for the identity arm). What is checked is that the row
        ids on disk are exactly the set ``work`` emitted, with nothing duplicated and nothing
        dropped by a truncated write.
        """
        if not (path.exists() and is_done(path)):
            return False
        expected = self._expected.pop(str(path), None)
        if expected is None:
            return True  # hand-built ctx / re-validation: the marker is the gate
        got = [
            r["sentence_id"]
            for batch in iter_parquet_batches(path, columns=["sentence_id"])
            for r in batch
        ]
        ok = len(got) == len(expected) and frozenset(got) == expected
        if not ok:
            _log.warning(
                "preprocess.translate_omt.validate.mismatch",
                stage=self.stage,
                path=str(path),
                rows=len(got),
                expected=len(expected),
            )
        return ok

    def merge(self, cfg: RunConfig, shard_paths: Sequence[Path]) -> None:
        """Fold the per-shard tables into one raw-translation artifact, Arrow-native.

        Shard names are ``rows_<start:012d>_<end:012d>``, so sorting them restores work-table
        row order; each shard is internally in that same order (see the canonical row order
        note above) and shards are document-atomic, so every document stays a contiguous row
        range in the merged table, which makes reconciliation's per-document read a
        merge-join over pruned row groups instead of a hash join over tens of millions of
        rows.
        """
        out = self.out_path(cfg)
        concat_parquet(out, sorted(shard_paths), schema=translation_raw_arrow_schema())
        _log.info(
            "preprocess.translate_omt.merge.done",
            stage=self.stage,
            shards=len(shard_paths),
            path=str(out),
        )


class OmtIdentityAdapter(_OmtAdapterBase):
    """English pass-through as its own CPU substage, never a GPU short-circuit.

    Over a third of the corpus's sentences are English and need no MT. Routed to the CPU
    template, it imports no model: ``bringup`` touches neither ``ctranslate2`` nor ``torch``.
    English is never merged, since the merge map carries no English rows, so every row here is
    a lone sentence with a null ``merge_unit_id``, which is also why this arm shards the
    sentence table rather than the map.
    """

    stage_prefix = "translate_omt_identity"
    template = "workqueue_worker_cpu.sbatch"
    part = "identity"
    shards_key = "translate_identity_shards"
    default_shards = _DEFAULT_IDENTITY_SHARDS

    def bringup(self, cfg: RunConfig) -> _IdentityCtx:
        """No model and no GPU: just the two table paths and the provenance hashes."""
        return _IdentityCtx(
            documents=self.documents_path(cfg),
            sentences=self.sentences_path(cfg),
            model_config_hash=self.model_config_hash,
            merge_hash=self.merge_hash,
        )

    def work(self, ctx: _IdentityCtx, shard: Path) -> Path:
        """This shard's English sentences, copied verbatim out of their document's text.

        Emitted in the sentence table's own row order, which is already canonical and never
        re-sorted (see the canonical row order note above).
        """
        r = spine.ShardRange.from_payload(_read_payload(shard))
        texts = spine.load_document_texts(ctx.documents, r)
        created_at = _now()
        rows: list[dict[str, Any]] = []
        for row in spine.iter_row_range(
            ctx.sentences, r.row_start, r.row_end, columns=list(_SENTENCE_COLUMNS)
        ):
            if row["lang"] != "en":
                continue
            doc_text = texts[row["document_id"]]
            rows.append(
                {
                    "sentence_id": row["sentence_id"],
                    "document_id": row["document_id"],
                    "variant": VARIANT,
                    # identity: the native English span, verbatim. It can never carry a join
                    # marker, since English is never merged, so raw and final coincide here.
                    "text_raw": doc_text[int(row["start"]) : int(row["end"])],
                    "source_lang": _FLORES["en"],
                    "merge_unit_id": None,
                    "merge_constituent_count": 1,
                    "model_id": IDENTITY_MODEL_ID,
                    "model_config_hash": ctx.model_config_hash,
                    "merge_hash": ctx.merge_hash,
                    "created_at": created_at,
                }
            )
        self.stats.emit("translate.identity_rows", float(len(rows)), lang="en")
        return self._write(shard, rows)


class OmtTranslateAdapter(_OmtAdapterBase):
    """The non-English units through the resident CTranslate2 replica(s).

    A "unit" is whatever ``translation.config.translate_units`` says it is: the merge
    map's ``§``-joined groups (``merged``, the NLLB arm) or one sentence span each
    (``sentences``, the OPUS-MT arm). The mode picks the work table too, since the merge map
    is not read at all in sentence mode, which is why it is resolved once in ``__init__``.
    """

    stage_prefix = "translate_omt"
    template = "workqueue_worker.sbatch"
    part = "data"
    shards_key = "translate_shards"
    default_shards = _DEFAULT_TRANSLATE_SHARDS
    #: The ``execution`` key naming the ShapeSet this substage's replica is allocated
    #: against. The value is never hardcoded here: a GPU shape is a machine fact and a run
    #: choice, so it is read from config like every other one.
    shape_key = "translate_shape_key"

    def __init__(
        self, cfg: RunConfig, *, base: str | os.PathLike[str] | None = None, stats: Any = None
    ) -> None:
        super().__init__(cfg, base=base, stats=stats)
        self.shape = str(cfg.blocks.get("execution", {}).get(self.shape_key, "") or "")
        self.units = str(_tcfg(cfg).get("translate_units", _DEFAULT_TRANSLATE_UNITS))
        if self.units not in _UNIT_MODES:
            raise ValueError(
                f"translation.config.translate_units {self.units!r} is not one of "
                f"{sorted(_UNIT_MODES)}; the MT call's granularity must be stated, never "
                "inferred from the model name"
            )

    @property
    def sentence_mode(self) -> bool:
        """True when one MT call is one sentence span (no merge map, no marker)."""
        return self.units == UNITS_SENTENCES

    def work_table(self, cfg: RunConfig) -> Path:
        """The table this arm shards, which follows from the unit mode.

        ``merged``: the merge map, whose rows are the units and which is co-ordered with
        the sentence spine, so a contiguous range of it is a document-atomic shard.
        ``sentences``: the spine itself, for the same reason, and the merge map
        is then never opened, which is what "this arm does not merge" has to mean.
        """
        if self.inventory is not None:
            # Translating an already-reconciled inventory is inherently one to one, since
            # every row is one contiguous span, so the merge map is never opened.
            return self.inventory / "sentences.parquet"
        return self.sentences_path(cfg) if self.sentence_mode else self.merge_map_path(cfg)

    @staticmethod
    def _probe_langs(tc: Mapping[str, Any]) -> list[str]:
        """The source languages bring-up must prove the tokenizer contract on.

        ``source_langs`` when the config declares it (it always does), else the single
        configured probe language, so a config that has not been updated still gets the
        old, one-language behaviour rather than silently probing nothing.
        """
        langs = [str(x) for x in (tc.get("source_langs") or [])]
        return langs or [str(tc.get("marker_probe_lang", _DEFAULT_MARKER_PROBE_LANG))]

    def bringup(self, cfg: RunConfig) -> _TranslateCtx:
        """Stand up the resident MT replica once, from the serving singleton registry.

        The model comes from ``serving.registry.build_clients``, since a stage never
        instantiates one, and the import is function-local so this module stays CUDA-free.
        """
        from ragtime.serving.registry import build_clients

        tc = _tcfg(cfg)
        mt = build_clients(cfg).mt
        budget = int(tc.get("bucket_token_budget", _DEFAULT_BUCKET_TOKEN_BUDGET))
        marker = str(tc.get("merge_marker", MARKER))
        # Check the join marker is still one atomic piece in this tokenizer, before a single
        # unit is built. A tokenizer/model change that fused or dropped it would otherwise
        # degrade silently: every merged unit would come back unsplittable and the
        # granularity loss would show up only as a counter nobody read.
        #
        # Every configured direction is probed, not just one: a per-direction arm has a
        # different sentencepiece model per language, so a marker that is atomic in
        # opus-mt-es-en tells you nothing about opus-mt-zh-en. On the single-checkpoint
        # NLLB arm the extra probes hit the same cached tokenizer and cost nothing.
        #
        # Skipped entirely in sentence mode: no unit ever has two constituents, so no
        # marker is inserted, nothing is split back, and asserting a property of a string
        # this arm never builds would be pointless, and would gratuitously fail a perfectly
        # usable checkpoint whose vocabulary happens not to contain `§`.
        if not self.sentence_mode:
            for lang in self._probe_langs(tc):
                mt.assert_marker_atomic(marker, lang)
        return _TranslateCtx(
            mt=mt,
            documents=self.documents_path(cfg),
            merge_map=self.work_table(cfg),
            model_id=tc.get("omt_model", getattr(mt, "model", "")),
            model_config_hash=self.model_config_hash,
            merge_hash=self.merge_hash,
            target_lang=str(tc.get("target_lang", _DEFAULT_TARGET_LANG)),
            beam_size=int(tc.get("beam_size", _DEFAULT_BEAM_SIZE)),
            len_ratio_a={str(k): float(v) for k, v in dict(tc.get("len_ratio_a", {})).items()},
            len_ratio_b=float(tc.get("len_ratio_b", _DEFAULT_LEN_RATIO_B)),
            max_decoding_cap=int(tc.get("max_decoding_cap", _DEFAULT_MAX_DECODING_CAP)),
            max_input_length=int(tc.get("max_input_length", _DEFAULT_MAX_INPUT_LENGTH)),
            no_repeat_ngram_size=int(
                tc.get("no_repeat_ngram_size", _DEFAULT_NO_REPEAT_NGRAM)
            ),
            tier=Tier(
                token_budget=budget,
                max_items=int(
                    tc.get("max_sentences_per_bucket", _DEFAULT_MAX_SENTENCES_PER_BUCKET)
                ),
            ),
            ct2_max_batch_tokens=budget,
            marker=marker,
        )

    def units_for_shard(self, ctx: _TranslateCtx, r: spine.ShardRange) -> list[TranslationUnit]:
        """This shard's MT units, from whichever work table the unit mode selected.

        ``merged``: merge-map rows, grouped per document and validated against the spine.
        ``sentences``: spine rows, one unit each, English skipped.
        """
        texts = spine.load_document_texts(ctx.documents, r)
        units: list[TranslationUnit] = []
        columns = _SENTENCE_COLUMNS if self.sentence_mode else _MAP_COLUMNS
        rows = spine.iter_row_range(ctx.merge_map, r.row_start, r.row_end, columns=list(columns))
        for doc_id, doc_rows in spine.group_by_document(rows):
            doc_text = texts.get(doc_id)
            if doc_text is None:
                raise LookupError(
                    f"{doc_id}: no row in documents.parquet for a document in this shard's "
                    "work-table range; the tables are not from the same corpus build"
                )
            if self.sentence_mode:
                units.extend(build_sentence_units(doc_rows, doc_text, stats=self.stats))
            else:
                units.extend(build_units(doc_rows, doc_text, marker=ctx.marker, stats=self.stats))
        return units

    def work(self, ctx: _TranslateCtx, shard: Path) -> Path:
        """Read, unit, tokenize once, order, bucket, translate, then write the raw rows.

        One row per unit, markers intact. Nothing is split, fused or scrubbed here: the
        stage's whole job is to spend the GPU hours once and record exactly what came back.
        """
        r = spine.ShardRange.from_payload(_read_payload(shard))
        units = self.units_for_shard(ctx, r)
        items = [
            MtInput(
                sentence_id=u.unit_id,
                src_lang=u.src_lang,
                text=u.text,
                tokens=ctx.mt.tokenize(
                    u.text,
                    u.src_lang,
                    max_input_length=ctx.max_input_length,
                    stats=self.stats,
                ),
            )
            for u in units
        ]
        texts: dict[str, str] = {}
        for bucket in buckets(items, ctx.tier):
            texts.update(
                ctx.mt.translate_batch(
                    bucket,
                    target_lang=ctx.target_lang,
                    beam_size=ctx.beam_size,
                    len_ratio_a=ctx.len_ratio_a,
                    len_ratio_b=ctx.len_ratio_b,
                    max_decoding_cap=ctx.max_decoding_cap,
                    max_input_length=ctx.max_input_length,
                    no_repeat_ngram_size=ctx.no_repeat_ngram_size,
                    tier=Tier(
                        token_budget=ctx.ct2_max_batch_tokens, max_items=ctx.tier.max_items
                    ),
                    stats=self.stats,
                )
            )
        created_at = _now()
        # Rows come from the ordered unit list, never from the translated dict, so bucket
        # completion order cannot reach the file (see the canonical row order note above).
        rows = [
            unit_row(
                u,
                texts[u.unit_id],
                model_id=_model_id_for(ctx.model_id, u.src_lang),
                model_config_hash=ctx.model_config_hash,
                merge_hash=ctx.merge_hash,
                created_at=created_at,
                marker=ctx.marker,
                stats=self.stats,
            )
            for u in units
        ]
        for unit in units:
            self.stats.emit(
                "translate.sentences_translated",
                float(len(unit.members)),
                lang=unit.members[0].lang,
            )
        return self._write(shard, rows)
