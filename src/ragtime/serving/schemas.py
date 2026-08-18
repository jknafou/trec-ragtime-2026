"""The fixed guided-decoding JSON Schema set, compiled once and reused everywhere.

``compile_schemas()`` returns a module-cached :class:`SchemaSet` of the schemas the LLM
emits under: the ``{rationale, action}`` discriminated union (search, submit_claim,
submit_answer, abstain), ``{rationale, nuggets}``, the coverage-audit gap detector, the
loop aggregator, and the dedup gate.

There are two emission shapes, pinned per schema, because vLLM's native field and OpenAI's
``response_format`` differ on optionality. ``llm.guided_json`` honours the per-schema
choice rather than one global setting:

* ``structured_outputs``, the native vLLM field, allows genuinely optional keys. It is used
  for the ``{rationale, action}`` union, where only ``rationale`` and ``action`` are
  required and a key meaningful for a single ``action`` literal, such as ``passage_id`` for
  ``submit_claim``, is simply absent for the others. The semantics are "key present means
  meaningful", so a consumer never has to check an inapplicable key for ``None``.

* ``response_format`` json_schema with ``strict: true`` requires every property to appear in
  ``required``, so an inapplicable field is expressed as an explicit ``null`` and the
  property type becomes ``["string", "null"]``. It is used for the flat schemas
  (``nuggets``, ``coverage_audit``, ``aggregator``, ``dedup``), where the semantics are the
  opposite: the key is always present and ``null`` means inapplicable.

Schemas are flat literal dicts with no ``$ref`` and ``additionalProperties: false``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CompiledSchema",
    "SchemaSet",
    "action_schema_for",
    "actions_of",
    "compile_schemas",
]

# Emission-shape markers (read by ``llm.guided_json`` to build the right request).
MODE_STRUCTURED = "structured_outputs"  # native vLLM/xgrammar; true-optional keys
MODE_RESPONSE_FORMAT = "response_format"  # OpenAI json_schema strict; nullable-not-absent


# The fixed schema literals.
#
# `{rationale, action}` is the RAG-loop controller union, discriminated by the `action` enum
# with per-action optional fields in a flat shape the grammar backend handles well: a
# `search` payload simply omits the submit_claim and submit_answer keys.
#
# The bounds below are the only brake on generation length. `max_tokens` is never sent,
# because a cap on a thinking model truncates JSON mid-span, so for a grammar-constrained
# call the schema is the only thing that can stop a generation, and vLLM does enforce
# `maxItems` and `maxLength` in both emission modes.
#
# Bounding fields one at a time does not work: bounding the array leaves a runaway free to
# grow in `rationale`, and bounding that leaves `nugget_id`. Every string and array below is
# therefore bounded. These are safety bounds, set far above any well-formed value so they
# can never bind a good generation, and they exist only to make a pathological one
# terminable. `_ACTION_SCHEMA` matters most, since it runs k-parallel per round, so an
# unbounded generation there multiplies by k.
MAX_RATIONALE_CHARS = 2000     # a paragraph of reasoning; well-formed ones are far shorter
MAX_QUESTION_CHARS = 400       # a single-sentence nugget question
MAX_ID_CHARS = 128             # "2061#n2"-shaped ids, with room to spare
MAX_ENUMISH_CHARS = 32         # short controlled tokens (a language code, an action name)
MAX_CLAIM_CHARS = 2000         # one claim sentence
MAX_SPAN_CHARS = 4000          # a verbatim quoted span, the longest legitimate free text
MAX_ANSWER_CHARS = 4000        # one answer sentence + its support
MAX_QUERY_CHARS = 1000         # one search query string
MAX_LIST_ITEMS = 64            # any generated list (nuggets, new_nuggets, selected)
MAX_REASON_CHARS = 1000        # the dedup gate's short justification


_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rationale": {"type": "string", "maxLength": MAX_RATIONALE_CHARS},
        "action": {
            "type": "string",
            "enum": ["search", "submit_claim", "submit_answer", "abstain"],
        },
        # search
        "query": {"type": "string", "maxLength": MAX_QUERY_CHARS},
        "lang": {"type": "string", "maxLength": MAX_ENUMISH_CHARS},
        # submit_claim: `answer` and `sentence` ride inside this action, alongside the span
        # that grounds them, rather than in a separate step. Emitting the short value only at
        # close would leave a multi-valued answer with no per-value short form, so
        # `common.schemas.Answer.answer` would hold a whole claim sentence.
        "claim": {"type": "string", "maxLength": MAX_CLAIM_CHARS},
        "passage_id": {"type": "string", "maxLength": MAX_ID_CHARS},
        "span": {"type": "string", "maxLength": MAX_SPAN_CHARS},
        "answer": {"type": "string", "maxLength": MAX_ANSWER_CHARS},
        "sentence": {"type": "string", "maxLength": MAX_ANSWER_CHARS},
        # submit_answer carries no value of its own; it is the model saying it is done.
        "nugget_id": {"type": "string", "maxLength": MAX_ID_CHARS},
    },
    "required": ["rationale", "action"],
}

#: The action union's own enum, named so `action_schema_for` validates a menu against the
#: SCHEMA rather than against a second hand-written list that could drift from it.
_ACTION_ENUM: tuple[str, ...] = tuple(_ACTION_SCHEMA["properties"]["action"]["enum"])

#: The fields an action cannot be useful without, applied by :func:`action_schema_for` only
#: when the menu has narrowed to that single action. They are keyed by action name rather than
#: derived, because which fields an action needs is a contract with ``pipeline.rag_loop`` and
#: not a property the flat union can express.
#:
#: ``submit_answer`` is absent, since it carries no payload of its own; the short
#: answer and the report sentence ride with each claim, so requiring anything here would make
#: the terminal turn unemittable. ``abstain`` likewise carries nothing. ``search`` is present
#: because a search with no query is already rejected as a malformed action, and requiring the
#: field turns a wasted turn into an unrepresentable one.
_SINGLE_ACTION_REQUIRED: dict[str, tuple[str, ...]] = {
    "search": ("query",),
    "submit_claim": ("claim", "passage_id", "span"),
}

#: Upper bound, in characters, on the fields a narrowed menu requires. It is applied together
#: with ``required``, and the pairing is the point: requiring a field without bounding it turns
#: a cheap failure into an expensive one, because a model forced to emit a span it does not
#: have can decode without limit when ``max_tokens`` is never set.
#:
#: The values are derived rather than picked. Across around ninety real spans, committed and
#: attempted, the longest was 370 characters, so 1500 is four times the longest ever produced
#: and still covers quoting an entire packed passage. ``claim`` and ``passage_id`` are bounded
#: on the same principle, since any required string is a potential runaway channel.
#:
#: This does not change the commit gate: claim commit still performs the same symmetric NFC
#: substring match, and a span at the bound is checked exactly as before.
_SINGLE_ACTION_MAXLEN: dict[str, int] = {
    "span": 1500,
    "claim": 2000,
    "passage_id": 200,
    "query": 500,
}

# `{rationale, nuggets}` is the decomposition output. It is strict, so every property is
# required and `nugget_id` is nullable, minted downstream, rather than absent.
#: Bounds on the nuggets schema. All three are safety bounds rather than run choices: they sit
#: far above any legitimate value, so they cannot bind a well-formed generation, and exist only
#: to make a runaway terminable.
#:
#: ``max_tokens`` is never sent, because a cap on a thinking model truncates JSON mid-object,
#: so for a grammar-constrained call the schema is the only thing that can bound output length.
#: Without ``maxItems`` the model can emit nuggets until it exhausts the context window.
#: Bounding the array alone is not enough either, since the runaway is not the item count but
#: the unbounded strings, which is why the rationale and question are bounded as well.
#:
#: 64 items is roughly 2.7 times the widest legitimate ``decomposition.k_band`` row. This is a
#: module constant rather than a config leaf because ``compile_schemas()`` takes no config and
#: is an identity-stable singleton the k RAG loops share; making the bound track ``k_band`` per
#: topic would be a deliberate API change.
NUGGETS_MAX_ITEMS = 64
NUGGET_QUESTION_MAX_CHARS = 400
NUGGETS_RATIONALE_MAX_CHARS = 2000

_NUGGETS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rationale": {"type": "string", "maxLength": NUGGETS_RATIONALE_MAX_CHARS},
        "nuggets": {
            "type": "array",
            "maxItems": NUGGETS_MAX_ITEMS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "nugget_id": {"type": ["string", "null"], "maxLength": MAX_ID_CHARS},
                    "question": {"type": "string", "maxLength": NUGGET_QUESTION_MAX_CHARS},
                    # `weight` must be declared here or it is unemittable:
                    # `additionalProperties: false` under strict structured output means the
                    # model cannot return a key this schema omits, and the self-critique prompt
                    # asks for a weight in [0.0, 1.0] on every surviving nugget. Without it,
                    # every round-0 nugget would score 0.0, sorting the facets taken directly
                    # from the request below the audit-born ones and making the whole seed bank
                    # structurally non-vital. It is nullable, as in `_COVERAGE_SCHEMA`, so a
                    # model with no opinion can say so rather than invent a number.
                    "weight": {"type": ["number", "null"]},
                },
                "required": ["nugget_id", "question", "weight"],
            },
        },
    },
    "required": ["rationale", "nuggets"],
}

# The coverage-audit delta carries mark-covered, gap-detect and prune in one object.
#
# There is no model-declared `saturated` flag. The stop rule is novelty
# saturation over `min_new` and `low_streak`, precisely because the mark-covered judge is an
# LLM that can over-claim coverage and premature stopping is the main failure mode of these
# systems. Leaving a `saturated` flag in the grammar would put a future caller one `if` away
# from ending the coverage loop on the model's own say-so.
#
# Mark-covered rides in the same object as the rest of the delta, on the three-way
# full/partial/none scale the track's own evaluator uses. A separate call per nugget would
# multiply the shared vLLM's round cost by the bank size.
#
# The AND/OR aggregation boundary is a prompt instruction rather than a schema shape, since
# `common.Nugget` carries no member list for an AND nugget. The code-side guard against an
# over-claiming judge lives in `pipeline.decompose.coverage_audit.apply_delta`, where a nugget
# with zero committed answers never flips to answered whatever the label says.
#: Safety bound on the audit's per-nugget label list and its prune list, sized above the bank a
#: saturating run reaches and only open nuggets are judged. Like `NUGGETS_MAX_ITEMS` it sits far
#: enough above any legitimate value that it cannot bind a good generation and exists only so a
#: pathological one terminates. An open nugget the model omits simply stays open, which is the
#: conservative direction.
COVERAGE_MAX_ITEMS = 256
_COVERAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rationale": {"type": "string", "maxLength": MAX_RATIONALE_CHARS},
        "coverage": {
            "type": "array",
            "maxItems": COVERAGE_MAX_ITEMS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "nugget_id": {"type": "string", "maxLength": MAX_ID_CHARS},
                    "coverage": {"type": "string", "enum": ["full", "partial", "none"]},
                },
                "required": ["nugget_id", "coverage"],
            },
        },
        "add": {
            "type": "array",
            "maxItems": MAX_LIST_ITEMS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question": {"type": "string", "maxLength": MAX_QUESTION_CHARS},
                    # Nullable-not-absent, per this family's strict convention: a
                    # request-driven gap genuinely has no trigger passage, and `null` says so
                    # where an omitted key would be indistinguishable from a model that forgot.
                    "trigger_passage_id": {
                        "type": ["string", "null"],
                        "maxLength": MAX_ID_CHARS,
                    },
                    "weight": {"type": ["number", "null"]},
                },
                "required": ["question", "trigger_passage_id", "weight"],
            },
        },
        "prune": {
            "type": "array",
            "maxItems": COVERAGE_MAX_ITEMS,
            "items": {"type": "string", "maxLength": MAX_ID_CHARS},
        },
    },
    "required": ["rationale", "coverage", "add", "prune"],
}

# Loop aggregator.
_AGGREGATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rationale": {"type": "string", "maxLength": MAX_RATIONALE_CHARS},
        "selected": {"type": "array", "maxItems": MAX_LIST_ITEMS,
                     "items": {"type": "string", "maxLength": MAX_ID_CHARS}},
    },
    "required": ["rationale", "selected"],
}

# The dedup gate, a terminal confirmation step. It is strict and nullable: `reason` is
# required but may be null when no reason applies, which pins the "always present, null means
# inapplicable" convention for the flat family.
_DEDUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "duplicate": {"type": "boolean"},
        "paraphrase_match": {"type": "boolean"},
        "entity_match": {"type": "boolean"},
        "reason": {"type": ["string", "null"], "maxLength": MAX_REASON_CHARS},
    },
    "required": ["duplicate", "paraphrase_match", "entity_match", "reason"],
}


# `{rationale, on_topic}` is the decompose admission gate: a filter over already-generated
# candidate nuggets rather than a generator. Generation and admission are separate calls, so a
# nugget that should not exist is dropped rather than never proposed.
_ON_TOPIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        # A binary gate needs only a short justification, so this takes the dedup gate's bound
        # rather than the paragraph-sized bound the generative schemas use.
        "rationale": {"type": "string", "maxLength": MAX_REASON_CHARS},
        "on_topic": {"type": "boolean"},
    },
    "required": ["rationale", "on_topic"],
}


@dataclass(frozen=True)
class CompiledSchema:
    """One compiled schema plus the emission shape ``guided_json`` must use for it.

    ``mode`` is :data:`MODE_STRUCTURED` (true-optional keys, for the action union)
    or :data:`MODE_RESPONSE_FORMAT` (strict json_schema, nullable-not-absent).
    """

    name: str
    schema: Mapping[str, Any]
    mode: str


@dataclass(frozen=True)
class SchemaSet:
    """The fixed set of compiled schemas that ``compile_schemas`` caches."""

    action: CompiledSchema
    nuggets: CompiledSchema
    coverage_audit: CompiledSchema
    aggregator: CompiledSchema
    dedup: CompiledSchema
    on_topic: CompiledSchema
    claim_importance: CompiledSchema

    def by_name(self) -> dict[str, CompiledSchema]:
        """Name -> compiled schema, for iteration in tests and callers."""
        return {
            "action": self.action,
            "nuggets": self.nuggets,
            "coverage_audit": self.coverage_audit,
            "aggregator": self.aggregator,
            "dedup": self.dedup,
            "on_topic": self.on_topic,
            "claim_importance": self.claim_importance,
        }


_CACHE: SchemaSet | None = None


# `{rationale, coverage}` is the per-(question, claim) claim-importance judge used by the
# citation scorer.
#
# The full/partial/none enum is the same vocabulary the coverage audit uses, since
# `importance.COVERAGE_TO_IMPORTANCE` maps it to {1.0, 0.5, 0.0} and one notion of "how much of
# this nugget is answered" is better than two that drift. `importance.claim_importance` raises
# on a value outside the enum rather than defaulting, so an unapplied grammar surfaces instead
# of inventing a score.
#
# `rationale` is required and comes first, so the judge states its reading before committing to
# a label, matching every other judge in the project.
_CLAIM_IMPORTANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rationale": {"type": "string", "maxLength": MAX_REASON_CHARS},
        "coverage": {"type": "string", "enum": ["full", "partial", "none"]},
    },
    "required": ["rationale", "coverage"],
}


def compile_schemas() -> SchemaSet:
    """Return the module-cached :class:`SchemaSet`, compiled once and reused.

    Two calls return the same object, so a stage compiles the fixed schema set once and the
    k RAG loops share it with no per-call recompile.
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = SchemaSet(
            action=CompiledSchema("rag_action", _ACTION_SCHEMA, MODE_STRUCTURED),
            nuggets=CompiledSchema("nuggets", _NUGGETS_SCHEMA, MODE_RESPONSE_FORMAT),
            coverage_audit=CompiledSchema("coverage_audit", _COVERAGE_SCHEMA, MODE_RESPONSE_FORMAT),
            aggregator=CompiledSchema("aggregator", _AGGREGATOR_SCHEMA, MODE_RESPONSE_FORMAT),
            dedup=CompiledSchema("dedup_and_gate", _DEDUP_SCHEMA, MODE_RESPONSE_FORMAT),
            on_topic=CompiledSchema("on_topic_gate", _ON_TOPIC_SCHEMA, MODE_RESPONSE_FORMAT),
            claim_importance=CompiledSchema(
                "claim_importance", _CLAIM_IMPORTANCE_SCHEMA, MODE_RESPONSE_FORMAT
            ),
        )
    return _CACHE


# Per-menu action schemas: the grammar-level phase gate.
#
#: Cache keyed by the menu tuple. The schema is one of a small fixed set (gather,
#: gather plus abstain, forced terminal), each compiled once by the grammar backend and
#: reused, so changing the exposed actions mid-loop is just picking which precompiled schema to
#: pass on that request, and the KV-cache prefix stays shared across turns.
#:
#: A real loop can produce at most five distinct menus, so the cache is bounded by construction.
#: It is keyed by the menu rather than by a phase name because the menu is the grammar, and two
#: phases with the same menu must share one compiled schema.
_ACTION_MENU_CACHE: dict[tuple[str, ...], CompiledSchema] = {}


def action_schema_for(menu: Sequence[str]) -> CompiledSchema:
    """Return the ``{rationale, action}`` union with ``action`` narrowed to ``menu``.

    This is where the phase gate becomes unrepresentable rather than merely checked: an action
    outside the menu is not in the enum, so grammar-constrained decoding cannot emit it. The
    loop re-asserts it in code anyway, through ``phase_gate.assert_allowed``, because a
    structured-output request can be silently ignored on this stack and a bypass raises nothing.

    Only the enum changes. Every other property stays present, because the union is deliberately
    flat: per-action fields are optional, so narrowing the enum is sufficient and no per-menu
    property surgery is needed. The schema name carries the menu, so a log line says which
    grammar a turn ran under.
    """
    key = tuple(menu)
    if not key:
        raise ValueError("action menu is empty: a turn must offer at least one action")
    unknown = [a for a in key if a not in _ACTION_ENUM]
    if unknown:
        raise ValueError(f"unknown action(s) {unknown}; the union declares {_ACTION_ENUM}")
    hit = _ACTION_MENU_CACHE.get(key)
    if hit is not None:
        return hit
    schema = dict(_ACTION_SCHEMA)
    props = dict(schema["properties"])
    props["action"] = {"type": "string", "enum": list(key)}
    schema["properties"] = props
    # When the menu is exactly one action, that action's own fields become required too.
    #
    # The union is flat and every per-action field is optional, which is right while the menu is
    # ambiguous, since `search` must not be forced to carry a `span`. Once the menu has
    # collapsed to a single action there is no ambiguity left, and leaving the fields optional
    # makes an empty action grammar-legal: a `submit_claim` with no claim, passage id or span
    # validates, reaches the commit gate and is rejected on its first precondition, and the
    # retry re-decodes against the same permissive grammar.
    #
    # Narrowing the enum makes the wrong action unrepresentable; this makes an unusable instance
    # of the right action unrepresentable, which is the same idea one level down.
    #
    # `minLength: 1` accompanies `required`, because an empty string is the same failure as an
    # absent key: the commit gate's first check is `if not span`, which both satisfy.
    #
    # This does not relax the commit gate. Every check still runs, unchanged, on the same
    # NFC-symmetric comparison; it only stops the model spending a generation on an object that
    # could never have passed them.
    required = list(schema.get("required") or ())
    if len(key) == 1:
        needed = _SINGLE_ACTION_REQUIRED.get(key[0], ())
        if needed:
            required = [*required, *[f for f in needed if f not in required]]
            props = dict(props)
            for f in needed:
                if isinstance(props.get(f), dict) and props[f].get("type") == "string":
                    # `minLength` and `maxLength` together: an empty field and an unbounded
                    # one are both failures, and fixing only the first trades a fast failure
                    # for a slow one. The base union already bounds every string, so this
                    # tightens an existing bound rather than introducing the only one.
                    bound: dict[str, int] = {"minLength": 1}
                    if f in _SINGLE_ACTION_MAXLEN:
                        bound["maxLength"] = _SINGLE_ACTION_MAXLEN[f]
                    props[f] = {**props[f], **bound}
            schema["properties"] = props
    schema["required"] = required

    # A multi-action menu becomes a `oneOf` of the per-action branches, so the same guarantee
    # reaches the first attempt and not only the retry. Most empty-span claims arrive on the
    # first attempt under a multi-action menu, so narrowing only the single-action path fixes
    # a path that already worked.
    #
    # JSON Schema's conditional (`if`/`then`) is the natural way to say "span is required when
    # action is submit_claim", but this stack accepts it with HTTP 200 and enforces nothing:
    # probed against the real compiled schema with an adversarial prompt and a permissive
    # control, `if`/`then` and `allOf` plus `if`/`then` both left the span empty in every
    # sample, while `oneOf` branches enforced it in every sample. `oneOf` is therefore the only
    # spelling used here.

    #
    # Each branch is exactly the single-action schema for its own action, so the two paths
    # cannot drift: `submit_claim` requires a bounded claim, passage id and span, `search`
    # requires a query, and an action with no required fields contributes an unchanged branch.
    if len(key) > 1 and any(_SINGLE_ACTION_REQUIRED.get(a) for a in key):
        schema = {"oneOf": [dict(action_schema_for((a,)).schema) for a in key]}

    compiled = CompiledSchema(f"rag_action[{'|'.join(key)}]", schema, MODE_STRUCTURED)
    _ACTION_MENU_CACHE[key] = compiled
    return compiled


def actions_of(schema: CompiledSchema | Mapping[str, Any]) -> tuple[str, ...]:
    """Return the action menu a compiled action schema offers, whatever shape it has.

    There is one owner for "which menu is this", because there are two shapes. A single-action
    menu is a plain object whose ``action`` enum names it, while a multi-action menu is a
    ``oneOf`` of per-action branches with no top-level ``properties`` at all, so a caller
    reaching in with ``schema["properties"]["action"]["enum"]`` breaks on the second shape.

    Reading the menu is not a test-only need: a log line naming the grammar a turn ran under,
    and any assertion that the phase gate reached the grammar, both want it. It therefore lives
    next to the code that builds the shapes and cannot drift from them.
    """
    raw = schema.schema if isinstance(schema, CompiledSchema) else schema
    branches = raw.get("oneOf")
    if branches:
        out: list[str] = []
        for branch in branches:
            for action in branch.get("properties", {}).get("action", {}).get("enum", ()):
                if action not in out:
                    out.append(str(action))
        return tuple(out)
    return tuple(raw.get("properties", {}).get("action", {}).get("enum", ()))
