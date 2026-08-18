"""``grow_nuggets``: one operation, parameterized by round.

Decompose and the coverage audit are the same function with one conditional on ``round``: same
contract, same schema, same model, same post-processing, with round 0 simply being "round r with
an empty bank and no passages". Keeping it one function makes the fairness invariant automatic
rather than maintained, since two modules would let the seed and audit prompts drift apart.

Round 0 is retrieval-free and request-only::

    prompt <- facet_enumeration(problem_statement, background, k_band)
    new    <- LLM.guided_json(prompt)            # the draft
    new    <- self_critique_dedup_coverage(new)  # one call: dedup + coverage + weight
    return weight_and_dedup(bank + new)          # derive `vital`, then merge duplicates

Retrieval-freeness is structural rather than a convention: this branch takes no retrieval client,
``passages`` must be ``None``, and nothing in ``pipeline/decompose/`` imports ``ragtime.retrieval``
or ``pipeline.rag_loop`` at all.

Rounds 1 and later are the coverage audit, evidence-informed and reuse-only::

    tagged <- attach the loops' answers/retrieved onto the bank
    delta  <- LLM.guided_json(gap_audit(...))     # {coverage, add, prune}
    kept   <- apply(delta): close full-and-answered; prune the flagged, never the answered
    add    <- [g for g in delta.add if on_topic(g, problem_statement, passages)]
    return weight_and_dedup(kept + add)

``passages`` is then a :class:`~ragtime.pipeline.decompose.coverage_audit.Evidence` built from
this round's loop results, and is required: the mirror of round 0's ``passages is None`` check, so
neither branch can silently run the other's contract.

Every LLM call goes through the injected ``ClientBundle``, the one shared vLLM; this module
constructs no client and defines no concurrency. Round 0 is a single call sequence and an audit
round is a single call plus one admission call per proposal. The fan belongs to
``pipeline.round_loop``, over the RAG loops, and rounds are sequential because they share the
growing bank.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from ragtime.common import Nugget, Statistics, get_logger, nfkc_len

from . import bank as bank_ops
from . import coverage_audit
from .dedup import dedup_nuggets
from .exemplars import select_exemplars
from .fairness_anchor import ROUND0_BANK_HASH, bank_hash_counter
from .kband import limit_to_k
from .on_topic import on_topic
from .prompts import SEED_SYSTEM, decoding_kwargs, dedup_confirm_prompt, seed_prompt
from .prompts import self_critique_prompt as _self_critique_prompt
from .weighting import weight_and_dedup

__all__ = [
    "BANK_SIZE",
    "GAP_EVIDENCE_DRIVEN",
    "GAP_REQUEST_DRIVEN",
    "THIN_SEED",
    "THIN_SEED_MIN_CHARS",
    "grow_nuggets",
]

#: Round-0 bank size, the seed half of the reported final k per variant and seed.
BANK_SIZE = "decompose.bank_size"
#: The degenerate-seed flag.
THIN_SEED = "decompose.thin_seed"

#: How many question-nuggets the seed draft contained, and whether that count hit the compiled
#: schema's ``maxItems``. Without :data:`SEED_SATURATED`, "the model wanted this many" and "the
#: grammar cut it off here" are indistinguishable after the fact.
SEED_DRAFTED = "decompose.seed_drafted"
#: Emissions rejected for not being a question. With no thinking channel the model uses the
#: `question` field as one, and the schema cannot stop it, because it bounds the field at 400
#: characters and the deliberation fits. The bank therefore gates on shape, the same way
#: claim-commit gates on a verbatim span: reject and count, never repair. Repairing would put our
#: wording into a Task-3 deliverable and hide the rate at which it happens.
MALFORMED_REJECTED = "decompose.malformed_rejected"
SEED_SATURATED = "decompose.seed_saturated"

#: New nuggets split by origin: a facet a retrieved passage surfaced (``trigger_passage_id``
#: present) against one the problem statement demanded and retrieval never surfaced. The first
#: says evidence reuse works, the second says the request spine is still driving breadth, and a
#: single "nuggets added" counter cannot tell them apart.
GAP_EVIDENCE_DRIVEN = "decompose.gap_evidence_driven"
GAP_REQUEST_DRIVEN = "decompose.gap_request_driven"

#: Below this many NFKC characters a ``problem_statement`` is treated as degenerate and the seed
#: falls back to the band's ``k_min``.
#:
#: This is a degeneracy floor rather than a tuned run choice, so the config carries no knob for
#: it. Real topics run to hundreds of characters, so nothing near this value is a judgement call
#: about a genuine request. Measured in ``nfkc_len``, the character-count normalizer, and never in
#: ``nfc``, which keeps the budget counter and the verbatim span-commit normalizer apart.
THIN_SEED_MIN_CHARS = 40

_log = get_logger("pipeline.decompose.grow_nuggets")


def _nuggets_ceiling(clients: Any) -> int:
    """Return ``maxItems`` of the nuggets array on the schema this bundle will send, or 0.

    Read rather than re-declared. Returns 0, meaning "cannot tell, do not warn", if the bundle is
    a stub whose schema is not introspectable: the saturation warning is diagnostic and must never
    be the thing that breaks a caller.
    """
    try:
        schema = clients.schemas.nuggets.schema
        return int(schema["properties"]["nuggets"]["maxItems"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return 0


async def grow_nuggets(
    problem_statement: str,
    background: str,
    bank: tuple[Nugget, ...],
    passages: Any | None,
    # `round` shadows the builtin: it is the documented parameter name of this
    # operation, and renaming it here would break the published contract.
    round: int,
    *,
    cfg: Any,
    clients: Any,
    topic_id: str,
    limit: int,
    seed: int,
    stats: Statistics | None = None,
    title: str = "",
) -> tuple[Nugget, ...]:
    """Grow the nugget bank by one round.

    At ``round == 0`` this is retrieval-free seed enumeration from the request and ``passages``
    must be ``None``. At ``round >= 1`` it is the coverage audit and ``passages`` must be a
    ``coverage_audit.Evidence`` carrying this round's loop results.

    ``limit`` is the topic's character budget, which enters decompose here and only here, as the
    breadth hint at the seed. ``seed`` is the per-cell seed value from
    ``orchestration.determinism.expand_seeds``, a parameter rather than a read of ``cfg`` because
    ``RunConfig.seeds`` is the seed count rather than a value, and the fan over seeds belongs to
    ``orchestration``.

    ``stats`` is injected and defaults to a fresh instance, so counters are isolated per call
    rather than accumulating in a module singleton.

    ``title`` is the topic's short request label. It is keyword-only with an empty default, so the
    positional signature is untouched and a topic without one renders byte-identical prompts. It
    is threaded no further than this package, because the RAG loop is request-blind by
    construction and ``round_loop._as_mapping`` hands a loop exactly ``nugget_id`` and
    ``question``.
    """
    if round < 0:
        raise ValueError(f"grow_nuggets round must be >= 0; got {round}")
    stats = stats if stats is not None else Statistics()
    if round != 0:
        # The one conditional the whole design rests on: same contract, same schema family, same
        # model, same post-processing tail, with round 0 as "round r with an empty bank and no
        # passages". Two functions would let the seed and audit prompts drift apart.
        return await _audit_round(
            problem_statement,
            background,
            bank,
            passages,
            round,
            cfg=cfg,
            clients=clients,
            topic_id=topic_id,
            seed=seed,
            stats=stats,
            title=title,
        )
    if passages is not None:
        raise ValueError(
            "grow_nuggets(round=0) must be called with passages=None: the seed is "
            "retrieval-free by construction (decomposition.retrieval_access.seed: false), and "
            "that is what makes round 0 identical across the three translation variants."
        )
    dcfg = cfg.blocks["decomposition"]
    _assert_seed_guardrails(dcfg)

    variant = getattr(cfg, "passage_lang", None)
    k_min, k_max = limit_to_k(int(limit), dcfg["k_band"], stats=stats)

    thin = nfkc_len(problem_statement.strip()) < THIN_SEED_MIN_CHARS
    if thin:
        stats.emit(THIN_SEED, 1.0, round=0)
    # A degenerate seed falls back to the band's k_min and proceeds; the audit rounds recover
    # coverage from evidence.
    band = (k_min, k_min) if thin else (k_min, k_max)

    few_shot = dcfg["few_shot"]
    exemplars = select_exemplars(str(few_shot["exemplar_set"]), int(few_shot["n"]))
    # `decomposition.background` is an ablation knob belonging to a separate config family;
    # `false` omits the section entirely.
    background_text = background if bool(dcfg.get("background", True)) else ""

    draft = await clients.llm.generate(
        clients.schemas.nuggets,
        seed_prompt(problem_statement, background_text, band, exemplars, title=title),
        seed,
        system=SEED_SYSTEM,
        **decoding_kwargs(cfg),
    )
    # Emit the item count, and warn when a draft lands on the schema ceiling: at that point the
    # grammar rather than the model chose where to stop, so the result is a truncated enumeration
    # rather than an answer. Without the count, "the model emitted the maximum" and "the bound is
    # not enforced" look identical from the outside.
    _drafted_questions = _questions(draft)
    _malformed = _malformed_count(draft)
    if _malformed:
        stats.emit(MALFORMED_REJECTED, float(_malformed), round=0)
    stats.emit(SEED_DRAFTED, float(len(_drafted_questions)), round=0)
    # The ceiling is read off the compiled schema in force, never re-declared here: a second copy
    # of the number would stop matching the one the grammar enforces, and this check exists to
    # detect the grammar binding.
    _ceiling = _nuggets_ceiling(clients)
    if _ceiling and len(_drafted_questions) >= _ceiling:
        stats.emit(SEED_SATURATED, 1.0, round=0)
        _log.warning(
            "decompose.seed.saturated_schema_bound",
            topic=topic_id,
            seed=seed,
            drafted=len(_drafted_questions),
            ceiling=_ceiling,
            k_band=band,
            detail=(
                "the draft hit the schema's maxItems, so generation was cut by the grammar "
                "rather than finished by the model: a truncated enumeration, not an answer"
            ),
        )
    drafted = bank_ops.mint(
        topic_id,
        _drafted_questions,
        start_index=bank_ops.next_index(bank),
        origin_round=0,
    )

    new = await _self_critique(
        drafted,
        problem_statement=problem_statement,
        background=background_text,
        k_band=band,
        cfg=cfg,
        clients=clients,
        topic_id=topic_id,
        seed=seed,
        start_index=bank_ops.next_index(bank),
        title=title,
    )

    out = await weight_and_dedup(
        bank_ops.add(bank, new),
        cfg=cfg,
        dedup_fn=_dedup_fn(cfg=cfg, clients=clients, seed=seed, stats=stats),
        stats=stats,
    )

    stats.emit(BANK_SIZE, float(len(out)), round=0, **_slices(variant=variant, seed=seed))
    stats.emit(
        ROUND0_BANK_HASH,
        bank_hash_counter(bank_ops.bank_fingerprint(out)),
        round=0,
        **_slices(variant=variant, seed=seed),
    )
    return out


async def _audit_round(
    problem_statement: str,
    background: str,
    bank: tuple[Nugget, ...],
    passages: Any,
    round: int,
    *,
    cfg: Any,
    clients: Any,
    topic_id: str,
    seed: int,
    stats: Statistics,
    title: str = "",
) -> tuple[Nugget, ...]:
    """The ``round >= 1`` branch: the coverage audit::

        tagged <- mark_answered(bank, committed_claims(passages))
        prompt <- gap_audit(problem_statement, background, tagged, passages)
        delta  <- LLM.guided_json(prompt)           # {coverage, add, prune}
        add    <- [q for q in delta.add if on_topic(q, problem_statement, passages)]
        bank'  <- (tagged - delta.prune) + add
        return weight_and_dedup(bank')

    ``mark_answered`` is split in two here. ``attach_evidence`` folds the loops' ``answers`` and
    ``retrieved`` onto the bank before the call, because the auditor must see what was found to
    judge coverage over it, and ``apply_delta`` moves ``status`` afterwards: the judge decides
    ``full``, and the code decides whether ``full`` is allowed to close anything. Splitting them
    is what makes the over-claim guard expressible at all.

    ``passages`` is a :class:`~ragtime.pipeline.decompose.coverage_audit.Evidence` rather than
    raw text: the audit reads only what the loops already retrieved and answered, and issues no
    queries of its own, which this module could not do anyway since it imports no retrieval
    client.

    The ``on_topic`` gate is the same one round 0 would use, with no second gate for later
    rounds, so a nugget minted at round 3 passes the same admission judgement as a round-0 one
    and bank drift over rounds is a countable rate.
    """
    if passages is None:
        raise ValueError(
            f"grow_nuggets(round={round}) requires passages: rounds >= 1 are the coverage "
            "audit and are evidence-informed by construction "
            "(decomposition.retrieval_access.audit: true). Passing None would run a "
            "request-only round under the audit prompt, which is a second seed, not an audit."
        )
    if not isinstance(passages, coverage_audit.Evidence):
        raise TypeError(
            "grow_nuggets(round>=1) takes a coverage_audit.Evidence for `passages`; build it "
            "with `coverage_audit.evidence_from_results(round.results, render=...)`. The audit "
            "reads the loops' own evidence (audit_evidence: reuse), so an arbitrary text blob "
            f"here would be evidence from nowhere. Got {type(passages).__name__}."
        )
    dcfg = cfg.blocks["decomposition"]
    _assert_audit_guardrails(dcfg)
    variant = getattr(cfg, "passage_lang", None)

    tagged = coverage_audit.attach_evidence(bank, passages)
    delta = await coverage_audit.audit(
        tagged,
        passages,
        problem_statement=problem_statement,
        background=background if bool(dcfg.get("background", True)) else "",
        cfg=cfg,
        clients=clients,
        seed=seed,
        round=round,
        stats=stats,
        title=title,
    )
    kept = coverage_audit.apply_delta(tagged, delta, passages, stats=stats, round=round)

    # The admission gate, one call per proposal, exactly as round 0's nuggets would face it.
    # Sequential rather than fanned: these calls share the one vLLM with the k loops and the
    # scorer, and the audit runs between rounds, so there is nothing to overlap with.
    admitted: list[coverage_audit.Gap] = []
    for gap in delta.gaps:
        ok = await on_topic(
            gap.question,
            problem_statement,
            _gate_context(passages, gap),
            cfg=cfg,
            clients=clients,
            seed=seed,
            stats=stats,
            round=round,
            title=title,
        )
        if ok:
            admitted.append(gap)

    out = kept
    for gap in admitted:
        # Minted one at a time because `trigger_passage_id` is per nugget: the trajectory log
        # records which passage surfaced which facet, and a batch mint would stamp them all with
        # the same id.
        out = bank_ops.add(
            out,
            bank_ops.mint(
                topic_id,
                [(gap.question, gap.weight)],
                start_index=bank_ops.next_index(out),
                origin_round=round,
                trigger_passage_id=gap.trigger_passage_id,
            ),
        )
        stats.emit(
            GAP_EVIDENCE_DRIVEN if gap.evidence_driven else GAP_REQUEST_DRIVEN, 1.0, round=round
        )

    out = await weight_and_dedup(
        out,
        cfg=cfg,
        # Open-only dedup. Round 0 uses `_dedup_fn` bare because every round-0 nugget is
        # unanswered; from round 1 the bank is status-mixed and a status-blind merge corrupts it.
        dedup_fn=_open_only_dedup(_dedup_fn(cfg=cfg, clients=clients, seed=seed, stats=stats)),
        stats=stats,
    )
    stats.emit(BANK_SIZE, float(len(out)), round=round, **_slices(variant=variant, seed=seed))
    return out


def _open_only_dedup(inner: Any) -> Any:
    """Restrict the paraphrase and embedding dedup to the open (``unanswered``) nuggets.

    The rule is that only an open nugget may be a dedup candidate, and only an open nugget may be
    the survivor that absorbs one. Answered and pruned nuggets are held out of the merge entirely:
    never merged away, never a survivor.

    This is a real restriction rather than a tidy-up. ``dedup_nuggets`` walks the bank in order
    and ``_absorb`` keeps the survivor's identity and status, which is right on a round-0 bank
    where everything is unanswered and corrupting on a status-mixed one:

    * a freshly minted gap absorbed into a pruned survivor makes the facet permanently unaskable.
      It is never fanned, its question never reaches Task 3, and because the bank size does not
      move, the novelty curve reports the round as quiet;
    * an answered nugget merged into an earlier open one destroys its ``nugget_id`` while
      ``rag_loop/{id}.jsonl`` keeps referring to it, and the survivor keeps
      ``status="unanswered"``, so committed content is orphaned and the work is redone.

    Both would also defeat
    :func:`~ragtime.pipeline.decompose.coverage_audit.apply_delta`'s prune refusal, since the very
    next call in the same function could drop the nugget anyway.

    The trade-off is deliberate: a new gap that duplicates an answered nugget now survives as an
    open duplicate and gets fanned. That costs one redundant loop, and the audit's own ``prune``
    can flag it next round. Letting the merge happen would instead delete a facet, which is
    unrecoverable, and a cheap redundant question beats a lost one.

    Bank order is preserved exactly: the full bank is walked in place and each open nugget is
    replaced by its survivor, which may have absorbed evidence, or dropped if it merged away. That
    keeps ``bank.next_index`` monotone and two runs at one seed comparable.
    """

    async def _dedup_open_only(bank: tuple[Nugget, ...]) -> tuple[Nugget, ...]:
        open_bank = tuple(n for n in bank if n.status == bank_ops.STATUS_UNANSWERED)
        if len(open_bank) == len(bank):
            return await inner(bank)
        survivors = {n.nugget_id: n for n in await inner(open_bank)}
        return tuple(
            survivors.get(n.nugget_id, n)
            for n in bank
            if n.status != bank_ops.STATUS_UNANSWERED or n.nugget_id in survivors
        )

    return _dedup_open_only


def _gate_context(evidence: Any, gap: coverage_audit.Gap) -> str | None:
    """Return the passage text offered to ``on_topic`` for a later-round proposal, or ``None``.

    The triggering passage when there is one, else the round's evidence, offered as context for
    judging topicality only and never as evidence the nugget must be supported by.
    """
    if not evidence.passages:
        return None
    if gap.trigger_passage_id:
        for pid, text in evidence.passages:
            if pid == gap.trigger_passage_id:
                return text
    return "\n\n".join(text for _pid, text in evidence.passages)


def _assert_audit_guardrails(dcfg: Any) -> None:
    """Refuse a config that asks the audit for behaviour this code does not implement.

    The mirror of :func:`_assert_seed_guardrails`, and for the same reason: each check guards a
    config drift rather than a bug. ``retrieval_access.audit: false`` would ask for a request-only
    audit round, which is a second seed; ``audit_evidence`` other than ``reuse`` would ask the
    auditor to issue its own queries, which this module structurally cannot do, since it imports
    no retrieval client.
    """
    if not dcfg.get("retrieval_access", {}).get("audit", True):
        raise ValueError(
            "decomposition.retrieval_access.audit is false, but rounds >= 1 are the "
            "evidence-informed coverage audit by construction. A retrieval-free audit round "
            "would be a second seed decomposition under the audit prompt."
        )
    evidence_mode = dcfg.get("audit_evidence", "reuse")
    if evidence_mode != "reuse":
        raise ValueError(
            f"decomposition.audit_evidence={evidence_mode!r} is not implemented; the only "
            "supported mode is 'reuse' (the audit reads only what the loops already retrieved "
            "and answered). Honouring the config's letter and not its meaning would let a run "
            "record claim the auditor searched when it cannot."
        )
    gate = dcfg.get("grounding_gate", "on_topic")
    if gate != "on_topic":
        raise ValueError(
            f"decomposition.grounding_gate={gate!r} is not implemented; the only supported "
            "admission gate is 'on_topic' (pipeline/decompose/on_topic.py)."
        )


async def _self_critique(
    drafted: tuple[Nugget, ...],
    *,
    problem_statement: str,
    background: str,
    k_band: tuple[int, int],
    cfg: Any,
    clients: Any,
    topic_id: str,
    seed: int,
    start_index: int = 0,
    title: str = "",
) -> tuple[Nugget, ...]:
    """Run the single post-draft call: dedup and coverage revisions plus ``weight``, in one object.

    Re-presents the drafted questions and asks the model to merge near-duplicates, add a nugget
    for any request facet not yet covered, and score each surviving nugget's ``weight`` in [0, 1],
    all in the same ``{rationale, nuggets}`` response. This is a closed three-item scope rather
    than an open-ended "reconsider your answer" pass, and folding ``weight`` in costs no extra
    constrained-decoding call.

    ``weight`` rides here rather than on the draft because ``weight_and_dedup`` runs on
    self-critique's output, so scoring the draft would assign weights to nuggets self-critique may
    still merge or drop.

    Ids are minted on the output rather than carried from the draft: at round 0 nothing references
    a nugget id yet, and self-critique may rewrite or drop any drafted question, so a carried id
    could end up on a different question.
    """
    obj = await clients.llm.generate(
        clients.schemas.nuggets,
        _self_critique_prompt(
            problem_statement, background, [n.question for n in drafted], k_band, title=title
        ),
        seed,
        system=SEED_SYSTEM,
        **decoding_kwargs(cfg),
    )
    return bank_ops.mint(
        topic_id,
        _weighted_questions(obj),
        start_index=start_index,
        origin_round=0,
    )


def _dedup_fn(*, cfg: Any, clients: Any, seed: int, stats: Statistics):
    """Bind ``dedup_nuggets`` to the shared dense encoder and the shared LLM.

    ``embed`` is ``clients.query_dense.embed`` with ``mode="dense"`` pinned here, at the point the
    client is sliced, so the choice is visible at the call site and ``dedup.py`` never names a
    mode or a model. ``llm_confirm`` is the ``dedup`` gate schema issued through the same
    ``clients.llm`` singleton. Neither is constructed inside ``dedup.py``, which imports no
    ``serving`` module at all.
    """
    dedup_cfg = cfg.blocks["decomposition"]["dedup"]

    async def _confirm(kept: str, candidate: str) -> bool:
        obj = await clients.llm.generate(
            clients.schemas.dedup,
            dedup_confirm_prompt(kept, candidate),
            seed,
            **decoding_kwargs(cfg),
        )
        return bool(obj["duplicate"])

    return partial(
        dedup_nuggets,
        cosine_cutoff=float(dedup_cfg["cosine_cutoff"]),
        llm_paraphrase_merge=bool(dedup_cfg["llm_paraphrase_merge"]),
        # `query_dense`, not `embedder`: the latter is built from the `retrieval.dense` leaf that
        # most configs do not carry, and sentence-transformers constructs successfully from an
        # empty model id, so the failure would surface only at the first `encode()`. `query_dense`
        # carries the resident encoder that built the index, keeping dedup in retrieval's embedding
        # space. Its device, via `index_build.config.query_encode_device`, is not a free machine
        # knob: a CPU vector and a CUDA vector differ, and this cosine is compared against a fixed
        # cutoff, so a CPU arm is a different arm rather than a cheaper one.
        embed=partial(clients.query_dense.embed, mode="dense"),
        llm_confirm=_confirm,
        stats=stats,
    )


def _assert_seed_guardrails(dcfg: Any) -> None:
    """Refuse a config that asks round 0 for behaviour this code does not implement.

    Both checks are narrow and cheap, and both guard a drift rather than a bug: flipping
    ``retrieval_access.seed`` to true expects a retrieving seed, which would break the fairness
    anchor and which this branch structurally cannot do, and a ``grounding_gate`` other than
    ``on_topic`` names an admission policy that does not exist. Failing loudly beats honouring
    the config's letter and not its meaning.
    """
    seed_access = dcfg.get("retrieval_access", {}).get("seed", False)
    if seed_access:
        raise ValueError(
            "decomposition.retrieval_access.seed is true, but round 0 is retrieval-free "
            "by construction: it takes no retrieval client, and a retrieving seed would "
            "read variant-specific passages and break the round-0 fairness anchor."
        )
    gate = dcfg.get("grounding_gate", "on_topic")
    if gate != "on_topic":
        raise ValueError(
            f"decomposition.grounding_gate={gate!r} is not implemented; the only "
            "supported admission gate is 'on_topic' (pipeline/decompose/on_topic.py). "
            "Silently accepting an unimplemented gate would drop the drift guard."
        )


def _slices(*, variant: str | None, seed: int | None) -> dict[str, Any]:
    """Canonical stats slices, omitting the ones this call site does not know."""
    out: dict[str, Any] = {}
    if variant is not None:
        out["variant"] = variant
    if seed is not None:
        out["seed"] = seed
    return out


def question_of(text: str) -> str:
    """Return the question inside a model emission, or ``""`` if there is none.

    A nugget is one single-sentence English question, so the question ends at its first ``?``.
    What comes after it, on this model, is the model's own deliberation, because with no thinking
    channel it uses this field as one.

    Cutting here is legitimate where repairing a claim span is not. The verbatim rule protects
    text that belongs to the corpus: a claim's span must be exactly what the document says, so
    trimming it would make our wording into the evidence. A nugget question is model-authored text
    this pipeline owns end to end, and nothing downstream matches it against a source, so taking
    the prefix up to the first ``?`` extracts what the model asked rather than inventing,
    extending or rewording it.

    Rejecting these outright was tried first and was worse: it emptied the bank completely,
    because every question in the draft carried the artifact, and an empty round-0 bank fails the
    fairness anchor and produces no Task-3 output. Counting keeps the cut honest:
    ``MALFORMED_REJECTED`` records how often it fires, so a prompt regression shows up as a number
    instead of as silence.
    """
    cut = text.find("?")
    return text[: cut + 1].strip() if cut != -1 else ""


def is_question(text: str) -> bool:
    """Return whether this is one question, which is what a nugget is.

    A nugget is one single-sentence English question, and every downstream consumer treats it as
    one: Task 3 publishes it verbatim, the RAG loop answers it, the coverage audit matches against
    it. A string that does not end in ``?`` is not a question.

    The weakest check that catches the failure: a trailing ``?`` after strip.
    Anything cleverer, such as sentence counting or a no-parentheses rule, would start rejecting
    legitimate questions, and the point is a floor on well-formedness rather than a style guide.
    """
    return bool(question_of(text.strip()))


def _questions(obj: dict[str, Any]) -> list[str]:
    return [
        q
        for n in obj.get("nuggets", ())
        if (q := question_of(str(n.get("question", "")).strip()))
    ]


def _malformed_count(obj: dict[str, Any]) -> int:
    """Return how many emissions were dropped for not being questions."""
    return sum(
        1
        for n in obj.get("nuggets", ())
        if (raw := str(n.get("question", "")).strip()) and question_of(raw) != raw
    )


def _weighted_questions(obj: dict[str, Any]) -> list[tuple[str, float]]:
    """``[(question, weight), ...]`` from a self-critique response.

    A null or absent ``weight``, which is the shape the draft call emits, degrades to 0.0 rather
    than raising: the bank is still valid, just uniformly non-vital, and
    ``weighting.clamp_weight`` owns the range discipline.
    """
    out: list[tuple[str, float]] = []
    for n in obj.get("nuggets", ()):
        question = question_of(str(n.get("question", "")).strip())
        if not question:
            continue
        weight = n.get("weight")
        out.append((question, 0.0 if weight is None else float(weight)))
    return out
