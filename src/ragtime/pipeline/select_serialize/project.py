"""``project``: the one entry point. Dedup once, fan to three projections, own the submission.

The pipeline is dedup, then caps, then the three task projections, then validation as a hard gate,
then the write, then the manifest, and nothing else: no retrieval, no generation, no new content.
Every value emitted was decided upstream; this stage only chooses, orders and truncates. Its single
model call is the answer-dedup confirm gate, run greedily on the one shared vLLM with a
canonicalized pair order, so the whole projection is deterministic and identical across variants.
The only reason the three variants' outputs differ is that their input banks already differ, which
is what keeps the serializer from confounding the translation comparison.

There is one refusal, :class:`SubmissionRejected`: a file our gates reject is not published, since
an invalid run is rejected outright by the organisers and publishing one is worse than publishing
none. It is raised by the official validator on Tasks 1 and 3 and by the Task-2 self-check alike,
because a caller that has to learn which track raises which class will eventually catch the wrong
one.

Citation scores are a ranking signal, not a gate: nothing is discarded on their value. An absent
scorer reorders Task 1's assessment priority and costs Task 2 one of its two factors, and takes no
answer and no document out of either. An uncited answer is still excluded, by a presence check in
``task1.admit`` rather than by a score. Task 2 is the one projection that ranks ON them:
``task2_claims.claim_support_rows`` scores a document by the persisted
``claim_importance x nugget_importance`` of the claims citing it, and a topic whose scorer never
ran is ranked on ``nugget_importance`` alone, recorded rather than invented.

Order matters. Dedup runs once, before all three projections. The ``k_t3`` and ``answers_cap``
caps are then applied once and feed both Tasks 1 and 3, while Task 2 ranks over the uncapped
deduped bank, because Task-2 relevance is "supports at least one nugget" rather than "was
submitted to Task 3".
"""

from __future__ import annotations

import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragtime.common import Statistics, get_logger
from ragtime.common.layout import Layout
from ragtime.common.topics import Topic, load_topics

from . import dedup as dedup_mod
from .claim_evidence import reranker_by_doc
from .knobs import SerializeKnobs
from .load import assemble_bank, citation_scores, last_complete_round, topic_dirs
from .submission import manifest as manifest_mod
from .submission import trec_selfcheck
from .submission import write as write_mod

from .submission.envelope import Envelope, EnvelopeError, envelopes
from .submission.validate import (
    FORMAT_NUGGETS,
    FORMAT_REPORT,
    SubmissionRejected,
    Verdict,
    normalize_topics,
)
from .submission.validate import validate as run_validator
from .task1 import task1_report
from .task2_claims import claim_support_rows
from .task3 import apply_caps, task3_bank

__all__ = ["TRACKS", "ProjectResult", "project"]

_log = get_logger("pipeline.select_serialize")

#: ``config.outputs`` track -> which task it is. The names are the ones the shipped configs use.
TRACKS = {"report_generation": 1, "retrieval": 2, "nuggetization": 3}


@dataclass(frozen=True, slots=True)
class ProjectResult:
    """What one run produced, and what the gate could and could not check."""

    paths: tuple[Path, ...]
    verdicts: tuple[Verdict, ...]
    manifest_path: Path | None
    coverage: dict[str, Any] = field(default_factory=dict)


async def project(
    cfg: Any,
    *,
    cell_dir: str | Path,
    layout: Layout,
    topics_path: str | Path,
    submission_root: str | Path,
    embed: Callable[[list[str]], Any] | None = None,
    confirm: Callable[[str, str], Awaitable[bool]] | None = None,
    stats: Statistics | None = None,
    variant: str | None = None,
    seed: int | None = None,
    drop_orphan_answers: bool = False,
    recompute_t2: bool = False,
) -> ProjectResult:
    """Project one finished cell into this run's declared submission files.

    ``layout`` is the cell-level layout, which carries ``config.outputs`` and so resolves the
    submission and manifest paths; per-topic layouts are derived from the completed topic
    directories. ``embed`` and ``confirm`` are sliced from the one ``serving`` ``ClientBundle`` by
    the caller, because this module builds no client. ``submission_root`` is required for the
    reason :mod:`.submission.write` explains.
    """
    knobs = SerializeKnobs.from_cfg(cfg)
    stats = stats if stats is not None else Statistics()
    deliverables = envelopes(cfg)
    if recompute_t2:
        # Task 2 only, synthesising the envelope when the config declares none.
        #
        # A flag rather than an `outputs:` entry, because a config that declares Tasks 1 and 3 only
        # may still hold exactly the retrieval behaviour a Task-2 arm needs when it is
        # byte-identical on both knobs. Adding a Task-2 output to that config would edit the record
        # of what the scored run declared, in order to describe a rebuild done afterwards; the flag
        # keeps the config honest and makes the rebuild an explicit, logged act.
        t2 = [env for env in deliverables if env.task == 2]
        if not t2:
            run_id = f"T2-{getattr(cfg, 'run_id', 'run')}"
            if len(run_id) > 25:  # the track caps run_id at 25 characters on Tasks 1 and 2
                raise EnvelopeError(
                    f"--recompute-t2 would synthesise run_id {run_id!r} ({len(run_id)} chars); "
                    "the track caps it at 25. Shorten cfg.run_id or declare the output explicitly."
                )
            # Priority 0 means no declared assessment order, which is the honest value here:
            # `envelope.py` reads priority from the path so there is exactly one source, and a
            # synthesised deliverable has no declared position.
            #
            # This path is the fallback rather than the method. If a config already declares the
            # output, use that config against the cells you have: it carries run_id, priority and
            # run_desc, and the launcher sees it too, so the shard bookkeeping and the concat
            # agree. Synthesis exists only for a run with genuinely no declared Task 2 anywhere.
            t2 = [
                Envelope(
                    track="retrieval",
                    task=2,
                    path=f"submissions/retrieval/{getattr(cfg, 'run_id', 'run')}.txt",
                    team_id=str(getattr(cfg, "team_id", "") or ""),
                    run_id=run_id,
                    run_desc=(
                        "Task-2 rebuild from the finished cells of run "
                        f"{getattr(cfg, 'run_id', 'run')} via --recompute-t2; the config declares "
                        "no task-2 output and was not modified. Assessment priority is undeclared."
                    ),
                    priority=0,
                )
            ]
            _log.info("select_serialize.t2_envelope_synthesised", path=t2[0].path, run_id=run_id)
        deliverables = tuple(t2)
    tasks = {env.task for env in deliverables}
    unknown = sorted({env.track for env in deliverables} - set(TRACKS))
    if unknown:
        raise EnvelopeError(f"outputs declare unknown track(s) {unknown}")

    topics = {t.topic_id: t for t in load_topics(topics_path)}
    dirs = topic_dirs(cell_dir)
    if not dirs:
        raise FileNotFoundError(f"no completed topic directories under {Path(cell_dir) / 'topics'}")

    t1_rows, t3_rows, t2_rows_all = [], [], []
    covered_topics: list[str] = []
    topics_with_citation_scores = 0
    for topic_dir in dirs:
        topic_id = topic_dir.name
        topic = topics.get(topic_id)
        if topic is None:
            raise KeyError(f"cell holds topic {topic_id!r}, which is not in {topics_path}")
        topic_layout = Layout(run_dir=topic_dir, outputs=getattr(cfg, "outputs", None))
        scores = citation_scores(topic_layout)
        if scores is None:
            # The counter must say whether scores were FOUND, so it is decided by `scores` alone.
            # The log line stays conditional on Task 1 because only Task 1 orders citations by
            # score, but folding that condition into the counter made a run with no Task 1 -- every
            # `mlir-*` run -- report scores as present for topics that had none.
            if 1 in tasks:
                # Not an error: nothing is discarded on a score threshold, so Task 1 does not
                # depend on citation scores existing. An uncited answer is still excluded, by a
                # presence check in `task1.admit` rather than by a score, and `references` remains
                # a ranking signal.
                _log.info(
                    "select_serialize.no_citation_scores",
                    topic=topic_id,
                    note="T1 ordering falls back to weight alone; nothing is dropped on score",
                )
        else:
            topics_with_citation_scores += 1

        orphans: list[tuple[str, str]] = []
        bank = assemble_bank(
            topic_layout, topic_id=topic_id, answer_score=knobs.answer_score, scores=scores,
            drop_orphan_answers=drop_orphan_answers, dropped=orphans,
        )
        if orphans:
            # `drop_orphan_answers` trades a hard failure for a quiet one unless every drop is
            # reported, so this logs the topic, the count and the exact keys.
            _log.info(
                "select_serialize.orphan_answers_dropped",
                topic=topic_id,
                dropped=str(len(orphans)),
                keys=str([f"{nid}:{key[:60]}" for nid, key in orphans]),
                note="a bank answer with no loop twin has no citations and cannot be emitted; "
                     "the rest of the topic is kept",
            )
        bank = await _dedup(bank, knobs, embed=embed, confirm=confirm, stats=stats)
        kept = apply_caps(bank, knobs, stats=stats, variant=variant, seed=seed)

        for env in deliverables:
            if env.task == 1:
                t1_rows.append(
                    task1_report(
                        kept, topic, knobs,
                        team_id=env.team_id, run_id=env.run_id, run_desc=env.run_desc,
                        stats=stats, variant=variant, seed=seed,
                    )
                )
            elif env.task == 2:
                # The claim-support projection is the Task-2 ranking, and the only one. It ranks
                # from what the run actually committed rather than from what it merely retrieved:
                # a document scores the persisted `claim_importance x nugget_importance` product,
                # looked up from the citation scorer rather than re-derived; a document cited by
                # more than one claim accumulates, its scores added; and ties are broken by the
                # reranker score of the tied documents rather than by the doc-id. The uncapped
                # deduped bank is the input, not the capped list.
                t2_rows_all.extend(
                    claim_support_rows(
                        bank, topic, knobs, run_id=env.run_id,
                        scores=scores,
                        reranker=reranker_by_doc(
                            topic_layout,
                            [n.nugget_id for n in bank],
                            through_round=last_complete_round(topic_layout),
                        ),
                        stats=stats, variant=variant, seed=seed,
                    )
                )
            elif env.task == 3:
                t3_rows.append(
                    task3_bank(
                        kept, topic, knobs,
                        team_id=env.team_id, run_id=env.run_id, run_desc=env.run_desc,
                        stats=stats, variant=variant, seed=seed,
                    )
                )
        covered_topics.append(topic_id)

    manifest_mod.assert_collection(
        [topics[tid] for tid in covered_topics], knobs.collection_id
    )
    return _emit(
        cfg, knobs, deliverables,
        rows={1: t1_rows, 2: t2_rows_all, 3: t3_rows},
        layout=layout, submission_root=submission_root, topics_path=topics_path,
        covered=covered_topics, all_topics=topics,
        topics_with_citation_scores=topics_with_citation_scores,
        variant=variant, seed=seed, stats=stats,
    )


async def _dedup(
    bank: Sequence[Any],
    knobs: SerializeKnobs,
    *,
    embed: Callable[[list[str]], Any] | None,
    confirm: Callable[[str, str], Awaitable[bool]] | None,
    stats: Statistics,
) -> tuple[Any, ...]:
    """Run the dedup guard once, before the three projections.

    ``embed`` and ``confirm`` are required: a missing client silently skipping the guard would be
    a check that cannot fail, and the determinism claim rests on the guard running.
    """
    if embed is None or confirm is None:
        raise ValueError(
            "the dedup guard needs the resident embedder and the shared vLLM confirm gate, both "
            "sliced from the one ClientBundle and never constructed here. Skipping it silently "
            "would make the projection non-deterministic."
        )
    merged = dedup_mod.merge_questions_embedding(
        bank, knobs.dedup_q_cutoff, embed=embed, stats=stats, log=_log
    )
    out = []
    for nugget in merged:
        out.append(
            await dedup_mod.merge_answers(
                nugget, knobs.dedup_a_cutoff, embed=embed, confirm=confirm, stats=stats
            )
        )
    return tuple(out)


def _emit(
    cfg: Any,
    knobs: SerializeKnobs,
    deliverables: Sequence[Envelope],
    *,
    rows: dict[int, list[Any]],
    layout: Layout,
    submission_root: str | Path,
    topics_path: str | Path,
    covered: Sequence[str],
    all_topics: dict[str, Topic],
    topics_with_citation_scores: int,
    variant: str | None,
    seed: int | None,
    stats: Statistics,
) -> ProjectResult:
    """Validate first, then write. A rejected run is never published."""
    written: list[Path] = []
    verdicts: list[Verdict] = []
    entries: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="ragtime-serialize-") as staging_s:
        staging = Path(staging_s)
        normalized = (
            normalize_topics(topics_path, staging / "topics.normalized.jsonl")
            if knobs.validate
            else None
        )
        for env in deliverables:
            target = write_mod.resolve(layout, env.track, root=submission_root)
            payload = rows[env.task]
            verdict: Verdict | None = None
            # What our own Task-2 gate found. Recorded because for a Task-2 submission it is the
            # only gate there is, and a manifest that says only "not applicable, the validator
            # covers no part of a TREC run" describes the check that did not run while staying
            # silent about the one that did.
            selfcheck: dict[str, Any] | None = None

            if env.task == 2:
                staged = write_mod.write_trec_run(staging / "t2.txt", payload)
                # The Task-2 gate refuses rather than logging. The official validator covers no
                # part of a TREC run, refusing one by name, so this self-check is the whole of
                # Task-2 correctness. `topics=` rides `knobs.validate` because that is the switch
                # the Task-1 and Task-3 coverage check already rides, so `validate: false` cannot
                # mean "no gates" on two tracks and "some gates" on the third. Emptiness is not
                # gated: zero rows is not a coverage question.
                try:
                    rows = trec_selfcheck.check_run_file(
                        staged,
                        top_docs=knobs.top_docs,
                        topics=all_topics if knobs.validate else None,
                    )
                except trec_selfcheck.TrecFormatError as exc:
                    # Raised as the same refusal a validator rejection raises: one exception type
                    # for "this file is not publishable", whichever gate said so.
                    absent = sorted(set(all_topics) - set(covered))
                    raise SubmissionRejected(
                        f"{env.track}: the Task-2 self-check rejected the staged run; nothing "
                        f"published. {len(covered)} of {len(all_topics)} declared topics had a "
                        f"completed cell directory"
                        + (f" (no cell for {len(absent)} of them)" if absent else "")
                        + f".\n{exc}"
                    ) from exc
                selfcheck = {
                    "rows": rows,
                    "topics_declared": len(all_topics),
                    "topic_coverage_checked": knobs.validate,
                    # Record that the rule ran, not just that the file passed. `forbid_ties`
                    # defaults to true and this call takes the default, but a future call site
                    # passing false would loosen the gate invisibly: the file would still be
                    # published and the manifest would look identical.
                    "strictly_descending_checked": True,
                }
                write_mod.write_trec_run(target, payload)
            else:
                fmt = FORMAT_REPORT if env.task == 1 else FORMAT_NUGGETS
                trec_selfcheck.check_ids(_emitted_ids(payload, env.task))
                staged = write_mod.write_jsonl_rows(staging / f"t{env.task}.jsonl", payload)
                if knobs.validate:
                    verdict = run_validator(staged, fmt, topics_path=normalized)
                    verdicts.append(verdict)
                    if not verdict.ran:
                        # A gate that cannot run is not a gate that passed. This branch is the
                        # likely one rather than the hypothetical, since the validator is a
                        # vendored clone and compute nodes have no network to fetch it. Recording
                        # "not run" in the manifest is honest bookkeeping but would let a run whose
                        # own config says `validate: true` publish unvalidated files. To serialize
                        # without the validator, say so in the config with `validate: false`, so
                        # the run record shows the choice was made.
                        raise SubmissionRejected(
                            f"{env.track}: serialize.validate is ON but the official "
                            f"rag-run-validator could NOT BE RUN, so nothing was published. "
                            f"A gate that cannot run has not passed.\n{verdict.stderr}"
                        )
                    if not verdict.ok:
                        raise SubmissionRejected(
                            f"{env.track}: rag-run-validator rejected the staged file "
                            f"(exit {verdict.returncode}); nothing published.\n{verdict.stdout}"
                        )
                write_mod.write_jsonl_rows(target, payload)

            written.append(target)
            entries.append(
                {
                    "task": env.task,
                    "track": env.track,
                    "path": env.path,
                    "written_to": str(target),
                    "run_id": env.run_id,
                    "run_desc": env.run_desc,
                    "team_id": env.team_id,
                    "priority": env.priority,
                    "variant": variant,
                    "seed": seed,
                    "lines": len(payload),
                    "validator": (
                        verdict.to_dict() if verdict is not None
                        else {"verdict": "not applicable: the validator covers no part of a "
                                         "TREC run file",
                              "trec_selfcheck": selfcheck} if env.task == 2
                        else {"verdict": "not run: serialize.validate is off"}
                    ),
                }
            )

    coverage = {
        "topics_in_file": len(all_topics),
        "topics_covered": len(covered),
        # A record rather than a gate: with `serialize.validate` on, a topic absent from the
        # emitted Task-2 bytes is already a `SubmissionRejected` above, and Tasks 1 and 3 report
        # the same fact through the validator. A manifest that cannot say what was skipped is
        # worse than one that can.
        "topics_missing": sorted(set(all_topics) - set(covered)),
        "topics_with_citation_scores": topics_with_citation_scores,
        # Records whether citation scores existed, because that changes how Task 1 orders its
        # citations, though not which citations are emitted. A reader of the manifest
        # must be able to tell a score-ordered run from a weight-ordered one. The score is
        # `nugget_importance x claim_importance`, so "absent" means no scored artifact was found
        # for at least one covered topic and ordering fell back to the nugget weight alone.
        "citation_scores": (
            "present" if topics_with_citation_scores == len(covered)
            else "absent: citations ordered by weight alone"
        ),
        "t1_budget_bound": stats.total("serialize.t1_sentences_dropped_budget") > 0,
    }
    _log.info("select_serialize.coverage", **{k: str(v) for k, v in coverage.items()})

    manifest_path: Path | None = None
    try:
        # This run's own fragment, never the family file. `project()` is a per-run call whose
        # entries are only this run's files, so writing them to the family path would make the
        # last of several concurrent runs the only one described. The fragment key is
        # `layout.run_dir.name`, the cell key the artifact tree is already organised by, so this
        # introduces no second naming scheme and cannot collide across cells. The family file is
        # produced by the explicit reduce in `submission.manifest`.
        declared_manifest = layout.submission_manifest_fragment(layout.run_dir.name)
    except (KeyError, ValueError):  # no outputs, or no single submission root
        declared_manifest = None
    if declared_manifest is not None:
        manifest_path = (
            declared_manifest if declared_manifest.is_absolute()
            else Path(submission_root) / declared_manifest
        )
        # `manifest_path` is `<root>/manifest.d/<cell>.json`, so the reduce's root is two levels
        # up. Derived rather than spelled: passing the wrong root finds no fragments and fails
        # silently, so the one place that knows the real path is the one place that writes the
        # instruction.
        manifest_root = manifest_path.parent.parent
        manifest_mod.write_manifest(
            manifest_path,
            manifest_mod.manifest_document(
                entries=entries,
                run_mode=knobs.run_mode,
                collection_id=knobs.collection_id,
                config_hashes=_hashes(cfg),
                coverage=coverage,
                manifest_root=manifest_root,
            ),
        )
        # The reduce is not wired into the DAG, because it would race the concurrent
        # fragment writes. The step is therefore stated three ways so it cannot be missed: here in
        # the run log, inside the fragment's own bytes, and as a check a caller can run before an
        # upload.
        _log.info(
            "select_serialize.manifest_fragment",
            fragment=str(manifest_path),
            family_manifest="not assembled: this is one run's fragment",
            next_step=manifest_mod.reduce_command(manifest_root),
        )
    return ProjectResult(
        paths=tuple(written), verdicts=tuple(verdicts),
        manifest_path=manifest_path, coverage=coverage,
    )


def _emitted_ids(rows: Sequence[Any], task: int) -> list[str]:
    """Return every doc-id that will appear in a Task-1 or Task-3 file, for the pre-write sweep."""
    ids: list[str] = []
    for row in rows:
        if task == 1:
            ids.extend(row.references)
            for response in row.responses:
                ids.extend(response.citations)
        else:
            for nugget in row.nugget_bank:
                for answer in nugget.answers:
                    ids.extend(answer.references)
    return ids


def _hashes(cfg: Any) -> dict[str, Any]:
    """Return the run's per-block config hashes, from the one hasher in ``config.hashing``.

    Imported lazily and tolerantly: the manifest is a record, and a hashing failure must not stop
    a validated submission from being written.
    """
    try:
        from ragtime.config.hashing import all_hashes

        return dict(all_hashes(cfg))
    except Exception as exc:  # noqa: BLE001 - recorded in the manifest, never fatal
        return {"error": f"{type(exc).__name__}: {exc}"}
