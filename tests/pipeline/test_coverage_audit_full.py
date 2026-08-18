"""Full tier for the coverage audit: two real audit rounds against one vLLM instance.

The small tier proves every mechanism with canned deltas: the over-claim guard, the prune
refusal, saturation on the configured streak, the two-line record. What it cannot prove is that
the model, under grammar-constrained decoding, emits a delta this code can act on at all -- that
the `coverage` labels come back inside the three-way enum and name real `nugget_id`s, that
`add[]` carries questions the `on_topic` gate then admits, and that a second round fans the
nuggets the first round left open. Those are prompt-and-schema properties, and a canned response
answers none of them.

The assertions are biconditionals rather than expectations about what the model will say. A
sampled judge may label everything `partial`, or propose nine gaps, or none, so asserting that a
nugget was closed would be asserting a coin flip. This test records the delta the model actually
emitted -- `RecordingLlm` wraps the real `LlmClient.generate`, with no substitution and no canned
payload -- and asserts the rule the code claims to apply to it:

    a nugget is `answered`  iff  it was labelled `full` and has >= 1 committed answer
    a nugget is `pruned`    iff  the delta named it and it has no committed answers

Whatever the model says, one branch of each is exercised, and any mis-wiring fails the test: a
label read from the wrong key, an over-claim slipping through, an ungated gap minted. The two
liveness assertions that are not biconditional (at least one gap proposed, at least one nugget
still open in round 2) are engineered rather than hoped for; see `SEED_QUESTIONS` and
`PROBLEM_STATEMENT`.

Round 0 is seeded from its artifact instead of being decomposed, for two reasons. The published
fixture index is semantically inert by design: every query returns the same five Russian
port-bulletin filler passages, so a real 2026 report request decomposed against it yields nuggets
nothing can answer, no claim is ever committed, no nugget can legally be marked `full`, and the
central property of this file could not be exercised at all. Round-0 decompose on a real report
request is also already covered by the decompose tier, and re-paying it here would spend a fifth
of the per-test budget re-proving another test's property. Seeding via
`Layout.decompose_round(0)` is not a shortcut around the code: it is the resume path
`run_rounds._round_zero` ships.

Budget: 900 s per test, central estimate ~790 s. `_BUDGET_NOTE` carries the derivation and the
one input that is unmeasured (seconds per loop turn). A breach is a finding, not flakiness, and
the remedy is to split the test, never to delete an assertion.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Layout, Statistics
from ragtime.pipeline import driver
from ragtime.pipeline.decompose import bank as bank_ops
from ragtime.pipeline.decompose import coverage_audit, saturated
from ragtime.serving import compile_schemas
from ragtime.serving.llm import LlmClient
from tests.pipeline.conftest import REAL_CONFIG, _plain
from tests.retrieval.conftest import context_for, retrieval_cfg

pytestmark = pytest.mark.full

#: The hard round cap for this gate. Two rounds is the minimum that can show a nugget being
#: re-fanned after an audit left it open, and the maximum that fits the per-test budget.
R_MAX = 2

#: `rag_loop.budget` for this gate only, and a declared cut listed in NOT_COVERED. The shipped
#: budget is `min_searches 2 / max_iters 8 / max_searches 5`, and two real rounds at that shape
#: are ~1100-1300 s (estimated from the 528.14 s single loop), which breaches the cap.
#: One search, then one claim, then one answer is the shortest path a loop can take to a
#: committed claim, which is all this gate needs; the shipped budget's single-round fan is
#: measured elsewhere and is not re-measured here.
GATE_BUDGET = {"min_searches": 1, "max_iters": 3, "max_searches": 1, "token_budget": 20000}

#: The fixture's dense encoder is a sha256-derived stand-in (`tests/preprocess/test_index_small.
#: _vec`) whose components are all non-negative, so any two of its vectors sit at a high cosine,
#: nothing like BGE-M3's geometry. Running the shipped 0.80 cutoff against it would measure the
#: stand-in, propose spurious pairs, and add an unbounded LLM-confirm tail to the budget. Raised
#: here and declared; real dedup is covered by the decompose tier, on the real encoder.
GATE_COSINE_CUTOFF = 0.999

#: A report request written to match the fixture corpus. It is broad against a narrow seed bank,
#: so the audit has obvious request-driven gaps to propose; that is what makes the "at least one
#: gap proposed" assertion an engineered property rather than a hope.
PROBLEM_STATEMENT = (
    "Prepare a report on recent developments at the port: what was newly built or added, how "
    "cargo and freight volumes changed, which official bulletins and report numbers document "
    "those changes, who operates the facilities, when the works were completed, and what the "
    "reported figures mean for the port's capacity."
)
BACKGROUND = (
    "The reader is a logistics analyst who needs the documented facts and their sources, not "
    "commentary."
)

#: The seed bank. Two questions are answerable from what the inert fixture returns; the third is
#: answerable from nothing in the fixture. That third nugget is load-bearing twice over:
#:
#: * it guarantees at least one nugget is still open in round 2 (a loop with no evidence commits
#:   nothing, and the over-claim guard then refuses to close it even if the judge says `full`),
#:   which is what makes the re-fan and the two-line record testable at all;
#: * it is the only way a `full`-with-no-answers case can arise from a real judge, which is the
#:   left-hand branch of the first biconditional.
SEED_QUESTIONS = (
    ("What do the port bulletins report about new quays?", 0.9),
    ("What figure is given for the growth in cargo turnover?", 0.8),
    ("What is the annual operating budget of the port authority?", 0.5),
)
UNANSWERABLE_INDEX = 2

TOPIC_ID = "fixture-topic"

_BUDGET_NOTE = (
    "estimate ~790 s = round-1 fan ~175 s (3 loops CONCURRENT under ceiling 4, "
    "max_iters 3) + audit call ~120 s + on_topic ~90 s (sequential, one per proposed gap) + "
    "round-2 fan ~175 s + audit ~120 s + on_topic ~90 s + fixture index build ~20 s. "
    "Dominant uncertainty, stated rather than hidden: seconds-per-TURN is unmeasured, the only "
    "anchor is 528.14 s for one whole loop at the shipped budget and that job's "
    "turn count was not recorded, so the per-turn figure could be ~66 s or ~132 s. "
    "TAIL RISK: on_topic is one call per PROPOSED gap and the schema permits 64, so a "
    "pathological audit round could add several hundred seconds. Either overrun is a FINDING "
    "and the remedy is to move round 2 to a named pre-launch run), never a deleted assertion."
)

NOT_COVERED = (
    (
        "the SHIPPED rag_loop.budget. This gate runs "
        f"{GATE_BUDGET} so two REAL audit rounds fit the 900 s cap; the shipped budget "
        "(min_searches 2 / max_iters 8 / max_searches 5) is ~1100-1300 s for two rounds. The "
        "shipped budget's single-round fan is measured: step 1, 612.73 s at k=8."
    ),
    (
        f"the shipped decomposition.R_max. This gate pins R_max={R_MAX}; rounds 3..8 of a real "
        "saturating topic are not exercised, and neither is a bank that grows past a few tens."
    ),
    (
        "the shipped decomposition.dedup.cosine_cutoff (0.80). Raised to "
        f"{GATE_COSINE_CUTOFF} because the fixture's dense encoder is a sha256 stand-in with "
        "non-negative components, so its cosine scale is not BGE-M3's. Real paraphrase dedup is "
        "decompose's gate, on the real encoder."
    ),
    (
        "round-0 decompose from a real 2026 topic. Seeded from Layout.decompose_round(0) (the "
        "resume path) with questions answerable from the inert fixture; real-topic decompose and "
        "the round-0 fairness anchor are decompose's gate."
    ),
    (
        "the request itself is written for this fixture, not taken from topics/**. A real "
        "request against filler passages commits no claims, and every coverage property here "
        "needs at least one committed claim to exist."
    ),
    (
        "corpus-scale retrieval and retrieval QUALITY. This is the SMALL published fixture and "
        "it is semantically inert by design; quality is retrieval's gate."
    ),
    (
        "the dense-embedding half of dedup on the real encoder: this gate binds "
        "clients.query_dense to the fixture's salted stand-in, because build_clients needs a "
        "full shipped config while the fixture index needs the fixture config, and pointing a "
        "real BGE-M3 at a fixture index is an encoder-identity mismatch, not a measurement."
    ),
    (
        "the mlir run.kind branch of driver.drive: dispatch, depth-from-config and the "
        "no-RAG-loop guarantee are covered in the small tier; a real mlir run needs the real "
        "index and belongs to a Task-2 gate, not here."
    ),
    (
        "multi-pair behaviour and ensure_corpus. One topic on one pair by construction; "
        "ensure_corpus is UNBUILT (its corpus keys belong to preprocess, which pipeline may not "
        "import) and drive takes ctx as a required keyword instead."
    ),
    (
        "the R_max-vs-saturation branch NOT taken on this run. Both are asserted, but only one "
        "can happen per run; the measurements JSON records which, so a reader can see whether "
        "the saturation branch has ever actually executed against the real model."
    ),
)

_MEASUREMENTS: dict[str, Any] = {}


class RecordingLlm:
    """The real `LlmClient`, with every call observed. A wrapper, never a substitute.

    Nothing is canned and nothing is intercepted: `generate` delegates to the real client and
    returns what it returned. The recording is what lets the assertions be biconditionals over
    the deltas the model actually emitted, instead of expectations about what a sampled judge
    ought to say.
    """

    def __init__(self, inner: LlmClient) -> None:
        self._inner = inner
        self.model = inner.model
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def generate(self, schema: Any, prompt: str, seed: int, **kwargs: Any) -> dict:
        obj = await self._inner.generate(schema, prompt, seed, **kwargs)
        self.calls.append((getattr(schema, "name", str(schema)), obj))
        return obj

    def responses(self, schema_name: str) -> list[dict[str, Any]]:
        return [obj for name, obj in self.calls if name == schema_name]


def _endpoint() -> str:
    """Return the shared vLLM endpoint, failing if it is unset rather than skipping."""
    url = os.environ.get("RAGTIME_VLLM_URL", "")
    if not url:
        pytest.fail(
            "RAGTIME_VLLM_URL is unset. This tier requires the long-lived vLLM service. "
            "Start slurm/vllm_service.sbatch and point RAGTIME_VLLM_URL at its endpoint."
        )
    return url


@pytest.fixture(scope="module", autouse=True)
def _boundary(request: pytest.FixtureRequest):
    """Print the coverage boundary at setup, and the measurements at teardown.

    Both writes go through the terminal reporter, which reaches the job log past capture, but
    only from setup and teardown. A mid-test `write_line` is swallowed on a passing test, so the
    numbers below are written to a JSON file inside the test and only echoed here.
    """
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line("")
        reporter.write_line("NOT_COVERED_BY_THIS_GATE (tests/pipeline/test_coverage_audit_full.py):")
        for item in NOT_COVERED:
            reporter.write_line(f"  - {item}")
        reporter.write_line("")
        reporter.write_line(f"BUDGET: {_BUDGET_NOTE}")
    yield
    if reporter is not None and _MEASUREMENTS:
        reporter.write_line("")
        reporter.write_line("two real audit rounds (also written to the run dir):")
        for line in json.dumps(_MEASUREMENTS, indent=2, ensure_ascii=False).splitlines():
            reporter.write_line(f"  {line}")


@pytest.fixture(scope="module")
def llm() -> RecordingLlm:
    return RecordingLlm(
        LlmClient(
            model=os.environ.get("RAGTIME_VLLM_MODEL", "Qwen/Qwen3.5-122B-A10B-FP8"),
            base_url=_endpoint(),
        )
    )


def _gate_cfg(tmp_path: Path) -> Any:
    """The fixture retrieval config, plus the real `decomposition`/`rag_loop` blocks.

    The two blocks are deep-copied from `config/e2e-original.yml` rather than hand-written, so
    the prompts, the decoding knobs, `min_new`/`low_streak`, `vital_cutoff` and the pinned
    `emission`/`abstain` values are the shipped ones. Exactly three leaves move, each declared
    in NOT_COVERED: `R_max`, `rag_loop.budget`, `dedup.cosine_cutoff`.
    """
    from ragtime.config import load

    cfg = retrieval_cfg(tmp_path)
    blocks = _plain(dict(load(REAL_CONFIG).blocks))
    cfg.blocks["decomposition"] = blocks["decomposition"]
    cfg.blocks["rag_loop"] = blocks["rag_loop"]
    cfg.blocks["claim_commit"] = blocks["claim_commit"]
    cfg.blocks["decomposition"]["R_max"] = R_MAX
    cfg.blocks["decomposition"]["dedup"]["cosine_cutoff"] = GATE_COSINE_CUTOFF
    cfg.blocks["rag_loop"]["budget"] = dict(GATE_BUDGET)
    cfg.kind = driver.KIND_E2E
    cfg.run_id = "e2e-original"
    cfg.outputs = ()
    return cfg


def test_fm08b_02_two_real_audit_rounds_against_one_instance(built, llm, tmp_path) -> None:
    """Seed a bank, run two real audit rounds through `drive`, and check the delta was obeyed."""
    cfg = _gate_cfg(tmp_path)
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path / "base")
    stats = Statistics()
    ctx = context_for(built, cfg)

    seed_bank = bank_ops.mint(TOPIC_ID, list(SEED_QUESTIONS), origin_round=0)
    unanswerable_id = seed_bank[UNANSWERABLE_INDEX].nugget_id
    bank_ops.write_bank(layout.decompose_round(0), seed_bank)

    started = time.perf_counter()
    run_dir = driver.drive(
        cfg,
        {
            "topic_id": TOPIC_ID,
            "problem_statement": PROBLEM_STATEMENT,
            "background": BACKGROUND,
            "limit": 5000,
        },
        "original",
        0,
        ctx=ctx,
        clients=_clients(llm, built),
        llm=llm,
        layout=layout,
        stats=stats,
    )
    wall = time.perf_counter() - started

    # ----------------------------------------------------------------------------------- #
    # The artifacts, and the curve they encode.
    # ----------------------------------------------------------------------------------- #
    assert run_dir == layout.run_dir
    assert layout.success().exists(), "a completed topic must be marked done, LAST"
    banks = _persisted_banks(layout)
    assert len(banks) == R_MAX + 1, (
        f"expected round 0 plus {R_MAX} audit rounds on disk; got rounds {sorted(banks)}"
    )
    novelty = tuple(len(banks[r]) - len(banks[r - 1]) for r in range(1, len(banks)))

    # ----------------------------------------------------------------------------------- #
    # 1. The audit call happened, once per round, and came back inside its contract.
    # ----------------------------------------------------------------------------------- #
    deltas = llm.responses("coverage_audit")
    assert len(deltas) == R_MAX, (
        f"the audit is ONE call per round: expected {R_MAX}, got {len(deltas)}"
    )
    round1 = deltas[0]
    labelled = {e["nugget_id"]: e["coverage"] for e in round1["coverage"]}
    assert labelled, "the real model returned no coverage labels at all: the audit judged nothing"
    assert set(labelled.values()) <= set(coverage_audit.COVERAGE_LABELS), (
        f"labels outside the 3-way AutoNuggetizer scale: {sorted(set(labelled.values()))}"
    )
    assert set(labelled) <= {n.nugget_id for n in seed_bank}, (
        "the auditor labelled a nugget_id that is not in the bank"
    )

    # ----------------------------------------------------------------------------------- #
    # 2. The biconditional: a nugget closes iff it was labelled `full` and has a committed
    #    answer. Both code-side refusals are covered, whichever branch the sampled judge took.
    # ----------------------------------------------------------------------------------- #
    after1 = {n.nugget_id: n for n in banks[1]}
    for nid, nugget in after1.items():
        if nugget.origin_round != 0:
            continue
        full = labelled.get(nid) == coverage_audit.COVERAGE_FULL
        has_answer = bool(nugget.answers)
        expected = bank_ops.STATUS_ANSWERED if (full and has_answer) else "not answered"
        assert (nugget.status == bank_ops.STATUS_ANSWERED) == (full and has_answer), (
            f"{nid}: label={labelled.get(nid)!r} answers={len(nugget.answers)} "
            f"status={nugget.status!r}: a nugget closes IFF `full` AND answered "
            f"(expected {expected!r})"
        )

    # ... and the prune biconditional: named by the delta and answer-less, or it stays.
    named = set(round1["prune"])
    for nid, nugget in after1.items():
        if nugget.origin_round != 0:
            continue
        prunable = nid in named and not nugget.answers
        assert (nugget.status == bank_ops.STATUS_PRUNED) == prunable, (
            f"{nid}: named_for_prune={nid in named} answers={len(nugget.answers)} "
            f"status={nugget.status!r}: 'answered != deleted' was not honoured"
        )

    # ----------------------------------------------------------------------------------- #
    # 3. Gaps: proposed, gated, and only then minted into round 2's bank.
    # ----------------------------------------------------------------------------------- #
    proposed = [str(a["question"]) for a in round1["add"]]
    assert proposed, (
        "the audit proposed no gap nuggets at all. The request is broad against a "
        "3-nugget bank, so this is a finding about the audit prompt, not flakiness."
    )
    minted = [n for n in banks[1] if n.origin_round == 1]
    assert minted, "no proposed gap survived to the bank: the admission path minted nothing"
    for nugget in minted:
        assert any(nugget.question in q or q in nugget.question for q in proposed), (
            f"a round-1 nugget appeared that no proposal contains: {nugget.question!r}"
        )
        assert nugget.nugget_id not in {n.nugget_id for n in seed_bank}, (
            "a gap nugget reused a seed nugget's id: ids are monotone and never recycled"
        )
    gate_calls = len(llm.responses("on_topic_gate"))
    assert gate_calls >= len(minted), (
        f"{len(minted)} gap nuggets were minted but only {gate_calls} on_topic calls were made: "
        "a proposal reached the bank without passing decompose's admission gate"
    )

    # ----------------------------------------------------------------------------------- #
    # 4. Round 2 really re-fanned what round 1 left open, and both rounds' records survive.
    # ----------------------------------------------------------------------------------- #
    still_open = [
        n.nugget_id
        for n in banks[1]
        if n.status == bank_ops.STATUS_UNANSWERED and n.origin_round == 0
    ]
    assert unanswerable_id in still_open, (
        f"{unanswerable_id} is answerable from nothing in the fixture, so it committed no claim "
        "and the over-claim guard must have kept it open: it did not"
    )
    record = layout.rag_loop(unanswerable_id)
    assert record.exists(), f"no loop record for the re-fanned nugget: {record}"
    rounds_in_record = [json.loads(line)["round"] for line in record.read_text("utf-8").splitlines()]
    assert rounds_in_record == [1, 2], (
        f"the re-fanned nugget's record holds rounds {rounds_in_record}, not [1, 2]: round 1's "
        "evidence was overwritten when round 2 ran (the step-1 defect this pins against a real "
        "two-round run, not only a unit test)"
    )

    # ----------------------------------------------------------------------------------- #
    # 5. The loop ended for exactly one stated reason, re-derived from decompose's own predicate.
    # ----------------------------------------------------------------------------------- #
    dcfg = cfg.blocks["decomposition"]
    fired = saturated(novelty, min_new=dcfg["min_new"], low_streak=dcfg["low_streak"])
    incomplete = stats.value(round_loop_incomplete(), seed=0) == 1.0
    assert fired != incomplete, (
        f"novelty={novelty}: the loop must EITHER saturate OR flag coverage-incomplete, never "
        f"both and never neither (saturated={fired}, coverage_incomplete={incomplete})"
    )
    if fired:
        assert not saturated(
            novelty[:-1], min_new=dcfg["min_new"], low_streak=dcfg["low_streak"]
        ), f"novelty={novelty}: saturation fired LATER than the configured streak allows"

    # ----------------------------------------------------------------------------------- #
    # The measurements, written to a file: a mid-test terminal write is swallowed on a pass.
    # ----------------------------------------------------------------------------------- #
    _MEASUREMENTS.update(
        {
            "perf": (
                f"{wall:.1f} s for 1 topic, {R_MAX} audit rounds over the fixture index "
                f"(gate budget {GATE_BUDGET} s)"
            ),
            "wall_s": round(wall, 1),
            "r37_budget_s": 900,
            "r37_headroom_s": round(900 - wall, 1),
            "rounds_on_disk": sorted(banks),
            "bank_size_per_round": {str(r): len(b) for r, b in sorted(banks.items())},
            "novelty": list(novelty),
            "stop_reason": "saturated" if fired else "R_max (coverage-incomplete)",
            "labels_round1": labelled,
            "gaps_proposed_round1": len(proposed),
            "gaps_minted_round1": len(minted),
            "on_topic_calls": gate_calls,
            "still_open_after_round1": still_open,
            "llm_calls_by_schema": _call_histogram(llm),
        }
    )
    path = layout.run_dir / "gate_m08b_step2.json"
    path.write_text(json.dumps(_MEASUREMENTS, indent=2, ensure_ascii=False), encoding="utf-8")
    assert path.exists() and json.loads(path.read_text("utf-8"))["novelty"] == list(novelty), (
        "the measurements file is the only durable record of this run's numbers"
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clients(llm: RecordingLlm, built: Any) -> Any:
    """A `ClientBundle`-shaped holder carrying exactly what the audit path touches.

    `build_clients` is not used: it validates a full shipped config, down to
    `translation.config.omt_model`, while the fixture index needs the fixture config, and
    pointing a real BGE-M3 at a fixture index would be an encoder-identity mismatch rather than
    a measurement. The LLM and the compiled schemas here are the real ones; only `query_dense`
    is the fixture's stand-in, and that is declared in NOT_COVERED.
    """
    import types

    return types.SimpleNamespace(
        llm=llm,
        schemas=compile_schemas(),
        query_dense=built.clients.index_dense,
    )


def _persisted_banks(layout: Layout) -> dict[int, tuple[Any, ...]]:
    """Every `decompose/round_{r}.jsonl` on disk, read back through the bank's own reader."""
    from ragtime.common.io import is_done

    out: dict[int, tuple[Any, ...]] = {}
    for r in range(R_MAX + 1):
        path = layout.decompose_round(r)
        if is_done(path):
            out[r] = bank_ops.read_bank(path)
    return out


def round_loop_incomplete() -> str:
    from ragtime.pipeline.round_loop import COVERAGE_INCOMPLETE

    return COVERAGE_INCOMPLETE


def _call_histogram(llm: RecordingLlm) -> dict[str, int]:
    hist: dict[str, int] = {}
    for name, _obj in llm.calls:
        hist[name] = hist.get(name, 0) + 1
    return hist
