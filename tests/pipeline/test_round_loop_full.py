"""Full tier for the round fan: k concurrent RAG loops against one vLLM.

The test asserts that the fan is concurrent, and reports what k concurrent loops against one
instance cost. That cost is what every capacity decision is denominated in: how many GPU pairs
the run needs, how many seeds are affordable, whether the retrieval admission ceiling has to
exist at all.

It measures the fan rather than a whole topic. A topic is decompose plus fan; decompose costs
roughly three minutes and is sequential by necessity, since round 0 must finish before any loop
starts, which is also what keeps the fairness anchor intact. The unknown is the fan, so the topic
cost composes as roughly 180 s plus the loop wall reported here.

Isolating the fan also avoids a config problem that would make the test lie. `build_clients`
requires a full shipped config, down to `translation.config.omt_model`, while the small published
index needs the fixture config, and pointing a real BGE-M3 client at a fixture index would be an
encoder-identity mismatch rather than a measurement. `run_loop` needs no client bundle at all,
because `llm` and `ctx` are injected, so the fan runs exactly as production runs it.

The k questions are distinct, and that is load-bearing. vLLM runs with `--enable-prefix-caching`,
so fanning k copies of one question would share almost the entire prompt prefix, the cache would
serve k-1 of them at a fraction of the true cost, and the speedup would be an artifact of the
benchmark rather than a property of the fan. Distinct questions share only the system prefix,
which is what a real nugget bank does too.

The retrieval fixture is semantically inert by design -- every query returns the same five filler
passages at the same scores -- so varying the question changes the prompt without changing what
comes back, which is the isolation this measurement needs.

Budget: one test under a 900 s ceiling. A single loop measures a little over 500 s, so k=8 run
sequentially would take about 70 minutes. The test fits its budget only if the fan works, and a
breach is a finding rather than flakiness.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from ragtime.common import Layout, Statistics
from ragtime.pipeline.records import write_round_records
from ragtime.pipeline.round_loop import run_round
from ragtime.serving.llm import LlmClient
from tests.retrieval.conftest import context_for, retrieval_cfg

pytestmark = pytest.mark.full

REPO = Path(__file__).resolve().parents[2]

#: k for this measurement. A real topic yields 8-14 nuggets (k_band `(5000, 8, 14)`, and every
#: released 2026 topic carries `limit: 5000`), so 8 is the low end of the real range, and the
#: budget has to hold at a size the run will actually schedule.
K = 8

NOT_COVERED = (
    (
        "the DECOMPOSE half of a topic: measured separately (~3 min) and "
        "sequential by necessity. A topic composes as ~180 s + the loops_s reported here."
    ),
    (
        "corpus-scale retrieval: this is the SMALL published fixture and it is semantically "
        "inert by design (every query returns the same five filler passages). "
        "Retrieval QUALITY is retrieval's gate, already green on the real index."
    ),
    "the coverage-audit round loop (round >= 1): step 1 is round 0 plus the fan.",
    (
        "multi-pair behaviour: one topic on one pair by construction; the per-pair registry does "
        "not exist yet."
    ),
    (
        "the retrieval admission ceiling as a SEPARATE semaphore: this run bounds loops and "
        "searches under one ceiling and reports search_wall_s so the second ceiling can be sized "
        "from data rather than invented."
    ),
)

#: Distinct questions, all answerable from what the inert fixture returns: the Russian
#: port-bulletin filler documents. Distinctness defeats the prefix cache; answerability keeps the
#: loops doing real work, searching and then committing, instead of searching to the cap and
#: abstaining, which is what a question with no grounding available produces.
QUESTIONS = (
    "What do the port bulletins report about new quays?",
    "What figure is given for the growth in cargo turnover?",
    "Which port report number is referenced in the bulletins?",
    "What do the bulletins say was newly added to the port?",
    "By what percentage did the freight volume change?",
    "What does the port document describe as recently constructed?",
    "What quantity is reported for the change in shipping volume?",
    "What infrastructure does the port bulletin list?",
)


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
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line("")
        reporter.write_line("NOT_COVERED_BY_THIS_GATE (tests/pipeline/test_round_loop_full.py):")
        for item in NOT_COVERED:
            reporter.write_line(f"  - {item}")
    yield


@pytest.fixture(scope="module")
def llm() -> LlmClient:
    return LlmClient(
        model=os.environ.get("RAGTIME_VLLM_MODEL", "Qwen/Qwen3.5-122B-A10B-FP8"),
        base_url=_endpoint(),
    )


def test_fm08b_01_k_loops_fan_concurrently_against_one_instance(
    built, llm, tmp_path, request
) -> None:
    """Fan k=8 distinct nuggets through `run_round`, write the records, and report the cost."""
    cfg = retrieval_cfg(tmp_path)
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path / "base")
    stats = Statistics()
    bank = [
        {"nugget_id": f"fixture#n{i}", "question": q, "status": "open"}
        for i, q in enumerate(QUESTIONS[:K])
    ]

    # The same construction the RAG loop's full tier uses: the published fixture index under this
    # test's query-time config. `built` is re-exported into `tests/pipeline/conftest.py` so that a
    # pipeline test can request it.
    ctx = context_for(built, cfg)
    started = time.perf_counter()
    result = asyncio.run(
        run_round(
            bank, cfg, llm=llm, ctx=ctx, ceiling=K,
            round_no=0, seed=0, stats=stats,
            passage_lang=getattr(cfg, "passage_lang", None),
        )
    )
    wall = time.perf_counter() - started

    paths = write_round_records(layout, result, cfg, topic_id="fixture-topic", seed=0)

    # --- the wiring -------------------------------------------------------------------
    assert len(result.results) == K, "every nugget must produce a record, never a hole"
    assert not result.errors, f"loops raised: {result.errors}"
    for p in paths:
        assert p.exists(), f"record not written: {p}"
    assert any(getattr(r, "searches", 0) > 0 for r in result.results), (
        "no loop searched: the fan ran but the loops did no retrieval work"
    )

    # --- the numbers ------------------------------------------------------------------
    per_loop = result.sequential_wall_s / max(K, 1)
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    lines = [
        "",
        "k concurrent loops against one vLLM",
        f"  k (nuggets)        {K}",
        f"  fanned wall        {result.wall_s:8.1f} s   (harness {wall:.1f} s)",
        f"  sequential would be{result.sequential_wall_s:8.1f} s",
        f"  mean loop          {per_loop:8.1f} s   (the RAG loop single-loop reference: 528.1 s)",
        f"  ACHIEVED SPEEDUP   {result.speedup:8.2f}x  <- 1.0 means the fan bought NOTHING",
        (
            f"  retrieval wall     {result.search_wall_s:8.1f} s   "
            f"({100 * result.search_wall_s / max(result.wall_s, 1e-9):.1f}% of the round)"
        ),
        "",
        (
            f"  round wall: {result.wall_s:.1f} s for 1 round of {K} concurrent loops "
            "over the fixture index"
        ),
        "",
        (
            f"  A TOPIC therefore costs ~180 s decompose + {result.wall_s:.0f} s loops "
            f"= ~{(180 + result.wall_s) / 60:.1f} min"
        ),
        f"  1,545 e2e cells at that rate = ~{1545 * (180 + result.wall_s) / 3600:.0f} pair-hours",
        "",
    ]
    if reporter is not None:
        for line in lines:
            reporter.write_line(line)

    # A floor, not a performance target: if the fanned round costs what the loops cost end to
    # end, the semaphore is serialising, which is a defect rather than slowness.
    assert result.speedup > 1.0, (
        f"the fan produced NO concurrency (speedup {result.speedup:.2f}x over {K} loops), "
        "they ran effectively one after another"
    )
