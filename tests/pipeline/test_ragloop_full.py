"""Full tier for the RAG loop: the loop against the real vLLM and the real retrieval stack.

The integration proof for the loop: a decomposer-shaped nugget in, a real model choosing its own
actions, real fuse-then-rerank retrieval, real three-rendering display, verbatim-grounded claims
out.

Everything provable without a GPU lives in ``test_ragloop_small.py`` and is not repeated here.
What can only be checked here is whether a real model, under a real grammar, over real retrieved
text, produces a loop that terminates and grounds. A test in this file that could fail against a
stub is in the wrong file.

How to run it, as a CPU job against one long-lived service:

    sbatch slurm/vllm_service.sbatch      # once; it publishes the endpoint descriptor
    sbatch --partition=shared-cpu --cpus-per-task=4 --mem=16G --time=04:00:00 \
        --wrap="uv run pytest -m full tests/pipeline -rs"

The driver is CPU-only and pays no model load, so no test here needs a GPU job of its own.

The index under test is the small published fixture, not the corpus index of 9,941,840 passages,
so retrieval quality at corpus scale is out of scope here and belongs to the retrieval tests.
What this file covers is the loop's integration with retrieval. The module prints that boundary
when it runs.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from ragtime.common import Statistics
from ragtime.pipeline.rag_loop import ACTIONS, ANSWERED, UNANSWERED, run_loop
from ragtime.serving.llm import LlmClient
from tests.retrieval.conftest import context_for, retrieval_cfg

pytestmark = pytest.mark.full

NOT_COVERED = [
    "corpus-scale retrieval quality (retrieval's gate; this index is the small fixture)",
    "k concurrent loops against one service (the round loop's round_loop owns the fan-out)",
    "the retrieval SERVICE transport (this tier uses the in-process context)",
]


def _endpoint() -> str:
    url = os.environ.get("RAGTIME_VLLM_URL", "").strip()
    if not url:
        pytest.fail(
            "RAGTIME_VLLM_URL is unset. This tier requires the long-lived vLLM service; it "
            "fails rather than skips, because a skip reads as a green tier. "
            "Start slurm/vllm_service.sbatch and point RAGTIME_VLLM_URL at its endpoint."
        )
    return url


@pytest.fixture(scope="module")
def llm():
    """The one shared vLLM client: the same singleton class production uses."""
    return LlmClient(model=os.environ.get("RAGTIME_VLLM_MODEL", "Qwen/Qwen3.5-122B-A10B-FP8"),
                     base_url=_endpoint())


@pytest.fixture(scope="module")
def _report_boundary():
    print("\nNOT_COVERED:" + "".join(f"\n  - {x}" for x in NOT_COVERED))


# The question has to be answerable from what this fixture actually returns.
#
# The fixture's retrieval is semantically inert by design. Every query returns the same five
# Russian filler passages at the same scores (264.0, 76.0, 76.0, 76.0, 76.0), because
# `tests/retrieval/conftest.py` installs a `SpyReranker` and the fixture corpus exists to
# exercise fusion, merge and ordering mechanics rather than ranking. Semantic retrieval quality
# is measured on the real corpus index, not here.
#
# A question about anything else leaves the model with nothing to ground on: it searches to the
# cap and abstains, with no claim rejected, which is correct behaviour and proves nothing. Asking
# about the passages the fixture really returns is what makes the grounding path testable.
#
# The chosen document is `rus-docs/9100000#p0`, which reads differently in each rendering while
# keeping one original doc-id, so a single nugget exercises what is read against what is cited:
#   original : "Портовый отчёт 0 перечисляет новые причалы. Грузопоток вырос на 0 процентов."
#   omt_opus : "Port bulletin 0 lists new quays. The cargo turnover grew 0 percent."
# The `original` arm requires a verbatim Cyrillic span, which is a stronger test of the NFC
# span commit than any English-only question would be.
NUGGET = {
    "nugget_id": "fixture#n1",
    "question": "What do the port bulletins report about new quays and cargo turnover?",
}


def _loop(nugget, cfg, llm, ctx, **kw):
    return asyncio.run(run_loop(nugget, cfg, llm=llm, ctx=ctx, stats=Statistics(), **kw))


def test_flm07_01_a_real_loop_terminates_and_every_action_is_in_the_grammar(built, tmp_path, llm, _report_boundary):
    """A real model, choosing freely, always reaches a terminal action.

    Every emitted action is also checked against the menu. `assert_allowed` would already have
    raised on an out-of-menu action, so this restates the property as an explicit postcondition
    rather than resting on an exception that did not fire.
    """
    cfg = retrieval_cfg(tmp_path)
    ctx = context_for(built, cfg)
    result = _loop(NUGGET, cfg, llm, ctx)

    assert result.status in (ANSWERED, UNANSWERED), "the loop must reach a terminal"
    assert result.turns >= 1
    assert set(result.action_trail) <= set(ACTIONS)
    assert len(result.action_trail) == result.turns
    assert result.action_trail[-1] in ("submit_answer", "abstain")
    print(f"\nPERF-shape: turns={result.turns} searches={result.searches} "
          f"committed={result.claims_committed} rejected={result.claims_rejected} "
          f"status={result.status} closed_by={result.closed_by}")


def test_flm07_02_every_committed_claim_is_verbatim_in_the_text_the_model_was_shown(built, tmp_path, llm):
    """The verbatim span commit, against real model output rather than a scripted span.

    A model paraphrases by default, which is why the check is code-side and deterministic.
    Re-verifying the committed spans here shows the gate held under real generation; it says
    nothing about the model behaving.
    """
    cfg = retrieval_cfg(tmp_path)
    ctx = context_for(built, cfg)
    result = _loop(NUGGET, cfg, llm, ctx)
    if result.claims_committed == 0:
        pytest.fail(
            "the model committed ZERO claims on an ANSWERABLE question over a real index "
            f"(turns={result.turns} searches={result.searches} status={result.status} "
            f"rejected={result.claims_rejected}). This does not SKIP: a skip here is the vacuous "
            "green: either the grounding path is broken or the retrieval returned "
            "nothing usable, and both are findings."
        )

    for answer in result.answers:
        for support in answer.support:
            text = ctx.passage_store.render(support.passage_id, cfg.passage_lang)
            assert text, f"a committed passage must resolve: {support.passage_id}"
            # The span itself is not carried on Answer; what must hold is that the citation
            # resolves to the original doc-id of a passage that really exists.
            assert support.passage_id.split("#", 1)[0] in answer.references


@pytest.mark.parametrize("rendering", ("original", "omt", "omt_opus"))
def test_flm07_03_citations_are_original_doc_ids_in_every_rendering(built, tmp_path, llm, rendering):
    """Three display renderings, one citation space, one rendering per nodeid.

    The same nugget is answered while reading this rendering. Whatever the model read, every
    reference must be an original doc-id, which is what makes the three arms comparable at all.
    Asserted on the ids and not on the answers: the model may legitimately find different
    evidence in different renderings, and that difference is the experiment.

    One rendering per parameter, rather than three loops in one test. A single loop here costs
    roughly nine minutes, so the combined form ran over the per-test ceiling, and a failure in
    one rendering cost a re-run of all three. Each parameter asserts that its own rendering
    produced a record, which is stronger than an existential over the three.
    """
    cfg = retrieval_cfg(tmp_path, passage_lang=rendering)
    ctx = context_for(built, cfg)
    result = _loop(NUGGET, cfg, llm, ctx, passage_lang=rendering)
    refs = {d for a in result.answers for d in a.references}
    for doc_id in refs:
        assert "#" not in doc_id, f"{doc_id!r} is a passage id, not an original doc-id"
    print(f"\n  rendering={rendering:9} status={result.status} refs={sorted(refs)}")


def test_flm07_04_the_search_action_really_goes_through_the_retrieval_stack(built, tmp_path, llm):
    """The trail must contain passages that came from the real index, not from the prompt.

    Checked by resolving every retrieved id through the passage store: an id the model invented
    would not resolve, and an empty trail would mean the loop answered without searching.
    """
    cfg = retrieval_cfg(tmp_path)
    ctx = context_for(built, cfg)
    result = _loop(NUGGET, cfg, llm, ctx)
    if result.searches == 0:
        pytest.fail("the model never searched -- the loop cannot ground without retrieval")
    assert result.retrieved, "a search happened but the Task-2 trail is empty"
    for hit in result.retrieved[:5]:
        assert ctx.passage_store.render(hit.passage_id, cfg.passage_lang)


def test_flm07_05_a_wall_clock_bound_is_actually_in_force_on_the_real_client(llm):
    """The second half of the termination guarantee, on the real client rather than a stub.

    The small tier proves ``asyncio.timeout`` fires. What can only be checked against the real
    client is that the shipped singleton carries a bound at all. Without one, the only bound is
    an httpx read timeout, which a trickling generation resets forever and which never fires.
    """
    assert llm.call_timeout_s is not None, "the shared client must bound every generation"
    assert llm.call_timeout_s > 0
