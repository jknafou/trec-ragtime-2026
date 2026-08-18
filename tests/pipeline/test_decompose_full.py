"""Full tier: seed decompose against the one shared vLLM.

FT-A2, FT-B50, FT-C2, FT-D1, FT-D2. What only a full test can prove is that the round-0
fairness anchor holds on real model output rather than on canned stub output. FT-A1's stub
version proves the mechanism; a sampled model at ``temperature=0.7`` is where a leak between
family siblings would actually show.

The tier runs against an already-running shared vLLM, with ``RAGTIME_VLLM_URL`` pointing at it.
It never starts a second one, and it fails loudly rather than skipping when no endpoint answers.

Two arms, one code path, chosen by ``--query-encode-device``:

* shipped arm (no flag): the dense query encoder runs where the config says (``cuda``), so this
  measures the configuration that will actually run. It needs a card.
* CPU-client arm (``--query-encode-device cpu``): the same encoder, same checkpoint, same
  revision, on CPU. The tier is then a plain client of the shared vLLM and needs no GPU
  allocation of its own::

      sbatch --partition=shared-cpu --cpus-per-task=4 --mem=16G --time=02:00:00 \
          --wrap="uv run pytest -m full tests/pipeline/test_decompose_full.py \
                  --query-encode-device cpu -rs"

  The CPU arm does not cover the absolute nugget bank a shipped run produces: a CPU forward
  pass and a CUDA one give different vectors, and dedup thresholds one against a fixed
  ``cosine_cutoff``, so that bank is measured only on the GPU arm. Every assertion here is
  comparative or structural, and both sides of each comparison run on the same device, which is
  why the CPU arm is a valid harness tier and not a substitute for the shipped one. The tier
  prints its arm at collection.

Nothing else in this tier touches a local model: the generation model is remote over HTTP, and
the reranker, sparse and late-interaction clients are constructed lazily and never called
here, so they load nothing on either arm.

Nothing here measures decomposition quality; there are no qrels and no dev set. Every assertion
is a liveness or consistency property: the bank is non-empty and well-shaped, the same
``(config, seed, topic)`` reproduces, and two family siblings agree.
"""

from __future__ import annotations

import asyncio
import copy
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from ragtime.common.topics import CANONICAL_TOPICS_REL
from ragtime.config import load
from ragtime.orchestration.determinism import expand_seeds
from ragtime.pipeline.decompose import assert_seed_parity, grow_nuggets
from ragtime.pipeline.decompose import bank as bank_ops
from ragtime.serving import build_clients

pytestmark = pytest.mark.full

_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_A = _REPO_ROOT / "config" / "e2e-original.yml"
CONFIG_B = _REPO_ROOT / "config" / "e2e-omt.yml"
TOPICS = _REPO_ROOT / CANONICAL_TOPICS_REL

#: How many real topics the aggregate gate exercises. Bounded, because the gate is a liveness
#: and consistency check against the shared vLLM, not a corpus-scale run.
N_TOPICS = 2

#: Seeds per run, read from the shipped config so the sweep cannot drift from `cfg.seeds`.
_N_SEEDS = len(expand_seeds(load(CONFIG_A)))


def _sweep_cells(config) -> list[tuple[int, int]]:
    """The (topic_index, seed_index) cells this run's FT-D2 covers. See ``--seed-sweep``.

    The default, ``gate``, is the two cells that add a property FT-A2 does not already prove:
    parity at a topic other than the first, and at a seed other than the first. (0, 0) is
    deliberately absent, because it is FT-A2's own cell.

    The full sweep is 2 topics x 5 seeds x 2 config siblings = 20 real seed decompositions,
    roughly three quarters of a 77-minute tier, re-proving across every seed what FT-A2 proves at
    one and what FT-D1 proves for same-seed reproducibility. It still exists behind
    ``--seed-sweep full`` and is run once before a scored launch. FT-A2 itself is unchanged: it
    is the runtime proof of the fairness invariant, and only its repetitions are gone.
    """
    if config.getoption("--seed-sweep") == "full":
        return [(t, s) for t in range(N_TOPICS) for s in range(_N_SEEDS)]
    return [(1, 0), (0, 1)]


def pytest_generate_tests(metafunc) -> None:
    if "sweep_cell" in metafunc.fixturenames:
        cells = _sweep_cells(metafunc.config)
        metafunc.parametrize("sweep_cell", cells, ids=[f"t{t}s{s}" for t, s in cells])
        # The selected cells and the exclusion list have to survive the runner's flags. A bare
        # `print()` here is swallowed, because pytest captures stdout during collection unless
        # `-s` is passed. `pytest_report_header` is no good either, because runs pass
        # `--no-header`. The terminal reporter writes past capture and past both flags, and is
        # the only form that reaches the job log.
        sweep = metafunc.config.getoption("--seed-sweep")
        lines = [
            f"FT-D2 sweep={sweep} cells={cells}",
            "NOT_COVERED_BY_THIS_GATE: "
            f'["ft_d2 cells outside {cells} - MOVED to --seed-sweep full, run once pre-launch, '
            'not deleted"]' if sweep != "full" else
            "NOT_COVERED_BY_THIS_GATE: []  (--seed-sweep full: every cell covered)",
        ]
        reporter = metafunc.config.pluginmanager.get_plugin("terminalreporter")
        for line in lines:
            if reporter is not None:
                reporter.write_line(line)
            else:  # pragma: no cover - only when the terminal plugin is disabled
                print(line)


def _endpoint() -> str:
    return os.environ.get("RAGTIME_VLLM_URL", "http://localhost:8000/v1")


@pytest.fixture(scope="session")
def live_vllm() -> str:
    """The shared vLLM endpoint, or a hard failure.

    An unreachable service fails the tier rather than skipping it. A skip still reads as a green
    summary line, so it would hide the one condition this tier exists to exercise.
    """
    url = _endpoint().rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            if resp.status != 200:
                pytest.fail(f"vLLM at {url} answered HTTP {resp.status}: the service is unhealthy")
    except (urllib.error.URLError, OSError) as exc:
        pytest.fail(
            f"no shared vLLM at {url} ({exc}). This tier requires the real instance: start "
            "slurm/vllm_service.sbatch and point RAGTIME_VLLM_URL at it. It does not skip: "
            "a skipped full tier reads as a green one."
        )
    return _endpoint()


#: The arm, resolved once from ``--query-encode-device`` (see ``tests/conftest.py``). Module
#: state rather than a fixture argument, because every test here builds its own bundle inline
#: and threading a parameter through five call sites is five places to forget it.
_QUERY_ENCODE_DEVICE = ""


@pytest.fixture(scope="session", autouse=True)
def announce_arm(pytestconfig):
    """Resolve the arm and print it, together with what that arm does not cover.

    Printed rather than only written down here: a coverage boundary that lives only in a
    docstring never reaches the reader of a passing run's log.
    """
    global _QUERY_ENCODE_DEVICE
    _QUERY_ENCODE_DEVICE = str(pytestconfig.getoption("--query-encode-device") or "")
    if _QUERY_ENCODE_DEVICE:
        print(
            f"\nARM: query_encode_device={_QUERY_ENCODE_DEVICE!r} (in-memory override of the "
            "shipped config leaf; the file is untouched).\n"
            "NOT_COVERED: the absolute nugget bank a shipped run produces, a CPU forward "
            "pass and a CUDA one are different vectors, and dedup thresholds one against a "
            "fixed cosine_cutoff. Comparative assertions (parity, reproducibility) are "
            "unaffected: both sides of each comparison run on this same device."
        )
    else:
        print("\nARM: query_encode_device as configured (shipped: cuda), needs a card.")
    yield


def _clients(cfg: Any) -> Any:
    """``build_clients`` on this arm's device: the production factory, never a fake.

    The only difference between the arms is the value of one real config leaf, applied to an
    in-memory copy of the shipped blocks, as FT-B50 does for ``decomposition.background``.
    Editing a committed ``config/*.yml`` is a fairness-family decision and never a test's.
    """
    if not _QUERY_ENCODE_DEVICE:
        return build_clients(cfg)
    blocks = _plain(dict(cfg.blocks))
    index_build = blocks.setdefault("index_build", {})
    index_build.setdefault("config", {})["query_encode_device"] = _QUERY_ENCODE_DEVICE
    return build_clients(_MutableCfg(blocks, cfg.passage_lang))


@pytest.fixture(scope="session")
def real_topics():
    from ragtime.common import load_topics

    return load_topics(TOPICS)


def _seed_bank(cfg: Any, clients: Any, topic: Any, seed: int):
    return asyncio.run(
        grow_nuggets(
            topic.problem_statement,
            topic.background,
            (),
            None,
            0,
            cfg=cfg,
            clients=clients,
            topic_id=topic.topic_id,
            limit=topic.limit,
            seed=seed,
        )
    )


def _assert_well_shaped(bank) -> None:
    assert bank, "the seed bank must not be empty"
    assert len({n.nugget_id for n in bank}) == len(bank), "nugget ids are unique"
    for n in bank:
        assert n.question.strip().endswith("?"), n.question
        assert n.origin_round == 0
        assert n.trigger_passage_id is None
        assert n.status == "unanswered"
        assert n.retrieved == () and n.answers == ()
        assert 0.0 <= n.weight <= 1.0
        assert n.aggregator_type == "OR"


# --------------------------------------------------------------------------- #
# FT-A2: the runtime fairness anchor on real model output
# --------------------------------------------------------------------------- #
def test_ft_a2_round0_bank_hash_equal_across_family_siblings_real(live_vllm, real_topics):
    cfg_a, cfg_b = load(CONFIG_A), load(CONFIG_B)
    # Precondition, the config layer's half: the two siblings share the decompose-relevant blocks.
    assert cfg_a.blocks["decomposition"] == cfg_b.blocks["decomposition"]
    assert cfg_a.blocks["llm"] == cfg_b.blocks["llm"]
    assert cfg_a.passage_lang != cfg_b.passage_lang, "they must be different variants"

    topic = real_topics[0]
    seed = expand_seeds(cfg_a)[0]
    banks = {
        cfg_a.passage_lang: _seed_bank(cfg_a, _clients(cfg_a), topic, seed),
        cfg_b.passage_lang: _seed_bank(cfg_b, _clients(cfg_b), topic, seed),
    }
    for bank in banks.values():
        _assert_well_shaped(bank)
    assert_seed_parity(banks, seed=seed)


# --------------------------------------------------------------------------- #
# FT-D1: reproducibility (a stronger property than cross-variant equality)
# --------------------------------------------------------------------------- #
def test_ft_d1_same_seed_real_run_twice_same_bank_fingerprint(live_vllm, real_topics):
    cfg = load(CONFIG_A)
    topic = real_topics[0]
    seed = expand_seeds(cfg)[0]
    first = _seed_bank(cfg, _clients(cfg), topic, seed)
    second = _seed_bank(cfg, _clients(cfg), topic, seed)
    _assert_well_shaped(first)
    assert bank_ops.bank_fingerprint(first) == bank_ops.bank_fingerprint(second)


# --------------------------------------------------------------------------- #
# FT-C2: the common, config and serving layers wired into decompose, against the real instance
# --------------------------------------------------------------------------- #
def test_ft_c2_real_topic_real_config_real_bundle(live_vllm, real_topics):
    cfg = load(CONFIG_A)
    clients = _clients(cfg)
    topic = real_topics[1]
    bank = _seed_bank(cfg, clients, topic, expand_seeds(cfg)[0])
    _assert_well_shaped(bank)
    assert all(n.nugget_id.startswith(f"{topic.topic_id}#n") for n in bank)


# --------------------------------------------------------------------------- #
# FT-B50, the `background: false` ablation arm does not crash the real path
# --------------------------------------------------------------------------- #
def test_ft_b50_background_off_completes_on_the_real_model(live_vllm, real_topics):
    """The background on/off ablation is a separate config family; no shipped file sets false.

    Rather than editing a committed config, which is a fairness-family decision, this drives the
    same real code path with an in-memory copy of the real blocks whose
    ``decomposition.background`` is ``false``, which is what such a family's file would carry.
    """
    cfg = load(CONFIG_A)
    off = _MutableCfg(_plain(dict(cfg.blocks)), cfg.passage_lang)
    off.blocks["decomposition"]["background"] = False
    bank = _seed_bank(off, _clients(cfg), real_topics[0], expand_seeds(cfg)[0])
    _assert_well_shaped(bank)


class _MutableCfg:
    """``RunConfig``-shaped, with mutable blocks (the real one is frozen)."""

    def __init__(self, blocks: dict, passage_lang: str) -> None:
        self.blocks = blocks
        self.passage_lang = passage_lang


def _plain(obj: Any) -> Any:
    if hasattr(obj, "items"):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return copy.deepcopy(obj)


# --------------------------------------------------------------------------- #
# FT-D2, the aggregate gate: a few real topics x the configured seeds
# --------------------------------------------------------------------------- #
def test_ft_d2_seed_decompose_topics_x_configured_seeds(live_vllm, real_topics, sweep_cell):
    """One (topic, seed) cell of the cross-variant parity sweep, one nodeid per cell.

    Each cell is two real decompositions, around five minutes, and is runnable on its own::

        pytest "tests/pipeline/test_decompose_full.py::test_ft_d2_seed_decompose_topics_x_configured_seeds[t1s0]"

    Which cells ran is recorded by the runner rather than counted inside this test: a count
    computed by the very test that produced the cells cannot notice a cell that was never
    selected. The two scale-free equal-budget preconditions live in
    ``test_decompose_seed_small.py``, since they need no model at all.
    """
    topic_i, seed_i = sweep_cell
    cfg_a, cfg_b = load(CONFIG_A), load(CONFIG_B)
    seeds = expand_seeds(cfg_a)
    topic = real_topics[topic_i]
    seed = seeds[seed_i]
    banks = {
        cfg_a.passage_lang: _seed_bank(cfg_a, _clients(cfg_a), topic, seed),
        cfg_b.passage_lang: _seed_bank(cfg_b, _clients(cfg_b), topic, seed),
    }
    for bank in banks.values():
        _assert_well_shaped(bank)
    assert_seed_parity(banks, seed=seed)
    # The final k is reported, never asserted against a target: k is adaptive, and is a
    # cross-condition confound rather than a quality bar.
    print(f"round-0 k: topic={topic.topic_id} seed={seed} k={len(banks[cfg_a.passage_lang])}")
