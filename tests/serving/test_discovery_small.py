"""Node discovery, over captured SLURM text rather than a live cluster: the parser
(multi-GRES nodes, ``--json`` against ``sinfo``, cross-cluster name collisions, the
missing-VRAM fallback, driver_version), the boundary that keeps ``config/serving`` outside
the fairness glob, the committed-profile-first against explicit-refresh split, and the
profile roundtrip from the discovery writer to the bring-up reader (J3)."""

from __future__ import annotations

import glob
import os
from dataclasses import asdict
from pathlib import Path

import pytest
from ruamel.yaml import YAML  # the loader the serving config files are read with

from ragtime.serving.capacity import capacity_for, mt_capacity
from ragtime.serving.modelspec import default_models_path, load_specs
from ragtime.serving.node import discovery
from ragtime.serving.node.discovery import NodeType

pytestmark = pytest.mark.small

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVING_DIR = _REPO_ROOT / "config" / "serving"

#: ``config/serving/`` holds three kinds of file, and only ``<cluster>.yml`` is a capacity
#: profile. The other two are registries with their own document shapes, so they are named
#: here rather than inferred from "not models.yml", which is what the profile sweep below
#: used to assume: a launch-shape registry has no ``nodes:`` key at all, so the sweep tried to
#: recompute capacity over nothing. Naming them explicitly keeps the sweep automatic for a new
#: cluster profile while forcing a new registry to be declared on purpose, and each one is
#: covered by its own contract in
#: ``test_the_non_profile_serving_registries_load_under_their_own_contract``.
_NON_PROFILE_YML = {
    "models.yml",  # hand-authored model-spec table -> ragtime.serving.modelspec
    "shapes.yml",  # hand-maintained launch-shape record; no Python module owns it
}
_CLUSTER_PROFILES = sorted(
    p for p in _SERVING_DIR.glob("*.yml") if p.name not in _NON_PROFILE_YML
)


def _node_from_entry(entry: dict) -> NodeType:
    return NodeType(**{k: entry[k] for k in NodeType.__dataclass_fields__ if k in entry})


# --------------------------------------------------------------------------- #
# E. Node discovery parser.
# --------------------------------------------------------------------------- #
def test_discovery_parses_multi_gres_node(scontrol_multi_gres):
    nodes = discovery.parse_nodes_json(scontrol_multi_gres, "baobab")
    assert len(nodes) == 1
    assert nodes[0].gpu_count == 4  # summed 2+2, not just the first segment
    assert nodes[0].gpu_model == "h100"


def test_discovery_prefers_json_over_plus_aggregated_sinfo(
    scontrol_blackwell, sinfo_plus_aggregated
):
    json_nodes = discovery.parse_nodes_json(scontrol_blackwell, "bamboo")
    sinfo_nodes = discovery.parse_nodes_oneline(sinfo_plus_aggregated, "bamboo")
    # --json gives the exact per-node figure (1 GPU); the `+` sinfo row is lossy.
    assert json_nodes[0].gpu_count == 1
    assert sinfo_nodes and "+" in sinfo_plus_aggregated  # aggregated -> not authoritative

    # discover_nodes with a runner that returns the --json fixture uses it (primary).
    def runner(cmd):
        if "--json" in cmd:
            return scontrol_blackwell
        return sinfo_plus_aggregated

    got = discovery.discover_nodes(["bamboo"], runner=runner)
    assert got[0].gpu_count == 1 and got[0].vram_gb == pytest.approx(95.6, abs=0.5)


def test_discovery_keys_by_cluster_and_node_name_collision(scontrol_blackwell):
    # same NodeName across two clusters -> two distinct records (keyed by cluster).
    def runner(cmd):
        return scontrol_blackwell  # both clusters return a node named gpu008

    got = discovery.discover_nodes(["bamboo", "baobab"], runner=runner)
    assert {n.cluster for n in got} == {"bamboo", "baobab"}
    assert len(got) == 2  # not collapsed into one


def test_discovery_missing_vram_per_gpu_fallback(scontrol_missing_vram):
    nodes = discovery.parse_nodes_json(scontrol_missing_vram, "baobab")
    assert nodes[0].vram_gb > 0  # model->VRAM fallback, never None/0


def test_discovery_node_type_includes_driver_version(scontrol_blackwell):
    nodes = discovery.parse_nodes_json(scontrol_blackwell, "bamboo")
    assert nodes[0].driver_version == "610.43.02"


# --------------------------------------------------------------------------- #
# F. Fairness boundary: config/serving/ stays outside the family glob.
# --------------------------------------------------------------------------- #
def test_family_guard_glob_excludes_config_serving_subdir():
    cwd = os.getcwd()
    try:
        os.chdir(_REPO_ROOT)
        swept = glob.glob(os.path.join("config", "*.yml"))  # the exact non-recursive pattern
    finally:
        os.chdir(cwd)
    assert not any("serving" in p for p in swept)
    # and the serving files DO exist (so the exclusion is meaningful, not vacuous).
    assert (_REPO_ROOT / "config" / "serving" / "models.yml").exists()
    assert list((_REPO_ROOT / "config" / "serving").glob("*.yml"))


def test_the_non_profile_serving_registries_load_under_their_own_contract():
    """The files excluded from the capacity-profile sweep are covered, not skipped.

    ``_NON_PROFILE_YML`` is what keeps ``shapes.yml`` (a launch-shape record, no ``nodes:``
    key) out of a sweep that recomputes ``capacity_for`` per node, so this is the test that
    stops that exclusion from being a hiding place: each excluded file must exist and must
    satisfy its own contract. It also refuses a vacuous sweep: at least one real
    ``<cluster>.yml`` has to be left over for the regression guard to have anything to check.

    ``models.yml`` has an owning module and is read through it. ``shapes.yml`` does not: it
    is maintained by hand and read by people, so its contract is asserted directly on the
    parsed document.
    """
    assert _CLUSTER_PROFILES, "no <cluster>.yml left: the profile sweep would be vacuous"
    for name in sorted(_NON_PROFILE_YML):
        assert (_SERVING_DIR / name).exists(), name  # excluding a ghost proves nothing

    llm, mt = load_specs()  # models.yml, through modelspec
    assert llm and mt

    doc = YAML(typ="safe").load((_SERVING_DIR / "shapes.yml").read_text(encoding="utf-8"))
    shapes = list(doc.get("shapes") or ())
    assert shapes, "shapes.yml declares no shape"
    for shape in shapes:
        assert 1 <= int(shape["gpus"]) <= int(shape["of"]), shape["id"]
        verified = dict(shape.get("verified") or {})
        assert set(verified) <= {"llm", "retrieval", "coresident"}, shape["id"]
        for check, entry in verified.items():
            # The rule the file is maintained under, re-checked against what is COMMITTED:
            # a `pass` may never be derived from arithmetic, only measured.
            assert not (
                entry.get("status") == "pass" and entry.get("basis") == "derived"
            ), f"{shape['id']}/{check}"
    resident = dict(doc.get("index_resident_gib") or {})
    assert resident and all(float(v) > 0 for v in resident.values())


def test_config_serving_files_are_valid_yaml_outside_glob():
    llm, mt = load_specs()  # parses config/serving/models.yml via ruamel safe loader
    assert "Qwen/Qwen3.5-122B-A10B-FP8" in llm
    assert "facebook/nllb-200-3.3B" in mt
    for p in (_REPO_ROOT / "config" / "serving").glob("*.yml"):
        assert p.read_text(encoding="utf-8").strip()  # non-empty, well-formed


# --------------------------------------------------------------------------- #
# G. Committed-profile-primary discovery (monkeypatched subprocess).
# --------------------------------------------------------------------------- #
def test_bringup_prefers_committed_profile_over_live_subprocess(
    tmp_path, monkeypatch, node_96gb_blackwell
):
    import subprocess

    def _boom(*a, **k):
        raise AssertionError("bring-up must NOT shell out to scontrol/sacctmgr")

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(subprocess, "Popen", _boom)

    # write a committed profile, then resolve knobs from it (no subprocess).
    llm, _mt = load_specs()
    spec = llm["Qwen/Qwen3-32B-FP8"]
    knobs = capacity_for(node_96gb_blackwell, spec)
    entry = discovery.profile_entry(node_96gb_blackwell, knobs, {"batch_size": 32})
    discovery.write_profile(tmp_path / "bamboo.yml", "bamboo", [entry])

    got = discovery.resolve_knobs("bamboo", "rtx_pro_6000", profile_dir=tmp_path)
    assert got["tensor_parallel_size"] == knobs.tensor_parallel_size


def test_explicit_refresh_stage_calls_live_discovery(scontrol_blackwell):
    calls: list = []

    def runner(cmd):
        calls.append(cmd)
        return scontrol_blackwell

    discovery.discover_nodes(["bamboo"], runner=runner)
    # exactly one scontrol call for the one cluster (the refresh path).
    assert len([c for c in calls if "scontrol" in c]) == 1


# --------------------------------------------------------------------------- #
# J3. discovery-writer -> bringup-reader profile roundtrip shape.
# --------------------------------------------------------------------------- #
def test_serving_cluster_yml_roundtrip_shape(tmp_path, node_96gb_blackwell):
    llm, mt = load_specs()
    llm_knobs = capacity_for(node_96gb_blackwell, llm["Qwen/Qwen3-32B-FP8"])
    mt_knobs = mt_capacity(node_96gb_blackwell, mt["facebook/nllb-200-3.3B"])
    entry = discovery.profile_entry(node_96gb_blackwell, llm_knobs, mt_knobs)
    path = discovery.write_profile(tmp_path / "bamboo.yml", "bamboo", [entry])

    back = discovery.read_profile(path)
    node = back["nodes"][0]
    for key in (
        "cluster",
        "partition",
        "gpu_model",
        "gpu_count",
        "vram_gb",
        "compute_capability",
        "driver_version",
        "cpu_ram_gb",
        "cpu_count",
    ):
        assert key in node
    assert "tensor_parallel_size" in node["llm"]  # LLM capacity knobs
    assert "batch_size" in node["mt"]  # MT capacity knobs


def test_default_models_path_points_at_repo_config_serving():
    p = default_models_path()
    assert p.name == "models.yml" and p.parent.name == "serving"
    assert p.exists()


# --------------------------------------------------------------------------- #
# Every committed <cluster>.yml knob block must equal the calculator's output for the model
# and node it names, so no profile can be hand-copied or mislabelled. This is what catches a
# 32B number filed under the 122B.
# --------------------------------------------------------------------------- #
def _assert_knobs_equal(stored: dict, expected: dict, ctx: str) -> None:
    for key, exp in expected.items():
        got = stored[key]
        if isinstance(exp, float):
            assert got == pytest.approx(exp), f"{ctx}: {key} {got} != {exp}"
        else:
            assert got == exp, f"{ctx}: {key} {got} != {exp}"


@pytest.mark.parametrize("prof_path", _CLUSTER_PROFILES)
def test_committed_cluster_profile_matches_calculator(prof_path):
    llm_specs, mt_specs = load_specs()
    prof = discovery.read_profile(prof_path)
    assert prof.get("nodes"), f"{prof_path.name} has no node entries"
    for entry in prof["nodes"]:
        node = _node_from_entry(entry)
        ctx = f"{prof_path.name}:{node.gpu_model}"

        llm_model = entry["llm"]["model"]
        assert llm_model in llm_specs, f"{ctx}: unknown llm model {llm_model!r}"
        expected_llm = asdict(capacity_for(node, llm_specs[llm_model]))
        stored_llm = {k: v for k, v in entry["llm"].items() if k != "model"}
        _assert_knobs_equal(stored_llm, expected_llm, f"{ctx} llm")

        mt_model = entry["mt"]["model"]
        assert mt_model in mt_specs, f"{ctx}: unknown mt model {mt_model!r}"
        expected_mt = asdict(mt_capacity(node, mt_specs[mt_model]))
        stored_mt = {k: v for k, v in entry["mt"].items() if k != "model"}
        _assert_knobs_equal(stored_mt, expected_mt, f"{ctx} mt")


def test_committed_bamboo_profile_122b_tp_sanity():
    """The 122B (FP8 ~113.7 GiB) must be TP>=2 on every card it does not fit solo,
    and TP=1 only where it fits solo (H200-141G) or the node is a single GPU."""
    prof = discovery.read_profile(_SERVING_DIR / "bamboo.yml")
    for entry in prof["nodes"]:
        node = _node_from_entry(entry)
        llm = entry["llm"]
        if node.gpu_count > 1 and llm["kv_cache_tokens"] == 0:
            # doesn't fit even multi-GPU -> TP is the max head-divisible divisor (>=2)
            assert llm["tensor_parallel_size"] >= 2, node.gpu_model
        if node.gpu_model == "nvidia_h200_nvl":
            assert llm["tensor_parallel_size"] == 1 and llm["kv_cache_tokens"] > 0


# --------------------------------------------------------------------------- #
# 3rd discovery fallback tier: committed profile when the control plane is down.
# --------------------------------------------------------------------------- #
def test_discovery_falls_back_to_committed_profile_when_unreachable(tmp_path, node_96gb_blackwell):
    llm, mt = load_specs()
    entry = discovery.profile_entry(
        node_96gb_blackwell,
        capacity_for(node_96gb_blackwell, llm["Qwen/Qwen3-32B-FP8"]),
        mt_capacity(node_96gb_blackwell, mt["facebook/nllb-200-3.3B"]),
    )
    discovery.write_profile(tmp_path / "bamboo.yml", "bamboo", [entry])

    def dead_runner(cmd):
        raise OSError("control plane unreachable")

    got = discovery.discover_nodes(["bamboo"], runner=dead_runner, profile_dir=tmp_path)
    assert got and got[0].gpu_model == "rtx_pro_6000"  # not silently dropped
    assert got[0].cluster == "bamboo"


def test_discovery_unresolved_cluster_returns_empty_not_crash(tmp_path):
    def dead_runner(cmd):
        raise OSError("down")

    # no committed profile in tmp_path -> cluster is unresolved, not a crash.
    got = discovery.discover_nodes(["ghost"], runner=dead_runner, profile_dir=tmp_path)
    assert got == []
