"""SLURM node and cluster discovery by parsing ``scontrol show node --json``.

It enumerates node types across clusters so the capacity calculator can size serving
knobs per node, using only the standard library plus a ruamel writer for the committed
profile.

The rules it follows:

* ``scontrol`` rejects ``-M all``, so the loop runs per cluster; the JSON payload's
  ``cluster_name`` is empty, so ``cluster`` is stamped from the loop variable.
* Records are keyed by ``(cluster, node)``, since node names are not globally unique.
* GRES is parsed by regex as ``gpu:<model>:<count>``, with repeated segments summed.
* VRAM comes from ``VramPerGpu:no_consume:<N>[MG]``, falling back to a model-to-VRAM
  table rather than ``None`` or 0.
* ``NodeType.driver_version`` is parsed so a stale-driver node can be excluded.
* ``--json`` is the primary source; the ``sinfo -o`` aggregated parse is a lossy
  fallback.
* The committed ``config/serving/<cluster>.yml`` is this module's own output, written by
  :func:`write_profile` from :func:`profile_entry` over ``serving.capacity``'s knobs, and
  read back by :func:`nodes_from_profile` as the last discovery tier. A refresh means
  calling :func:`write_profile` directly; there is no CLI stage for it.

The committed profile is re-derived from ``serving.capacity`` by
``tests/serving/test_discovery_small.py``'s ``test_committed_cluster_profile_matches_calculator``,
so a hand-edited number cannot survive there.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ragtime.common import get_logger
from ruamel.yaml import YAML

_log = get_logger("serving.discovery")

# Failures tolerated while degrading gracefully: a missing binary, a non-zero exit, or
# an unparsable payload. Never a bare `Exception`.
_DISCOVERY_ERRORS = (OSError, subprocess.SubprocessError, ValueError, KeyError)

__all__ = [
    "NodeType",
    "default_profile_dir",
    "discover_nodes",
    "enumerate_clusters",
    "nodes_from_profile",
    "parse_nodes_json",
    "parse_nodes_oneline",
    "profile_entry",
    "read_profile",
    "resolve_knobs",
    "write_profile",
]

# ruamel writer (round-trip) for the committed profile; safe loader for reads.
_YAML_WRITE = YAML()
_YAML_WRITE.default_flow_style = False
_YAML_READ = YAML(typ="safe")

_GRES_GPU = re.compile(r"gpu:([A-Za-z0-9_.\-]+):(\d+)")
_VRAM = re.compile(r"VramPerGpu:no_consume:(\d+)\s*([MG])", re.IGNORECASE)
_DRIVER = re.compile(r"DriverVersion[:=]\s*([\d.]+)", re.IGNORECASE)
# Compute capability appears two ways across clusters: an admin `ComputeCapability:
# 12.0` comment, or a SLURM feature `COMPUTE_CAPABILITY_8_6`.
_COMPUTE = re.compile(r"ComputeCapability[:=]\s*([\d.]+)", re.IGNORECASE)
_COMPUTE_FEATURE = re.compile(r"COMPUTE_CAPABILITY_(\d+)_(\d+)", re.IGNORECASE)

# Fallback per-GPU VRAM (GiB) keyed by a lowercased GPU-model substring, used only
# when a node reports no VramPerGpu. Conservative, admin-editable.
_MODEL_VRAM_GB: dict[str, float] = {
    "rtx_pro_6000": 95.6,
    "h200": 141.0,
    "h100": 80.0,
    "a100": 80.0,
    "a100_40": 40.0,
    "rtx_5090": 32.0,
    "v100": 32.0,
    "titan": 24.0,
    "rtx_3090": 24.0,
}


@dataclass(frozen=True, slots=True)
class NodeType:
    """One discovered GPU node type, keyed logically by ``(cluster, gpu_model,
    gpu_count, vram_gb)``. ``node_name`` is a representative member."""

    cluster: str
    partition: str
    gpu_model: str
    gpu_count: int
    vram_gb: float
    compute_capability: str
    driver_version: str
    cpu_ram_gb: float
    cpu_count: int
    node_name: str = ""


def _fallback_vram(gpu_model: str) -> float:
    key = gpu_model.lower()
    for sub, gb in _MODEL_VRAM_GB.items():
        if sub in key:
            return gb
    return 0.0


def _parse_gres(gres: str) -> tuple[str, int]:
    """Sum repeated ``gpu:<model>:<count>`` segments into ``(model, total count)``."""
    model = ""
    total = 0
    for m in _GRES_GPU.finditer(gres or ""):
        model = model or m.group(1)
        total += int(m.group(2))
    return model, total


def _scan(*fields: Any) -> str:
    """Join candidate string fields for regex scanning.

    VRAM, driver version and compute capability live in the GRES string, the admin
    comment or the SLURM feature list, and which one varies by cluster.
    """
    return " ".join(str(f) for f in fields if f)


def _norm_partition(value: Any) -> str:
    """Normalize a partition field (SLURM `--json` gives a list) to a stable string."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(p) for p in value)
    return str(value or "")


def _parse_cc(blob: str) -> str:
    """Compute capability from either a `ComputeCapability:X.Y` comment or a
    `COMPUTE_CAPABILITY_X_Y` SLURM feature."""
    m = _COMPUTE.search(blob)
    if m:
        return m.group(1)
    fm = _COMPUTE_FEATURE.search(blob)
    return f"{fm.group(1)}.{fm.group(2)}" if fm else ""


def _one_node_from_json(node: dict[str, Any], cluster: str) -> NodeType | None:
    gres = node.get("gres") or node.get("Gres") or ""
    gpu_model, gpu_count = _parse_gres(gres)
    if gpu_count == 0:
        return None  # not a GPU node
    blob = _scan(
        node.get("comment"),
        node.get("Comment"),
        node.get("features"),
        node.get("active_features"),
        node.get("available_features"),
        gres,
    )
    vm = _VRAM.search(blob)
    if vm:
        n, unit = int(vm.group(1)), vm.group(2).upper()
        vram_gb = round(n / 1024, 2) if unit == "M" else float(n)
    else:
        vram_gb = _fallback_vram(gpu_model)
    dm = _DRIVER.search(blob)
    real_mem = node.get("real_memory") or node.get("RealMemory") or 0
    return NodeType(
        cluster=cluster,
        partition=_norm_partition(node.get("partitions") or node.get("Partitions")),
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        vram_gb=vram_gb,
        compute_capability=_parse_cc(blob),
        driver_version=dm.group(1) if dm else "",
        cpu_ram_gb=round(float(real_mem) / 1024, 2) if real_mem else 0.0,
        cpu_count=int(node.get("cpus") or node.get("CPUTot") or 0),
        node_name=str(node.get("name") or node.get("NodeName") or ""),
    )


def parse_nodes_json(payload: str | dict[str, Any], cluster: str) -> list[NodeType]:
    """Parse ``scontrol show node --json`` output for one cluster.

    ``cluster`` is stamped from the caller's loop variable, since the payload's
    ``cluster_name`` is empty. Records are de-duplicated by node type within the
    cluster, keeping a representative ``node_name``.
    """
    data = json.loads(payload) if isinstance(payload, str) else payload
    nodes = data.get("nodes", data if isinstance(data, list) else [])
    seen: dict[tuple, NodeType] = {}
    for raw in nodes:
        nt = _one_node_from_json(raw, cluster)
        if nt is None:
            continue
        # Dedup by physical node type. Partition is intentionally excluded: the same card
        # shape spanning two partitions is one serving-capacity type.
        key = (nt.cluster, nt.gpu_model, nt.gpu_count, nt.vram_gb)
        seen.setdefault(key, nt)
    return list(seen.values())


def parse_nodes_oneline(text: str, cluster: str) -> list[NodeType]:
    """Lossy fallback that parses ``sinfo -o '%n %G %m %c'`` rows.

    A ``+`` suffix on a GRES count means sinfo aggregated several node shapes into one
    row, and the per-node count cannot be recovered, which is why ``--json`` is preferred.
    """
    out: list[NodeType] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("nodelist", "hostnames")):
            continue
        parts = line.split()
        gres_field = next((p for p in parts if p.startswith("gpu:")), "")
        aggregated = "+" in gres_field
        gpu_model, gpu_count = _parse_gres(gres_field.replace("+", ""))
        if gpu_count == 0:
            continue
        out.append(
            NodeType(
                cluster=cluster,
                partition="",
                gpu_model=gpu_model,
                # A `+` row is aggregated, so the count is a lower bound.
                gpu_count=gpu_count,
                vram_gb=_fallback_vram(gpu_model),
                compute_capability="",
                driver_version="",
                cpu_ram_gb=0.0,
                cpu_count=0,
                node_name=parts[0] if aggregated is False else "",
            )
        )
    return out


def _default_runner(cmd: Sequence[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def enumerate_clusters(runner: Callable[[Sequence[str]], str] = _default_runner) -> list[str]:
    """Enumerate cluster names via ``sacctmgr -n -P show clusters`` (never hardcoded)."""
    try:
        out = runner(["sacctmgr", "-n", "-P", "show", "clusters", "format=Cluster"])
    except _DISCOVERY_ERRORS as exc:
        _log.warning("discovery.enumerate_clusters_failed", error=type(exc).__name__)
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def default_profile_dir() -> Path:
    """Path to the repository's ``config/serving/`` directory."""
    return Path(__file__).resolve().parents[4] / "config" / "serving"


def discover_nodes(
    clusters: Sequence[str] | None = None,
    *,
    runner: Callable[[Sequence[str]], str] = _default_runner,
    profile_dir: str | Path | None = None,
) -> list[NodeType]:
    """Enumerate GPU node types across ``clusters``, or across all of them via ``sacctmgr``.

    This is the explicit-refresh path, called directly rather than through the CLI, and it shells
    out. Per cluster it tries three tiers in order: ``scontrol -M <c> show node --json``,
    then the lossy ``sinfo -o`` parse, then the committed ``config/serving/<cluster>.yml``
    as a last resort when the control plane is unreachable, so a cluster is never silently
    dropped. Records are keyed by cluster and node type, so two clusters sharing a node
    name yield two distinct entries.
    """
    cluster_list = list(clusters) if clusters is not None else enumerate_clusters(runner)
    prof_dir = Path(profile_dir) if profile_dir is not None else default_profile_dir()
    result: list[NodeType] = []
    for c in cluster_list:
        try:
            out = runner(["scontrol", "-M", c, "show", "node", "--json"])
            result.extend(parse_nodes_json(out, c))
            continue
        except _DISCOVERY_ERRORS as exc:
            _log.warning("discovery.scontrol_json_failed", cluster=c, error=type(exc).__name__)
        try:
            out = runner(["sinfo", "-M", c, "-h", "-o", "%n %G %m %c"])
            nodes = parse_nodes_oneline(out, c)
            if nodes:
                result.extend(nodes)
                continue
        except _DISCOVERY_ERRORS as exc:
            _log.warning("discovery.sinfo_fallback_failed", cluster=c, error=type(exc).__name__)
        # Tier 3: the committed profile, when the control plane is unreachable.
        committed = prof_dir / f"{c}.yml"
        if committed.exists():
            _log.warning("discovery.using_committed_profile", cluster=c, path=str(committed))
            result.extend(nodes_from_profile(committed))
        else:
            _log.warning("discovery.cluster_unresolved", cluster=c)
    return result


# The committed profile is the primary bring-up source; reader and writer use one library.
def _knob_dict(knobs: Any) -> dict[str, Any]:
    return dict(knobs) if isinstance(knobs, dict) else dict(asdict(knobs))


def profile_entry(
    node: NodeType,
    llm_knobs: Any,
    mt_knobs: Any,
    *,
    llm_model: str | None = None,
    mt_model: str | None = None,
) -> dict[str, Any]:
    """Build one committed-profile node entry.

    The entry holds the ``NodeType`` fields plus the LLM and MT capacity knobs, which is
    the shape ``vllm_server.bringup`` reads back. When ``llm_model`` or ``mt_model`` are
    given they are stamped into the respective knob block, so the profile records which
    model each knob set sizes; a regression test recomputes ``capacity_for`` from the model
    spec and asserts equality, so a hand-copied or mislabelled profile fails.
    """
    entry: dict[str, Any] = dict(asdict(node))
    llm = _knob_dict(llm_knobs)
    mt = _knob_dict(mt_knobs)
    if llm_model is not None:
        llm = {"model": llm_model, **llm}
    if mt_model is not None:
        mt = {"model": mt_model, **mt}
    entry["llm"] = llm
    entry["mt"] = mt
    return entry


def nodes_from_profile(path: str | Path) -> list[NodeType]:
    """Reconstruct ``NodeType`` records from a committed ``<cluster>.yml`` profile.

    This is the third discovery tier, used when a cluster's control plane is unreachable.
    """
    prof = read_profile(path)
    fields = set(NodeType.__dataclass_fields__)
    out: list[NodeType] = []
    for entry in prof.get("nodes", []):
        kwargs = {k: entry[k] for k in fields if k in entry}
        out.append(NodeType(**kwargs))
    return out


def write_profile(path: str | Path, cluster: str, entries: Sequence[dict[str, Any]]) -> Path:
    """Write ``config/serving/<cluster>.yml`` via the ruamel round-trip dumper."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"cluster": cluster, "nodes": [dict(e) for e in entries]}
    with p.open("w", encoding="utf-8") as fh:
        _YAML_WRITE.dump(doc, fh)
    return p


def read_profile(path: str | Path) -> dict[str, Any]:
    """Read a committed ``<cluster>.yml`` profile with the safe loader, no subprocess."""
    return _YAML_READ.load(Path(path).read_text(encoding="utf-8")) or {}


def resolve_knobs(
    cluster: str,
    gpu_model: str,
    *,
    profile_dir: str | Path,
) -> dict[str, Any]:
    """Look up a node type's committed LLM knobs on the bring-up path.

    Reads ``<profile_dir>/<cluster>.yml`` and never shells out to ``scontrol`` or
    ``sacctmgr``. Raises ``KeyError`` if the profile lacks the node type, which means an
    explicit :func:`write_profile` refresh is required. There is no CLI stage for this.
    """
    prof = read_profile(Path(profile_dir) / f"{cluster}.yml")
    for entry in prof.get("nodes", []):
        if gpu_model in str(entry.get("gpu_model", "")):
            return dict(entry.get("llm", {}))
    raise KeyError(
        f"no committed knobs for gpu_model={gpu_model!r} on cluster={cluster!r}; "
        f"call serving.node.discovery.write_profile to refresh config/serving/{cluster}.yml"
    )
