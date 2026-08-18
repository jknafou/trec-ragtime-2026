"""The ``sbatch`` submitter: dependency wiring and the array flag, via subprocess.

The coarse grain of the design: one array task is one claimable cell. This module owns
``afterok``/``afterany`` wiring and the GPU OR-constraint and defines no claim or
heartbeat primitives, which belong to the dynamic ``workqueue``. Nothing here imports
vLLM or SLURM Python bindings; it shells out, so a test environment without ``sbatch``
simply never calls these functions.

``gpu=True`` and therefore :func:`gpu_constraint_args` are not on the deployed path:
``cli`` submits every node with ``gpu=False`` because each template carries its own
``#SBATCH --gres`` and ``--constraint`` lines, and a second constraint from here would
conflict with them. The OR-set is kept because it is a hardware-acceptance statement
about which cards clear the semantic bucket token budget, and it is cross-checked against
the serving shape calibration; the templates repeat the same set in their own headers.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

__all__ = ["gpu_constraint_args", "submit", "submit_array"]

# A broad OR-constraint so a replica lands on any calibrated card. Each card in the set
# has a measured max_batch_tokens clearing the semantic bucket token budget of 16384;
# the RTX 3090 is excluded because its ceiling of 9216 does not, which would re-split
# every bucket and stop a shard composing identically across cards.
_GPU_CONSTRAINT = (
    "nvidia_a100_80gb_pcie|nvidia_h100_nvl|nvidia_h200_nvl|nvidia_rtx_pro_6000_blackwell"
    "|nvidia_geforce_rtx_5090"
)


def gpu_constraint_args() -> list[str]:
    """Return the ``--gres`` and ``--constraint`` args for a GPU cell."""
    return ["--gres=gpu:1", f"--constraint={_GPU_CONSTRAINT}"]


def _build_argv(
    script: str,
    *,
    after: Sequence[int] = (),
    after_any: Sequence[int] = (),
    array: int | None = None,
    export: str | None = None,
    gpu: bool = True,
    extra: Sequence[str] = (),
) -> list[str]:
    argv = ["sbatch", "--parsable"]
    if array is not None:
        argv.append(f"--array=0-{array - 1}")
    if after and after_any:
        raise ValueError(
            "pass either after (afterok) or after_any (afterany), not both: a job "
            "cannot be gated on a parent both succeeding and merely finishing"
        )
    if after:
        argv.append("--dependency=afterok:" + ":".join(str(j) for j in after))
    elif after_any:
        argv.append("--dependency=afterany:" + ":".join(str(j) for j in after_any))
    if gpu:
        argv.extend(gpu_constraint_args())
    if export:
        argv.append(f"--export={export}")
    argv.extend(extra)
    argv.append(script)
    return argv


def _parse_jobid(stdout: str) -> int:
    """Parse the numeric jobid from ``sbatch --parsable`` output (``<id>[;cluster]``)."""
    token = stdout.strip().split(";", 1)[0]
    return int(token)


def submit(
    script: str,
    *,
    after: Sequence[int] = (),
    after_any: Sequence[int] = (),
    export: str | None = None,
    gpu: bool = True,
    extra: Sequence[str] = (),
) -> int:
    """Submit ``script`` via ``sbatch --parsable``, wiring dependencies; return the jobid.

    ``after`` is ``afterok``, the right edge when the child needs the parent's
    artifact. ``after_any`` is ``afterany`` ("start when the parent finishes, however
    it finished") and exists for the pull queue's ``drive`` node: ``afterok`` on an
    N-task worker array is satisfied only if every task exits 0, so a single preempted
    task would strand the merge as ``DependencyNeverSatisfied``. ``drive`` already
    refuses to merge when ``failed/`` is non-empty or the queue is undrained, which is
    a stronger, artifact-level guard.
    """
    argv = _build_argv(
        script, after=after, after_any=after_any, export=export, gpu=gpu, extra=extra
    )
    result = subprocess.run(argv, capture_output=True, text=True, check=True)
    return _parse_jobid(result.stdout)


def submit_array(
    script: str,
    n: int,
    *,
    after: Sequence[int] = (),
    after_any: Sequence[int] = (),
    export: str | None = None,
    gpu: bool = True,
    extra: Sequence[str] = (),
) -> int:
    """Submit ``script`` as a static ``0..n-1`` array, one task per cell; return the jobid."""
    argv = _build_argv(
        script, after=after, after_any=after_any, array=n, export=export, gpu=gpu, extra=extra
    )
    result = subprocess.run(argv, capture_output=True, text=True, check=True)
    return _parse_jobid(result.stdout)
