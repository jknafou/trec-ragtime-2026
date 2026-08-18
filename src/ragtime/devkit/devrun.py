"""`DevRun`: a dev-scoped Layout, plus the containment guarantees around it.

The whole trick, and the reason this module is short, is that `common.Layout` already separates
the tree it writes to (`run_dir`) from the tree it reads the family-shared corpus and index from
(`base`). `decompose_round` and `rag_loop` resolve off `run_dir`; `corpus_dir`, `index_dir` and
`passage_store_path` resolve off `base`. So a dev run writes into a dev tree and reads the real
index with no new path machinery at all.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragtime.common import Layout
from ragtime.common.io import is_done, mark_dev
from ragtime.config import all_hashes
from ragtime.config.hashing import config_hash
from ragtime.orchestration.cli import artifact_root
from ragtime.orchestration.run_identity import run_family

__all__ = ["DevRun", "dev_root", "resolve_dev_run"]

#: Env var naming the dev artifact root. Defaults to a sibling of the production root, never a
#: child, because a child would be reachable by a production glob and the whole containment story
#: rests on dev artifacts being unaddressable rather than merely unwanted.
DEV_ROOT_ENV = "RAGTIME_DEV_ROOT"

#: Set for the lifetime of a devkit process. `orchestration.cli` refuses to launch a real run when
#: it is set, and the submission writer must refuse to write. Belt and braces, because
#: `Layout.submission()` ignores `run_dir` and is the one path that escapes the root split.
DEV_HARNESS_ENV = "RAGTIME_DEV_HARNESS"


def dev_root(production_root: Path) -> Path:
    """The dev root: `$RAGTIME_DEV_ROOT`, else a sibling of the production root."""
    env = os.environ.get(DEV_ROOT_ENV)
    if env:
        return Path(env)
    return production_root.parent / f"{production_root.name.rstrip('/')}-dev"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - provenance must never break the run
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class DevRun:
    """One dev iteration: a config, a (topic, seed) cell, and a dev-scoped Layout."""

    cfg: Any
    topic_id: str
    seed: int
    label: str
    layout: Layout
    prod_root: Path
    dev_dir: Path
    config_path: str

    def bank_path(self, round_index: int = 0) -> Path:
        return self.layout.decompose_round(round_index)

    def loop_path(self, nugget_id: str) -> Path:
        return self.layout.rag_loop(nugget_id)

    def dev_only_dir(self) -> Path:
        """`<devrun>/dev/`, a level `Layout` has no method for.

        The seam between decompose and the loops has no production artifact by design, since the
        fan-out happens in process, so materialising it must not look like a pipeline artifact to
        any production reader. Putting it under a directory `Layout` cannot name is what
        guarantees that.
        """
        d = self.dev_dir / "dev"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def assert_writes_are_contained(self) -> None:
        """Every path we write must be under the dev root; `base` is read-only.

        The failure this prevents is a dev stage rebuilding a missing corpus or index cell and
        silently contaminating the family-shared tree that every scored run reads.
        """
        dev = self.dev_dir.resolve()
        for p in (self.bank_path(0), self.dev_only_dir()):
            rp = Path(p).resolve()
            if dev not in rp.parents and rp != dev:
                raise RuntimeError(
                    f"devkit would write outside the dev root: {rp} not under {dev}. "
                    "Refusing, because a dev run must never touch the family-shared corpus."
                )
        prod = self.prod_root.resolve()
        if prod == dev or prod in dev.parents:
            raise RuntimeError(
                f"dev root {dev} is inside the production root {prod}. It must be a sibling: a "
                "child is reachable by production globs, which defeats containment."
            )


def resolve_dev_run(
    config_path: str,
    topic_id: str,
    seed: int,
    *,
    label: str = "dev",
    injected: tuple[Path, ...] = (),
    extra_meta: dict[str, Any] | None = None,
) -> DevRun:
    """Build the dev-scoped run and stamp its marker before anything is written."""
    from ragtime.config import load as load_config

    os.environ[DEV_HARNESS_ENV] = "1"
    cfg = load_config(config_path)
    prod = Path(artifact_root(cfg))
    root = dev_root(prod)
    # Key the dev path by the whole config, using the project's one canonical hasher rather than a
    # second sha256 call site. `all_hashes` returns per-block hashes and has no "config" key, so
    # asking it for one would silently yield a constant and collide every config into one
    # directory.
    cfg12 = config_hash(dict(cfg.blocks))[:12]
    # A shape that cannot parse as a production cell key (no `__` separator).
    dev_dir = root / label / cfg12 / str(topic_id) / f"seed{seed}"
    layout = Layout(
        run_dir=dev_dir,
        base=prod,                      # read the real corpus and index; never write there
        family=run_family(cfg),
        chunker_hash=all_hashes(cfg)["chunker"],
    )
    meta: dict[str, Any] = {
        "harness": "ragtime.devkit",
        "label": label,
        "config": config_path,
        "config_hash12": cfg12,
        "topic_id": topic_id,
        "seed": seed,
        "git_sha": _git_sha(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "user": os.environ.get("USER"),
        "production_root": str(prod),
        "injected": [
            {"path": str(p), "sha256": _sha256(Path(p)), "had_success": is_done(p)}
            for p in injected
        ],
    }
    meta.update(extra_meta or {})
    mark_dev(dev_dir, meta)
    run = DevRun(
        cfg=cfg, topic_id=str(topic_id), seed=int(seed), label=label,
        layout=layout, prod_root=prod, dev_dir=dev_dir, config_path=config_path,
    )
    run.assert_writes_are_contained()
    return run
