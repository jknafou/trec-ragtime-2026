"""One whole corpus rendering, three legs plus the reranker, resident on GPUs and answering a queue.

The retrieval service loads one rendering of the index once and answers every query from that
resident state, so a client pays the cold load (tens of minutes) once for a whole run instead of
once per query. It edits no production code: placement and the PLAID knobs are applied as runtime
patches, or as decisions made before the lazy production objects build their backends.

Shape
-----
The default shape is N independent replicas, one whole stack per card. A query is dispatched to one
replica and finishes entirely there (late interaction, fusion and rerank), so nothing is reconciled
across cards because nothing is split across them, and a batch of N queries costs about 1/N of the
serial wall. The opposite arrangement, sharding one query's language cells across cards
(``--plaid-shard``), measured slower overall: it moved late interaction 2.95 to 2.66 s but blew the
rerank 0.48 to 1.94 s and the total 3.65 to 4.88 s, because the sharded cells land on the card the
reranker lives on and pin it at 100 %. Splitting one query creates contention where replicating
whole stacks creates none. It is affordable because a whole stack measures 36.87 GiB (PLAID 20.50,
reranker and encoders 16.37) of a 95.59 GiB card, and because the roughly 210 GiB of dense and
sparse index is shared copy-on-write from the parent and the PLAID blobs are one shared page cache,
so two replicas cost about 410 GiB of host memory rather than 820. The parent never initialises
CUDA at all: it warms the CPU legs, forks the replicas, and then only dispatches.

``--replicas 0`` selects the older single in-parent stack, whose components are placed on cards one
by one, because a device *list* replicates a PLAID index rather than sharding it: with
``device=["cuda:0","cuda:1"]`` the same 14 parts allocated 17.903 GiB on *each* card and card 1 did
no work. That layout put PLAID alone on one card and the three query encoders plus the reranker on
the other. The arithmetic that first put an encoder beside PLAID (12.1 residency + 4.2 encoder +
2.7 scratch = 19 GiB on a 24 GiB card) missed that ``--li-workers`` *multiplies* the scratch: every
part searched concurrently holds its own 0.9 to 2 GiB working set, so a width-4 fan needs about
8 GiB rather than 2.7, and the load OOMed. A per-part scratch figure must never be added once to a
concurrent fan.

Four implementation facts carry the rest of the module:

1. PyLate's ``FastPlaid.__call__`` re-reads two 131,072-entry id-map pickles from disk on every
   call. :func:`patch_idmap_cache` memoizes them per object, measured 52.93x on GPU and
   byte-identical: it is a pure identity round-trip, since ordinals are re-resolved through our
   own ``idmap.parquet`` anyway.
2. ``Qwen/Qwen3-Reranker-4B`` is a causal LM whose relevance signal is a yes/no token logit.
   Loading it through ``AutoModelForSequenceClassification`` discards the LM head and randomly
   initialises the classifier, so two loads of the same checkpoint disagree and the ranking is a
   permutation of the input pool. :class:`~ragtime.serving.reranker.Reranker` uses the checkpoint's
   own head with Qwen's instruct template; ``--verify-reranker`` proves it by loading twice and
   comparing.
3. ``pyseismic-lsr`` does not release the GIL inside ``search``, so language cells and index parts
   searched in one process serialize however wide the thread fan is: a thread fan achieved
   0.99 to 1.00 cores at widths 2, 6, 12 and 23, and four sparse cells that cost 0.49 to 0.71 s
   each came back as 5.2 to 7.6 s summed. The host-resident legs are therefore fanned over forked
   processes, one per index part, which inherit the already-warm readers copy-on-write and never
   touch CUDA; that took one real cell from 2.9678 s to 0.1657 s at width 23, a 17.91x speed-up at
   20.43 cores, with ids and scores bit-identical.
4. PLAID mmaps the residual and code blobs, roughly 94 GiB per rendering, and faults them randomly
   4 KiB at a time. On shared storage every fault is a network round trip: 335,899 major faults in
   about 870 s, roughly 1.5 MB/s of useful pages. Neither a sequential ``read()`` prefault nor a
   larger memory grant fixes it (the 350, 460 and 560 G arms measured 951, 869 and 873 s of late
   interaction, i.e. flat): the pages are not being evicted, they are being fetched. The fix is to
   move the bytes to node-local storage (``--plaid-local``), where a fault is cheap. See
   :func:`mirror_plaid`.

One service serves one rendering
--------------------------------
``--index`` is required and closed-set (``original``, ``omt``, ``omt_opus``). There is no default,
because a default is how a service ends up answering from the wrong index while looking correct.

The argument is Knob 1 (``retrieval.index``, the index that is searched), never Knob 2
(``passage_lang``, what the LLM reads). Retrieval returns ids and scores and reads no text, so the
reading rendering is a client-side ``display()`` concern that never enters this process. The two
axes are independent:

===================== ====================== ====================
run                   retrieval.index (here) passage_lang (client)
===================== ====================== ====================
e2e-original          original               original
e2e-omt               original               omt
e2e-omt-weak          original               omt_opus
mlir-original         original               original
mlir-omt              omt                    original
mlir-omt-weak         omt_opus               original
===================== ====================== ====================

All three controlled e2e runs search ``original``, so the whole e2e roster needs one rendering
resident and only the ``mlir-*`` family needs a service per lane. Against the roughly 324 GiB of
host memory one service occupies, and one service per node, that is what makes the fleet
affordable.

Three independent guards keep a service from answering from the wrong index: ``--index`` must agree
with the config's own ``retrieval.index`` (or the launch fails); the descriptor published for
clients carries the ``(rendering, index_hash)`` pair, which is what
:func:`~ragtime.devkit.rsvc_registry.live_services` filters on; and a request that names a different
rendering is refused rather than served.

Protocol
--------
A filesystem queue rather than HTTP, because the compute nodes here route outbound HTTP through a
proxy that answers 504 for every cross-node request. A request is one JSON file dropped in
``<queue>/in/``::

    {"id": "q1", "query": "...", "index": "original", "top_k": 20, "rerank_depth": 100,
     "repeat": 3}

The reply lands atomically in ``<queue>/out/<id>.json`` (temp then rename, so a reader never sees
half). The service publishes a descriptor (``<registry>/<name>.json``) carrying the queue path,
rendering, index hash, pid, node and a heartbeat, and deletes it on exit; see :mod:`.rsvc_registry`.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # annotation-only; the runtime import stays lazy at the construction sites,
    # because this module is importable without the heavy serving stack present.
    from ragtime.serving.reranker import Reranker

GIB = 1024**3

#: The three leg names, spelled here only so this module can talk about them before importing
#: production code. The values are asserted against ``preprocess.index``'s at bring-up.
DENSE, SPARSE, LATE = "dense", "sparse", "late_interaction"

#: The closed set of renderings a service may serve. Mirrors ``common.passage_store.RENDERINGS``
#: and ``config.schema.KNOB_VALUES``, and is re-asserted against both at bring-up rather than
#: trusted: a fourth spelling here would let a typo start a service against nothing.
RENDERINGS: tuple[str, ...] = ("original", "omt", "omt_opus")


# --------------------------------------------------------------------------- #
# Runtime patches. Nothing here edits production code.
# --------------------------------------------------------------------------- #
def patch_idmap_cache() -> int:
    """Memoize PyLate's two per-call id-map unpickles. Byte-identical, and measured 52.93x on GPU.

    Returns how many methods were patched, 2 on pylate 1.6.0. A 0 here is a loud signal that the
    vendor renamed them and the speed-up is not in effect; the caller records it.
    """
    from pylate.indexes import fast_plaid as fp

    patched = 0
    for name in ("_load_documents_ids_to_plaid_ids", "_load_plaid_ids_to_documents_ids"):
        orig = getattr(fp.FastPlaid, name, None)
        if orig is None:
            continue
        attr = f"__rsvc_cache_{name}"

        def make(orig=orig, attr=attr):  # closure binds the vendor method at patch time
            def cached(self):
                got = getattr(self, attr, None)
                if got is None:
                    got = orig(self)
                    object.__setattr__(self, attr, got)
                return got

            return cached

        setattr(fp.FastPlaid, name, make())
        patched += 1
    return patched


def _parse_shard(spec: str) -> dict[str, str]:
    """`en=cuda:0,es=cuda:0,ru=cuda:1,zh=cuda:1` -> {lang: device}. Empty -> {}."""
    out: dict[str, str] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        lang, _, dev = part.partition("=")
        if not lang.strip() or not dev.strip():
            raise ValueError(f"bad --plaid-shard entry {part!r}; want lang=device")
        out[lang.strip()] = dev.strip()
    return out


def patch_plaid_kwargs(
    *,
    n_full_scores: int | None,
    show_progress: bool = False,
    low_memory: bool | None = None,
    shard: dict[str, str] | None = None,
) -> None:
    """Inject the PLAID knobs ``_open_plaid`` does not pass.

    ``show_progress=False`` removes a progress bar per part per query and is byte-identical.

    ``n_full_scores`` changes results below 4096, and the loss grows as it falls (measured curve:
    4096 identical top-20, 2048 -0.4 passages, 1024 -2.2, 256 -5.0), so it is a parameter rather
    than a default: ``None`` leaves the vendor's 8192 alone and the service records which value it
    ran.

    ``low_memory`` must be assigned rather than ``setdefault``-ed, because ``_open_plaid`` sets
    ``kwargs["low_memory"] = True`` outright on every CUDA query path, so a default would never win.
    ``None`` leaves production's choice untouched. ``low_memory=False`` is the
    full-residuals-in-VRAM form, byte-identical to ``True`` in the returned ranking (measured 20/20
    on the same queries); the only thing it buys is speed, at a documented 100-105 GiB per
    rendering against around 12 GiB.
    """
    from pylate import indexes

    orig = indexes.PLAID.__init__

    def patched(self, *a, **kw):  # same signature as the vendor __init__ it replaces
        kw.setdefault("show_progress", show_progress)
        if n_full_scores is not None:
            kw.setdefault("n_full_scores", int(n_full_scores))
        if low_memory is not None:
            kw["low_memory"] = bool(low_memory)
        if shard:
            # Route each language cell to its own card. `_open_plaid` sets one device for every part
            # from a single environment variable, so all 78 parts land on one GPU and the four cells
            # then contend for it, which is why `cell_fan` achieved 3.5x real overlap and only 1.13x
            # of wall-clock gain. The language is recoverable from `index_folder`
            # (`.../<variant>/<lang>/part-NNNNN/late_interaction`), so routing by path needs no
            # change to the production open path.
            folder = str(kw.get("index_folder") or (a[0] if a else ""))
            for lang, dev in shard.items():
                if f"/{lang}/" in folder:
                    kw["device"] = dev
                    break
        return orig(self, *a, **kw)

    indexes.PLAID.__init__ = patched


# --------------------------------------------------------------------------- #
# Page-cache prefault.
# --------------------------------------------------------------------------- #
def plaid_blobs(cells: dict[str, Any]) -> list[Path]:
    """Every mmap'd PLAID blob under the open cells, largest first.

    Read off the handles' own ``shard_dir``s rather than globbed from the index root, so a prefault
    can never warm a cell this service is not serving.
    """
    out: list[Path] = []
    for handle in cells.values():
        for part in handle.parts:
            engine = Path(part.shard_dir) / LATE
            out.extend(engine.rglob("merged_*.npy"))
            out.extend(engine.rglob("*.pkl"))
    return sorted({p for p in out if p.exists()}, key=lambda p: -p.stat().st_size)


def mirror_plaid(cells: dict[str, Any], root: Path, *, threads: int = 8) -> dict[str, Any]:
    """Copy every cell's ``late_interaction`` tree to node-local storage and search that.

    This exists because prefaulting the page cache does not work, and the reason is not memory.
    Measured at ``--mem=560G`` with the cgroup limit unset and 534 GiB free on the node: the
    service's own cgroup held 216 GiB anonymous plus 61 GiB of file cache and
    ``workingset_refault_file`` was 0, so nothing was being evicted and re-read. Yet a query still
    sat with four threads in ``folio_wait_bit_common`` and both GPUs at 0 %, having taken 335,899
    major faults in about 870 s. That is roughly 1.5 MB/s of useful pages: PLAID's access pattern
    is random 4 KiB faults, and on a network filesystem each one is a round trip.

    A sequential ``read()`` cannot fix it, because it streams at 1.7 GB/s but does not leave the
    data where a later mmap fault can find it, and raising ``--mem`` cannot fix it either: the 350,
    460 and 560 G arms measured 951, 869 and 873 s of late interaction, i.e. flat. The only fix is
    to put the bytes somewhere a fault is cheap: tmpfs, where the pages are memory and there is no
    I/O at all, or node-local NVMe.

    The mirror is a plain copy, and the search reads the same bytes from a different path, so it
    cannot change a result. ``idmap.parquet`` is still read from the original tree by
    ``LegHandle.passage_ids``: it is small, read once, and leaving it in place keeps the mirror to
    exactly the blobs that are mmap'd.
    """
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    root.mkdir(parents=True, exist_ok=True)
    assert_not_tmpfs(root)
    jobs: list[tuple[Path, Path]] = []
    for lang, handle in sorted(cells.items()):
        for i, part in enumerate(handle.parts):
            src = Path(part.shard_dir) / LATE
            dst = root / lang / f"part-{i:05d}" / LATE
            jobs.append((src, dst))

    started = time.perf_counter()

    def copy(job: tuple[Path, Path]) -> int:
        src, dst = job
        if dst.exists():  # a restart on the same node reuses the mirror
            return 0
        tmp = dst.with_name(dst.name + f".tmp-{os.getpid()}")
        shutil.copytree(src, tmp)
        tmp.replace(dst)
        return sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())

    with ThreadPoolExecutor(max_workers=max(1, threads), thread_name_prefix="rsvc-mirror") as pool:
        copied = sum(pool.map(copy, jobs))
    elapsed = time.perf_counter() - started
    mapping = {str(src): str(dst) for src, dst in jobs}
    return {
        "root": str(root),
        "parts": len(jobs),
        "copied_gib": round(copied / GIB, 2),
        "seconds": round(elapsed, 1),
        **verify_mirror(jobs),
        "_mapping": mapping,
    }


def fstype_of(path: Path) -> str:
    """The filesystem type ``path`` lives on, by longest-prefix match in ``/proc/mounts``."""
    try:
        target = str(Path(path).resolve())
    except OSError:  # pragma: no cover - telemetry only
        return ""
    best, kind = -1, ""
    try:
        for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
            bits = line.split()
            if len(bits) < 3:
                continue
            point = bits[1]
            if (target == point or target.startswith(point.rstrip("/") + "/")) and len(point) > best:
                best, kind = len(point), bits[2]
    except OSError:  # pragma: no cover - telemetry only
        return ""
    return kind


def assert_not_tmpfs(root: Path) -> None:
    """Refuse to stage the roughly 200 GiB of PLAID blobs on tmpfs. Measured fatal, not theoretical.

    ``/dev/shm`` looks like the ideal target, since the pages are memory and a fault costs nothing,
    and that is exactly the trap: tmpfs pages are unevictable and are charged to our cgroup. A job
    staging 199.65 GiB of blobs there alongside roughly 210 GiB of anonymous index against a 460 G
    grant was killed with ``OUT_OF_MEMORY 0:125``, because when the cgroup hit its limit there was
    nothing reclaimable; the kernel cannot drop a tmpfs page the way it drops a page-cache page.

    Node-local disk holds the same bytes and degrades gracefully: under pressure the kernel evicts
    page cache and the next fault is a local read rather than an out-of-memory abort. So the choice
    between them is not a performance preference, and it is checked here rather than left to whoever
    writes the next sbatch.
    """
    kind = fstype_of(root)
    if kind in {"tmpfs", "ramfs"}:
        raise RuntimeError(
            f"--plaid-local {root} is on {kind}, and the ~200 GiB of PLAID blobs must not be staged "
            "there: tmpfs pages are unevictable and charged to this cgroup, so blobs + the ~210 "
            "GiB anonymous index exceed the grant with nothing reclaimable and the job is "
            "OOM-killed (observed: OUT_OF_MEMORY 0:125). Use node-local disk (/hosttmp), whose "
            "page cache degrades gracefully instead."
        )


def verify_mirror(jobs: list[tuple[Path, Path]]) -> dict[str, Any]:
    """Every source file must exist in the mirror at the same size, or bring-up dies here.

    A silently short mirror is the worst outcome available in this module: :func:`patch_plaid_open`
    re-points the engine at the mirrored path, so a missing blob does not degrade to a slow read off
    shared storage, it breaks a part. And a broken part does not surface at load: a service was
    observed publishing ``ready: true`` and advertising all 78 parts with nine part loads already
    failed, and the first witness was a query. Checking it costs about 500 stat calls against a
    roughly 20-minute bring-up, so it runs unconditionally rather than behind a flag.

    Sizes, not hashes: this is a plain ``copytree``, so a size mismatch is the only corruption the
    copy path can produce, and hashing 200 GiB would cost more than making the mirror.
    """
    missing: list[str] = []
    wrong_size: list[str] = []
    files = 0
    for src, dst in jobs:
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            files += 1
            want = path.stat().st_size
            got = dst / path.relative_to(src)
            try:
                have = got.stat().st_size
            except OSError:
                missing.append(str(got))
                continue
            if have != want:
                wrong_size.append(f"{got} ({have} != {want})")
    if missing or wrong_size:
        raise RuntimeError(
            f"PLAID mirror incomplete: {len(missing)} missing and {len(wrong_size)} wrong-size "
            f"of {files} files under {len(jobs)} parts. Refusing to continue: the engine reads "
            "the mirror and not the original, so serving this would publish `ready: true` over a "
            f"partial index. first_missing={missing[:3]} first_wrong_size={wrong_size[:3]}"
        )
    return {"verified_files": files}


class Background:
    """One bring-up phase running in a thread, whose failure is fatal at join time.

    A thread is safe here for exactly one phase. The cold load is two independent chains, not one
    line: ``warm_cpu_legs`` reads the dense (FAISS) and sparse (Seismic) indexes into anonymous
    memory, while :func:`mirror_plaid` copies the late-interaction blobs to node-local storage.
    Different files, different legs, no ordering between them, and the mirror is pure ``shutil`` and
    ``os`` filesystem work, so it cannot initialise CUDA in the pre-fork window. Both phases spend
    their time in the kernel with the GIL released, which is why threads rather than another process
    are enough, and they measurably move bytes at very different rates (around 330 MB/s of many
    small structured reads against around 1.3 GB/s of large sequential copy), so neither saturates
    the filesystem alone.

    It must be joined before :class:`CellPool` forks, and :meth:`result` is the only way to read
    it, so joining is not a step that can be forgotten. Two reasons, both load-bearing: forking a
    process with a live worker thread leaves the child holding locks nobody will release, and a
    mirror that failed must abort bring-up rather than let the service publish ``ready: true`` over
    an index whose late-interaction leg is not there. ``future.result()`` re-raises the worker's
    exception, with its original traceback, in the parent.
    """

    def __init__(self, name: str, fn: Any, *args: Any, **kw: Any) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.name = name
        self.started = time.perf_counter()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"rsvc-bg-{name}")
        self._future = self._pool.submit(fn, *args, **kw)

    def result(self) -> Any:
        """Block until the phase finishes, re-raise its exception, and leave no thread behind."""
        try:
            return self._future.result()
        finally:
            self._pool.shutdown(wait=True)

    def waited_s(self) -> float:
        return round(time.perf_counter() - self.started, 1)


def cuda_witnesses() -> dict[str, Any]:
    """Has this process touched CUDA yet? Two witnesses, neither of which initialises it.

    ``torch.cuda.is_initialized()`` reads a module-level flag and creates no context, and torch is
    consulted only if it is already in ``sys.modules``, so this can never be the thing that imports
    it. The stronger, library-independent witness is the kernel's: a CUDA context always opens
    ``/dev/nvidiactl`` and ``/dev/nvidia<N>``, so an fd table with no ``/dev/nvidia*`` entry is
    proof that no context exists whichever library might have made one. Live thread names come along
    because the same moment needs both facts; see :func:`assert_pre_fork`.
    """
    torch_flag: bool | None = None
    mod = sys.modules.get("torch")
    if mod is not None:
        try:
            torch_flag = bool(mod.cuda.is_initialized())
        except Exception:  # noqa: BLE001 - a witness must never be the thing that fails
            torch_flag = None
    fds: list[str] = []
    try:
        for entry in sorted(Path("/proc/self/fd").iterdir()):
            try:
                target = os.readlink(str(entry))
            except OSError:  # the fd closed between listdir and readlink
                continue
            if target.startswith("/dev/nvidia"):
                fds.append(target)
    except OSError:  # pragma: no cover - telemetry only
        pass
    return {
        "torch_cuda_initialized": torch_flag,
        "nvidia_fds": sorted(set(fds)),
        "threads": sorted(t.name for t in threading.enumerate()),
    }


def assert_pre_fork(what: str) -> dict[str, Any]:
    """Refuse to fork if CUDA is live in this process. Checked, not reasoned about.

    The cell workers are forked so they inherit the dense and sparse index copy-on-write, and that
    is only sound because they never touch CUDA: a forked child cannot use a CUDA context created
    before the fork, and the damage does not appear at fork time, it appears at query time as a
    corrupt or hung child. Overlapping the mirror with the CPU warm moves work around inside exactly
    this window, so the ordering stops being a comment and becomes an assertion.
    """
    got = cuda_witnesses()
    if got["nvidia_fds"] or got["torch_cuda_initialized"]:
        raise RuntimeError(
            f"CUDA is already live in this process before {what}: "
            f"nvidia_fds={got['nvidia_fds']} torch_cuda_initialized="
            f"{got['torch_cuda_initialized']}. Forking now would hand the cell workers a CUDA "
            "context they cannot use, and the failure would surface at query time rather than "
            "here. Every phase before the fork must be CPU/filesystem only."
        )
    return got


def patch_plaid_open(mapping: dict[str, str]) -> None:
    """Point the late-interaction leg's engine at the mirrored path. Same bytes, cheaper faults.

    Registered through the hook production owns rather than by reassigning the leg class from
    outside, so the indirection is visible to anyone reading that class, survives a signature
    change, and cannot compose unpredictably with a second patch.
    """
    from ragtime.preprocess.index import set_leg_dir_mirror

    set_leg_dir_mirror(mapping)


def prefault(paths: list[Path], *, threads: int = 8, chunk: int = 32 << 20) -> dict[str, Any]:
    """Stream every file through ``read()`` so the kernel holds it in page cache.

    Sequential ``read()`` is the whole trick: PLAID's own access pattern is random faults over an
    mmap, measured at about 163 MB/s on this filesystem, where a sequential stream runs at GB/s.
    This does not pin the pages, which remain evictable; it only means the first query is not the
    one that pays.
    """
    from concurrent.futures import ThreadPoolExecutor

    total = sum(p.stat().st_size for p in paths)
    started = time.perf_counter()

    def stream(path: Path) -> int:
        got = 0
        with open(path, "rb", buffering=0) as fh:
            while True:
                block = fh.read(chunk)
                if not block:
                    break
                got += len(block)
        return got

    # Measured on our cgroup, not the node: a node-wide reading cannot tell "we retained nothing"
    # from "someone else's 341 GiB of page cache moved", and it has raised exactly that false alarm.
    before = host_memory()
    key = "cg_file_gib" if "cg_file_gib" in before else "page_cache_gib"
    cache_before = before.get(key, 0.0)
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        read = sum(pool.map(stream, paths))
    elapsed = time.perf_counter() - started
    cache_after = host_memory().get(key, 0.0)
    retained = round(cache_after - cache_before, 1)
    want = round(total / GIB, 2)
    out = {
        "files": len(paths),
        "bytes_gib": want,
        "seconds": round(elapsed, 1),
        "mb_per_s": round(read / max(elapsed, 1e-9) / (1 << 20), 1),
        "page_cache_retained_gib": retained,
        "measured_on": "cgroup" if key.startswith("cg_") else "node",
    }
    # A prefault that streams the whole blob set and retains nothing has not warmed anything: the
    # cgroup evicted the pages as fast as they arrived, which is what happens when `--mem` cannot
    # hold the ~220 GiB of anonymous index plus the blobs. That is not a slow service, it is a
    # service whose every query will sit faulting from shared storage -- diagnosed at --mem=350G,
    # where a single warm-up query had not returned after 10 minutes with both GPUs at 0 %. Saying
    # so at boot is far cheaper than diagnosing it from a stalled query.
    if retained < 0.5 * want:
        out["WARNING"] = (
            f"retained only {retained} GiB of {want} GiB; the page cache is being evicted, so "
            "PLAID will fault from beegfs on every query. Raise --mem (need roughly "
            "anonymous-index + blobs, ~330 GiB total for one rendering) or serve fewer renderings "
            "per node."
        )
    return out


# --------------------------------------------------------------------------- #
# The reranker, using the checkpoint's own head.
# --------------------------------------------------------------------------- #


def merge_parts(
    handle: Any, leg: str, rep: Any, top_k: int, workers: int, omp: int = 0
) -> list:
    """Search one leg of a whole cell from an already-computed representation.

    Byte-identical to ``preprocess.index.query_lang_leg`` for the same ``rep``: the same per-part
    ``search_with_rep``, the same raw-score merge across parts, which share one score space, the
    same total ``(-score, passage_id)`` order, and the same per-part top-k, since the global top-k
    is a subset of the union of the per-part top-ks. It exists only because the query is encoded in
    the parent, on the GPU, and shipped here, so the cell worker must not re-encode.

    ``omp`` is applied inside the pool worker, and that is not a detail. OpenMP's ``nthreads-var``
    is a per-thread ICV: ``omp_set_num_threads`` binds only the calling thread, and every thread the
    pool creates starts from the ``OMP_NUM_THREADS`` environment value instead. Measured on a
    32-core allocation with ``OMP_NUM_THREADS=32``, setting it on the main thread and then fanning
    over a 14-part cell::

        fan 1: omp 1/2/4/8/16 -> 0.84 / 2.01 / 3.89 / 7.67 / 15.75 cores   (obeyed exactly)
        fan 2: omp 1/2/4/8/16 -> 22.8 / 21.9 / 24.2 / 22.5 / 21.5 cores    (ignored entirely)

    So a service that thought it had capped FAISS at one thread was running it at about eleven per
    fan worker. Calling it per task is what makes the cap real.
    """
    from ragtime.preprocess.index import search_cell_with_rep

    # A delegation rather than a second implementation, so the equivalence claimed above is true by
    # construction. The per-task OMP cap this function pioneered was ported into `_fan_parts`, where
    # the production fan needed it too.
    return search_cell_with_rep(handle, leg, rep, top_k, workers=workers, omp=omp)


def warm_cpu_legs(cells: dict[str, Any], legs: tuple[str, ...], *, workers: int = 4) -> dict[str, Any]:
    """Materialize the dense and sparse readers and id maps for every part. No CUDA is touched.

    This must happen before the fork: what the children inherit copy-on-write is exactly what is
    resident here, and the roughly 208 GiB of FAISS plus Seismic state is the whole reason a
    process per part is affordable at all.

    Concurrent because it measured 854.9 s sequentially, 64 % of a 1331 s cold load, and because it
    is a filesystem read rather than a computation: ``faiss.read_index`` and Seismic's Rust loader
    both release the GIL while streaming. Each part is an independent object with its own caches, so
    the only shared state is the filesystem. Width is bounded because 78 concurrent multi-GiB reads
    would spike well past the memory the finished state occupies.
    """
    from concurrent.futures import ThreadPoolExecutor

    started = time.perf_counter()
    jobs = [(part, leg) for handle in cells.values() for part in handle.parts for leg in legs]

    def warm(job: tuple[Any, str]) -> None:
        part, leg = job
        part.reader(leg)
        part.passage_ids(leg)

    if workers <= 1:
        for job in jobs:
            warm(job)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rsvc-warm") as pool:
            list(pool.map(warm, jobs))
    return {
        "parts": sum(len(h.parts) for h in cells.values()),
        "legs": list(legs),
        "workers": workers,
        "seconds": round(time.perf_counter() - started, 1),
    }


def _faiss_omp(threads: int) -> None:
    """Set FAISS's own OpenMP width in this thread, delegating to the one implementation.

    ``OMP_NUM_THREADS`` is read once when libgomp initialises, so it can only be swept by launching
    a service per rung, and a rung costs a full cold load. ``faiss`` exposes the setter, which is
    what makes the dense leg's OMP axis measurable against one resident index. A no-op for the
    sparse leg: Seismic is Rust and Rayon.
    """
    from ragtime.preprocess.index import _faiss_omp as _one_implementation

    _one_implementation(threads)


def part_slices(n_parts: int, width: int) -> tuple[tuple[int, int], ...]:
    """Partition ``n_parts`` into at most ``width`` contiguous ``[lo, hi)`` slices.

    Contiguity is the whole correctness argument of the part fan. A cell's merge concatenates its
    parts' hit lists and sorts them on ``(-score, passage_id)``; contiguous slices mean the
    concatenation of the slices, taken in slice-index order, is the concatenation of the parts in
    part order, byte-identical to the sequential loop. A round-robin or work-stealing partition
    would produce the same set in a different sequence, and any tie the sort could not break would
    then resolve by which worker happened to answer first. ``preprocess.index._fan_parts`` obeys the
    same rule one level down, for the same reason.

    ``width`` is clamped to ``[1, n_parts]`` and empty slices are dropped, so a width larger
    than the part count degrades to one part per slice instead of to idle workers.
    """
    n = max(0, int(n_parts))
    if n == 0:
        return ()
    width = max(1, min(int(width), n))
    bounds = ((i * n // width, (i + 1) * n // width) for i in range(width))
    return tuple((lo, hi) for lo, hi in bounds if lo < hi)


def device_resident_legs(handle: Any, legs: Sequence[str]) -> frozenset[str]:
    """Which of ``legs`` this cell searches on a card: the legs a part fan must not split.

    A part-slice fan buys its speed-up by turning threads into processes, and a process that
    searches a device-resident leg pays a whole CUDA context on that card. Measured with the dense
    leg pinned to a card, four cell processes held 11966/10544/7764/10984 MiB against 38810 MiB of
    fp32 dense tensors, i.e. 612 MiB of context per process. One process per part is 78 of them:
    46.6 GiB of context on top of 37.9 GiB of tensors, 84.5 GiB on an 80 GiB card. That is not a
    slow service but a CUDA OOM at first query, and on the PLAID path an OOM arrives as an
    uncatchable Rust ``PanicException`` that takes the whole service down.

    So the rule is: host-resident legs are partitioned, device-resident legs are not. The
    device-resident leg is searched whole, by the cell's first slice worker, sequentially before
    that worker's own sparse slice, which is exactly the floor the fan is designed against,
    ``sparse(W) + dense``. Splitting the dense leg off into its own concurrent process is a
    different change, making the term ``max(sparse(W), dense)``, and is not made here.

    Detected by capability rather than by re-reading the config: a lazy device-resident matrix
    carries the ``device`` it will materialise on, while a ``faiss`` index and a Seismic reader
    carry nothing. Duck-typing keeps this module free of an import edge into a leg implementation,
    and keeps it correct on a tree where that module does not exist at all. It reads only
    ``parts[0]``: a cell's parts are opened by one ``LegHandle.open`` per leg, so they cannot
    disagree, and ``warm_cpu_legs`` has already materialised the reader, so this costs a dict
    lookup and touches no CUDA.
    """
    parts = getattr(handle, "parts", ()) or ()
    if not parts:
        return frozenset()
    resident = []
    for leg in legs:
        try:
            device = str(getattr(parts[0].reader(leg), "device", "") or "")
        except Exception:  # noqa: BLE001 - an unopenable leg is not a device-resident one
            device = ""
        if device and device != "cpu":
            resident.append(leg)
    return frozenset(resident)


def merge_slices(per_slice: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    """Fold one cell's slice replies back into the single per-cell reply the caller expects.

    ``per_slice`` is indexed by slice, never by arrival: the caller receives in slice order, so this
    function never learns which worker finished first and cannot be made to depend on it. Per leg it
    concatenates in that order, applies the same total ``(-score, passage_id)`` sort the whole-cell
    path applies, and truncates to ``top_k``.

    The truncation is not an optimisation, it is the contract. The fused ranker downstream is
    rank-based RRF, so a pool of ``slices x top_k`` entries would fuse differently from a pool of
    ``top_k``. And it is lossless for the same reason the per-part top-k is: the global top-k of a
    cell is a subset of the union of its slices' top-ks. The order is identical because
    ``(-score, passage_id)`` is a total order, since a passage id occurs in exactly one part, so
    sorting any permutation of the surviving multiset yields the same list.

    ``wall`` is the max across slices, because they ran concurrently and their walls do not sum, and
    ``cpu_s`` is the sum, because CPU seconds do. ``cpu_s / wall`` therefore keeps meaning exactly
    what it meant for one process: the number of cores the leg achieved.
    """
    if len(per_slice) == 1:
        return per_slice[0]
    legs = [leg for leg in per_slice[0] if not leg.startswith("_")]
    merged: dict[str, Any] = {}
    for leg in legs:
        hits: list[tuple[str, float]] = []
        wall = 0.0
        cpu_s = 0.0
        err: str | None = None
        for got in per_slice:
            entry = got.get(leg)
            if entry is None:
                continue
            hits.extend(entry[0])
            wall = max(wall, float(entry[1]))
            if entry[2] and err is None:
                err = entry[2]
            if len(entry) > 3:
                cpu_s += float(entry[3])
        hits.sort(key=lambda kv: (-kv[1], kv[0]))
        merged[leg] = (hits[: int(top_k)], round(wall, 4), err, round(cpu_s, 4))
    merged["_fan"] = per_slice[0].get("_fan")
    merged["_slices"] = len(per_slice)
    return merged


def _cell_loop(
    handle: Any,
    conn: Any,
    part_workers: int,
    *,
    bounds: tuple[int, int] | None = None,
    whole_legs: frozenset[str] = frozenset(),
    skip_legs: frozenset[str] = frozenset(),
) -> None:
    """One forked worker: ``(reps, top_k, opts) -> {leg: (hits, wall, err, cpu_s)}``.

    The unit of work is a contiguous part slice of one language cell, ``bounds``, not the whole
    cell: ``pyseismic-lsr`` holds the GIL for the whole of ``SeismicIndex.search``, so the thread
    fan one level down achieved 0.99 to 1.00 cores at widths 2, 6, 12 and 23. The fan ran, and every
    worker in it was blocked. Processes are the mechanism that works: on a real 23-part cell of
    2,906,906 passages, threads measured 2.9678 s against 0.8293 s at fork width 4, 0.2993 s at 12
    and 0.1657 s at 23, a 17.91x speed-up at 20.43 cores, with ids and scores bit-identical across
    every arm.

    ``whole_legs`` are searched over the whole cell rather than this slice, and ``skip_legs`` are
    not searched at all: the two halves of "exactly one worker per cell owns the device-resident
    leg", see :func:`device_resident_legs`. A skipped leg still gets an entry, empty and at zero
    wall, so the gather never has to know which worker owned what.

    ``cpu_s`` is :func:`time.process_time` across the leg, the CPU seconds this process burned in
    all its threads. ``cpu_s / wall`` is therefore the number of cores the leg actually used,
    measured rather than inferred, and it is the only signal that separates "the fan is running"
    from "the fan is running and every worker is blocked on the GIL". It is per process, which is
    why it stays honest under a fan of processes and lied under a fan of threads.
    """
    import dataclasses

    n_parts = len(getattr(handle, "parts", ()) or ())
    lo, hi = (0, n_parts) if bounds is None else (int(bounds[0]), int(bounds[1]))
    # The slice handle is built once, outside the request loop: it is a shallow view over the same
    # already-warm `LegHandle` objects (a `dataclasses.replace` on a slots dataclass, not a
    # re-open), so it re-loads nothing and re-reads nothing from shared storage.
    mine = (
        handle
        if (lo, hi) == (0, n_parts)
        else dataclasses.replace(handle, parts=tuple(handle.parts)[lo:hi])
    )
    while True:
        try:
            msg = conn.recv()
        except (EOFError, KeyboardInterrupt):
            return
        if msg is None:
            return
        opts: dict[str, Any] = {}
        if len(msg) == 3:
            reps, top_k, opts = msg
        else:  # the 2-tuple spelling, kept so an older client is not broken
            reps, top_k = msg
        fan = int(opts.get("part_workers") or part_workers)
        omp = int(opts.get("omp") or 0)
        _faiss_omp(omp)  # binds this thread; `merge_parts` binds each pool worker
        out: dict[str, Any] = {}
        for leg, rep in reps.items():
            if leg in skip_legs:
                # Another slice of this cell owns it whole. An explicit empty entry, not a missing
                # key: `merge_slices` then folds every worker identically and the gather has no
                # per-leg special case to get wrong.
                out[leg] = ([], 0.0, None, 0.0)
                continue
            target = handle if leg in whole_legs else mine
            started = time.perf_counter()
            cpu0 = time.process_time()
            try:
                hits = merge_parts(target, leg, rep, int(top_k), fan, omp)
                err = None
            except Exception as exc:  # noqa: BLE001 - a bad leg must not kill the worker
                hits, err = [], f"{type(exc).__name__}: {exc}"
            out[leg] = (
                hits,
                round(time.perf_counter() - started, 4),
                err,
                round(time.process_time() - cpu0, 4),
            )
        out["_fan"] = (fan, omp)
        conn.send(out)


class CellPool:
    """One forked process per ``(language cell, contiguous part slice)``.

    Fork, not spawn: a spawned child would re-load the whole index. Forked after
    :func:`warm_cpu_legs` and before the parent initialises CUDA, so the index is shared
    copy-on-write and no child ever inherits a CUDA context it could corrupt.

    The partition is on the part, not on the language: the unit of work is the index part (78
    here: en 23, es 20, ru 14, zh 21), and the inner per-part fan is a ``ThreadPoolExecutor``
    against a library that holds the GIL, so forking per language cell would give four real
    processes with everything inside them serial. This module's own instrumentation names that
    shape: ``sparse_cores`` sitting at exactly 1.00 in every cell at every width is what it looks
    like.

    Two properties are structural rather than reviewed:

    * Slices are contiguous and the gather is keyed by slice index (:func:`part_slices`,
      :func:`merge_slices`), so concatenation stays in part order and no tie can resolve by arrival.
      That is what makes the fan invisible in the output.
    * Device-resident legs are not partitioned (:func:`device_resident_legs`), because a process
      that searches one costs a whole CUDA context on that card: 612 MiB measured, or 46.6 GiB
      across 78 workers, on top of 37.9 GiB of tensors.

    ``part_procs`` is the total process budget across all cells: ``<= 0``, the default, means one
    process per part, the measured optimum, and a positive value is shared out in proportion to each
    cell's part count. Beware quantisation when capping: a cell's wall is
    ``ceil(parts / slices) x per-part``, so a width that does not divide the part count wastes a
    whole round -- widths 12 and 16 both read about 0.29 s on a 23-part cell. One process per part
    avoids the question entirely.
    """

    def __init__(
        self,
        cells: dict[str, Any],
        *,
        part_workers: int,
        part_procs: int = 0,
        legs: Sequence[str] = (DENSE, SPARSE),
    ) -> None:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        self.langs = sorted(cells)
        self._conns: dict[str, list[Any]] = {}
        self._procs: dict[str, list[Any]] = {}
        self.plan: dict[str, Any] = {}
        total_parts = sum(len(cells[lang].parts) for lang in self.langs) or 1
        budget = int(part_procs)
        started = time.perf_counter()
        for lang in self.langs:
            handle = cells[lang]
            n_parts = len(handle.parts)
            width = n_parts if budget <= 0 else max(1, round(budget * n_parts / total_parts))
            slices = part_slices(n_parts, width)
            # Only worth naming when the cell is actually split: at one slice the whole/skip split
            # is a no-op and the worker is exactly a per-language process.
            whole = device_resident_legs(handle, legs) if len(slices) > 1 else frozenset()
            self._conns[lang] = []
            self._procs[lang] = []
            for i, (lo, hi) in enumerate(slices):
                parent_conn, child_conn = ctx.Pipe(duplex=True)
                proc = ctx.Process(
                    target=_cell_loop,
                    args=(handle, child_conn, part_workers),
                    kwargs={
                        "bounds": (lo, hi),
                        # Slice 0 owns every device-resident leg whole; every other slice is told
                        # not to touch it, so exactly one CUDA context per cell exists.
                        "whole_legs": whole if i == 0 else frozenset(),
                        "skip_legs": frozenset() if i == 0 else whole,
                    },
                    name=f"rsvc-cell-{lang}-{i:02d}",
                    daemon=True,
                )
                proc.start()
                child_conn.close()
                self._conns[lang].append(parent_conn)
                self._procs[lang].append(proc)
            self.plan[lang] = {
                "parts": n_parts,
                "slices": [[lo, hi] for lo, hi in slices],
                "whole_legs": sorted(whole),
            }
        self.workers = sum(len(v) for v in self._procs.values())
        self.fork_s = round(time.perf_counter() - started, 2)

    def alive(self) -> dict[str, bool]:
        """One verdict per cell: every one of its slice workers is up. A short cell is dead."""
        return {
            lang: all(proc.is_alive() for proc in procs) for lang, procs in self._procs.items()
        }

    def scatter(self, reps: dict[str, Any], top_k: int, opts: dict[str, Any] | None = None) -> None:
        """Send the same reps to every slice worker and return without waiting.

        Split from :meth:`gather` because the caller drives PLAID on the GPU in between, and a
        combined send-and-receive would block there and destroy that overlap.
        """
        payload = (reps, int(top_k), dict(opts or {}))
        for conns in self._conns.values():
            for conn in conns:
                conn.send(payload)

    def gather(self, top_k: int) -> dict[str, dict[str, Any]]:
        """Collect every slice in slice order and fold each cell back to one reply.

        ``recv`` is called in slice order, so the list handed to :func:`merge_slices` is indexed by
        slice and never by arrival. That costs nothing, since the workers run concurrently and
        waiting for slice 0 first still ends when the slowest ends, and it is the reason a tie in
        ``(-score, passage_id)`` cannot resolve by which worker was quicker.
        """
        return {
            lang: merge_slices([conn.recv() for conn in conns], top_k)
            for lang, conns in sorted(self._conns.items())
        }

    def search(
        self, reps: dict[str, Any], top_k: int, opts: dict[str, Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Fan the same reps to every cell, then collect. The scatter is what parallelizes."""
        self.scatter(reps, top_k, opts)
        return self.gather(top_k)

    def close(self) -> None:
        for conns in self._conns.values():
            for conn in conns:
                try:
                    conn.send(None)
                except (BrokenPipeError, OSError):
                    pass
        for procs in self._procs.values():
            for proc in procs:
                proc.join(timeout=10)


# --------------------------------------------------------------------------- #
# The rerank fan: N reranker instances, one per card, over one deduplicated pool.
# --------------------------------------------------------------------------- #
def batch_aligned_slices(n: int, shards: int, batch: int) -> list[tuple[int, int]]:
    """Split ``n`` candidates into at most ``shards`` contiguous slices, on batch boundaries.

    Alignment is what makes a sharded rerank bit-identical rather than merely equivalent.
    ``Reranker.score`` batches in input order with no sorting, ``prompts[i : i + batch]``, and every
    row in a batch is padded to that batch's longest row, so a row's logits depend on which rows
    share its batch. Cut the pool at an arbitrary index and the batches either side of the cut
    acquire different membership, different padding, a different reduction and last-bit different
    scores; the ``(-score, passage_id)`` sort then reorders near-ties, and the change is invisible
    until it moves a rank. Cut only at multiples of ``batch`` and every batch a shard forms is the
    same batch the single instance would have formed, so the scores are the same bits, which is a
    property rather than a tolerance.

    Balance is by batch count, so the slowest instance sets the wall by at most one batch. The
    remainder batch, when ``n % batch``, rides inside its own slice and is never split.
    """
    n, batch = int(n), max(1, int(batch))
    if n <= 0:
        return []
    n_batches = (n + batch - 1) // batch
    width = max(1, min(int(shards), n_batches))
    out = []
    for i in range(width):
        lo_b, hi_b = i * n_batches // width, (i + 1) * n_batches // width
        if lo_b < hi_b:
            out.append((lo_b * batch, min(hi_b * batch, n)))
    return out


def _rerank_loop(model: str, device: str, batch_size: int, max_length: int, conn: Any) -> None:
    """One reranker instance, pinned to one card, in its own process. Loads, then serves.

    A process and not a thread, and that is the load-bearing choice. Threads would share this
    interpreter, and the rerank payload is not purely GPU-bound: it tokenizes and it drives the
    forward pass from Python. One thread per instance would hand the whole fan to the GIL and
    measure about 1x on N cards, which is the same failure the sparse cell fan has one layer up.
    Separate processes have no shared interpreter and each gets its own CUDA context on its card.

    Two things travel back that the caller cannot obtain any other way. The ready record carries the
    card the weights actually landed on, read off a parameter rather than off the argument that
    asked for it, because a declared placement is not a placement. Each reply carries ``t0``/``t1``
    on the wall clock, because the concurrency claim is that N instances' scoring intervals overlap,
    and intervals are only comparable across processes on a shared clock.
    """
    try:
        import torch

        from ragtime.serving.reranker import Reranker

        rr = Reranker(model, device, batch_size=batch_size, max_length=max_length)
        t = time.perf_counter()
        rr.load()
        param = next(rr._model.parameters())
        conn.send({"ready": True, "device": device, "pid": os.getpid(),
                   "actual_device": f"{param.device.type}:{param.device.index}",
                   "dtype": str(param.dtype),
                   "load_s": round(time.perf_counter() - t, 1),
                   "vram_alloc_gib": round(torch.cuda.memory_allocated(device) / 2**30, 3),
                   "vram_reserved_gib": round(torch.cuda.memory_reserved(device) / 2**30, 3)})
    except Exception as exc:  # noqa: BLE001 - a dead instance must report, not just vanish
        import traceback

        try:
            conn.send({"ready": False, "device": device, "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()[-2000:]})
        except (BrokenPipeError, OSError):
            pass
        return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, KeyboardInterrupt):
            return
        if msg is None:
            return
        query, texts = msg
        t0 = time.time()
        cpu0 = time.process_time()
        try:
            scores = rr.score(query, list(texts)) if texts else []
            err = None
        except Exception as exc:  # noqa: BLE001 - one bad shard fails the query, not the fleet
            scores, err = [], f"{type(exc).__name__}: {exc}"
        conn.send({"scores": scores, "error": err, "t0": t0, "t1": time.time(),
                   "cpu_s": round(time.process_time() - cpu0, 4), "n": len(texts),
                   "pid": os.getpid(), "device": device})


class RerankPool:
    """``N`` reranker instances, one per card, scoring disjoint slices of one pool.

    The phases of a query are sequential (legs, fuse, rerank), so during the rerank the cards
    holding the retrieval legs sit idle, and during the legs the rerank card sits idle. Measured on
    the standing query: ``late_interaction`` 0.858 s and ``rerank`` 4.929 s of a 7.860 s service
    wall, so on a six-card node five of the six do nothing for about 63 % of every query. Putting a
    reranker on the idle cards costs VRAM they would otherwise waste and buys the rerank a factor
    of N.

    One instance per card is an invariant rather than a preference, and it is refused here rather
    than measured. Two instances on one card share its SMs: the kernels interleave, the wall is the
    sum, and the only thing that scales is the VRAM bill. It is refused twice: once on the argv, for
    a repeated device, and once on the loaded weights, for two instances whose parameters report the
    same card or an instance whose parameters did not land where it asked. The second check is the
    one that catches a placement that is specified correctly and executed wrongly.

    Bit-identical, by alignment. Slices are contiguous and cut on ``batch_size`` boundaries
    (:func:`batch_aligned_slices`), so every shard re-forms the same batches the single instance
    would have formed. The gather is keyed by shard index, never arrival order, and the caller's
    ``(-score, passage_id)`` sort is untouched. Scores are one model's ``log P(yes)`` on one scale,
    which is what makes them comparable across instances at all. :meth:`score_solo` lets that
    identity be measured on the live service instead of only argued.

    The pool is a set-scorer. Its input is the fused candidate pool, which ``common.fusion.rrf`` has
    already collapsed to unique ids: twelve overlapping leg pools become one dict keyed by
    ``passage_id``, about 900 of them. Scoring a duplicate would waste a slot and could hand one id
    two different scores, since two shards mean two batch compositions, so the caller asserts
    uniqueness going in and coming out.
    """

    def __init__(self, model: str, devices: list[str], *, batch_size: int, max_length: int,
                 log: Any) -> None:
        import multiprocessing as mp

        self.devices = [d.strip() for d in devices if d and d.strip()]
        if not self.devices:
            raise ValueError("RerankPool needs at least one device")
        bad = [d for d in self.devices if not d.startswith("cuda:") or not d[5:].isdigit()]
        if bad:
            raise SystemExit(
                f"--rerank-devices names {bad}, which is not an ordinal-bearing CUDA device. A "
                "bare 'cuda' cannot be proven distinct from another bare 'cuda', and a CPU "
                "instance would set the wall for the whole fan; spell every card explicitly."
            )
        dupes = sorted({d for d in self.devices if self.devices.count(d) > 1})
        if dupes:
            raise SystemExit(
                f"--rerank-devices names {dupes} more than once. One reranker instance per card "
                "is an invariant: two instances on one card share its SMs, so they serialise and "
                "you pay 2x the VRAM for ~1x the throughput. Refusing to start rather than "
                "publishing a fleet that looks N-wide and is not."
            )
        self.batch_size = int(batch_size)
        self.last_shards: list[dict[str, Any]] = []
        self.last_scatter_s = 0.0
        self._poisoned: str | None = None
        assert_pre_fork("the rerank pool's fork")
        ctx = mp.get_context("fork")
        self._conns: list[Any] = []
        self._procs: list[Any] = []
        started = time.perf_counter()
        for i, device in enumerate(self.devices):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                target=_rerank_loop,
                args=(model, device, int(batch_size), int(max_length), child_conn),
                name=f"rsvc-rerank-{i}-{device.replace(':', '')}",
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(proc)
        # Every instance loads concurrently on its own card, so this wait is about one load, not N.
        self.ready: list[dict[str, Any]] = []
        for i, conn in enumerate(self._conns):
            try:
                got = conn.recv()
            except EOFError as exc:
                self.close()
                raise RuntimeError(
                    f"rerank instance {i} ({self.devices[i]}) died during load without reporting "
                    "ready, refusing to publish a service that would rerank on fewer cards than "
                    "it advertises"
                ) from exc
            if not got.get("ready"):
                self.close()
                raise RuntimeError(
                    f"rerank instance {i} ({self.devices[i]}) failed to load: {got.get('error')}\n"
                    f"{got.get('traceback', '')}"
                )
            self.ready.append(got)
            log("rerank_instance_ready", instance=i, device=self.devices[i], boot=got)
        # The placement, verified on the weights. Everything above proves the argv was sane; this
        # proves the load obeyed it. Two instances reporting one card would serialise on that
        # card's SMs while the fleet advertised N-wide throughput, silently.
        placed: dict[str, int] = {}
        for i, got in enumerate(self.ready):
            actual = str(got.get("actual_device"))
            if actual in placed:
                self.close()
                raise SystemExit(
                    f"rerank instances {placed[actual]} and {i} both loaded their weights onto "
                    f"{actual} (asked for {self.devices[placed[actual]]} and {self.devices[i]}). "
                    "Two instances on one card share its SMs and serialise; refusing to serve a "
                    "fleet that is not as wide as it claims."
                )
            if actual != got.get("device"):
                self.close()
                raise SystemExit(
                    f"rerank instance {i} was asked for {got.get('device')} but its weights "
                    f"landed on {actual}: the placement the argv states is not the placement "
                    "that happened. Refusing rather than serving an unproven layout."
                )
            placed[actual] = i
        self.bring_up_s = round(time.perf_counter() - started, 1)
        log("rerank_pool_placed", devices=self.devices, placed=sorted(placed),
            pids=[r["pid"] for r in self.ready],
            vram_alloc_gib=[r["vram_alloc_gib"] for r in self.ready],
            bring_up_s=self.bring_up_s)

    def alive(self) -> list[bool]:
        return [p.is_alive() for p in self._procs]

    def _poison(self, why: str) -> None:
        """Mark the fan permanently unusable and raise. There is no partial rerank fan.

        Every instance holds the same model, so a survivor cannot cover for a corpse: the slice the
        dead instance held would simply be missing from the merge. Worse, a half-scattered fan
        leaves the instances that did take a slice answering into a pipe nobody reads, so the next
        query's gather would splice this query's scores in at those indices, which is a silently
        wrong ranking and far worse than a dead service.
        """
        self._poisoned = why
        raise RuntimeError(
            f"the rerank fan is no longer trustworthy and has been poisoned: {why}. It will "
            "refuse every further query; restart the service to recover."
        )

    def _check_usable(self) -> None:
        if self._poisoned:
            raise RuntimeError(f"rerank fan poisoned earlier: {self._poisoned}")
        dead = [f"instance {i} ({self.devices[i]}, exitcode {p.exitcode})"
                for i, p in enumerate(self._procs) if not p.is_alive()]
        if dead:
            self._poison("dead before dispatch: " + "; ".join(dead))

    def score(self, query: str, texts: list[str]) -> tuple[list[float], dict[str, Any]]:
        """Score the whole pool across every instance. Returns ``(scores, telem)`` in order."""
        if not texts:
            self.last_shards = []
            return [], {"instances": 0}
        self._check_usable()
        bounds = batch_aligned_slices(len(texts), len(self._conns), self.batch_size)
        # Scatter: an instance with no slice is told so explicitly, so the gather below can recv
        # from every connection and stay index-keyed instead of guessing which ones replied.
        scatter_started = time.perf_counter()
        for i, conn in enumerate(self._conns):
            lo, hi = bounds[i] if i < len(bounds) else (0, 0)
            try:
                conn.send((query, texts[lo:hi]))
            except (BrokenPipeError, OSError) as exc:
                self._poison(
                    f"scatter to instance {i} ({self.devices[i]}) failed "
                    f"({type(exc).__name__}: {exc}); {i} of {len(self._conns)} slices were "
                    "already dispatched, so the pipes are out of step"
                )
        self.last_scatter_s = round(time.perf_counter() - scatter_started, 4)
        scores: list[float] = []
        shards, errs = [], []
        for i, conn in enumerate(self._conns):  # Keyed by index: shard i's reply is read at i
            while not conn.poll(5.0):
                if not self._procs[i].is_alive():
                    self._poison(
                        f"instance {i} ({self.devices[i]}) died with its slice in flight "
                        f"(exitcode {self._procs[i].exitcode})"
                    )
            got = conn.recv()
            lo, hi = bounds[i] if i < len(bounds) else (0, 0)
            if got.get("error"):
                errs.append(f"instance {i} ({self.devices[i]}): {got['error']}")
            elif len(got["scores"]) != hi - lo:
                errs.append(
                    f"instance {i} ({self.devices[i]}) returned {len(got['scores'])} scores for "
                    f"{hi - lo} candidates"
                )
            scores.extend(got["scores"])
            if hi > lo:
                shards.append({"shard": i, "device": self.devices[i], "pid": got["pid"],
                               "range": [lo, hi], "n": hi - lo, "t0": got["t0"], "t1": got["t1"],
                               "wall_s": round(got["t1"] - got["t0"], 4), "cpu_s": got["cpu_s"]})
        if errs:
            raise RuntimeError("; ".join(errs))
        if len(scores) != len(texts):
            raise RuntimeError(
                f"rerank pool produced {len(scores)} scores for {len(texts)} candidates; the "
                "partition did not cover the pool exactly once"
            )
        self.last_shards = shards
        return scores, {
            "instances": len(bounds),
            "devices": self.devices[: len(bounds)],
            "slices": [list(b) for b in bounds],
            "scatter_s": self.last_scatter_s,
            "shards": shards,
            "overlap": self.overlap(),
        }

    def score_solo(self, query: str, texts: list[str]) -> list[float]:
        """Score the whole pool on instance 0: the single-instance shape, for a live comparison.

        This is what the fan claims to be bit-identical to: the same checkpoint, batch size, max
        length, query and service, on one card. Comparing against it settles the question by
        measurement instead of by an argument about whether batch composition moved.
        """
        if not texts:
            return []
        self._check_usable()
        conn = self._conns[0]
        try:
            conn.send((query, list(texts)))
        except (BrokenPipeError, OSError) as exc:
            self._poison(f"solo pass could not reach instance 0: {type(exc).__name__}: {exc}")
        while not conn.poll(5.0):
            if not self._procs[0].is_alive():
                self._poison("instance 0 died during the solo verification pass")
        got = conn.recv()
        if got.get("error"):
            raise RuntimeError(f"solo pass failed on instance 0: {got['error']}")
        if len(got["scores"]) != len(texts):
            raise RuntimeError(
                f"solo pass returned {len(got['scores'])} scores for {len(texts)} candidates"
            )
        return [float(s) for s in got["scores"]]

    def overlap(self) -> dict[str, Any]:
        """How much of the last fan was actually concurrent, from the shards' own intervals.

        ``busiest`` is the largest number of shards whose ``[t0, t1)`` intervals contain a common
        instant; ``concurrency`` is the time-weighted mean number in flight, the sum of shard walls
        over the span. A fan that serialised reports ``busiest: 1`` and ``concurrency`` near 1.0
        however fast it was, which is the point: a wall-time drop alone is not evidence that N
        cards worked at once.
        """
        if not self.last_shards:
            return {}
        events = []
        for shard in self.last_shards:
            events.append((shard["t0"], 1))
            events.append((shard["t1"], -1))
        events.sort()
        live = busiest = 0
        for _, delta in events:
            live += delta
            busiest = max(busiest, live)
        span = max(s["t1"] for s in self.last_shards) - min(s["t0"] for s in self.last_shards)
        total = sum(s["wall_s"] for s in self.last_shards)
        return {
            "shards": len(self.last_shards),
            "busiest": busiest,
            "span_s": round(span, 4),
            "sum_shard_wall_s": round(total, 4),
            "concurrency": round(total / span, 2) if span > 0 else None,
            "slowest_shard_s": round(max(s["wall_s"] for s in self.last_shards), 4),
            "fastest_shard_s": round(min(s["wall_s"] for s in self.last_shards), 4),
        }

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=30)


# --------------------------------------------------------------------------- #
# The GPU replicas: one complete stack per card, nothing reconciled across cards.
# --------------------------------------------------------------------------- #
class ReplicaDied(RuntimeError):
    """A replica died with a job in flight: a fleet-level failure, not a query error.

    Distinguished from a query that merely raised, which leaves the replica alive and fails one
    request, because the two need opposite handling: a bad query is the client's problem and the
    service keeps serving, whereas a dead replica means every remaining answer would come from a
    fleet smaller than the one this service published. It carries which replica and which job, so
    the message can name both.
    """

    def __init__(self, index: int, device: str, key: Any, detail: str = "") -> None:
        self.index = index
        self.device = device
        self.key = key
        super().__init__(
            f"replica {index} ({device}) died with job {key!r} in flight; failing the request "
            f"rather than returning a short list as if it were complete{detail}"
        )


class ReplicaPool:
    """``N`` forked processes, each a whole independent stack pinned to one card.

    Replicate whole stacks; do not split one query across cards. Both were measured on two
    Blackwell cards: sharding a single query's language cells across the two (``--plaid-shard``)
    moved late interaction 2.95 to 2.66 s but blew the rerank 0.48 to 1.94 s and the total 3.65 to
    4.88 s, because the ru and zh cells landed on the card the reranker lives on and pinned it at
    100 %. Splitting one query creates contention between components that were never independent;
    replicating the whole stack creates none, because a query is dispatched to one card and finishes
    entirely there, PLAID, fusion and rerank included. Nothing is reconciled across replicas because
    nothing is split across them.

    What is shared and what is duplicated, which is the reason a replica is a fork and not a
    second service:

    ===================================== =========== =========================================
    dense + sparse indexes (~210 GiB)     shared      copy-on-write from the parent's `warm_cpu`
    PLAID blobs (199.65 GiB)              shared      the same mmap'd files, one page cache
    PLAID VRAM residency + reranker       duplicated  36.87 GiB measured per card, of 95.59
    ===================================== =========== =========================================

    So two replicas cost about 410 GiB of host memory rather than 820, and 36.87 GiB of a 95.59 GiB
    card, which is what makes a replica per card affordable at all. A card too small to hold one
    whole stack cannot host a replica; that is not checked here, because VRAM is not knowable
    before the load, but it is why the flag defaults to 1.

    Every replica is forked before any CUDA exists in the parent. That is the same invariant the
    cell workers rely on, and it is absolute rather than incidental here: the parent in replica mode
    never initialises CUDA at all, it only dispatches. :func:`assert_pre_fork` proves it at the fork
    point instead of asserting it in a comment.

    Replicas are not daemonic, because a daemonic process may not have children and each replica
    forks its own :class:`CellPool` (see :meth:`Service._replica_body`), so the parent's ``finally``
    is what reaps them.
    """

    def __init__(self, service: Any, devices: list[str], log: Any) -> None:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        self.devices = list(devices)
        self._conns: list[Any] = []
        self._procs: list[Any] = []
        started = time.perf_counter()
        for i, device in enumerate(self.devices):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                # The forked body is a bound method of the Service; with the `fork` start method
                # nothing here is pickled, so it inherits the warmed cells by memory.
                target=service._replica_body,
                args=(i, device, child_conn),
                name=f"rsvc-replica-{i}",
                daemon=False,
            )
            proc.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(proc)
        # Every replica loads its own PLAID residency and reranker concurrently, so this wait is
        # about one replica's GPU load, not N. A replica that dies during it closes its pipe, and
        # the EOFError below is raised here rather than at the first query.
        self.ready: list[dict[str, Any]] = []
        try:
            for i, conn in enumerate(self._conns):
                try:
                    got = conn.recv()
                except EOFError as exc:
                    raise RuntimeError(
                        f"replica {i} ({self.devices[i]}) died during bring-up without reporting "
                        "ready, refusing to publish a service that would answer from a subset "
                        "of its cards"
                    ) from exc
                if got.get("error"):
                    raise RuntimeError(
                        f"replica {i} ({self.devices[i]}) failed to bring up: {got['error']}\n"
                        f"{got.get('traceback', '')}"
                    )
                self.ready.append(got)
                log("replica_ready", replica=i, device=self.devices[i], boot=got)
        except BaseException:
            # Replicas are not daemonic, so a live one would make the interpreter hang forever in
            # `multiprocessing`'s exit handler, joining a child blocked on a pipe nobody will write
            # to again. A failed bring-up must exit, loudly and promptly.
            self.close()
            raise
        self.bring_up_s = round(time.perf_counter() - started, 1)
        # The dispatch state belongs to the pool rather than to one batch, because the serve loop
        # multiplexes several requests over the same fleet: a replica that finishes a job must be
        # free for the next job from any request, and "which conn is free" cannot be a local of one
        # batch if that is to hold. `_idle` is a FIFO of free connections, `_busy` maps an in-flight
        # connection to the opaque key its result belongs to.
        self._idle: list[Any] = list(self._conns)
        self._busy: dict[Any, Any] = {}
        self._dead: str = ""

    def alive(self) -> list[bool]:
        return [p.is_alive() for p in self._procs]

    # -- the one dispatch policy: least-busy, keyed, order-free ---------------------------- #
    def _width(self, limit: int | None) -> int:
        if limit is None:
            return len(self._conns)
        width = int(limit)
        if not 1 <= width <= len(self._conns):
            raise ValueError(
                f"use_replicas={limit!r} but this service has {len(self._conns)} replicas; a "
                "batch may use between 1 and all of them"
            )
        return width

    def free(self, limit: int | None = None) -> int:
        """How many of the first ``limit`` replicas are idle right now.

        The caller uses this to decide how much work to hand over, which is what makes the policy
        least-busy rather than pre-assigned: nothing is allocated to a replica until that replica is
        the one asking for work.
        """
        width = self._width(limit)
        return sum(1 for conn in self._idle if self._conns.index(conn) < width)

    def dispatch(self, key: Any, job: dict[str, Any], limit: int | None = None) -> bool:
        """Give ``job`` to an idle replica, tagged with ``key``. False if none is free.

        ``key`` is opaque and travels with the result, so a caller may place results wherever its
        own ordering requires. That is the seam that lets dispatch be least-busy while replies stay
        in submission order.
        """
        if self._dead:
            raise RuntimeError(
                f"refusing to dispatch: {self._dead}. This service published a fleet of "
                f"{len(self._conns)} replicas and can no longer answer from it; relaunch it "
                "rather than serving silently from a smaller one"
            )
        width = self._width(limit)
        for pos, conn in enumerate(self._idle):
            if self._conns.index(conn) < width:
                self._idle.pop(pos)
                conn.send(job)
                self._busy[conn] = key
                return True
        return False

    def harvest(self, timeout: float | None = None) -> list[tuple[Any, dict[str, Any]]]:
        """Collect whatever has finished, as ``(key, result)``, freeing those replicas.

        Completion order: the caller re-orders by ``key``. Waiting in submission order
        would idle a replica that finished early behind one that has not, which is the head-of-line
        blocking this layout exists to avoid.

        A replica that died raises :class:`ReplicaDied` and poisons the pool. The alternative,
        quietly continuing on the survivors, is a service that keeps answering while advertising a
        fleet it no longer has.
        """
        from multiprocessing.connection import wait

        if not self._busy:
            return []
        out: list[tuple[Any, dict[str, Any]]] = []
        for conn in wait(list(self._busy), timeout=timeout):
            key = self._busy.pop(conn)
            which = self._conns.index(conn)
            try:
                got = conn.recv()
            except (EOFError, OSError) as exc:
                self._dead = f"replica {which} ({self.devices[which]}) died"
                raise ReplicaDied(which, self.devices[which], key) from exc
            self._idle.append(conn)
            out.append((key, got))
        return out

    def run_batch(self, jobs: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
        """Run ``jobs`` across the replicas and return results in submission order.

        ``limit`` caps how many of the resident replicas a batch may use. It exists so that "does N
        replicas divide the wall?" can be answered inside one warm process: a second allocation
        would move page-cache retention as well as the replica count, and retention is measured to
        dominate query latency (37 % retention gave 25.6 s/query where 100 % gave 12.4 s on a weaker
        card). Same process, same cache, same index, only the fleet width moves.

        Order is a correctness property rather than a nicety. A client indexes the reply
        positionally and the fused scores it carries are floats, so returning completion order
        would silently re-pair queries with answers. Results land in ``out[i]`` by the index the job
        was submitted with, so the dispatch policy cannot affect it.

        A replica that dies mid-batch raises here, naming the replica; it must never come back as a
        short list that looks like a complete answer.

        This is the exclusive spelling: it assumes no other caller is using the pool concurrently,
        because it drains :meth:`harvest` to completion. The serve loop uses
        :meth:`dispatch`/:meth:`harvest` directly when it multiplexes several requests, so both
        paths share one dispatch policy and differ only in who owns the fleet.
        """
        self._width(limit)  # validate before sending anything
        out: list[Any] = [None] * len(jobs)
        nxt = 0
        inflight = 0
        while nxt < len(jobs) or inflight:
            while nxt < len(jobs) and self.dispatch(nxt, jobs[nxt], limit):
                nxt += 1
                inflight += 1
            for i, got in self.harvest():
                inflight -= 1
                if got.get("error"):
                    which = got.get("replica")
                    raise RuntimeError(
                        f"replica {which} ({got.get('replica_device')}) on query {i}: "
                        f"{got['error']}"
                    )
                out[i] = got
        return out

    def close(self) -> None:
        """Idempotent. Ask every replica to stop, then make sure none outlives this call."""
        for conn in self._conns:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError, ValueError):
                pass
            try:
                conn.close()
            except OSError:
                pass
        for proc in self._procs:
            proc.join(timeout=30)
            if proc.is_alive():  # pragma: no cover - a wedged replica must not wedge the exit
                proc.terminate()
                proc.join(timeout=10)


def replica_devices(spec: str, count: int) -> list[str]:
    """The card each replica owns. An explicit list wins; otherwise ``cuda:0..cuda:N-1``.

    Never hardcoded to two: one, two and four cards all have to work, and a node whose free cards
    are not ``0..N-1`` needs to be able to say so (``--replica-devices cuda:2,cuda:3``).
    """
    if spec.strip():
        devices = [d.strip() for d in spec.split(",") if d.strip()]
        if count and len(devices) != count:
            raise SystemExit(
                f"--replica-devices lists {len(devices)} cards but --replicas is {count}; a "
                "replica owns exactly one card, so these must agree"
            )
        return devices
    return [f"cuda:{i}" for i in range(count)]


# --------------------------------------------------------------------------- #
# Shape / telemetry.
# --------------------------------------------------------------------------- #
def gpu_memory() -> list[dict[str, Any]]:
    """Per-card VRAM from ``nvidia-smi``: the number a budget is checked against.

    ``torch.cuda.memory_allocated`` reports only this process's tensors and would miss the CUDA
    context, cuBLAS workspaces and the allocator's cached blocks, which is most of what actually
    occupies a card.
    """
    import subprocess

    try:
        raw = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - telemetry only
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    cards = []
    for line in raw.strip().splitlines():
        bits = [b.strip() for b in line.split(",")]
        if len(bits) < 5:
            continue
        cards.append(
            {
                "index": int(bits[0]),
                "name": bits[1],
                "used_gib": round(float(bits[2]) / 1024, 2),
                "total_gib": round(float(bits[3]) / 1024, 2),
                "util_pct": float(bits[4]),
            }
        )
    return cards


def cgroup_memory() -> dict[str, Any]:
    """Our cgroup's memory: limit, current, and the page cache charged to us.

    ``/proc/meminfo`` is the wrong instrument here and has produced a false alarm: it reports the
    whole node, so on a shared node carrying someone else's page cache, a prefault that retained
    nothing and one that retained everything look identical. What decides whether PLAID faults is
    the ``file`` counter of our cgroup against our ``memory.max``, because a cgroup at its limit
    reclaims its own page cache however idle the node is.

    Returns ``{}`` when cgroup v2 is not readable, so the caller degrades to the node-wide figures
    rather than pretending to know.
    """
    base = Path("/sys/fs/cgroup")
    try:
        rel = ""
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            parts = line.split(":")
            if len(parts) == 3 and parts[1] == "":  # v2 unified line: "0::/path"
                rel = parts[2].lstrip("/")
        for candidate in ((base / rel) if rel else base, base):
            if (candidate / "memory.stat").exists():
                base = candidate
                break
        stat = {}
        for line in (base / "memory.stat").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(" ")
            stat[key] = float(value)
        raw_max = (base / "memory.max").read_text(encoding="utf-8").strip()
        out = {
            "cg_current_gib": round(
                float((base / "memory.current").read_text(encoding="utf-8").strip()) / GIB, 1
            ),
            "cg_anon_gib": round(stat.get("anon", 0.0) / GIB, 1),
            "cg_file_gib": round(stat.get("file", 0.0) / GIB, 1),
        }
        out["cg_max_gib"] = None if raw_max == "max" else round(float(raw_max) / GIB, 1)
        return out
    except (OSError, ValueError):  # pragma: no cover - telemetry only
        return {}


def torch_vram(reset: bool = False) -> dict[str, Any]:
    """Peak live bytes per card: the number that decides which card this fits on.

    ``nvidia-smi`` reports what the process holds, and PyTorch's caching allocator grows to fill
    whatever is available: on a 95.59 GiB card it climbed 23.05 -> 30.6 -> 48.5 GiB over a dozen
    batches and then sat flat at 48.5 while idle. None of that 48.5 is required;
    it is retained freed blocks, and on a smaller card the same workload simply caches less and
    recycles harder. So "does this fit a smaller card" is answered by ``max_memory_allocated``, the
    peak live figure, with ``max_memory_reserved`` alongside it: the gap between the two is the
    compressible slack, and quoting the reserved figure as a requirement would rule out cards that
    in fact fit.

    Two caveats, reported rather than hidden. This is the parent process only, which is the
    meaningful scope for the GPU work, since the late-interaction leg runs here and the forked cell
    workers never touch CUDA. And neither counter includes the CUDA context or cuBLAS workspaces,
    which are real, per-process and typically several hundred MiB; the ``nvidia_smi_gib`` column is
    carried next to them so the difference stays visible.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - telemetry only
        return {}
    if not torch.cuda.is_available():
        return {}
    smi = {c["index"]: c.get("used_gib") for c in gpu_memory() if "index" in c}
    out: dict[str, Any] = {}
    for i in range(torch.cuda.device_count()):
        out[f"cuda:{i}"] = {
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(i) / GIB, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(i) / GIB, 2),
            "current_allocated_gib": round(torch.cuda.memory_allocated(i) / GIB, 2),
            "nvidia_smi_gib": smi.get(i),
        }
        if reset:
            torch.cuda.reset_peak_memory_stats(i)
    return out


def host_memory() -> dict[str, Any]:
    """Anonymous RSS of this process, our cgroup's accounting, and the node's page-cache state."""
    info: dict[str, Any] = {
        "self_max_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20), 1)
    }
    info.update(cgroup_memory())
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            fields[key] = float(rest.strip().split()[0]) / (1 << 20)
        info.update(
            mem_total_gib=round(fields.get("MemTotal", 0), 1),
            mem_available_gib=round(fields.get("MemAvailable", 0), 1),
            page_cache_gib=round(fields.get("Cached", 0), 1),
        )
    except OSError:  # pragma: no cover - telemetry only
        pass
    return info


def thread_shape() -> dict[str, Any]:
    """Every thread-pool width this process actually has, read back from the libraries.

    The environment variable is what was asked for; ``faiss.omp_get_max_threads`` and
    ``torch.get_num_threads`` are what was got, and they have disagreed here before (a service on a
    14-core allocation ran FAISS at 4 because the sbatch hardcoded ``OMP_NUM_THREADS=4``). Both are
    recorded so the reader can see the gap. ``len(os.sched_getaffinity(0))`` is the cgroup's real
    core count, which is the only honest denominator: ``os.cpu_count`` reports the whole node.
    """
    got: dict[str, Any] = {
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "cpu_count": os.cpu_count(),
        "env": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "RAYON_NUM_THREADS",
                "SLURM_CPUS_PER_TASK",
            )
        },
    }
    try:
        import faiss

        got["faiss_omp"] = faiss.omp_get_max_threads()
    except Exception:  # noqa: BLE001
        got["faiss_omp"] = None
    try:
        import torch

        got["torch_intraop"] = torch.get_num_threads()
        got["torch_interop"] = torch.get_num_interop_threads()
    except Exception:  # noqa: BLE001
        got["torch_intraop"] = got["torch_interop"] = None
    return got


def shape() -> dict[str, Any]:
    """Stamp what this process actually got. A latency number without its shape is not evidence."""
    return {
        "node": os.uname().nodename,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "mem_mb": os.environ.get("SLURM_MEM_PER_NODE"),
        "gpus": os.environ.get("SLURM_GPUS_ON_NODE"),
        "pid": os.getpid(),
    }


def atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


class _Live:
    """One accepted request in flight: the unit the serve loop multiplexes over the fleet.

    It owns its own ``results`` list, indexed by the job's submission position. That is what makes
    least-busy dispatch safe here: a result is placed by the index it was submitted with, never
    appended as it completes, so which replica answered and in what order are both invisible to the
    client. The client zips scores back onto candidate ids positionally and RRF accumulates floats,
    so a reordered reply would not error, it would silently re-pair queries with answers.
    """

    __slots__ = (
        "_accepted", "_first", "_started", "done", "error", "exclusive", "inflight", "jobs",
        "nxt", "req", "results", "rid", "traceback", "width",
    )

    def __init__(self, rid: str, req: dict[str, Any]) -> None:
        self.rid = rid
        self.req = req
        self.jobs: list[dict[str, Any]] = []
        self.results: list[Any] = []
        self.width: int | None = None
        self.exclusive = False
        self.error: str | None = None
        self.traceback = ""
        self.nxt = 0  # next job to dispatch
        self.done = 0
        self.inflight = 0
        self._accepted = time.perf_counter()
        self._started: float | None = None
        self._first: float | None = None

    def start(self) -> None:
        """Mark the un-multiplexed run's start, so its ``batch_wall_s`` excludes any drain wait."""
        self._started = self._first = time.perf_counter()

    def wants(self) -> bool:
        return self.error is None and self.nxt < len(self.jobs)

    def sent(self) -> None:
        if self._first is None:
            self._first = time.perf_counter()
        self.nxt += 1
        self.inflight += 1

    def took(self, i: int, res: dict[str, Any]) -> None:
        self.results[i] = res
        self.inflight -= 1
        self.done += 1

    def fail(self, error: str, tb: str) -> None:
        """First error wins: the cause is more informative than whatever it knocked over."""
        if self.error is None:
            self.error = error
            self.traceback = tb

    @property
    def finished(self) -> bool:
        return self.inflight == 0 and (self.error is not None or self.done == len(self.jobs))

    def elapsed(self) -> float:
        return time.perf_counter() - (self._accepted if self._started is None else self._started)

    def queued_s(self) -> float:
        return 0.0 if self._first is None else self._first - self._accepted


# --------------------------------------------------------------------------- #
# The service.
# --------------------------------------------------------------------------- #
class Service:
    """One whole rendering, three legs plus the reranker, behind one filesystem queue."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.queue = Path(args.queue)
        self.ctx: Any = None
        self.pool: CellPool | None = None
        self.reranker: Reranker | None = None  # serving.Reranker (dedup phase 2)
        # N rerankers, one per card, when `--rerank-devices` asks for them. Mutually exclusive with
        # `self.reranker`: a parent instance and a pool instance on the same card is two instances
        # on one card, which is the placement `RerankPool` exists to refuse.
        self.rerank_pool: RerankPool | None = None
        self.replicas: ReplicaPool | None = None
        self.clients: Any = None
        self.replica_id: int | None = None
        self.rerank_model = ""
        self.boot: dict[str, Any] = {}
        self.cold_s = 0.0

    # -- bring-up ---------------------------------------------------------- #
    def bring_up(self) -> None:
        from ragtime.common import Statistics
        from ragtime.config import loader
        from ragtime.devkit.devrun import resolve_dev_run
        from ragtime.preprocess.index import DENSE_LEG, LATE_INTERACTION_LEG, SPARSE_LEG
        from ragtime.preprocess.index import index_hash as _index_hash
        from ragtime.preprocess.packing import packing_hash
        from ragtime.preprocess.reconcile import reconcile_hash
        from ragtime.retrieval import bring_up
        from ragtime.serving.registry import build_clients

        assert (DENSE_LEG, SPARSE_LEG, LATE_INTERACTION_LEG) == (DENSE, SPARSE, LATE)
        a = self.args
        t_boot = time.perf_counter()
        log = self._log

        # PLAID's query device is an environment override, because the config leaf would re-key the
        # index. It must be set before any cell opens.
        os.environ["RAGTIME_QUERY_PLAID_DEVICE"] = a.gpu_plaid
        idmap = patch_idmap_cache()
        _low_mem = None if a.plaid_low_memory == "default" else (a.plaid_low_memory == "true")
        _shard = _parse_shard(a.plaid_shard)
        patch_plaid_kwargs(n_full_scores=a.n_full_scores, low_memory=_low_mem, shard=_shard)
        log("patch", idmap_methods=idmap, n_full_scores=a.n_full_scores,
            plaid_device=a.gpu_plaid, low_memory=_low_mem, plaid_shard=_shard or None)
        if idmap != 2:
            log("WARNING", msg="id-map memoization did not apply; expect ~53x slower PLAID")

        cfg = loader.load(a.config)
        cfg, rendering_note = _resolve_rendering(cfg, a.index, allow_override=a.allow_index_override)
        log("rendering", **rendering_note)
        cfg, store_note = _apply_store_mirror(cfg, a.store_local)
        log("store_mirror", **store_note)
        run = resolve_dev_run(a.config, "_rsvc", 0, label="retrieval-service")
        idx = _index_hash(cfg)
        clients = self.clients = build_clients(cfg)

        # Card assignment. Every encoder is lazy, since the backend builds on first use, so setting
        # `.device` here decides placement without a production edit. A device *list* is
        # never used: it replicates the index rather than sharding it. In replica mode
        # these are re-assigned inside each child to that replica's own card (`_replica_body`), so
        # the per-component flags describe only the `--replicas 0` layout.
        #
        # torch's CPU intra-op pool is set here, before any encoder builds, and re-set inside each
        # replica child because a fork inherits the value but a replica may be given a different
        # share. It matters most when PLAID searches on CPU, where torch is the dominant leg; on
        # `--gpu-plaid cuda:N` it only sizes the host-side ops.
        if a.torch_threads:
            import torch

            torch.set_num_threads(int(a.torch_threads))
            log("torch_threads", asked=int(a.torch_threads), got=torch.get_num_threads())
        clients.mtd_colbert.device = a.gpu_mtd
        clients.index_dense.device = a.gpu_dense
        clients.milco.device = a.gpu_sparse
        log(
            "placement",
            plaid=a.gpu_plaid,
            mtd=a.gpu_mtd,
            dense=a.gpu_dense,
            sparse=a.gpu_sparse,
            rerank=a.gpu_rerank,
        )

        t = time.perf_counter()
        self.ctx = bring_up(
            cfg,
            clients,
            run.layout,
            recon_hash=reconcile_hash(cfg),
            pack_hash=packing_hash(cfg),
            idx_hash=idx,
            stats=Statistics(),
        )
        self.boot["open_cells_s"] = round(time.perf_counter() - t, 1)
        cells = self.ctx.cells
        log(
            "cells",
            index=self.ctx.index,
            parts={k: len(v.parts) for k, v in sorted(cells.items())},
            total_parts=sum(len(v.parts) for v in cells.values()),
            seconds=self.boot["open_cells_s"],
        )

        # The pre-fork window. Everything from here to the CellPool below is CPU and filesystem
        # only, and `assert_pre_fork` proves it rather than trusting the comment.
        #
        # 1. Start the blob mirror in the background. It is the second of the cold load's two
        #    independent chains (mirror -> warm_plaid -> reranker; warm_cpu -> fork), touches only
        #    the late-interaction blobs, and is pure `shutil`, so it cannot initialise CUDA in the
        #    pre-fork window. Overlapping it with `warm_cpu_legs` saves about the shorter of the
        #    two. `--plaid-local` is the mechanism that works; `--prefault` is kept because it is
        #    nearly free, but it does not fix mmap faults on shared storage. See `mirror_plaid`.
        overlap = a.mirror_overlap == "on"
        mirror_bg: Background | None = None
        if a.plaid_local and overlap:
            mirror_bg = Background(
                "plaid_local", mirror_plaid, cells, Path(a.plaid_local),
                threads=a.prefault_threads,
            )
            log("plaid_local_started", root=a.plaid_local, threads=a.prefault_threads,
                overlaps="warm_cpu_legs")

        # 2. Warm the CPU legs (FAISS and Seismic), still with no CUDA in this process.
        self.boot["warm_cpu"] = warm_cpu_legs(cells, (DENSE, SPARSE), workers=a.warm_workers)
        log("warm_cpu_legs", **self.boot["warm_cpu"], host=host_memory())

        # 3. Join the mirror, before anything forks. `result()` re-raises whatever the thread hit,
        #    so a failed mirror aborts bring-up here instead of becoming a `ready: true` service
        #    over a partial index. `residual_s` is the part of the mirror that did not hide under
        #    the warm, which is the honest measure of whether the overlap paid.
        if mirror_bg is not None:
            t_join = time.perf_counter()
            mirror = mirror_bg.result()
            mirror["overlapped"] = True
            mirror["residual_s"] = round(time.perf_counter() - t_join, 1)
            patch_plaid_open(mirror.pop("_mapping"))
            self.boot["plaid_local"] = mirror
            log("plaid_local", **mirror, host=host_memory())

        # 4. The fork point. Everything above is CPU and filesystem; everything CUDA is below, in a
        #    child. The guard is mechanical (no /dev/nvidia* fd, torch's CUDA flag unset), so
        #    re-ordering the phases above, which is exactly what the mirror overlap does, cannot
        #    quietly break the invariant that makes forking safe.
        self.rerank_model = a.rerank_model or self._config_rerank_model(cfg)
        self.boot["pre_fork"] = assert_pre_fork("the cell-worker / replica fork")
        log("pre_fork", **self.boot["pre_fork"])

        if a.replicas >= 1:
            # Replica mode. The parent never initialises CUDA at all: it forks N whole stacks, one
            # per card, and then only dispatches. Each replica already has the blob phase's patch
            # applied, from above, and does its own `warm_plaid` plus reranker on its own card,
            # concurrently with the others.
            if _shard:
                raise SystemExit(
                    "--plaid-shard and --replicas are two different answers to the same "
                    "question and must not be combined: sharding splits one query across cards, "
                    "which measured 3.65 -> 4.88 s overall, while replicas give each card a whole "
                    "stack. Pick one."
                )
            if mirror_bg is None:
                self._blob_phase(cells, log)
            devices = replica_devices(a.replica_devices, a.replicas)
            log("replicas_forking", devices=devices)
            self.replicas = ReplicaPool(self, devices, log)
            self.boot["replicas"] = {
                "devices": devices,
                "bring_up_s": self.replicas.bring_up_s,
                "boot": self.replicas.ready,
            }
        else:
            # The single-stack layout (`--replicas 0`): one stack in this process, answering
            # serially. Kept so the replica architecture can be compared against the shape every
            # earlier measurement was taken on, rather than against a memory of it.
            if a.cell_procs:
                self.pool = CellPool(
                    cells, part_workers=a.part_workers, part_procs=a.part_procs
                )
                log("cell_pool", langs=self.pool.langs, workers=self.pool.workers,
                    fork_s=self.pool.fork_s, plan=self.pool.plan, alive=self.pool.alive())
            # The rerank fleet forks here: after the cell pool and before `_load_gpu_stack`, which
            # is where CUDA starts in this process. Both fans depend on the same invariant, that a
            # child never inherits a CUDA context, and `RerankPool` asserts it again at its own
            # fork point rather than trusting this ordering to survive an edit. The instances then
            # load concurrently with the parent's PLAID load.
            if a.rerank_devices and not a.no_rerank:
                self.rerank_pool = RerankPool(
                    self.rerank_model,
                    [d for d in str(a.rerank_devices).split(",") if d.strip()],
                    batch_size=a.rerank_batch, max_length=a.rerank_max_len, log=log,
                )
                self.boot["rerank_pool"] = {
                    "devices": self.rerank_pool.devices,
                    "bring_up_s": self.rerank_pool.bring_up_s,
                    "boot": self.rerank_pool.ready,
                }
            if mirror_bg is None:
                self._blob_phase(cells, log)
            self._load_gpu_stack(a.gpu_plaid, a.gpu_rerank, log)

        self.cold_s = time.perf_counter() - t_boot
        self.index_hash = idx
        self.rendering_note = rendering_note
        self.boot["cold_load_s"] = round(self.cold_s, 1)
        # Whatever the argument said, the served rendering is whatever the context actually opened,
        # and every reply and descriptor quotes that.
        if self.ctx.index != a.index:
            raise RuntimeError(
                f"asked to serve {a.index!r} but the context opened {self.ctx.index!r}, "
                "refusing to publish a descriptor that would misdescribe which index is resident"
            )

    def _load_gpu_stack(self, plaid_device: str, rerank_device: str, log: Any) -> None:
        """Put one whole stack on the given cards: PLAID residency and reranker. CUDA starts here.

        Called in the parent only under ``--replicas 0``; in replica mode every call site is inside
        a forked child, which is what keeps the parent CUDA-free.

        Every part is checked to have produced both a reader and a non-empty id map, because "78
        parts opened" is not the same claim as "78 parts are searchable": a service can otherwise
        publish ``ready: true`` advertising all 78 parts with nine part loads already broken, and
        the first witness is a query. The count in the ready record is something that was counted.
        """
        a = self.args
        os.environ["RAGTIME_QUERY_PLAID_DEVICE"] = plaid_device
        t = time.perf_counter()
        broken: list[str] = []
        loaded = 0
        for handle in self.ctx.cells.values():
            for part in handle.parts:
                if part.reader(LATE) is None or not part.passage_ids(LATE):
                    broken.append(str(part.shard_dir))
                else:
                    loaded += 1
        if broken:
            raise RuntimeError(
                f"{len(broken)} of {loaded + len(broken)} late-interaction parts loaded without a "
                f"reader or with an empty id map: {broken[:5]}. Refusing to publish a descriptor "
                "that would advertise every part while silently searching a subset."
            )
        self.boot["warm_plaid_s"] = round(time.perf_counter() - t, 1)
        self.boot["warm_plaid_parts"] = loaded
        log("warm_plaid", seconds=self.boot["warm_plaid_s"], parts=loaded, device=plaid_device,
            gpu=gpu_memory())

        if self.rerank_pool is not None:
            # The pool owns reranking; loading the parent's own instance too would put a second
            # 7.92 GiB copy of the model on `rerank_device` for nothing, and if that card is one of
            # the pool's it would be the co-residency `RerankPool` refuses.
            log("reranker_skipped_pool_owns_it", devices=self.rerank_pool.devices,
                would_have_been=rerank_device)
            return

        from ragtime.serving.reranker import Reranker

        # `serving.Reranker`, not a devkit copy: the same checkpoint, and `rerank_prompt()` builds
        # a byte-identical prompt, so scores cannot move.
        self.reranker = Reranker(
            self.rerank_model,
            rerank_device,
            batch_size=a.rerank_batch,
            max_length=a.rerank_max_len,
        )
        if not a.no_rerank:
            t = time.perf_counter()
            self.reranker.load()
            self.boot["rerank_load_s"] = round(time.perf_counter() - t, 1)
            log("reranker", model=self.reranker.model, device=rerank_device,
                batch_size=a.rerank_batch, seconds=self.boot["rerank_load_s"])

    def _replica_body(self, index: int, device: str, conn: Any) -> None:
        """One replica: a whole stack on one card, in a forked child. Runs until the pipe closes.

        The child inherits the parent's warmed dense and sparse legs copy-on-write and the parent's
        already-applied PLAID patches, so it loads nothing off shared storage twice. It then, in
        this order: pins every component of the stack to its own card, forks its own cell workers
        while still CUDA-free, and initialises CUDA by loading PLAID and the reranker.

        Its own cell workers, not the parent's. A pool shared between replicas would put two
        replicas' queries in one queue behind the same processes, which is the contention that made
        cross-card sharding lose. The alternative of running the language cells as threads inside
        the replica is rejected on a measured pathology: ``pyseismic``'s ``search`` does not release
        the GIL, so four cells in one process serialize (0.49 to 0.71 s each becomes 5.2 to 7.6 s
        summed) and would also starve the thread driving PLAID. The cost of per-replica pools is
        page tables, not index bytes, since all 1 + N + 4N processes map the same copy-on-write
        pages. ``--cell-procs 0`` still selects the in-process form.
        """
        try:
            self.replica_id = index
            self.pool = None
            self.boot = {}
            os.environ["RAGTIME_QUERY_PLAID_DEVICE"] = device
            # A replica is a whole stack on one card, encoders included. They are lazy, and the
            # parent built none of them, so assigning here is what decides placement.
            self.clients.mtd_colbert.device = device
            self.clients.index_dense.device = device
            self.clients.milco.device = device

            def log(event: str, **kw: Any) -> None:
                # `replica` only. Injecting `device` here collides with `_load_gpu_stack`'s own
                # `device=` keyword and kills the replica after its whole 15-minute load; the phase
                # name already says which device it is talking about.
                self._log(event, replica=index, **kw)

            # Before the cell fork, so the children inherit the small env-derived FAISS width and
            # only this parent gets the wide torch pool.
            if self.args.torch_threads:
                import torch

                torch.set_num_threads(int(self.args.torch_threads))
                log("torch_threads", asked=int(self.args.torch_threads),
                    got=torch.get_num_threads())
            assert_pre_fork(f"replica {index}'s own cell-worker fork")
            if self.args.cell_procs:
                # A replica forks its own pool, so `part_procs` is per replica: N replicas times
                # this budget is what the node actually runs, which is why the flag's default of
                # one process per part belongs to a single-stack service and a replica fleet
                # should state a cap.
                self.pool = CellPool(
                    self.ctx.cells,
                    part_workers=self.args.part_workers,
                    part_procs=self.args.part_procs,
                )
                log("replica_cell_pool", device=device, langs=self.pool.langs,
                    workers=self.pool.workers, fork_s=self.pool.fork_s, plan=self.pool.plan,
                    alive=self.pool.alive())
            self._load_gpu_stack(device, device, log)
            conn.send({**self.boot, "pid": os.getpid()})
        except Exception as exc:  # noqa: BLE001 - a replica must report its death, not just die
            import traceback

            try:
                conn.send({"error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()[-2000:]})
            except (BrokenPipeError, OSError):
                pass
            return

        while True:
            try:
                job = conn.recv()
            except (EOFError, KeyboardInterrupt):
                break
            if job is None:
                break
            try:
                got = self.query(
                    job["query"],
                    top_k=job["top_k"],
                    rerank_depth=job["rerank_depth"],
                    li_workers=job.get("li_workers"),
                    cell_fan=job.get("cell_fan"),
                    part_workers=job.get("part_workers"),
                    omp=job.get("omp"),
                    rerank=job.get("rerank"),
                )
                if job.get("reset_peak", True):
                    got["vram"] = torch_vram(reset=True)
                got["replica"] = index
                got["replica_device"] = device
            except Exception as exc:  # noqa: BLE001 - one bad query must not kill the replica
                import traceback

                got = {"error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()[-2000:],
                       "replica": index, "replica_device": device}
            conn.send(got)
        if self.pool is not None:
            self.pool.close()

    def run_queries(
        self, jobs: list[dict[str, Any]], limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Answer every job, in submission order, across whatever stacks this service has.

        The one entry point shared by the queue handler and the self-bench, so "N replicas make a
        batch N times faster" is a property of the service rather than of one call site.
        """
        if self.replicas is not None:
            return self.replicas.run_batch(jobs, limit)
        return [
            self.query(
                job["query"],
                top_k=job["top_k"],
                rerank_depth=job["rerank_depth"],
                li_workers=job.get("li_workers"),
                cell_fan=job.get("cell_fan"),
                part_workers=job.get("part_workers"),
                omp=job.get("omp"),
                rerank=job.get("rerank"),
                verify_rerank=bool(job.get("verify_rerank", False)),
            )
            for job in jobs
        ]

    def _blob_phase(self, cells: dict[str, Any], log: Any) -> None:
        """Get the PLAID blobs somewhere a page fault is cheap: the synchronous spelling.

        Identical work to the backgrounded mirror in :meth:`bring_up`. It exists as its own method
        so the overlapped and un-overlapped paths cannot drift into doing different things, and so
        ``--mirror-overlap off`` can run it at its historical position, after the fork.
        """
        a = self.args
        if a.plaid_local:
            mirror = mirror_plaid(cells, Path(a.plaid_local), threads=a.prefault_threads)
            mirror["overlapped"] = False
            mirror["residual_s"] = mirror["seconds"]
            patch_plaid_open(mirror.pop("_mapping"))
            self.boot["plaid_local"] = mirror
            log("plaid_local", **mirror, host=host_memory())
        elif not a.no_prefault:
            self.boot["prefault"] = prefault(plaid_blobs(cells), threads=a.prefault_threads)
            log("prefault", **self.boot["prefault"], host=host_memory())

    @staticmethod
    def _config_rerank_model(cfg: Any) -> str:
        block = dict((getattr(cfg, "blocks", None) or {}).get("retrieval") or {})
        return str(dict(block.get("reranker") or {}).get("model") or "")

    def _log(self, event: str, **kw: Any) -> None:
        print(json.dumps({"rsvc": event, "t": round(time.time(), 3), **kw}, default=str), flush=True)

    # -- one query --------------------------------------------------------- #
    def query(
        self,
        text: str,
        *,
        top_k: int,
        rerank_depth: int,
        li_workers: int | None = None,
        cell_fan: int | None = None,
        part_workers: int | None = None,
        omp: int | None = None,
        rerank: bool | None = None,
        verify_rerank: bool = False,
    ) -> dict[str, Any]:
        """One fused query: encode once, fan the CPU legs to the cell processes, PLAID here, rerank.

        Per-leg walls are reported for attribution and they overlap by construction, because the
        CPU legs run in other processes while this one drives PLAID on the GPU, so they do not sum
        to the total and are not meant to.
        """
        from ragtime.common.fusion import rrf
        from ragtime.preprocess.index import search_with_rep
        from ragtime.retrieval.display import display

        # Both overrides exist because the cold load takes tens of minutes, so a fan-width sweep
        # must run against one warm service rather than one allocation per rung. They change no
        # returned list, since both fans collect in submission order, only how fast it is produced.
        ctx = self.ctx
        li_fan = int(self.args.li_workers if li_workers is None else li_workers)
        cell_fan = int(self.args.cell_fan if cell_fan is None else cell_fan)
        # The CPU-leg fan inside each cell process, and FAISS's OpenMP width. Both are per-query
        # for the same reason `cell_fan` is: the cold load is long enough that a sweep over them
        # must run against one warm service. Neither changes a returned list.
        part_fan = int(self.args.part_workers if part_workers is None else part_workers)
        omp_w = int(getattr(self.args, "cell_omp", 0) or 0) if omp is None else int(omp)
        do_rerank = (not self.args.no_rerank) if rerank is None else bool(rerank)
        depth = max(int(top_k), int(rerank_depth))
        timing: dict[str, float] = {}
        # Reset rather than accumulate: a query that does not rerank must report no fan evidence,
        # not the previous query's. An instrument that keeps printing the last good reading when
        # the thing it measures did not run is decoration.
        rerank_telem: dict[str, Any] = {}
        t_all = time.perf_counter()

        # 1. Encode once, on the GPU, in this process. The representations are what travel.
        t = time.perf_counter()
        reps: dict[str, Any] = {}
        impls = {leg.name: leg for leg in ctx.index_ctx.legs}
        for leg in (DENSE, SPARSE, LATE):
            reps[leg] = impls[leg].encode_query(ctx.index_ctx, text)
        timing["encode"] = round(time.perf_counter() - t, 4)

        cpu_reps = {DENSE: _to_numpy(reps[DENSE]), SPARSE: reps[SPARSE]}

        # 2. Scatter the CPU legs to the cell processes (non-blocking), then run PLAID here while
        #    they work. This overlap is the point of the whole layout.
        t_cpu = time.perf_counter()
        if self.pool is not None:
            # `scatter` is the deliberate split of `CellPool.search`, which sends and receives;
            # that would block here and destroy the PLAID overlap below.
            cell_opts = {"part_workers": part_fan, "omp": omp_w}
            self.pool.scatter(cpu_reps, depth, cell_opts)

        t = time.perf_counter()
        li_pools, li_per_cell = _fan_li_cells(
            ctx.cells, reps[LATE], depth, cell_fan, li_fan, search_with_rep
        )
        timing[LATE] = round(time.perf_counter() - t, 4)

        # 3. Gather.
        dense_pools: list[list] = []
        sparse_pools: list[list] = []
        cpu_per_cell: dict[str, Any] = {}
        if self.pool is not None:
            # The gather half of the split. It folds each cell's part slices back into one reply
            # first (`merge_slices`, in slice order, truncated to `depth`), so what arrives here is
            # what a single whole-cell worker would have returned.
            for lang, got in sorted(self.pool.gather(depth).items()):
                dense_pools.append(got[DENSE][0])
                sparse_pools.append(got[SPARSE][0])
                cpu_per_cell[lang] = {DENSE: got[DENSE][1], SPARSE: got[SPARSE][1]}
                if got.get("_slices"):
                    cpu_per_cell[lang]["slices"] = got["_slices"]
                # cpu_s / wall is the core count the leg achieved, straight off the cell process's
                # own clock. Reported next to the wall so a fan that is running but GIL-serialised
                # (Seismic) is visibly different from one that is not (FAISS).
                for leg in (DENSE, SPARSE):
                    if len(got[leg]) > 3:
                        cpu_per_cell[lang][f"{leg}_cpu_s"] = got[leg][3]
                        wall = got[leg][1] or 1e-9
                        cpu_per_cell[lang][f"{leg}_cores"] = round(got[leg][3] / wall, 2)
                for leg in (DENSE, SPARSE):
                    if got[leg][2]:
                        raise RuntimeError(f"cell {lang} leg {leg}: {got[leg][2]}")
        else:
            for lang, handle in sorted(ctx.cells.items()):
                cpu_per_cell[lang] = {}
                _faiss_omp(omp_w)
                for leg, pools in ((DENSE, dense_pools), (SPARSE, sparse_pools)):
                    t_cell = time.perf_counter()
                    cpu0 = time.process_time()
                    pools.append(
                        merge_parts(handle, leg, cpu_reps[leg], depth, part_fan, omp_w)
                    )
                    wall = time.perf_counter() - t_cell
                    cpu_per_cell[lang][leg] = round(wall, 4)
                    cpu_per_cell[lang][f"{leg}_cpu_s"] = round(time.process_time() - cpu0, 4)
                    cpu_per_cell[lang][f"{leg}_cores"] = round(
                        cpu_per_cell[lang][f"{leg}_cpu_s"] / (wall or 1e-9), 2
                    )
        timing["cpu_legs_wall"] = round(time.perf_counter() - t_cpu, 4)

        # 4. Fuse. Rank-based RRF across legs and languages, whose score spaces differ; the
        #    within-cell across-parts merge was a raw score sort one level down, in `merge_parts`.
        t = time.perf_counter()
        pools = [p for p in (*dense_pools, *sparse_pools, *li_pools) if p]
        kwargs = {} if ctx.knobs.rrf_k is None else {"k": ctx.knobs.rrf_k}
        fused = rrf(pools, **kwargs)
        timing["fuse"] = round(time.perf_counter() - t, 4)

        # 5. Rerank the top `rerank_depth` on the searched rendering (Knob 1), not `passage_lang`.
        ranked = [(pid, float(s)) for pid, s in fused]
        rerank_n = min(int(rerank_depth), len(ranked))
        if (self.reranker is not None or self.rerank_pool is not None) and do_rerank and rerank_n:
            t_fetch = time.perf_counter()
            head = [pid for pid, _ in ranked[:rerank_n]]
            # The pool is a set, and that is checked here. `common.fusion.rrf` accumulates into a
            # dict keyed by `passage_id`, so the twelve overlapping leg pools (three legs by four
            # language cells) are already collapsed to unique ids by the time they reach this line.
            # A duplicate would waste a slot and could be handed two different scores (two shards,
            # two batch compositions, two GEMM shapes, two low-order bits), which the
            # `(-score, passage_id)` sort below would then resolve arbitrarily. Deduplication
            # happens before sharding, so the slices are balanced over unique ids.
            if len(set(head)) != len(head):
                from collections import Counter

                dupes = [p for p, c in Counter(head).most_common(5) if c > 1]
                raise RuntimeError(
                    f"the rerank pool is a multiset: {len(head)} ids, {len(set(head))} unique "
                    f"(e.g. {dupes}). Scoring a duplicate wastes a slot and can produce two "
                    "different scores for one id, which the (-score, passage_id) sort would then "
                    "resolve non-deterministically."
                )
            texts = [txt for _, txt in display(ctx, head, ctx.index)]
            timing["store_fetch"] = round(time.perf_counter() - t_fetch, 4)
            t = time.perf_counter()
            if self.rerank_pool is not None:
                scores, rerank_telem = self.rerank_pool.score(text, texts)
                timing["rerank_instances"] = rerank_telem["instances"]
                if verify_rerank:
                    # One instance, the whole set: exactly the shape the fan replaces.
                    solo_started = time.perf_counter()
                    solo = self.rerank_pool.score_solo(text, texts)
                    deltas = [abs(a - b) for a, b in zip(scores, solo, strict=True)]
                    fan_rank = sorted(zip(head, scores, strict=True),
                                      key=lambda kv: (-kv[1], kv[0]))
                    solo_rank = sorted(zip(head, solo, strict=True),
                                       key=lambda kv: (-kv[1], kv[0]))
                    rerank_telem["verify"] = {
                        "n": len(scores),
                        "solo_s": round(time.perf_counter() - solo_started, 4),
                        "solo_device": self.rerank_pool.devices[0],
                        "scores_bit_identical": all(d == 0.0 for d in deltas),
                        "differing_scores": sum(1 for d in deltas if d != 0.0),
                        "max_abs_delta": max(deltas) if deltas else 0.0,
                        "ranking_identical": fan_rank == solo_rank,
                        "ids_identical": [p for p, _ in fan_rank] == [p for p, _ in solo_rank],
                    }
            else:
                scores = self.reranker.score(text, texts)
            rescored = sorted(zip(head, scores, strict=True), key=lambda kv: (-kv[1], kv[0]))
            ranked = [(p, float(s)) for p, s in rescored] + ranked[rerank_n:]
            timing["rerank"] = round(time.perf_counter() - t, 4)
            timing["rerank_depth"] = rerank_n

        total = time.perf_counter() - t_all
        # The merged output is a set too. The fan splices N slices back together, and a splice is
        # exactly where an id could be emitted twice (a boundary counted at both ends, an
        # off-by-one in the ranges). `RerankPool.score` checks that the split covers its input
        # once; this checks the result, which is the thing the caller reads.
        if len({pid for pid, _ in ranked}) != len(ranked):
            raise RuntimeError(
                f"the merged ranking repeats an id: {len(ranked)} entries, "
                f"{len({pid for pid, _ in ranked})} unique"
            )
        return {
            "wall_s": round(total, 4),
            "timing": timing,
            "li_per_cell": li_per_cell,
            "cpu_per_cell": cpu_per_cell,
            "hits": [(p, round(s, 5)) for p, s in ranked[: int(top_k)]],
            "fused_pool": len(fused),
            "li_workers": li_fan,
            "cell_fan": cell_fan,
            "part_workers": part_fan,
            "omp": omp_w,
            # If this still equals `timing["late_interaction"]` at cell_fan > 1, the cells did not
            # overlap and threads are the wrong mechanism. The sparse leg has exactly that
            # pathology: pyseismic holds the GIL and shows no speed-up at any thread width.
            "li_cell_sum_s": round(sum(li_per_cell.values()), 4),
            "reranked": bool(do_rerank and rerank_n),
            # The concurrency evidence, per query: every shard's card, pid, size and wall-clock
            # [t0, t1) interval, plus the derived `busiest` count of how many were in flight at
            # once. A wall-time drop alone could come from anything; N overlapping intervals on N
            # distinct cards is the claim, so the claim travels in the reply rather than being
            # reconstructed from a log afterwards.
            "rerank_pool": rerank_telem,
        }

    # -- the queue --------------------------------------------------------- #
    def serve(self) -> int:
        from .rsvc_registry import Heartbeat, descriptor_path, unpublish

        a = self.args
        qin, qout = self.queue / "in", self.queue / "out"
        for d in (qin, qout):
            d.mkdir(parents=True, exist_ok=True)

        try:
            self.bring_up()
        except BaseException:
            # `bring_up` forks non-daemonic replicas; a failure after that point would otherwise
            # hang the interpreter's exit joining children blocked on a pipe.
            if self.replicas is not None:
                self.replicas.close()
            if self.rerank_pool is not None:
                self.rerank_pool.close()
            raise
        ready = {
            "ready": True,
            "cold_load_s": round(self.cold_s, 1),
            "boot": self.boot,
            "index_hash": self.index_hash,
            # The two facts a client needs to pick a lane: which rendering is searched, and which
            # built index it is. `rsvc_registry.live_services` filters on the pair.
            "rendering": self.ctx.index,
            "rendering_source": self.rendering_note,
            # Reported for provenance only. A client chooses its own reading rendering at
            # display() time; this service never reads text except for the rerank pool.
            "passage_lang": self.ctx.passage_lang,
            "config": a.config,
            "queue": str(self.queue),
            # Which lane this is, and which allocation tier won it. Two attempts of one slot are
            # two descriptors; the launcher groups on `slot` to keep exactly one of them.
            "slot": a.slot or a.name,
            "tier": a.tier,
            # Which architecture the placement pinned: the SLURM feature id, not nvidia-smi's
            # marketing string, which `gpu` below carries. Retrieval is not bit-identical across
            # architectures, so a timing taken against this service belongs to this card set and
            # cannot be compared one-variable against another's.
            "gpu_model": os.environ.get("SVC_GPU_MODEL") or None,
            # Non-empty means this service is running a labelled degradation of the canonical
            # shape -- e.g. the 14-CPU A100 placement, measured 19.76 s/query against 5.60 s at
            # 24 cores per replica. A degraded lane may exist; a degraded lane indistinguishable
            # from the
            # canonical one in the artifact may not, because that is how its numbers end up quoted
            # as if they were the shipped shape's.
            "degraded": [
                d for d in (os.environ.get("SVC_DEGRADED") or "").split("|") if d and d != "none"
            ],
            "gpu": gpu_memory(),
            "host": host_memory(),
            "cells": {k: len(v.parts) for k, v in sorted(self.ctx.cells.items())},
            # How many independent stacks answer here, and on which cards. A client sizing a batch
            # needs this: the per-query latency is a replica's, the throughput is N of them.
            "replicas": (self.replicas.devices if self.replicas is not None else []),
            "threads": thread_shape(),
            "fans": {
                "cell_fan": a.cell_fan,
                "li_workers": a.li_workers,
                "part_workers": a.part_workers,
                "cell_procs": a.cell_procs,
                # The asked budget and the realised partition, both: `part_procs: 0` means one
                # process per part, so the number a reader wants (how many processes are actually
                # searching) is only in `cell_workers`, and which parts each of them owns is only
                # in `cell_plan`.
                "part_procs": a.part_procs,
                "cell_workers": (self.pool.workers if self.pool is not None else 0),
                "cell_plan": (self.pool.plan if self.pool is not None else {}),
            },
            **shape(),
        }
        atomic_json(self.queue / "READY.json", ready)
        desc = beat = None
        if a.registry:
            desc = descriptor_path(Path(a.registry), a.name)
            # A background beat, so a service answering a long batch never looks like a corpse.
            beat = Heartbeat(desc, ready, interval_s=a.heartbeat).start()
        self._log("READY", cold_load_s=round(self.cold_s, 1), gpu=ready["gpu"])

        if a.selfbench:
            # Never fatal. The cold load is far too expensive to throw away because the measurement
            # that follows it hit a bad rung; the service must survive and stay queryable so the
            # next rung can be tried against the same resident index.
            try:
                self.selfbench(a.selfbench, top_k=a.bench_top_k, rerank_depth=a.bench_depth)
            except Exception as exc:  # noqa: BLE001
                import traceback

                self._log("bench_failed", error=f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc()[-2000:])

        idle_since = time.time()
        try:
            if self.replicas is not None:
                # A fleet is shared across requests, not held by one: see `_serve_multiplexed`.
                return self._serve_multiplexed(qin, qout)
            while True:
                pending = sorted(p for p in qin.glob("*.json") if not p.name.startswith("."))
                if not pending:
                    if a.idle_exit and (time.time() - idle_since) > a.idle_exit:
                        self._log("idle_exit")
                        return 0
                    time.sleep(a.poll)
                    continue
                for req_path in pending:
                    idle_since = time.time()
                    self._handle(req_path, qout)
        finally:
            if beat is not None:
                beat.stop()
            if desc is not None:
                unpublish(desc)
            (self.queue / "READY.json").unlink(missing_ok=True)
            if self.replicas is not None:
                self.replicas.close()
            if self.pool is not None:
                self.pool.close()
            if self.rerank_pool is not None:
                self.rerank_pool.close()

    # -- the measurement, run in-process so it cannot be lost to a client's timeout ----- #
    def selfbench(self, n: int, *, top_k: int, rerank_depth: int) -> dict[str, Any]:
        """``n`` different real topics, warm, after one discarded warm-up. Not one query repeated.

        The distinction is load-bearing and has been got wrong here before: repeating one query is
        best-case cache locality, since it touches the same parts every pass, and it measured
        65.5 s where five different queries measured 138 s on the same index. Real traffic is
        different queries.

        Cold start is reported separately, in ``boot``, and never folded into a median: a fused
        mean of 156.3 s was once quoted for a set whose first query alone was 1653 s.
        """
        from ragtime.common.topics import load_topics

        topics = load_topics(self.args.topics)
        queries = [t.problem_statement[:300] for t in topics[: n + 1]]
        if len(queries) < 2:
            raise RuntimeError(f"{self.args.topics}: need at least 2 topics to bench")

        def job(q: str) -> dict[str, Any]:
            return {"query": q, "top_k": top_k, "rerank_depth": rerank_depth}

        # The warm-up is dispatched on its own so it warms one replica; the batch below is what
        # measures the fleet. With N replicas the remaining N-1 pay their warm-up inside the batch,
        # which is why `batch_wall_s` is reported next to the per-query walls rather than instead
        # of them.
        warm = self.run_queries([job(queries[0])])[0]
        self._log("bench_warmup_discarded", wall_s=warm["wall_s"], timing=warm["timing"])

        t_batch = time.perf_counter()
        runs = self.run_queries([job(q) for q in queries[1 : n + 1]])
        batch_wall = round(time.perf_counter() - t_batch, 4)
        for i, got in enumerate(runs):
            self._log(
                "bench_query",
                i=i,
                replica=got.get("replica"),
                wall_s=got["wall_s"],
                timing=got["timing"],
                li_per_cell=got["li_per_cell"],
                cpu_per_cell=got["cpu_per_cell"],
                top3=[p for p, _ in got["hits"][:3]],
            )

        walls = sorted(r["wall_s"] for r in runs)
        legs = sorted({k for r in runs for k in r["timing"]})
        report = {
            "rsvc": "BENCH",
            "n": len(walls),
            "median_s": walls[len(walls) // 2],
            "min_s": walls[0],
            "max_s": walls[-1],
            "walls": [r["wall_s"] for r in runs],
            # The throughput number. Per-query wall is a replica's latency and should be flat in
            # N; this is what N replicas are supposed to divide.
            "batch_wall_s": batch_wall,
            "batch_s_per_query": round(batch_wall / max(len(runs), 1), 4),
            "replicas": (self.replicas.devices if self.replicas is not None else []),
            "served_by": [r.get("replica") for r in runs],
            "mean_per_stage_s": {
                leg: round(
                    sum(float(r["timing"].get(leg, 0.0)) for r in runs) / len(runs), 4
                )
                for leg in legs
            },
            "cold": self.boot,
            "top_k": top_k,
            "rerank_depth": rerank_depth,
            "n_full_scores": self.args.n_full_scores,
            "cell_procs": bool(self.args.cell_procs),
            "cell_fan": self.args.cell_fan,
            "li_workers": self.args.li_workers,
            "part_workers": self.args.part_workers,
            "part_procs": self.args.part_procs,
            "cell_workers": (self.pool.workers if self.pool is not None else 0),
            "rerank_batch": self.args.rerank_batch,
            "reranked": not self.args.no_rerank,
            "gpu": gpu_memory(),
            # In replica mode CUDA lives in the children, so calling `torch_vram()` here would
            # initialise a context in the dispatcher, on a card a replica already owns, and report
            # zeros. Each run carries its own replica's peak instead.
            "vram": (
                {f"replica{r.get('replica')}": r.get("vram") for r in runs}
                if self.replicas is not None
                else torch_vram()
            ),
            "host": host_memory(),
            "shape": shape(),
            "TARGET_1s": (
                "MET"
                if walls[len(walls) // 2] <= 1.0
                else f"MISSED: best median {walls[len(walls) // 2]:.3f}s"
            ),
        }
        print(json.dumps(report, default=str), flush=True)
        atomic_json(self.queue / "BENCH.json", report)
        return report

    # -- one request: parse -> jobs -> (dispatch happens elsewhere) -> reply ------------ #
    def _accept(self, req_path: Path) -> _Live | None:
        """Parse one queued request into an in-flight unit, consuming the file. None if torn.

        Expanding the request here rather than inside the run lets the serve loop hold several of
        these at once and feed their jobs to whichever replica is free, so a request no longer owns
        the fleet for its whole duration.
        """
        try:
            req = json.loads(req_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None  # still being written; next cycle
        req_path.unlink(missing_ok=True)
        rid = str(req.get("id") or req_path.stem)
        live = _Live(rid, req)
        try:
            # A service answering from the wrong rendering is the worst failure available here:
            # nothing downstream errors, the Knob-1 comparison is silently corrupted, and it looks
            # like a result. So a request may name the rendering it means, and a mismatch is
            # refused rather than served. Naming it is optional only so an existing client that
            # already resolved the service through the registry's rendering filter is not broken.
            asked = req.get("index", req.get("rendering"))
            if asked is not None and str(asked) != self.ctx.index:
                raise ValueError(
                    f"this service searches {self.ctx.index!r} (index_hash "
                    f"{self.index_hash[:12]}) but the request asked for {asked!r}, refusing to "
                    "answer from a different rendering"
                )
            if req.get("reset_peak", True) and self.replicas is None:
                torch_vram(reset=True)  # so each rung reports its own peak, not the session's
            queries = req["query"] if isinstance(req.get("query"), list) else [req["query"]]
            top_k = int(req.get("top_k", 20))
            depth = int(req.get("rerank_depth", self.ctx.knobs.rerank_depth))
            # The job list is built in exactly the order the single-threaded loop ran it (repeat
            # outer, queries inner) and results land at those indices, so dispatch changes who
            # answers and never what position an answer comes back in.
            live.jobs = [
                {
                    "query": q,
                    "top_k": top_k,
                    "rerank_depth": depth,
                    "li_workers": req.get("li_workers"),
                    "cell_fan": req.get("cell_fan"),
                    "part_workers": req.get("part_workers"),
                    "omp": req.get("omp"),
                    "rerank": req.get("rerank"),
                    # Opt-in per request, never a service-wide default: it doubles the rerank cost
                    # (a whole extra single-instance pass) and exists to settle the fan-versus-solo
                    # question, not to run in every measurement it would distort.
                    "verify_rerank": bool(req.get("verify_rerank", False)),
                    "reset_peak": bool(req.get("reset_peak", True)),
                }
                for _ in range(int(req.get("repeat", 1)))
                for q in queries
            ]
            if not live.jobs:
                raise ValueError(
                    "request carries no queries, refusing rather than replying with an empty "
                    "`runs` list that a client would read as a successful search that found "
                    "nothing"
                )
            live.results = [None] * len(live.jobs)
            # `use_replicas` caps the fleet for this request, so the 1-vs-N comparison happens in
            # one warm process rather than across two allocations with different page-cache
            # retention, which dominates query latency. A request that names it is a measurement
            # of fleet width, so it runs with the fleet to itself; sharing would make "1 replica"
            # mean "1 replica plus whatever else was in flight".
            use = req.get("use_replicas")
            live.width = None if use is None else int(use)
            live.exclusive = use is not None or bool(req.get("exclusive"))
        except Exception as exc:  # noqa: BLE001 - a bad request must not kill a long load
            import traceback

            live.fail(f"{type(exc).__name__}: {exc}", traceback.format_exc()[-4000:])
        return live

    def _reply(self, live: _Live, qout: Path) -> None:
        """Write the reply for one finished request. Results are placed by submission index."""
        out: dict[str, Any] = {
            "id": live.rid,
            "request": live.req,
            "shape": shape(),
            "index_hash": self.index_hash,
            "rendering": self.ctx.index,
            "cold_load_s": round(self.cold_s, 1),
        }
        if live.error is not None:
            out["ok"] = False
            out["error"] = live.error
            out["traceback"] = live.traceback
        else:
            runs = live.results
            walls = sorted(r["wall_s"] for r in runs)
            batch_wall = round(live.elapsed(), 4)
            out["runs"] = runs
            out["summary"] = {
                "n": len(walls),
                "used_replicas": (
                    len(self.replicas.devices)
                    if (self.replicas and live.width is None)
                    else live.width
                ),
                "median": walls[len(walls) // 2],
                "min": walls[0],
                "max": walls[-1],
                # The number replicas are supposed to divide. Per-query walls are a replica's
                # latency and should be roughly flat in N; this is the throughput.
                "batch_wall_s": batch_wall,
                "batch_s_per_query": round(batch_wall / max(len(runs), 1), 4),
                "served_by": [r.get("replica") for r in runs],
                # How long this request waited for its first replica. Non-zero means the fleet was
                # busy with other requests when it arrived: the one number that separates "the
                # query is slow" from "the query queued", which a wall alone cannot.
                "queued_s": round(live.queued_s(), 4),
            }
            out["gpu"] = gpu_memory()
            # Peak live allocation is what decides the smallest card this fits on; the reserved
            # figure beside it is the caching allocator's retained slack, not a requirement. In
            # replica mode CUDA lives in the children, so each run carries its own replica's.
            out["vram"] = (
                {f"replica{r.get('replica')}": r.get("vram") for r in runs}
                if self.replicas is not None
                else torch_vram()
            )
            out["replicas"] = self.replicas.devices if self.replicas is not None else []
            out["ok"] = True
        atomic_json(qout / f"{live.rid}.json", out)
        self._log("answered", id=live.rid, ok=out["ok"],
                  summary=out.get("summary") or out.get("error", "")[:200])

    def _handle(self, req_path: Path, qout: Path) -> None:
        """Answer one request with the whole service to itself: the un-multiplexed spelling.

        Used for the single-stack service, which has no fleet to share, and for a request that
        asked to own the fleet, so both keep the timings they had before multiplexing existed.
        """
        live = self._accept(req_path)
        if live is None:
            return
        self._run_exclusive(live)
        self._reply(live, qout)

    def _run_exclusive(self, live: _Live) -> None:
        if live.error is not None:
            return
        live.start()
        try:
            live.results = self.run_queries(live.jobs, live.width)
        except Exception as exc:  # noqa: BLE001 - a bad request must not kill a long load
            import traceback

            live.fail(f"{type(exc).__name__}: {exc}", traceback.format_exc()[-4000:])

    def _serve_multiplexed(self, qin: Path, qout: Path) -> int:
        """The serve loop for a fleet: keep every replica busy across requests, not only within one.

        A loop of the form ``for req_path in pending: self._handle(req_path, qout)`` takes one whole
        request at a time. Within a request the dispatcher is already least-busy, but a request that
        holds the fleet means the k RAG loops, each of which issues its own single-query search, are
        served strictly one after another: an N-replica service answers one query at a time and N-1
        cards idle through every one of them. Balanced ``served_by`` counts do not show it, because
        each request is individually balanced across the replicas it used, namely one.

        Two policies, both deliberate:

        * Fair-queued across requests, at job granularity. Each cycle hands one job to each admitted
          request in turn, so a one-query search does not wait out a twenty-query batch that arrived
          a moment earlier. FIFO over requests would only move the blocking up a level.
        * A request that names ``use_replicas`` runs alone, after the fleet drains. It is a
          measurement of fleet width, and sharing would silently invalidate it.

        Reply order is untouched: every result is placed at ``live.results[i]`` by the index its job
        was submitted with, so who ran it and when it finished are both invisible downstream.
        """
        a = self.args
        idle_since = time.time()
        live: list[_Live] = []
        while True:
            # -- admit ---------------------------------------------------------------- #
            if not any(r.exclusive for r in live):
                for path in sorted(p for p in qin.glob("*.json") if not p.name.startswith(".")):
                    got = self._accept(path)
                    if got is None:
                        continue
                    if got.error is not None:  # a refusal: answer it now, it needs no replica
                        self._reply(got, qout)
                        continue
                    if got.exclusive and live:
                        # It must own the fleet, and the fleet is busy. It is already off the
                        # queue, so hold it here and admit nothing further until it has run.
                        live.append(got)
                        break
                    live.append(got)
                    if got.exclusive:
                        break
            if not live:
                if a.idle_exit and (time.time() - idle_since) > a.idle_exit:
                    self._log("idle_exit")
                    return 0
                time.sleep(a.poll)
                continue
            idle_since = time.time()

            # -- an exclusive request runs alone, once everything else has drained ----- #
            # Admission is closed while it is held, and a request only leaves `live` once it has
            # nothing in flight, so `live == [head]` is the drained state and needs no separate
            # check.
            if len(live) == 1 and live[0].exclusive:
                head = live.pop()
                self._run_exclusive(head)
                self._reply(head, qout)
                continue

            # -- dispatch: round-robin across requests, to whichever replica is free -- #
            # A request that has just been handed a job goes to the back of the queue. Scanning
            # `live` from the head every time instead would starve outright: the first request
            # would re-take every replica the moment it freed one, and a single-query search
            # arriving behind a twenty-query batch would wait out the whole batch, which is the
            # same head-of-line blocking moved one level down rather than removed.
            while True:
                r = next((x for x in live if not x.exclusive and x.wants()), None)
                if r is None:
                    break
                if not self.replicas.dispatch((r.rid, r.nxt), r.jobs[r.nxt], r.width):
                    break  # the fleet is full: what is left waits for a replica, not for a turn
                r.sent()
                live.append(live.pop(live.index(r)))

            # -- harvest: completion order in, submission order out --------------------- #
            try:
                done = self.replicas.harvest(timeout=a.poll)
            except ReplicaDied as exc:
                # The fleet is gone. Every request in flight fails naming the replica, and the
                # service stops advertising itself rather than answering from a fleet smaller
                # than the one it published.
                for r in live:
                    r.fail(f"{type(exc).__name__}: {exc}", "")
                    self._reply(r, qout)
                raise
            for (rid, i), res in done:
                r = next((x for x in live if x.rid == rid), None)
                if r is None:  # pragma: no cover - a result for a request already replied to
                    continue
                r.took(i, res)
                if res.get("error"):
                    # A query that raised leaves the replica alive: fail this request only, and
                    # name the replica so the failure is attributable rather than anonymous.
                    r.fail(
                        f"replica {res.get('replica')} ({res.get('replica_device')}) on query "
                        f"{i}: {res['error']}",
                        res.get("traceback", ""),
                    )
            for r in [x for x in live if x.finished]:
                self._reply(r, qout)
                live.remove(r)


def _resolve_rendering(cfg: Any, wanted: str, *, allow_override: bool) -> tuple[Any, dict[str, Any]]:
    """Pin which rendering this service searches, from an explicit launch argument.

    The argument is Knob 1, ``retrieval.index``, the index that is searched, and never Knob 2,
    ``passage_lang``, what the LLM reads. Retrieval returns ids and scores and reads no text, so
    the reading rendering is a client-side ``display()`` concern that must not enter the service at
    all. The two genuinely differ: all three controlled ``e2e-*`` runs search ``original`` while
    moving only ``passage_lang``, so an ``e2e-omt`` service is an original-index service. Only the
    ``mlir-*`` family moves this axis.

    The config remains the record, so a mismatch between the argument and the config's own
    ``retrieval.index`` is a hard error: silently overriding Knob 1 is exactly the
    fairness-relevant substitution the family guard exists to prevent. ``--allow-index-override``
    is the escape hatch, and it is recorded in the descriptor, so an artifact produced under one is
    identifiable rather than merely suspected.
    """
    import dataclasses as dc

    from ragtime.common.passage_store import RENDERINGS as STORE_RENDERINGS
    from ragtime.config.schema import KNOB_VALUES
    from ragtime.devkit.rsvc_registry import RENDERINGS as CLIENT_RENDERINGS

    if (
        set(RENDERINGS) != set(STORE_RENDERINGS)
        or set(RENDERINGS) != set(KNOB_VALUES)
        or set(RENDERINGS) != set(CLIENT_RENDERINGS)
    ):
        raise RuntimeError(
            f"rendering set drift: rsvc={RENDERINGS} store={STORE_RENDERINGS} "
            f"config={KNOB_VALUES} client={CLIENT_RENDERINGS}"
        )
    if wanted not in RENDERINGS:
        raise SystemExit(
            f"--index {wanted!r} is not one of {RENDERINGS}; a service is never started against "
            "a rendering that does not exist, and there is no default"
        )
    from_config = str(getattr(cfg, "retrieval_index", "") or "original")
    if wanted == from_config:
        return cfg, {"index": wanted, "from_config": from_config, "override": False}
    if not allow_override:
        raise SystemExit(
            f"--index {wanted!r} disagrees with {from_config!r}, which is what "
            "retrieval.index resolves to in this config. The config is the run record, so this "
            "is refused rather than silently honoured. Either launch the config whose Knob 1 is "
            f"{wanted!r} (the mlir-* family is the one that moves this axis), or pass "
            "--allow-index-override to make the substitution explicit and recorded."
        )
    return dc.replace(cfg, retrieval_index=wanted), {
        "index": wanted,
        "from_config": from_config,
        "override": True,
    }


def _apply_store_mirror(cfg, root):
    """Set ``execution.passage_store_mirror_root`` from ``--store-local``. Nothing else.

    The smallest possible change. ``retrieval.context.bring_up`` already calls
    ``store_mirror.mirror_root(cfg)``, and ``resolve_store_location`` already owns the path
    re-resolution, the ``_SUCCESS`` check, the origin-size comparison and the loud refusal. All
    this adds is a launcher-side way to set the leaf; without it a service reads the 64 GiB by-id
    passage store off shared storage on every query.

    ``blocks`` is copied rather than mutated: the loader may hand out a shared structure, and a
    service that edits its caller's config makes the run record stop describing the run.
    """
    import dataclasses as dc

    text = str(root or "").strip()
    if not text:
        return cfg, {"store_local": None}
    blocks = dict(getattr(cfg, "blocks", None) or {})
    execution = dict(blocks.get("execution") or {})
    execution["passage_store_mirror_root"] = text
    blocks["execution"] = execution
    return dc.replace(cfg, blocks=blocks), {"store_local": text}


def _to_numpy(rep: Any) -> Any:
    """Make a query representation cheap and safe to send down a pipe."""
    detach = getattr(rep, "detach", None)
    if detach is not None:
        return detach().cpu().numpy()
    return rep


def _fan_li_cells(
    cells: dict[str, Any],
    rep: Any,
    top_k: int,
    cell_workers: int,
    part_workers: int,
    search_with_rep: Any,
) -> tuple[list[list], dict[str, float]]:
    """Score the language cells of the late-interaction leg, concurrently when asked.

    ``li_workers`` is the fan inside one cell, over its parts; nothing fans the cells themselves,
    so without this the leg's wall is the sum over cells rather than the max. The arithmetic is
    exact rather than suggestive: ``sum(li_per_cell) == timing["late_interaction"]`` to four
    decimals in every bench record (1.557 + 1.476 + 0.879 + 1.330 = 5.2412 against a reported
    5.2413). Since the CPU legs already hide under this leg (``cpu_legs_wall == late_interaction``
    in every record), the sum-versus-max difference is the whole remaining budget.

    Order is preserved, and that is a correctness property rather than a style choice. Results are
    collected by ``executor.map`` in ``sorted(cells)`` order, never appended as they complete: RRF
    accumulates floats, so a different pool order can change the fused ranking in the last bits.
    The returned lists must be byte-identical to the sequential path, which is checked by running
    both widths against the same queries through the same process, not asserted.

    Width is a parameter defaulting to 1, the sequential behaviour, because the failure mode is
    fatal rather than graceful. Concurrent PLAID searches multiply transient VRAM, and a PLAID OOM
    arrives as ``pyo3_runtime.PanicException`` from Rust, which is not a catchable Python exception
    and takes the whole service down instead of failing one query. That killed a service outright
    at fan width 3 on a 24 GiB card.
    """
    from concurrent.futures import ThreadPoolExecutor

    langs = sorted(cells)
    per_cell: dict[str, float] = {}

    def one(lang: str) -> list:
        started = time.perf_counter()
        hits = _fan_li(cells[lang], rep, top_k, part_workers, search_with_rep)
        per_cell[lang] = round(time.perf_counter() - started, 4)
        return hits

    if cell_workers <= 1 or len(langs) == 1:
        return [one(lang) for lang in langs], per_cell
    with ThreadPoolExecutor(
        max_workers=min(cell_workers, len(langs)), thread_name_prefix="rsvc-li-cell-fan"
    ) as pool:
        pools = list(pool.map(one, langs))  # map gives submission order, never completion order
    return pools, per_cell


def _fan_li(handle: Any, rep: Any, top_k: int, workers: int, search_with_rep: Any) -> list:
    """The late-interaction fan over one cell's parts, on the GPU, in the parent process."""
    from concurrent.futures import ThreadPoolExecutor

    parts = handle.parts
    if workers <= 1 or len(parts) == 1:
        per_part = [search_with_rep(p, LATE, rep, top_k) for p in parts]
    else:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(parts)), thread_name_prefix="rsvc-li-fan"
        ) as pool:
            per_part = list(pool.map(lambda p: search_with_rep(p, LATE, rep, top_k), parts))
    out = [hit for hits in per_part for hit in hits]
    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out[: int(top_k)]


# --------------------------------------------------------------------------- #
# Reranker verification: two loads in one process must give identical scores.
# --------------------------------------------------------------------------- #
def verify_reranker(model: str, device: str, batch_size: int) -> int:
    """Prove the reranker is deterministic and not a random permutation. Loads twice.

    The ``CrossEncoder`` path fails both halves: its ``score.weight`` is freshly random per load,
    so two loads disagree, and it is unrelated to relevance, so the ranking is a permutation of
    the input pool. The two facts are checked separately, because passing one and failing the
    other would be a different bug.
    """
    query = "What has been the international response to the conflict in Sudan?"
    passages = [
        (
            "The UN Security Council met in emergency session to discuss the escalating conflict "
            "in Sudan, with several member states calling for an immediate ceasefire and "
            "humanitarian corridors into Khartoum."
        ),
        (
            "Sudan's warring factions agreed to a seven-day truce brokered by Saudi Arabia and "
            "the United States, though aid agencies reported continued fighting in Darfur."
        ),
        (
            "The recipe calls for two cups of flour, one egg, and a teaspoon of baking powder. "
            "Mix the dry ingredients before folding in the wet ones."
        ),
        (
            "Manchester United's new signing scored twice on his debut at Old Trafford, ending a "
            "four-match goalless run for the club."
        ),
    ]
    runs = []
    for _ in range(2):
        from ragtime.serving.reranker import Reranker  # the one reranker (dedup phase 2)

        rr = Reranker(model, device, batch_size=batch_size)
        rr.load()
        runs.append(rr.score(query, passages))
        del rr
    identical = runs[0] == runs[1]
    order = sorted(range(len(passages)), key=lambda i: (-runs[0][i], i))
    distinct = len(set(runs[0])) == len(runs[0])
    # Passages 0 and 1 are on-topic; 2 and 3 are not. A working reranker ranks {0,1} above {2,3}.
    non_random = set(order[:2]) == {0, 1}
    print(
        json.dumps(
            {
                "rsvc": "verify_reranker",
                "model": model,
                "device": device,
                "load_1": [round(s, 6) for s in runs[0]],
                "load_2": [round(s, 6) for s in runs[1]],
                "identical_across_loads": identical,
                "relevant_ranked_first": non_random,
                "all_scores_distinct": distinct,
                "order": order,
                "gpu": gpu_memory(),
                "VERDICT": "PASS" if (identical and non_random) else "FAIL",
            },
            default=str,
        ),
        flush=True,
    )
    return 0 if (identical and non_random) else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    # Function-local, like every other `ragtime` import in this module: rsvc must stay importable
    # without the heavy serving stack, and `ragtime.common.__init__` pulls in pyarrow and lmdb. The
    # topics default is not restated here, so there is one spelling of which topics file is
    # canonical and the dev tools cannot drift off the run record.
    from ragtime.common.topics import CANONICAL_TOPICS_REL

    ap = argparse.ArgumentParser(prog="python -m ragtime.devkit.rsvc")
    ap.add_argument("--config", default="config/e2e-original.yml")
    ap.add_argument(
        "--index",
        required=True,
        choices=list(RENDERINGS),
        help="which rendering this service searches: Knob 1 (retrieval.index), never "
             "passage_lang. Required and closed-set, with no default, because defaulting would "
             "let a mis-launched service answer from the wrong index and look correct.",
    )
    ap.add_argument(
        "--allow-index-override",
        action="store_true",
        help="permit --index to differ from the config's retrieval.index (recorded in the "
             "descriptor as override:true).",
    )
    ap.add_argument("--queue", default="")
    ap.add_argument("--registry", default="", help="directory the descriptor is published in")
    ap.add_argument("--name", default="", help="descriptor name (default: rendering-jobid)")
    ap.add_argument(
        "--slot",
        default="",
        help="the lane this service belongs to, when the launcher raced several allocation tiers "
        "for it. Every attempt of one slot gets its own --name and its own --queue, so a cancel "
        "that races a start can only delete its own descriptor, and this is what tells the "
        "launcher that two descriptors are two attempts at the same lane. Defaults to --name.",
    )
    ap.add_argument("--tier", default="", help="which allocation tier won this slot (provenance)")
    # Placement: one card per component. A device list replicates the index rather than sharding it.
    ap.add_argument("--gpu-plaid", default="cuda:0")
    ap.add_argument(
        "--plaid-low-memory",
        choices=("default", "true", "false"),
        default="default",
        help="override PLAID low_memory. `default` leaves production alone (True on any CUDA "
             "path). `false` is the full-residuals-in-VRAM form: byte-identical to True (20/20 on "
             "the same queries), faster, and documented at ~100-105 GiB per rendering, which "
             "exceeds one 95.59 GiB card.",
    )
    ap.add_argument(
        "--plaid-shard",
        default="",
        help="route language cells to separate cards, e.g. "
             "`en=cuda:0,es=cuda:0,ru=cuda:1,zh=cuda:1`. Empty means every cell on --gpu-plaid, "
             "the contended default.",
    )
    ap.add_argument(
        "--replicas",
        type=int,
        default=1,
        help="how many complete, independent stacks to fork, one per card. A query is dispatched "
             "to one replica and finishes entirely there (PLAID, fusion, rerank), so nothing is "
             "reconciled across cards and a batch of N queries costs about 1/N of the serial "
             "wall. Measured 36.87 GiB per stack (PLAID 20.50 + reranker/encoders 16.37), and the "
             "~210 GiB of dense+sparse is shared copy-on-write, so two replicas cost ~410 GiB of "
             "host RAM, not 820. `0` selects the single in-parent stack that every earlier "
             "measurement in this module was taken on.",
    )
    ap.add_argument(
        "--replica-devices",
        default="",
        help="one card per replica, e.g. `cuda:0,cuda:1`; defaults to cuda:0..cuda:N-1. Explicit "
             "so a node whose free cards are not 0..N-1 still works; never hardcoded to two.",
    )
    ap.add_argument(
        "--store-local",
        default="",
        help="root a staged by-id passage store is served from. Sets the existing config leaf "
             "`execution.passage_store_mirror_root`, which `retrieval.context.bring_up` already "
             "reads. Empty means the canonical store on shared storage, the shipped default. "
             "Measured `store_fetch` 0.0024-0.0041 s from /dev/shm against ~25.5 s off shared "
             "storage, on queries the service had never seen. Paid on every query, because no "
             "two queries share a candidate pool. An unusable mirror falls back to the origin "
             "loudly (`retrieval.store_mirror_refused` plus a warning), never silently.",
    )
    ap.add_argument("--gpu-mtd", default="cuda:1")
    ap.add_argument("--gpu-dense", default="cuda:1")
    ap.add_argument("--gpu-sparse", default="cuda:1")
    ap.add_argument("--gpu-rerank", default="cuda:1")
    # Fan widths.
    ap.add_argument("--cell-procs", type=int, default=1, help="1 = fork the cell-worker pool")
    ap.add_argument(
        "--part-procs",
        type=int,
        default=0,
        help="total cell-worker processes across all cells, per replica. `0`, the default, means "
             "one process per index part, which is the measured optimum: the unit of work is the "
             "part (78 here, en 23 / es 20 / ru 14 / zh 21), pyseismic-lsr holds the GIL for the "
             "whole of `search`, and the thread fan it replaces achieved 0.99-1.00 cores at widths "
             "2, 6, 12 and 23 (processes at W=23 were 17.91x at 20.43 cores, ids and scores "
             "bit-identical). A positive value is shared out in proportion to each cell's part "
             "count; beware quantisation, since a cell's wall is `ceil(parts/slices) x per-part`, "
             "so W=12 and W=16 both read ~0.29 s on 23 parts. Device-resident legs are never "
             "partitioned, because a process costs a whole CUDA context on that card, 612 MiB "
             "measured; see `device_resident_legs`.",
    )
    ap.add_argument("--part-workers", type=int, default=4, help="per-part fan inside a cell worker")
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help=(
            "torch's CPU intra-op pool in the replica parent (0 leaves OMP_NUM_THREADS's value). "
            "Set explicitly because OMP_NUM_THREADS carries the cell processes' FAISS budget, "
            "which is small; torch.set_num_threads is a true process global, unlike "
            "OpenMP's per-thread omp_set_num_threads, so the parent can raise its own pool "
            "without unbounding the children's."
        ),
    )
    ap.add_argument(
        "--cell-omp",
        type=int,
        default=0,
        help=(
            "FAISS's OpenMP width inside each cell process (0 leaves OMP_NUM_THREADS alone). "
            "Separate from the environment variable because that is also torch's intra-op width, "
            "and the two want different numbers: the parent drives PLAID through torch while the "
            "cell processes drive FAISS, in different processes with different budgets."
        ),
    )
    ap.add_argument(
        "--cell-fan",
        type=int,
        default=1,
        help="how many language cells the late-interaction leg scores at once. 1 is the "
             "sequential loop, whose wall is the sum over cells. Above 1 multiplies transient "
             "VRAM, and a PLAID OOM is an uncatchable Rust panic that takes the whole service "
             "down, so this defaults to the safe value and is raised deliberately.",
    )
    ap.add_argument("--li-workers", type=int, default=2,
                    help="per-part fan for PLAID. Each concurrent part holds its own ~0.9-2 GiB "
                         "scratch, so this multiplies VRAM: width 4 with the MTD encoder "
                         "co-resident OOMed a 24 GiB card")
    # Model and rerank knobs.
    ap.add_argument("--n-full-scores", type=int, default=None)
    ap.add_argument("--rerank-model", default="")
    ap.add_argument(
        "--rerank-devices",
        default="",
        help="fan the rerank over these cards, one instance per card, e.g. "
             "`cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5`. The fused candidate set, already "
             "deduplicated by RRF (12 overlapping pools -> ~900 unique ids), is split into "
             "batch-aligned balanced slices and scored "
             "concurrently, one forked process per card, then merged by shard index. Empty means "
             "the single in-parent instance on --gpu-rerank. A repeated card is refused at "
             "startup, and so is a load whose weights did not land where the argv said. Ignored "
             "with --replicas >= 1, where each replica owns its own reranker on its own card.",
    )
    ap.add_argument("--rerank-batch", type=int, default=8)
    ap.add_argument("--rerank-max-len", type=int, default=1024)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-prefault", action="store_true")
    ap.add_argument(
        "--plaid-local",
        default="",
        help="mirror the late-interaction blobs here and search that: /dev/shm, where the pages "
             "are memory and a fault costs nothing, or node-local NVMe. This is the fix that "
             "works; --prefault is not, because mmap faults on a network filesystem run at "
             "~1.5 MB/s regardless of free memory.",
    )
    ap.add_argument(
        "--mirror-overlap",
        choices=("on", "off"),
        default="on",
        help="run the --plaid-local blob mirror concurrently with the CPU-leg warm. They touch "
             "different legs and different files, and the mirror initialises no CUDA, so the "
             "pre-fork window stays CPU-only. `off` restores the strictly sequential order, and "
             "exists so the two can be compared for byte-identical results.",
    )
    ap.add_argument("--prefault-threads", type=int, default=8)
    ap.add_argument("--warm-workers", type=int, default=4,
                    help="concurrency for the CPU-leg cold load (854.9 s at width 1)")
    # Lifecycle.
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--heartbeat", type=float, default=30.0)
    ap.add_argument("--idle-exit", type=float, default=None)
    ap.add_argument("--verify-reranker", action="store_true")
    # In-process measurement.
    ap.add_argument("--selfbench", type=int, default=0, help="bench N topics right after READY")
    ap.add_argument("--bench-top-k", type=int, default=20)
    ap.add_argument("--bench-depth", type=int, default=20)
    ap.add_argument("--topics", default=CANONICAL_TOPICS_REL)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `--verify-reranker` loads no index, so requiring a rendering for it would be pointless.
    if "--verify-reranker" in argv and not any(x.startswith("--index") for x in argv):
        argv += ["--index", RENDERINGS[0]]
    a = build_parser().parse_args(argv)
    if a.verify_reranker:
        model = a.rerank_model or "Qwen/Qwen3-Reranker-4B"
        return verify_reranker(model, a.gpu_rerank, a.rerank_batch)
    if not a.queue:
        raise SystemExit("--queue is required unless --verify-reranker")
    if not a.name:
        a.name = f"rsvc-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    return Service(a).serve()


if __name__ == "__main__":
    sys.exit(main())
