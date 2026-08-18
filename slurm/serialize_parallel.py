"""Per-topic parallel, resumable select-and-serialize over the real ``project()``.

One shard is one (arm, topic). The map step calls
``ragtime.pipeline.select_serialize.project.project()`` on a symlink view of the cell holding
exactly that topic; the reduce step concatenates the per-topic files in the same lexicographic
topic order the sequential path uses, and runs the official validator on the result.

Nothing here reimplements serialization: every emitted byte comes out of ``project()``. This file
owns the fan, the resume check and the concatenation. Resume is the ``_SUCCESS`` companion that
``submission.write`` already writes, so a shard whose declared outputs are all complete is skipped
before any client is built.

    python slurm/serialize_parallel.py \\
        --spec e2e-original:config/e2e-original.yml:$ROOT/e2e-original__original__seed0 \\
        --out /scratch/serialize/e2e --workers 8
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Built once per worker process and reused for every shard it runs: the first embed call loads the
# query encoder on CPU, and paying that per topic would eat the speedup.
_CACHE: dict[str, Any] = {}


def _cfg_and_clients(config_path: str) -> tuple[Any, Any]:
    key = f"cfg::{config_path}"
    if key not in _CACHE:
        from ragtime.config import load
        from ragtime.serving.registry import build_clients

        cfg = load(config_path)
        _CACHE[key] = (cfg, build_clients(cfg))
    return _CACHE[key]


def topics_file_for(src: str, wanted: list[str], dst: Path) -> Path:
    """Write a topics file holding exactly `wanted`, normalized through `common.load_topics`.

    Normalized rather than copied: the shipped topics file is concatenated single-line JSONL,
    which the validator cannot parse.
    """
    from ragtime.common import load_topics

    rows = [dataclasses.asdict(t) for t in load_topics(src)]
    keep = [r for r in rows if str(r.get("topic_id")) in set(wanted)]
    missing = set(wanted) - {str(r.get("topic_id")) for r in keep}
    if missing:
        raise SystemExit(f"topics {sorted(missing)} are not in {src}")
    dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in keep) + "\n")
    return dst


def _declared(cfg: Any) -> list[dict[str, Any]]:
    """The `config.outputs` deliverables, as {task, track, path}."""
    from ragtime.pipeline.select_serialize.submission.envelope import envelopes

    return [{"task": e.task, "track": e.track, "path": e.path} for e in envelopes(cfg)]


def shard_dir(out: Path, arm: str, topic: str) -> Path:
    return Path(out) / arm / "per_topic" / topic


def shard_done(cfg: Any, out: Path, arm: str, topic: str) -> bool:
    """True when every declared output of this shard carries its `_SUCCESS` companion."""
    from ragtime.common.io import is_done

    root = shard_dir(out, arm, topic)
    paths = [Path(d["path"]) for d in _declared(cfg)]
    if not paths:
        return False
    return all(is_done(p if p.is_absolute() else root / p) for p in paths)


def _run_shard(shard: dict[str, Any]) -> dict[str, Any]:
    """Serialize one (arm, topic) through the real `project()`. Runs in a worker process."""
    t0 = time.time()
    # Set before any client is built: LlmClient reads RAGTIME_VLLM_URL in its constructor, so a
    # later export would silently address the default endpoint. The bundle is cached per process,
    # which pins a worker to the endpoint its first shard carried; since the driver round-robins
    # the url over shards, that gives one lane per worker.
    if shard.get("url"):
        os.environ["RAGTIME_VLLM_URL"] = shard["url"]
    os.environ.setdefault("no_proxy", "*")
    os.environ.setdefault("NO_PROXY", "*")

    from ragtime.common import Layout, Statistics
    from ragtime.orchestration.cli import _serialize_confirm, _serialize_embed, _topics_path
    from ragtime.pipeline.select_serialize.project import project

    arm, topic = shard["arm"], shard["topic"]
    out, cell = Path(shard["out"]), Path(shard["cell_dir"])
    cfg, clients = _cfg_and_clients(shard["config"])

    sd = shard_dir(out, arm, topic)
    sd.mkdir(parents=True, exist_ok=True)

    # A cell view holding exactly this topic. `project` refuses a cell holding a topic the topics
    # file does not list, and would serialize the neighbours if the view held them. Symlinks: the
    # run tree is read-only here.
    view = out / arm / ".view" / topic
    (view / "topics").mkdir(parents=True, exist_ok=True)
    link = view / "topics" / topic
    if not link.exists():
        link.symlink_to(cell / "topics" / topic)

    topics_file = topics_file_for(str(_topics_path(cfg)), [topic], sd / "topics.one.jsonl")

    # Count the model calls this stage makes without changing what it does: `project` takes embed
    # and confirm as injected callables, so the wrappers are pure pass-throughs.
    calls = {"embed": 0, "embed_texts": 0, "confirm": 0, "confirm_true": 0, "llm_s": 0.0}
    base_embed = _serialize_embed(clients)
    base_confirm = _serialize_confirm(cfg, clients, shard["seed"])

    def embed(texts: list[str]) -> Any:
        calls["embed"] += 1
        calls["embed_texts"] += len(texts)
        return base_embed(texts)

    async def confirm(a: str, b: str) -> bool:
        calls["confirm"] += 1
        t = time.time()
        try:
            return_value = await base_confirm(a, b)
        finally:
            calls["llm_s"] += time.time() - t
        calls["confirm_true"] += int(bool(return_value))
        return return_value

    layout = Layout(run_dir=view, outputs=getattr(cfg, "outputs", None), base=str(sd))
    stats = Statistics()
    result = asyncio.run(
        project(
            cfg,
            cell_dir=view,
            layout=layout,
            topics_path=topics_file,
            submission_root=str(sd),
            embed=embed,
            confirm=confirm,
            stats=stats,
            variant=cfg.passage_lang,
            seed=shard["seed"],
            drop_orphan_answers=shard.get("drop_orphan_answers", False),
            recompute_t2=shard.get("recompute_t2", False),
        )
    )
    rec = {
        "arm": arm,
        "topic": topic,
        "ok": True,
        "wall_s": round(time.time() - t0, 3),
        "paths": [str(p) for p in result.paths],
        "coverage": {k: str(v) for k, v in result.coverage.items()},
        "calls": calls,
        "pid": os.getpid(),
        "url": shard.get("url"),
    }
    (sd / "shard.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    return rec


def _is_conn_error(exc: BaseException) -> bool:
    """True when the shard's pinned endpoint has gone, rather than the topic being bad."""
    name = type(exc).__name__
    return "APIConnection" in name or "Connect" in name or "Connection error" in str(exc)


def _run_shard_guarded(shard: dict[str, Any]) -> dict[str, Any]:
    """Run one shard, reporting a failure instead of killing the fan.

    A dead endpoint is retried once against a different live url, because the fleet cycles pairs
    by design. A data error is not retried: it would fail identically everywhere.
    """
    import traceback

    try:
        return _run_shard(shard)
    except BaseException as exc:  # noqa: BLE001 - reported per shard, never swallowed
        first = f"{type(exc).__name__}: {exc}"
        if _is_conn_error(exc) and not shard.get("_retried"):
            try:
                alt = [u for u in live_urls(shard["registry"]) if u != shard.get("url")]
            except Exception:  # noqa: BLE001 - the registry is best-effort here
                alt = []
            if alt:
                _CACHE.clear()
                retry = dict(shard, url=alt[hash(shard["topic"]) % len(alt)], _retried=True)
                try:
                    rec = _run_shard(retry)
                    rec["retried_after"] = first
                    return rec
                except BaseException as exc2:  # noqa: BLE001
                    return {
                        "arm": shard["arm"], "topic": shard["topic"], "ok": False,
                        "error": f"{type(exc2).__name__}: {exc2} (after retry; first: {first})",
                        "traceback": traceback.format_exc()[-4000:],
                    }
        return {
            "arm": shard["arm"], "topic": shard["topic"], "ok": False,
            "error": first,
            "traceback": traceback.format_exc()[-4000:],
        }


def concat_arm(cfg: Any, out: Path, arm: str, topics: list[str], *, validate: bool) -> list[dict]:
    """Concatenate the per-topic files into one submission file per declared track.

    Order matters: `project()` over a whole cell emits rows in lexicographic topic order, and
    `sorted(topics)` reproduces it, so the concatenated bytes match the sequential path line for
    line. The merge is `common.io.concat_files`, which copies raw bytes, so a float formatting or
    key order difference between the two paths is structurally impossible.
    """
    from ragtime.common.io import concat_files
    from ragtime.orchestration.cli import _topics_path
    from ragtime.pipeline.select_serialize.submission.validate import (
        FORMAT_NUGGETS,
        FORMAT_REPORT,
    )
    from ragtime.pipeline.select_serialize.submission.validate import validate as run_validator

    reports = []
    covered = sorted(topics)
    for d in _declared(cfg):
        rel = Path(d["path"])
        parts: list[Path] = []
        used: list[str] = []
        lines = 0
        for topic in covered:
            part = shard_dir(out, arm, topic) / rel
            if not part.exists():
                continue
            parts.append(part)
            used.append(topic)
            lines += sum(1 for ln in part.read_text(encoding="utf-8").split("\n") if ln.strip())
        target = out / arm / rel
        concat_files(target, parts, skip_if_done=False)
        rec = {"arm": arm, "track": d["track"], "task": d["task"],
               "path": str(target), "lines": lines, "topics": len(used)}
        if validate and d["task"] in (1, 3) and used:
            tf = topics_file_for(
                str(_topics_path(cfg)), used, out / arm / f"topics.covered.t{d['task']}.jsonl"
            )
            v = run_validator(
                target, FORMAT_REPORT if d["task"] == 1 else FORMAT_NUGGETS, topics_path=tf
            )
            rec["validator"] = {"ran": v.ran, "ok": v.ok, "rc": v.returncode,
                                "stdout": v.stdout[-2000:], "stderr": v.stderr[-2000:]}
        reports.append(rec)
    return reports


def live_urls(registry: str) -> list[str]:
    """Every beating vLLM endpoint, read only.

    Claiming one would move the descriptor out of the fleet's free pool and starve the production
    run of a pair; serialize is a guest on these instances.
    """
    from ragtime.serving.vllm_registry import live_endpoints

    return [str(d["url"]) for d in live_endpoints(registry)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-topic parallel resumable select-and-serialize over the real project().",
    )
    ap.add_argument("--spec", action="append", required=True,
                    help="arm:config:cell_dir - repeatable; every arm's topics join one pool")
    ap.add_argument("--out", required=True, help="scratch output root, never the real run tree")
    ap.add_argument("--workers", type=int, default=8,
                    help="process pool size, and the ceiling on concurrent LLM requests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topics", default="", help="comma-separated subset (default: all complete)")
    ap.add_argument("--limit", type=int, default=0, help="first N topics per arm (0 = all)")
    ap.add_argument("--registry", default=os.environ.get("RAGTIME_VLLM_REGISTRY", ""),
                    help="vLLM endpoint registry directory")
    ap.add_argument("--urls", default="", help="comma-separated vLLM urls (default: the registry)")
    # An orphan answer has no loop twin, so it has no citations and can never be emitted; refusing
    # the whole topic for it loses good topics to save nothing. Every drop is still logged. The
    # library default stays strict and only this CLI opts out.
    ap.add_argument("--strict-orphan-answers", action="store_true",
                    help="fail the shard on a bank answer with no loop twin (library default)")
    ap.add_argument(
        "--recompute-t2", action="store_true",
        help="emit only Task 2, synthesising a task-2 deliverable if the config declares none",
    )
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--no-concat", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    urls = [u for u in a.urls.split(",") if u] or live_urls(a.registry)
    if not urls:
        print("no live vLLM endpoint - refusing to start (a shard would hang on localhost)")
        return 2
    print(f"endpoints: {len(urls)}  {urls}")

    from ragtime.config import load

    wanted = [t for t in a.topics.split(",") if t]
    shards, arms = [], {}
    for spec in a.spec:
        arm, config, cell = spec.split(":", 2)
        cfg = load(config)
        base = Path(cell) / "topics"
        topics = sorted(
            p.name for p in base.iterdir() if p.is_dir() and (p / "_SUCCESS").exists()
        )
        if wanted:
            topics = [t for t in topics if t in wanted]
        if a.limit:
            topics = topics[: a.limit]
        arms[arm] = (cfg, config, cell, topics)
        for t in topics:
            shards.append({"arm": arm, "config": config, "cell_dir": cell, "topic": t,
                           "out": str(out), "seed": a.seed, "registry": a.registry,
                           "drop_orphan_answers": not a.strict_orphan_answers,
                           "recompute_t2": a.recompute_t2})
        print(f"arm {arm}: {len(topics)} complete topic(s)")

    todo, skipped = [], 0
    for s in shards:
        cfg = arms[s["arm"]][0]
        if shard_done(cfg, out, s["arm"], s["topic"]):
            skipped += 1
            continue
        todo.append(s)
    for i, s in enumerate(todo):
        s["url"] = urls[i % len(urls)]

    print(f"shards: {len(shards)} total, {skipped} already serialized, {len(todo)} to do")
    t0 = time.time()
    results: list[dict] = []
    if todo:
        with ProcessPoolExecutor(max_workers=min(a.workers, len(todo))) as pool:
            futs = {pool.submit(_run_shard_guarded, s): s for s in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                results.append(rec)
                el = time.time() - t0
                if rec.get("ok"):
                    print(f"[{n}/{len(todo)}] {rec['arm']}/{rec['topic']} "
                          f"{rec['wall_s']}s confirm={rec['calls']['confirm']} "
                          f"llm={rec['calls']['llm_s']:.1f}s  elapsed={el/60:.1f}m", flush=True)
                else:
                    print(f"[{n}/{len(todo)}] FAILED {rec['arm']}/{rec['topic']} "
                          f"{rec.get('error')}", flush=True)
    map_s = time.time() - t0
    failed = [r for r in results if not r.get("ok")]
    print(f"\nmap: {len(todo)} shard(s) in {map_s:.1f}s ({map_s/60:.2f} min), "
          f"{len(failed)} failed, workers={a.workers}")
    if results:
        walls = [r["wall_s"] for r in results if r.get("ok")]
        conf = sum(r["calls"]["confirm"] for r in results if r.get("ok"))
        llm = sum(r["calls"]["llm_s"] for r in results if r.get("ok"))
        if walls:
            print(f"     per-shard wall: median {sorted(walls)[len(walls)//2]:.1f}s "
                  f"min {min(walls):.1f}s max {max(walls):.1f}s, serial-equivalent "
                  f"{sum(walls)/60:.2f} min")
            print(f"     llm: {conf} confirm call(s), {llm:.1f}s of LLM wall "
                  f"({llm/max(sum(walls),1e-9)*100:.1f}% of shard wall)")
    for r in failed[:10]:
        print(f"  failed {r['arm']}/{r['topic']}: {r.get('error')}\n{r.get('traceback','')}")

    summary = {"map_wall_s": map_s, "workers": a.workers, "shards": len(shards),
               "skipped": skipped, "ran": len(todo), "failed": len(failed), "results": results}
    if not a.no_concat:
        t1 = time.time()
        allrep = []
        for arm, (cfg, _config, _cell, topics) in arms.items():
            done = [t for t in topics if shard_done(cfg, out, arm, t)]
            allrep += concat_arm(cfg, out, arm, done, validate=not a.no_validate)
        summary["concat_wall_s"] = time.time() - t1
        summary["files"] = allrep
        print(f"\nreduce: {time.time()-t1:.1f}s")
        for r in allrep:
            v = r.get("validator")
            vs = ("validator pass" if v and v["ok"] else
                  f"validator FAIL rc={v['rc']} {v['stdout'][:200]}" if v else "not validated")
            print(f"  {r['arm']:16s} {r['track']:18s} {r['lines']:4d} line(s) "
                  f"{r['topics']:3d} topic(s)  {vs}\n    {r['path']}")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\ntotal {time.time()-t0:.1f}s   summary: {out/'summary.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
