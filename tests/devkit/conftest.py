"""A test double of the retrieval service's half of the filesystem protocol.

The client half (``devkit.rsvc_registry.ask_service`` / ``ask``) runs for real: resolving a
descriptor, filtering on rendering + ``index_hash``, re-checking the heartbeat while waiting. Only
the ranked list is faked, which is the one thing a GPU job would otherwise have to provide. The
refusals under test, a stale service and the wrong lane, are properties of the client, so a test
that faked the client would prove nothing about them.

The reply shape mirrors ``rsvc.Service._handle``: ``{id, request, ok, rendering, index_hash,
runs:[{hits, wall_s, fused_pool, ...}]}``. A drift in that shape surfaces here.
"""

from __future__ import annotations

import json
import threading
import time
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Layout
from ragtime.devkit.rsvc_registry import descriptor_path, publish

INDEX_HASH = "2072b711bfa2deadbeefcafe0000000000000000000000000000000000000000"
OTHER_INDEX_HASH = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"


def dev_cfg(
    *, index: str = "original", passage_lang: str = "original", depth: int = 100
) -> types.SimpleNamespace:
    """The minimum a ``retrieval_knobs`` read needs: the two knobs and the rerank depth.

    Hand-built rather than loaded from ``config/*.yml`` so a small test can move Knob 1 and Knob 2
    independently without touching a shipped run record (which is a fairness-family artifact and is
    never edited by a test).
    """
    return types.SimpleNamespace(
        blocks={
            "retrieval": {
                "index": index,
                "passage_lang": passage_lang,
                "rrf": {"k": 60},
                "reranker": {"model": "Qwen/Qwen3-Reranker-4B", "depth": depth},
            }
        },
        retrieval_index=index,
        passage_lang=passage_lang,
    )


@dataclass
class FakeService:
    """One service lane: a descriptor, a queue, and (optionally) a thread that answers it.

    ``serve=False`` publishes a descriptor with no answering thread, which is the case where a
    descriptor advertises a service that is no longer alive.
    """

    root: Path
    name: str = "fake"
    rendering: str = "original"
    #: What this service actually searches, when that differs from what its descriptor advertises.
    #: A descriptor is a claim a service writes about itself, not a fact, so the server-side
    #: refusal below only means something when the two can disagree.
    searches: str | None = None
    index_hash: str = INDEX_HASH
    hits: tuple[tuple[str, float], ...] = ()
    serve: bool = True
    #: Seconds subtracted from the published heartbeat; over 120 makes the descriptor stale.
    beat_age_s: float = 0.0
    #: What the reply claims, when a service that misreports its lane is what the test is about.
    reply_rendering: str | None = None
    reply_index_hash: str | None = None
    ok: bool = True
    error: str | None = None
    #: How long the request waited for its first replica, as the service reports it in its summary.
    #: It is the one number separating "the query is slow" from "the query queued", so a client
    #: that drops it cannot size an admission ceiling. The fake carries it rather than leaving the
    #: field absent.
    queued_s: float = 0.0
    seen: list[dict[str, Any]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    @property
    def registry(self) -> Path:
        return self.root / "registry"

    @property
    def queue(self) -> Path:
        return self.root / "queues" / self.name

    def start(self) -> FakeService:
        (self.queue / "in").mkdir(parents=True, exist_ok=True)
        (self.queue / "out").mkdir(parents=True, exist_ok=True)
        publish(
            descriptor_path(self.registry, self.name),
            {
                "ready": True,
                "rendering": self.rendering,
                "index_hash": self.index_hash,
                "queue": str(self.queue),
                "heartbeat": time.time() - self.beat_age_s,
            },
        )
        if self.serve:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def kill_descriptor(self, after_s: float = 0.0) -> None:
        """Delete the descriptor (what a SIGKILLed job leaves behind), optionally on a delay."""

        def go() -> None:
            time.sleep(after_s)
            descriptor_path(self.registry, self.name).unlink(missing_ok=True)

        threading.Thread(target=go, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        qin, qout = self.queue / "in", self.queue / "out"
        while not self._stop.is_set():
            for path in sorted(p for p in qin.glob("*.json") if not p.name.startswith(".")):
                try:
                    req = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                path.unlink(missing_ok=True)
                self.seen.append(req)
                self._answer(req, qout)
            self._stop.wait(0.02)

    @property
    def searched(self) -> str:
        """The rendering the service searches, which is what ``rsvc._handle`` compares against."""
        return self.searches or self.rendering

    def _answer(self, req: dict[str, Any], qout: Path) -> None:
        rid = str(req.get("id") or uuid.uuid4().hex)
        hits = [[pid, score] for pid, score in self.hits][: int(req.get("top_k", 20))]
        # Mirrors ``rsvc.Service._handle``: a request may name the rendering it means, and a
        # mismatch is refused rather than served. Faked here because it is the server half of the
        # Knob-1 contract, which a client-side check cannot stand in for.
        asked = req.get("index", req.get("rendering"))
        ok = bool(self.ok) and (asked is None or str(asked) == self.searched)
        out = {
            "id": rid,
            "request": req,
            "rendering": self.reply_rendering or self.rendering,
            "index_hash": self.reply_index_hash or self.index_hash,
            "ok": ok,
            "runs": [
                {
                    "wall_s": 0.01,
                    "timing": {"encode": 0.001},
                    "hits": hits,
                    "fused_pool": len(self.hits),
                    "reranked": True,
                }
            ],
            "summary": {"n": 1, "median": 0.01, "queued_s": self.queued_s},
        }
        if not ok:
            if self.ok:  # refused for its lane, not because this fake was told to fail
                out["error"] = (
                    f"ValueError: this service searches {self.searched!r} (index_hash "
                    f"{self.index_hash[:12]}) but the request asked for {asked!r}, refusing to "
                    "answer from a different rendering"
                )
            else:
                out["error"] = self.error or "ValueError: refused by the fake service"
            out.pop("runs")
        tmp = qout / f".{rid}.json.tmp"
        tmp.write_text(json.dumps(out), encoding="utf-8")
        tmp.replace(qout / f"{rid}.json")


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    """A dev-shaped Layout. Nothing here reads the corpus: the store is always injected."""
    return Layout(run_dir=tmp_path / "devrun", base=tmp_path / "base")


@pytest.fixture
def service(tmp_path: Path, passage_records):
    """A live ``original`` lane whose ranked list is the fixture store's own passage ids.

    Scores descend with rank, so the order the service returned is observable in the result and a
    client that re-sorted would be caught.
    """
    ids = [str(r["passage_id"]) for r in passage_records]
    svc = FakeService(
        root=tmp_path / "rsvc",
        hits=tuple((pid, 1.0 - i / 100.0) for i, pid in enumerate(ids)),
    ).start()
    yield svc
    svc.stop()


# --------------------------------------------------------------------------- #
# `built` publishes the small fixture index and is defined in tests/retrieval/conftest.py, which
# pytest does not apply to this directory. Re-exporting it here, rather than importing it inside
# the test module, keeps the test functions free to name their parameter `built` without shadowing
# the import: the shape ruff flags as F811.
# --------------------------------------------------------------------------- #
from tests.retrieval.conftest import built  # noqa: F401
