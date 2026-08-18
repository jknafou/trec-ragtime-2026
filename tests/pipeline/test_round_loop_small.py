"""Fast tests for the round fan: no vLLM, no index, no GPU.

Everything about the fan that is size-independent is answered here in seconds, so the expensive
tier only has to answer what needs a real service.

The four properties pinned here are the ones whose failure is invisible until something expensive
has already been paid for:

* a raising loop must not cancel its siblings: the failure mode costs k-1 multi-minute loops;
* the ceiling must come from the caller, not a literal: a phantom knob reads as configured;
* the record must carry both translation knobs: a record that cannot name its arm is
  indistinguishable from the same nugget answered in a different arm, and the three renderings
  share passage ids so nothing downstream would notice;
* `open_nuggets` must be a real predicate, not a pass-through.
"""

from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path
from typing import Any

import pytest

from ragtime.common import Layout, Statistics
from ragtime.pipeline import records, round_loop

pytestmark = pytest.mark.small


def _nugget(nid: str, question: str = "who?", status: str = "open") -> dict[str, Any]:
    return {"nugget_id": nid, "question": question, "status": status}


def _cfg(index: str = "original", passage_lang: str = "omt") -> Any:
    """A cfg-shaped object carrying only what the code under test reads."""
    return types.SimpleNamespace(
        blocks={"retrieval": {"index": index}},
        retrieval_index=index,
        passage_lang=passage_lang,
    )


# --------------------------------------------------------------------------- #
# 1. open_nuggets is a predicate, not a pass-through.
# --------------------------------------------------------------------------- #
def test_open_nuggets_selects_open_and_preserves_bank_order() -> None:
    bank = [
        _nugget("n1", status="open"),
        _nugget("n2", status="closed"),
        _nugget("n3", status="open"),
        _nugget("n4", status="answered"),
    ]
    got = round_loop.open_nuggets(bank)
    assert [n["nugget_id"] for n in got] == ["n1", "n3"], "order must be BANK order, not re-sorted"


def test_open_nuggets_can_say_no_to_everything() -> None:
    """The negative half: without it the test above passes against a function returning `bank`."""
    assert round_loop.open_nuggets([_nugget("n1", status="closed")]) == ()


# --------------------------------------------------------------------------- #
# 2. One raising loop must not cancel its siblings: the property of the wrap.
# --------------------------------------------------------------------------- #
def test_a_raising_loop_becomes_a_record_and_the_siblings_still_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`fanout.map` gathers without return_exceptions, so an unguarded raise kills the round.

    This is why `_guarded` exists. If the wrap is removed, this test fails rather than a
    production round losing k-1 loops to one bad nugget.
    """
    started: list[str] = []

    async def fake_run_loop(nugget, cfg, **kw):
        nid = nugget["nugget_id"]
        started.append(nid)
        await asyncio.sleep(0.01)
        if nid == "boom":
            raise RuntimeError("this loop exploded")
        return types.SimpleNamespace(
            nugget_id=nid, question="q", status="answered", answers=(),
            search_trail=[], turns=1, searches=1, claims_committed=0,
            claims_rejected=0, contested=False, closed_by="model",
        )

    monkeypatch.setattr(round_loop, "run_loop", fake_run_loop)
    bank = [_nugget("a"), _nugget("boom"), _nugget("c")]
    res = asyncio.run(
        round_loop.run_round(bank, _cfg(), llm=object(), ctx=object(), ceiling=3, stats=Statistics())
    )

    assert len(res.results) == 3, "a raising loop must appear as a RECORD, never as a hole"
    assert sorted(started) == ["a", "boom", "c"], "every sibling must still have run"
    statuses = {r.nugget_id: r.status for r in res.results}
    assert statuses == {"a": "answered", "boom": "error", "c": "answered"}
    assert res.errors and "boom" in res.errors[0] and "RuntimeError" in res.errors[0]


# --------------------------------------------------------------------------- #
# 3. The ceiling comes from the caller, and it is really enforced.
# --------------------------------------------------------------------------- #
def test_the_ceiling_actually_bounds_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tripwire that can fail: it observes peak in-flight, not just the returned count."""
    in_flight = 0
    peak = 0

    async def fake_run_loop(nugget, cfg, **kw):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return types.SimpleNamespace(
            nugget_id=nugget["nugget_id"], question="q", status="answered", answers=(),
            search_trail=[], turns=1, searches=0, claims_committed=0,
            claims_rejected=0, contested=False, closed_by="model",
        )

    monkeypatch.setattr(round_loop, "run_loop", fake_run_loop)
    bank = [_nugget(f"n{i}") for i in range(9)]
    res = asyncio.run(
        round_loop.run_round(bank, _cfg(), llm=object(), ctx=object(), ceiling=3)
    )
    assert len(res.results) == 9
    assert peak <= 3, f"ceiling=3 was not enforced; peak in-flight was {peak}"
    assert peak > 1, "nothing ran concurrently: the fan is not fanning"


def test_speedup_reports_achieved_concurrency_not_the_ceiling() -> None:
    """`speedup` reports achieved concurrency: it must divide the walls, not echo the ceiling."""
    r = round_loop.RoundResult(
        round_no=0, results=(), wall_s=10.0, sequential_wall_s=40.0
    )
    assert r.speedup == pytest.approx(4.0)
    assert round_loop.RoundResult(round_no=0, results=(), wall_s=0.0, sequential_wall_s=5.0).speedup == 0.0


# --------------------------------------------------------------------------- #
# 4. The record carries both translation knobs.
# --------------------------------------------------------------------------- #
def test_the_loop_record_names_the_arm_it_was_produced_in(tmp_path: Path) -> None:
    """Both translation knobs, from cfg. The three renderings share passage ids, so a record
    that cannot name its arm is indistinguishable from a different arm's record."""
    result = types.SimpleNamespace(
        nugget_id="n1", question="who?", status="answered", closed_by="model",
        answers=({"value": "42", "quoted_span": "the answer is 42"},),
        turns=3, searches=2, claims_committed=1, claims_rejected=0, contested=False,
    )
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path / "base")
    path = records.write_loop_record(
        layout, result, _cfg(index="original", passage_lang="omt_opus"),
        topic_id="2061", seed=3,
    )
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["retrieval_index"] == "original"      # which index was searched
    assert rec["passage_lang"] == "omt_opus"         # which rendering was read
    assert rec["topic_id"] == "2061" and rec["seed"] == 3
    assert rec["answers"][0]["quoted_span"] == "the answer is 42", "MAT-6: the span must survive"


def test_the_record_is_written_atomically_leaving_no_temp_behind(tmp_path: Path) -> None:
    result = types.SimpleNamespace(
        nugget_id="n9", question="q", status="answered", closed_by="model",
        answers=(), turns=1, searches=1, claims_committed=0, claims_rejected=0, contested=False,
    )
    layout = Layout(run_dir=tmp_path / "run", base=tmp_path / "base")
    path = records.write_loop_record(layout, result, _cfg(), topic_id="t", seed=0)
    leftovers = [p.name for p in path.parent.iterdir() if p.name.startswith(".")]
    assert leftovers == [], f"atomic write left a temp file behind: {leftovers}"
