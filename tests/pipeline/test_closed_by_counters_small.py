"""Every way a loop can end has one decided counter outcome: a census, not a sample.

`run_loop` once emitted
``STAT_CLOSED_BY_BUDGET if closed_by == CLOSED_BUDGET else STAT_CLOSED_BY_MODEL``, an if/else
across four terminal reasons. No count was lost outright, since `STAT_GENERATION_TIMEOUT` and
`STAT_MALFORMED_ACTION` are emitted at their own sites, so a serving fault stayed countable. The
damage was to `closed_by_model`, the one rate meant to say that the loop decided it was done: it
absorbed every timeout and every malformed action, so a regression in either would have moved the
apparent self-closure rate.

The defence is a census over the module's own `CLOSED_*` constants rather than one test per
reason. A per-reason test only covers the reasons someone remembered to write; a census goes red
for a reason that does not exist yet.

Not covered here:
  - that the counters are flushed (the driver's ordering tests own that)
  - the numeric rates themselves (the monitoring rollup)
"""

from __future__ import annotations

import pytest

from ragtime.pipeline.rag_loop import loop as loop_mod
from ragtime.pipeline.rag_loop import stats as S


def _reason_constants() -> dict[str, str]:
    """Every ``CLOSED_*`` terminal-reason constant the loop module defines."""
    return {
        name: value
        for name, value in vars(loop_mod).items()
        if name.startswith("CLOSED_") and isinstance(value, str)
    }


@pytest.mark.small
def test_every_terminal_reason_has_a_counter_decision() -> None:
    """A new terminal reason with no entry fails here rather than being mis-attributed silently.

    `CLOSED_STAT` maps a reason to `None` when it is counted elsewhere. That mapping is a
    deliberate entry, which is what makes the map exhaustive rather than merely populated; a
    reason must never fall through to a default.
    """
    reasons = _reason_constants()
    assert reasons, "no CLOSED_* constants found: the census is scanning the wrong module"
    missing = sorted(v for v in reasons.values() if v not in loop_mod.CLOSED_STAT)
    assert missing == [], (
        f"terminal reason(s) {missing} have no entry in CLOSED_STAT. Add one: map to a counter "
        "id, or to None if the reason is already counted at its own emit site. Do not let it "
        "fall through: that is the bug this test exists for."
    )


@pytest.mark.small
def test_the_map_names_only_real_counter_ids() -> None:
    """A mistyped id emits a counter nothing rolls up, and stays invisible until analysis time."""
    # `S.__all__` holds the constant names (`STAT_CLOSED_BY_MODEL`); the map holds their values
    # (`rag_loop.closed_by_model`). The comparison has to be against the values, or the test
    # fails for the wrong reason.
    declared = {getattr(S, name) for name in S.__all__ if name.startswith("STAT_")}
    assert declared, "no STAT_* ids found: this check would pass vacuously"
    for reason, stat in loop_mod.CLOSED_STAT.items():
        if stat is None:
            continue
        assert stat in declared, (
            f"{reason!r} maps to {stat!r}, which is not any STAT_* id stats.py declares: a "
            "typo here emits a counter nothing rolls up"
        )


@pytest.mark.small
def test_a_serving_timeout_is_NOT_counted_as_the_model_closing_itself() -> None:
    """A timeout is a serving fault, not the loop spending its allowance.

    `CLOSED_TIMEOUT` is recorded apart from `budget` for that reason: conflating them would hide
    a degenerate-generation regression inside a normal-looking backstop rate. The same argument
    applies to `closed_by_model`, and that is the conflation this pins down.
    """
    assert loop_mod.CLOSED_STAT[loop_mod.CLOSED_TIMEOUT] is None
    assert loop_mod.CLOSED_STAT[loop_mod.CLOSED_MALFORMED] is None
    assert loop_mod.CLOSED_STAT[loop_mod.CLOSED_MODEL] == S.STAT_CLOSED_BY_MODEL
    assert loop_mod.CLOSED_STAT[loop_mod.CLOSED_BUDGET] == S.STAT_CLOSED_BY_BUDGET


@pytest.mark.small
def test_the_two_fault_reasons_are_still_counted_SOMEWHERE() -> None:
    """Mapping a reason to `None` is only defensible because another site counts it.

    If those emits were removed, `None` would stop meaning "counted elsewhere" and start meaning
    "not counted at all", with the map looking exactly the same.
    """
    import inspect

    src = inspect.getsource(loop_mod)
    assert "STAT_GENERATION_TIMEOUT" in src, (
        "CLOSED_TIMEOUT maps to None because STAT_GENERATION_TIMEOUT is emitted at the timeout "
        "site; that emit is gone, so the reason is now counted nowhere"
    )
    assert "STAT_MALFORMED_ACTION" in src, (
        "CLOSED_MALFORMED maps to None because STAT_MALFORMED_ACTION is emitted at the malformed "
        "site; that emit is gone, so the reason is now counted nowhere"
    )
