"""What `driver.drive` must do before it writes `_SUCCESS`, and why the order is the correctness.

`_SUCCESS` is the resume witness: the artifact tree is the checkpoint, so a re-launch skips any
cell that carries it. Everything a finished cell owes must therefore be on disk before that marker
lands. Two things owe it, for the same structural reason and with different consequences:

* `flush_stats`, the monitoring counters. `Statistics` is in-memory and instance-scoped, so
  unflushed counters die with the SLURM array task. This one is unrecoverable: the submission
  files are unaffected, so the run looks successful while the behavioural measurements are gone,
  and no later pass can reconstruct them.
* `score_citations`, the citation scorer. The claim-importance judgement is made from the nugget's
  question and the claim sentence, with no passage shown. It supplies the `references` values,
  `nugget_importance x claim_importance`, which are the priority signal an assessor sees.
  Recoverable by re-running the scorer, but only if someone notices: a cell marked done with no
  scores serialises into a valid file whose ranking column is all 0.0.

These are asserted over the parsed source rather than by running a cell, because the property is
statement order inside a function that needs a GPU pair, a live vLLM and a retrieval service to
execute. An ordering bug is invisible to any test that only checks the artifacts of a successful
run: both orders produce the same files when nothing crashes. What separates them is what
survives a crash, a preemption or a resume in between.

Not covered here:
  - that `score_citations` produces correct scores (its own tests own that)
  - that a resume actually skips a `_SUCCESS` cell (orchestration's `already_done`)
"""

from __future__ import annotations

import ast
import inspect

import pytest

from ragtime.pipeline import driver

_TREE = ast.parse(inspect.getsource(driver))


def _fn(name: str) -> ast.AST:
    """The def of ``name``. Ordering claims are scoped to one function, never to the file.

    A file-wide line comparison is a layout test, not an ordering test, and it fails for the
    wrong reason as soon as a helper is defined below its caller: the helper's body then reads as
    if its calls happened at the point of definition rather than at the point of use.
    """
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"driver has no function named {name!r}")


def _call_lines(name: str, *, within: str = "drive") -> list[int]:
    """Line numbers of every call to ``name`` inside ``within``, by simple or attribute name."""
    out = []
    for node in ast.walk(_fn(within)):
        if isinstance(node, ast.Call):
            fn = node.func
            called = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if called == name:
                out.append(node.lineno)
    return sorted(out)


@pytest.mark.small
def test_the_counters_are_flushed_BEFORE_the_success_marker() -> None:
    """The other order lets a resume skip a cell whose counters were never written."""
    flush = _call_lines("flush_stats")
    mark = _call_lines("_mark_complete")
    assert flush, "driver must flush its counters: they die with the process otherwise"
    assert mark, "driver must write the completion marker"
    assert max(flush) < min(mark), (
        f"flush_stats at {flush} must precede _mark_complete at {mark}: `_SUCCESS` is the resume "
        "witness, so a cell that reads as done must already carry its counters on disk"
    )


@pytest.mark.small
def test_citations_are_scored_BEFORE_the_success_marker() -> None:
    """A cell marked done with no scores ships its citations tied at 0.0, and says nothing.

    `load.citation_scores` returns `None` when the scorer never ran, and select-and-serialize
    does not collapse that into `{}`, so the loss stays detectable one step from the file.
    """
    scored = _call_lines("score_citations")
    mark = _call_lines("_mark_complete")
    assert scored, (
        "driver must call score_citations: the scorer was once complete but reachable only from a "
        "standalone entry point, so a cell could finish with every citation tied at 0.0 and still "
        "read as done. Calling it inline from `drive` is what closed that, and this pins it"
    )
    assert max(scored) < min(mark), (
        f"score_citations at {scored} must precede _mark_complete at {mark}"
    )


@pytest.mark.small
def test_the_scorer_runs_AFTER_the_loops_not_before() -> None:
    """The scorer is a post-hoc pass: it reads the finished bank and the loop records, and never
    gates a loop. Calling it before the rounds would score an empty bank and cost a generation
    per answer that does not exist yet."""
    scored = min(_call_lines("score_citations"))
    rounds = _call_lines("_drive_e2e")  # `drive`'s own call into the coverage loop
    assert rounds, "drive must run the e2e coverage loop"
    assert scored > max(rounds), (
        f"score_citations at {scored} must come after _drive_e2e at {rounds}: it scores what "
        "the rounds produce"
    )


@pytest.mark.small
def test_a_scorer_failure_CANNOT_discard_a_finished_cell() -> None:
    """A ranking pass may not destroy a finished cell's work, which is minutes of a GPU pair.

    The scorer runs after the loops and gates nothing, so an exception must degrade the cell to
    unranked citations rather than abort it. Asserted structurally: the call sits inside a `try`
    whose handler does not re-raise.
    """
    tree = ast.parse(inspect.getsource(driver))
    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        names = {
            getattr(c.func, "id", None) or getattr(c.func, "attr", None)
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
        }
        if "score_citations" not in names:
            continue
        for handler in node.handlers:
            if any(isinstance(n, ast.Raise) for n in ast.walk(handler)):
                pytest.fail("the scorer's handler re-raises; a finished cell would be lost")
        guarded = bool(node.handlers)
    assert guarded, "score_citations must be wrapped in try/except so a ranking pass cannot abort a cell"
