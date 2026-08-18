"""Test-harness primitives shared by the small and full tiers.

A test has two kinds of content with very different validation costs. Harness content,
meaning spies, monkeypatches, fixture wiring and skip predicates, is correct or not
independently of data size and can be checked at fixture scale in milliseconds. Data and
scale assertions genuinely need the corpus.

Three ways a corpus-scale run can fail or go vacuously green for a defect in the test's own
machinery, all of them answerable in seconds at fixture scale:

* a spy that calls the patched attribute inside its own replacement re-resolves the name
  ``monkeypatch.setattr`` has already rebound, self-calls forever and raises
  ``RecursionError`` only after the whole rendering has loaded;
* a skip guard that probes a private entry point turns an ``AttributeError`` from a rename
  into a skip, so the integration test for a headline change never runs while the gate
  still reports success;
* a test that requests a fixture defined in a conftest pytest does not apply to its
  directory errors at setup, which looks red while executing nothing.

:func:`spy_through` exists so the first of those becomes unwritable: the original is
captured by construction, before the patch, and the caller is never handed a name it could
re-resolve.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Spy", "spy_through"]


@dataclass
class Spy:
    """The record of one :func:`spy_through` patch.

    ``real`` is the function as it was before the patch, captured eagerly, so nothing in the
    replacement's body ever needs to name the patched attribute.
    """

    target: Any
    name: str
    real: Callable[..., Any]
    spy: Callable[..., Any] | None = None
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.calls)

    @property
    def count(self) -> int:
        return len(self.calls)

    @property
    def args(self) -> list[tuple[Any, ...]]:
        """Positional arguments of each recorded call, in call order."""
        return [a for a, _ in self.calls]


def spy_through(
    monkeypatch: Any,
    target: Any,
    name: str,
    *,
    before: Callable[..., Any] | None = None,
    after: Callable[..., Any] | None = None,
) -> Spy:
    """Replace ``target.name`` with a spy that delegates to the original exactly once.

    What this buys over a hand-written spy is that the self-call is not expressible: ``real``
    is read off ``target`` here, before ``monkeypatch.setattr`` runs, and the replacement
    closes over that value. A caller who only supplies ``before`` and ``after`` hooks never
    gets the chance to write ``target.name(...)`` inside the replacement.

    ``before(*args, **kwargs)`` runs before delegating; ``after(result, *args, **kwargs)``
    runs after, and its return value is ignored, so the spy always returns what the real
    function returned and patching cannot change the answer.
    """
    real = getattr(target, name)

    @functools.wraps(real)
    def spy(*args: Any, **kwargs: Any) -> Any:
        spy_record.calls.append((args, kwargs))
        if before is not None:
            before(*args, **kwargs)
        result = real(*args, **kwargs)
        if after is not None:
            after(result, *args, **kwargs)
        return result

    spy_record = Spy(target=target, name=name, real=real, spy=spy)
    monkeypatch.setattr(target, name, spy)
    return spy_record
