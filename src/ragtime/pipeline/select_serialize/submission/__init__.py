"""Submission ownership: the validator hard gate, the T2 self-check, the manifest, the writer.

Producing a valid, submittable output is this stage's responsibility; there is no separate
submission module elsewhere in the tree.

This package re-exports nothing. Import from the module that owns the name --
``from .submission.validate import validate``, ``from .submission.envelope import envelopes`` --
which is what every caller here already does. A re-export would shadow the submodule it came from:
``validate`` is both a module and the function that module exports, so binding the function on the
package turns ``from .submission import validate`` into the function and breaks the next attribute
access on it.
"""

from __future__ import annotations
