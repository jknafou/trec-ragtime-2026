"""The two idempotent SLURM shapes, kept separate.

``launcher`` is the submission side: it builds the ``sbatch`` argv, wires
``afterok``/``afterany`` and can add a GPU OR-constraint. ``workqueue`` is the dynamic
self-claiming queue the submitted tasks then run against: atomic-rename claim, heartbeat
and reaper, validate before done, poison shards to ``failed/``, USR1 requeue. Neither
substitutes for the other; the templates carry only bring-up plus ``run --stage``.

Every deployed submission is an array of interchangeable workers pulling from that queue,
which is why ``cli`` passes ``gpu=False`` throughout and lets each template state its own
resources.
"""

from __future__ import annotations

from . import launcher, workqueue

__all__ = ["launcher", "workqueue"]
