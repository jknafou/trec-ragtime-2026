"""wtpsplit import smoke under an override, and lazy-import discipline.

``wtpsplit`` has to import cleanly under the project's pinned ``transformers`` override,
and importing ``preprocess.chunk`` must not pull it in: the SaT backend is loaded on first
use of a ``Segmenter``, which keeps the rest of the chunk tests dependency-free. Skipped
where the ``chunk`` extra is not installed.
"""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.small


def test_wtpsplit_imports_under_override() -> None:
    pytest.importorskip("wtpsplit")
    from wtpsplit import SaT  # noqa: F401, the import itself is the assertion


def test_importing_chunk_does_not_import_wtpsplit() -> None:
    """Importing the chunk module does not drag in the SaT backend."""
    sys.modules.pop("wtpsplit", None)
    import ragtime.preprocess.chunk  # noqa: F401

    assert "wtpsplit" not in sys.modules  # Segmenter lazy-imports it only on first split()
