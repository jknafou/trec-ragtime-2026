"""SaT sentence and span segmenter (``wtpsplit``) behind one interface.

``split(text) -> list[str]`` is multilingual and runs on CPU. It is used for chunking
and for span grounding, with the model id taken from config and the heavy import kept
lazy.

``split_spans(text) -> list[tuple[int, int]]`` returns the same sentences as
:meth:`Segmenter.split` but as ``(start, end)`` code-point offsets into ``text``
rather than as copies. That is the primitive the corpus data model is built on:
sentence text is stored exactly once, in its document, and every sentence is a span
of it.

The backend is the torch-free onnxruntime CPU path
(``ort_providers=["CPUExecutionProvider"]``). SaT is small and deterministic, so
onnx-CPU is a good default for every consumer and lets the light ``chunk`` extra stay
torch-free and cross-platform. ``ort_providers`` is overridable.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ragtime.common.logging import get_logger

__all__ = ["Segmenter", "SpanMismatchError", "spans_of"]

_log = get_logger(__name__)


class SpanMismatchError(RuntimeError):
    """Raised when a segment is not a verbatim span of its input.

    The corpus data model stores sentence text exactly once, in its document, and
    addresses every sentence as ``document.text[start:end]``. A segment that is not a
    substring of its input at the walked cursor would make that address wrong, and
    there is nothing sensible to fall back to. Callers count the event as
    ``chunk.sentence_span_mismatch`` and fail the shard.
    """

_DEFAULT_ORT_PROVIDERS = ("CPUExecutionProvider",)
# Retry budget for the transient import failures described in `Segmenter._backend`:
# four attempts with 0.5 s doubling, about 7.5 s worst case.
_IMPORT_ATTEMPTS = 4
_IMPORT_BACKOFF_S = 0.5


def spans_of(text: str, segs: Any) -> list[tuple[int, int]]:
    """Walk ``segs``, which must concatenate back to ``text``, into spans of ``text``.

    Each raw segment occupies ``text[pos:pos + len(raw)]``, so its span is that window
    trimmed inward past leading and trailing whitespace, which is exactly what
    :meth:`Segmenter._clean` returns. Whitespace-only segments are dropped the same
    way, so ``[text[a:b] for a, b in spans_of(text, segs)]`` equals
    ``Segmenter._clean(segs)`` byte for byte, and the whitespace joining two sentences
    stays outside both spans.

    Two checks apply, with no fallback: every span must slice back to its own stripped
    segment, and the cursor must land exactly on ``len(text)``. A ``find``-based
    recovery path is not offered, because it would silently mis-address a repeated
    sentence.

    Exposed at module level so a test double can produce the same span contract
    without owning a model.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    for raw in segs:
        start = pos
        pos += len(raw)
        seg = raw.strip()
        if not seg:  # whitespace-only or empty, dropped exactly as `_clean` drops it
            continue
        a = start + (len(raw) - len(raw.lstrip()))
        b = a + len(seg)
        if text[a:b] != seg:
            raise SpanMismatchError(
                f"segment is not a verbatim span of its input at [{a}, {b}): "
                f"{text[a:b]!r} != {seg!r}"
            )
        spans.append((a, b))
    if pos != len(text):
        raise SpanMismatchError(
            f"segments do not reproduce their input: walked {pos} of {len(text)} chars"
        )
    return spans


def _intra_op_threads() -> int:
    """CPU threads for onnxruntime: ``SLURM_CPUS_PER_TASK`` when set, else ``os.cpu_count()``.

    onnxruntime does not reliably auto-detect a SLURM cpuset, so the allocation is read
    explicitly.
    """
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v and v.isdigit() and int(v) > 0:
        return int(v)
    return max(1, os.cpu_count() or 1)


class Segmenter:
    """One resident SaT segmenter on the torch-free onnx-CPU backend.

    ``split`` handles a single string; ``split_batch`` segments a list of texts with
    byte-identical per-text boundaries. It runs per text, since multi-text
    SaT calls are slower on this backend (see the ``split_batch`` docstring). The chunk
    stage gets its throughput from its process pool, with each worker holding one
    segmenter.
    """

    __slots__ = ("_ort_providers", "_sat", "_threads", "model")

    def __init__(
        self,
        model: str = "sat-3l-sm",
        *,
        ort_providers: list[str] | tuple[str, ...] | None = None,
        intra_op_threads: int | None = None,
    ) -> None:
        self.model = model
        self._ort_providers = list(ort_providers or _DEFAULT_ORT_PROVIDERS)
        # None means auto-detect the SLURM or CPU allocation at bring-up.
        self._threads = intra_op_threads
        self._sat: Any = None

    def _backend(self) -> Any:
        if self._sat is None:
            # Retry both the import and the model build. Under a wide fan-out every pool
            # child resolves these module files off one shared network venv at the same
            # time, and the filesystem can return a spurious ENOENT for files that exist.
            # The condition is transient and a short backoff clears it.
            #
            # Retrying matters because an import error raised in a pool child can wedge
            # `multiprocessing.Pool.imap`, while the heartbeat thread keeps stamping, so
            # the reaper never reclaims the shard.
            #
            # Importing in the parent before fork is not an option: the parent is kept
            # model-free so that `fork` stays safe, since onnxruntime's native intra-op
            # pool does not survive a fork.
            last: Exception | None = None
            for attempt in range(_IMPORT_ATTEMPTS):
                try:
                    import onnxruntime as ort
                    from wtpsplit import SaT

                    so = ort.SessionOptions()
                    so.intra_op_num_threads = self._threads or _intra_op_threads()
                    self._sat = SaT(
                        self.model,
                        ort_providers=self._ort_providers,
                        ort_kwargs={"sess_options": so},
                    )
                    break
                # A missing package never appears on a later attempt, so retrying would
                # only bury the real message; the transient case raises
                # FileNotFoundError or OSError instead.
                except ModuleNotFoundError:
                    raise
                except (FileNotFoundError, OSError, ImportError) as exc:
                    last = exc
                    if attempt == _IMPORT_ATTEMPTS - 1:
                        raise
                    _log.warning(
                        "serving.segmenter.import_retry",
                        attempt=attempt + 1,
                        of=_IMPORT_ATTEMPTS,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    time.sleep(_IMPORT_BACKOFF_S * (2**attempt))
            if self._sat is None and last is not None:  # pragma: no cover - defensive
                raise last
        return self._sat

    @staticmethod
    def _clean(segs: Any) -> list[str]:
        return [s.strip() for s in segs if s and s.strip()]

    def split(self, text: str) -> list[str]:
        """Segment ``text`` into non-empty sentence/span strings (single-text path)."""
        return self._clean(self._backend().split(text))

    def split_batch(self, texts: list[str]) -> list[list[str]]:
        """Segment a list of texts into one sentence list per text.

        Implemented as a per-text loop over the one resident backend, so it is
        byte-identical to :meth:`split` by construction. A multi-text
        ``SaT.split(list)`` call is avoided because on the onnxruntime CPU backend
        wtpsplit pads every text in a call up to the call's maximum token length, so
        mixed-length batches spend most of their time on padding and the forward cost
        rises sharply for wide input tensors. Batch throughput comes from the chunk
        stage's process pool rather than from tensor batching.
        """
        backend = self._backend()
        return [self._clean(backend.split(t)) for t in texts]

    def split_spans(self, text: str) -> list[tuple[int, int]]:
        """Segment ``text`` into ``(start, end)`` code-point spans of ``text`` itself.

        Same sentences, order and drops as :meth:`split`; only the representation
        differs, so ``[text[a:b] for a, b in split_spans(t)] == split(t)``.

        Raises :class:`SpanMismatchError` if the model's raw segments do not concatenate
        back to ``text``; see :func:`spans_of`, which has no fallback path.
        """
        return spans_of(text, self._backend().split(text))

    def split_spans_batch(self, texts: list[str]) -> list[list[tuple[int, int]]]:
        """Span form of :meth:`split_batch`: one span list per text, same boundaries."""
        backend = self._backend()
        return [spans_of(t, backend.split(t)) for t in texts]
