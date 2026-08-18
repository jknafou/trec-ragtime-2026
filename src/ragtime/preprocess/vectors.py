"""The on-disk vector block format: the contract between ``vectorize`` and ``assemble``.

Index building is split into two ``saturate`` stages. vectorize encodes passages into this format
and nothing else; assemble reads this format and builds the FAISS / Seismic / PLAID artifacts
``preprocess.index``'s writers already know how to build. This module is the seam: the writer, the
reader, the census and the block/part arithmetic. It builds no index, opens no engine, loads no
model and resolves no path above the cell directory it is handed.

Three units, never conflated:

===============  ========================  ================================================
unit             size                      what it is
===============  ========================  ================================================
encode block     ``encode_block_passages``  the unit of a vector file and of a ``_SUCCESS``,
                 (hashed)                   and the atom of this module.
vectorize task   an ``execution`` leaf,     a run of consecutive blocks claimed by one
                 not hashed                 worker. Invisible here: retuning parallelism
                                            must not move a recipe hash.
assemble part    ``plaid_part_passages``    one FAISS, Seismic or PLAID artifact, and the
                 (hashed)                   unit :func:`stream_part` reads.
===============  ========================  ================================================

Blocks divide parts evenly, and :class:`BlockPlan` refuses to exist otherwise
(:class:`BlockGeometryError`). Otherwise a part's blocks would not be a contiguous run of whole
blocks, the sequential read below would become a partial-block gather, and the two hashed knobs
would silently interact.

Assembling one part means reading ``part_passages / block_passages`` consecutive block files, in
block order, each exactly once, and concatenating them. That is what makes 1.68 TB of intermediate
I/O cost ~20-30 min aggregate; a random gather over the same bytes would be ~17x read
amplification and would dominate the build. The read API is shaped around it:
:func:`stream_part` is the only entrypoint taking a part index, and it yields blocks in ascending
order, opening each file once, asserting that block *i*'s ordinal range begins exactly where block
*i-1*'s ended. :func:`read_block_dir` takes a *directory*, consults no census and reports
block-local ordinals; it is the primitive :func:`stream_part` is built from, not a random-access
API. Nothing takes a block index.

Blocks of one cell are claimed by different workers, so no single writer knows the whole cell.
Each block writes its own ``block.json`` witness as the last act of its work, recording the plan it
was built under, and :func:`read_cell_census` reconstructs the cell from those witnesses: a witness
for every ``block-*`` directory on disk, unanimity on the plan, each witness sitting in the
directory its own index names, indices covering exactly ``0..N-1``, and only the last block short.
Anything else raises :class:`VectorIntegrityError`. A fan over 15 of 16 blocks would otherwise
return a full-looking, silently truncated index. Same mechanism and same failure semantics as
``preprocess.index``'s ``read_shard_parts``, one level up.

``block.json`` carries no timestamp: a rebuilt block must be byte-identical or the artifact tree
stops being a checkpoint, and the full gate checks exactly that.

The per-leg formats:

* dense, raw ``vectors.npy``, ``(n, 1024)``, fp32. Not fp16: the leg is 98 GB either way (the
  FAISS index, not this intermediate, is the storage cost that matters), and fp16 would inject
  ~1e-3 relative error into an exact ``IndexFlatIP``, under the 1e-2 tolerance of
  ``_FaissWriter``'s unit-norm assertion.
* sparse, ``vectors.parquet`` (+zstd, the pinned :mod:`ragtime.common.io` options), storing
  :func:`ragtime.serving.sparse_milco.seismic_components`' own output, so ``assemble`` re-derives
  nothing and the components cannot be ordered differently on the two sides.
* late-interaction, a raw flat ``vectors.bin`` of ``float16[total_vectors, 128]`` plus a
  ``counts.parquet`` sidecar (``passage_id``, ``n_vectors``); per-passage lists are reconstructed
  by cumulative offsets. Not Arrow nested lists: 6.03e9 leaf vectors would carry 24-48 GB of
  offset buffers. Not compressed: fp16 embeddings are near-incompressible and decompressing 1.5 TB
  costs real CPU on the read side.

The store dtype is a parameter rather than a literal (:func:`format_for`'s ``dtype=``): every
block records its dtype in its witness and the reader reads it back off that witness, so a change
of default leaves already-written blocks readable. Endianness is pinned little (``<f4``/``<f2``).

What this module does not do: construct paths above ``cell_dir`` (``Layout`` owns the path scheme,
and the ``block-00000`` level is this format's own business); read config
(``encode_block_passages``/``plaid_part_passages`` are hashed leaves the adapters resolve and pass
in, defaulted here to nothing); apply unit-norm, truncation or id-set policy (``preprocess.index``
owns those and still runs them on the assemble side); or declare a second id-map schema, the
``idmap`` schema being ``preprocess.index.idmap_arrow_schema`` verbatim (:func:`idmap_schema`).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

from ragtime.common import get_logger
from ragtime.common.io import is_done, read_jsonl, read_parquet, success_marker, write_jsonl
from ragtime.common.io import write_parquet_stream as _write_parquet_stream
from ragtime.serving.sparse_milco import seismic_components

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator, Mapping, Sequence

__all__ = [
    "BLOCK_FILENAME",
    "BLOCK_GLOB",
    "COUNTS_FILENAME",
    "DEFAULT_DENSE_DTYPE",
    "DEFAULT_LATE_INTERACTION_DTYPE",
    "DEFAULT_SPARSE_DTYPE",
    "STORE_DTYPES",
    "BlockGeometryError",
    "BlockPlan",
    "CellCensus",
    "DenseVectors",
    "LateInteractionVectors",
    "PassageRef",
    "SparseVectors",
    "VectorBlock",
    "VectorFormat",
    "VectorIntegrityError",
    "block_dir_name",
    "format_for",
    "idmap_filename",
    "idmap_schema",
    "read_block_dir",
    "read_cell_census",
    "stream_part",
    "write_block",
]

_log = get_logger("preprocess.vectors")

#: The stem every block directory is named from: a cell holds ``block-00000``,
#: ``block-00001``, ...: never a bare directory, not even when there is exactly one block.
#: Same rule (and same reason) as ``Layout``'s ``part-00000`` and ``preprocess.index``'s
#: ``plaid-00000``: a reader that had to handle "one block is named differently" would have a
#: code path in which a missing block looks like the normal case.
_BLOCK_PREFIX = "block"
#: What a reader globs to find the blocks on disk, for the census cross-check. Exported so
#: the pattern and the name are the same literal, never two that agree by luck.
BLOCK_GLOB = f"{_BLOCK_PREFIX}-*"

#: The census witness, written by each block into its own directory as the last act of its
#: work, and published by the same atomic rename as the vectors beside it. Its presence is
#: what distinguishes a complete block from a crashed one.
BLOCK_FILENAME = "block.json"
#: The late-interaction sidecar: ``passage_id`` -> ``n_vectors``, the cumulative offsets that
#: cut the flat ``vectors.bin`` back into per-passage matrices.
COUNTS_FILENAME = "counts.parquet"

_VECTORS_STEM = "vectors"
_DENSE_SUFFIX = ".npy"
_SPARSE_SUFFIX = ".parquet"
_LATE_INTERACTION_SUFFIX = ".bin"

#: ``census dtype name -> the pinned little-endian numpy dtype string``. Endianness is pinned
#: rather than left native: this is a raw byte file read back by a different process on a
#: different node, so the bytes must be defined by the format, not by the writer's CPU.
STORE_DTYPES: dict[str, str] = {"float32": "<f4", "float16": "<f2"}

#: fp32, and not a preference: see the module docstring: the leg costs 98 GB either way and
#: fp16's ~1e-3 relative error sits under the 1e-2 tolerance of the unit-norm assertion that
#: would otherwise have caught it, against an index that is exact by construction.
DEFAULT_DENSE_DTYPE = "float32"
#: Seismic keys on ``float32`` values; there is no second dtype to reach for here.
DEFAULT_SPARSE_DTYPE = "float32"
#: fp16: halves 1.5 TB of intermediate and the per-part host buffer. The default only: the
#: fast-plaid acceptance spike may move it, which is why every block records what it used and
#: the reader believes the record rather than this constant.
DEFAULT_LATE_INTERACTION_DTYPE = "float16"


class BlockGeometryError(ValueError):
    """A block/part plan that cannot be built from, caught at construction.

    Distinct from :class:`VectorIntegrityError`: this one is a caller's arithmetic
    (a part size that is not a whole number of blocks, a non-positive size, a block index
    outside the plan) and is raised before anything is read or written; that one is the
    artifact tree disagreeing with itself, discovered on read.
    """


class VectorIntegrityError(RuntimeError):
    """The vector blocks on disk are not a cell that may be assembled.

    Never recovered from, and never downgraded to "assemble what we found": a part assembled
    from 15 of its 16 blocks is a full-looking index that is silently short by 8,192 passages,
    and nothing downstream, not the id maps (each is correct for its own block), not a score,
    not the manifest, could tell. The same semantics as ``preprocess.index``'s
    :class:`~ragtime.preprocess.index.IndexIntegrityError` one level up, for the same reason.
    """


def block_dir_name(block: int) -> str:
    """``block-00007``: One encode block's directory inside a cell directory.

    Zero-padded so the on-disk (lexicographic) order is the block order, which is what lets
    the census cross-check compare ``sorted(dirs)`` against the recorded names directly. Five
    digits covers 99,999 blocks = 819 M passages at the shipped block size.
    """
    return f"{_BLOCK_PREFIX}-{int(block):05d}"


def idmap_schema() -> Any:
    """``preprocess.index.idmap_arrow_schema()`` verbatim: never a second declaration.

    A block's ``idmap.parquet`` is the same four columns a published leg's is (``ordinal`` /
    ``passage_id`` / ``document_id`` / ``source_lang``), so it is the same schema object, from
    the same owner. That schema has no text column, which is what makes "no text leaks out of
    the vector store" structural rather than a discipline, at this level too.

    ``ordinal`` here is block-local (``0..n-1``, the row's position in its own block);
    part-local and cell-local ordinals are arithmetic (:class:`BlockPlan`), never stored twice.

    Imported inside the function: this module must stay importable no matter
    which of ``index``/``vectors`` a future ``assemble`` imports first. A module-level import
    would make ``index -> vectors`` a hard cycle, and the two are being built concurrently.
    """
    from .index import idmap_arrow_schema

    return idmap_arrow_schema()


def idmap_filename() -> str:
    """``preprocess.index.IDMAP_FILENAME``: one literal, read from its owner (see above)."""
    from .index import IDMAP_FILENAME

    return str(IDMAP_FILENAME)


class PassageRef(NamedTuple):
    """The two id columns of a row, in ``idmap`` order: ``(passage_id, document_id)``.

    A positional alias for those columns, not a new record type: the vector store carries no
    text, no length and no rendering, so there is nothing else about a passage it could hold.
    ``document_id`` is carried rather than re-derived at query time, exactly as the leg id maps
    carry it (the always-original citation).
    """

    passage_id: str
    document_id: str


# --------------------------------------------------------------------------- #
# The block <-> part arithmetic. One place, asserted, no second derivation.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BlockPlan:
    """How one cell's passages are cut into encode blocks and assemble PARTS.

    All three numbers come from outside: ``passages`` from the pinned
    ``(token_count, passage_id)`` order of the one text-free passage table, ``block_passages``
    from the hashed ``index_build.config.encode_block_passages`` leaf and ``part_passages``
    from the hashed ``plaid_part_passages`` one. This module defaults neither size, a second
    default here could drift from the config's, and the drift would be invisible (both sides
    would look self-consistent while cutting the corpus differently).

    Both axes cut the same ordinal order, and a part is a whole number of blocks, so a part's
    blocks are a contiguous run and its rows are their concatenation. That is the reason
    :func:`stream_part` can be sequential.
    """

    passages: int
    block_passages: int
    part_passages: int

    def __post_init__(self) -> None:
        if self.passages < 0:
            raise BlockGeometryError(f"passages must be >= 0, got {self.passages}")
        if self.block_passages < 1 or self.part_passages < 1:
            raise BlockGeometryError(
                f"block_passages={self.block_passages} and part_passages="
                f"{self.part_passages} must both be >= 1"
            )
        if self.part_passages % self.block_passages:
            raise BlockGeometryError(
                f"part_passages={self.part_passages} is not a whole number of blocks of "
                f"{self.block_passages}. Blocks must divide parts evenly: a part is "
                "assembled by concatenating a contiguous run of whole block files, so a "
                "remainder would turn that sequential read into a partial-block gather (~17x "
                "read amplification) and would make the two hashed knobs interact"
            )

    # -- blocks ------------------------------------------------------------- #
    @property
    def blocks(self) -> int:
        """How many blocks the cell holds. A cell with no passages still holds one (empty)
        block, so "a published cell has at least one block" needs no special case."""
        return max(1, ceil(self.passages / self.block_passages))

    @property
    def blocks_per_part(self) -> int:
        """Whole blocks in one part: exact by :meth:`__post_init__`."""
        return self.part_passages // self.block_passages

    def block_bounds(self, block: int) -> tuple[int, int]:
        """``[lo, hi)`` cell ordinals of ``block`` (half-open, clipped to ``passages``)."""
        self._check_block(block)
        lo = block * self.block_passages
        return lo, min(self.passages, lo + self.block_passages)

    def block_passage_count(self, block: int) -> int:
        """How many passages ``block`` holds: full except the last, which may be short."""
        lo, hi = self.block_bounds(block)
        return max(0, hi - lo)

    def part_of_block(self, block: int) -> int:
        """Which assemble part ``block`` belongs to."""
        self._check_block(block)
        return block // self.blocks_per_part

    # -- parts -------------------------------------------------------------- #
    @property
    def parts(self) -> int:
        """How many assemble parts the cell holds (at least one, as for blocks)."""
        return max(1, ceil(self.passages / self.part_passages))

    def part_bounds(self, part: int) -> tuple[int, int]:
        """``[lo, hi)`` cell ordinals of ``part``."""
        self._check_part(part)
        lo = part * self.part_passages
        return lo, min(self.passages, lo + self.part_passages)

    def part_passage_count(self, part: int) -> int:
        lo, hi = self.part_bounds(part)
        return max(0, hi - lo)

    def blocks_of_part(self, part: int) -> range:
        """The contiguous run of block indices ``part`` is assembled from, in read order."""
        self._check_part(part)
        lo = part * self.blocks_per_part
        return range(lo, min(self.blocks, lo + self.blocks_per_part))

    def part_ordinals(self, block: int) -> range:
        """``block``'s rows as part-local ordinals: what an engine is keyed on.

        The one conversion assemble needs, defined once here so no writer re-derives it: a
        part's own ordinal space starts at 0, and block ``b`` of that part starts exactly
        where block ``b-1`` ended (which :func:`stream_part` re-checks against the stream).
        """
        lo, hi = self.block_bounds(block)
        base = self.part_of_block(block) * self.part_passages
        return range(lo - base, hi - base)

    # -- guards -------------------------------------------------------------- #
    def _check_block(self, block: int) -> None:
        if not 0 <= int(block) < self.blocks:
            raise BlockGeometryError(
                f"block {block} is outside this cell's plan (0..{self.blocks - 1} for "
                f"{self.passages} passages at {self.block_passages}/block)"
            )

    def _check_part(self, part: int) -> None:
        if not 0 <= int(part) < self.parts:
            raise BlockGeometryError(
                f"part {part} is outside this cell's plan (0..{self.parts - 1} for "
                f"{self.passages} passages at {self.part_passages}/part)"
            )


# --------------------------------------------------------------------------- #
# The per-leg formats. Three implementations, one writer/reader driver.
# --------------------------------------------------------------------------- #
@runtime_checkable
class VectorFormat(Protocol):
    """How one leg's representations become bytes, and back.

    Narrow: a format knows its filenames, its dtype and its serialization, and
    nothing about plans, censuses, ids or paths. Every policy the index build has about these
    vectors (unit-norm, truncation counting, id-set integrity, engine parameters) stays in
    ``preprocess.index`` and still runs on the assemble side, a format that started
    validating would be a second, quieter copy of it.
    """

    leg: str
    dtype: str

    @property
    def vectors_name(self) -> str:
        """The payload filename (``vectors.npy`` / ``.parquet`` / ``.bin``)."""
        ...

    def payload_names(self) -> tuple[str, ...]:
        """Every file this format writes: what the witness's ``bytes`` is summed over."""
        ...

    def write(self, out_dir: Path, reps: Sequence[Any], refs: Sequence[PassageRef]) -> int | None:
        """Serialize ``reps`` into ``out_dir``; return the leaf-vector count or ``None``."""
        ...

    def read(self, block_dir: Path, passages: int) -> Any:
        """Deserialize exactly ``passages`` representations, in block ordinal order."""
        ...


def _np_dtype(dtype: str) -> str:
    try:
        return STORE_DTYPES[dtype]
    except KeyError:
        raise BlockGeometryError(
            f"unknown store dtype {dtype!r}; expected one of {sorted(STORE_DTYPES)}"
        ) from None


@dataclass(frozen=True, slots=True)
class DenseVectors:
    """``vectors.npy``: one ``(n, dim)`` array, in block ordinal order.

    Raw ``.npy`` rather than Parquet: the array is the record, a float matrix compresses
    ~nothing, and ``np.load`` hands the assemble side exactly the buffer ``faiss.add`` wants
    with no columnar round-trip. An empty block is written as ``(0, 0)``, a legal artifact
    (the dense leg publishes empty indexes too), not a failure.
    """

    leg: str
    dtype: str = DEFAULT_DENSE_DTYPE

    def __post_init__(self) -> None:
        # At construction, not at the first write: an unwritable dtype must fail before a
        # worker has spent an hour encoding 8,192 passages it cannot then store.
        _np_dtype(self.dtype)

    @property
    def vectors_name(self) -> str:
        return _VECTORS_STEM + _DENSE_SUFFIX

    def payload_names(self) -> tuple[str, ...]:
        return (self.vectors_name,)

    def write(self, out_dir: Path, reps: Sequence[Any], refs: Sequence[PassageRef]) -> int | None:
        import numpy as np

        dt = _np_dtype(self.dtype)
        arr = (
            np.zeros((0, 0), dtype=dt)
            if len(reps) == 0
            else np.asarray(reps, dtype=dt)
        )
        if arr.ndim != 2:
            raise BlockGeometryError(
                f"dense reps have shape {arr.shape}, expected (n, dim), one vector per passage"
            )
        if arr.shape[0] != len(reps):
            raise BlockGeometryError(
                f"dense reps became {arr.shape[0]} rows for {len(reps)} passages"
            )
        with open(out_dir / self.vectors_name, "wb") as f:
            np.save(f, arr, allow_pickle=False)
        return None

    def read(self, block_dir: Path, passages: int) -> Any:
        import numpy as np

        arr = np.load(block_dir / self.vectors_name, allow_pickle=False)
        if int(arr.shape[0]) != int(passages):
            raise VectorIntegrityError(
                f"{block_dir}: {self.vectors_name} holds {arr.shape[0]} vectors but the "
                f"witness claims {passages} passages, the block would address the wrong rows"
            )
        if arr.dtype != np.dtype(_np_dtype(self.dtype)):
            raise VectorIntegrityError(
                f"{block_dir}: {self.vectors_name} is {arr.dtype}, but the witness records "
                f"{self.dtype}, the stored bytes and the record disagree"
            )
        return arr


@dataclass(frozen=True, slots=True)
class SparseVectors:
    """``vectors.parquet``: :func:`seismic_components`' own output, one row per passage.

    The conversion from MILCO's ``{dim_id: weight}`` map to Seismic's ``(components, values)``
    string form happens here, once, at write time: it is order-defining (components are sorted
    by their string, which is what Seismic keys on) and it is where the U30 cap is enforced, so
    doing it on the assemble side too would be a second chance to do it differently.
    """

    leg: str
    dtype: str = DEFAULT_SPARSE_DTYPE

    def __post_init__(self) -> None:
        _np_dtype(self.dtype)  # see DenseVectors.__post_init__
        if self.dtype != DEFAULT_SPARSE_DTYPE:
            raise BlockGeometryError(
                f"the sparse leg stores {DEFAULT_SPARSE_DTYPE} values (Seismic's own type); "
                f"{self.dtype!r} would be upcast on the way into the index, i.e. a lossy "
                "round-trip that buys nothing"
            )

    @property
    def vectors_name(self) -> str:
        return _VECTORS_STEM + _SPARSE_SUFFIX

    def payload_names(self) -> tuple[str, ...]:
        return (self.vectors_name,)

    def _schema(self) -> Any:
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("ordinal", pa.int64()),
                pa.field("components", pa.list_(pa.string())),
                pa.field("values", pa.list_(pa.float32())),
            ]
        )

    def write(self, out_dir: Path, reps: Sequence[Any], refs: Sequence[PassageRef]) -> int | None:
        def rows() -> Iterator[dict[str, Any]]:
            for i, weights in enumerate(reps):
                components, values = seismic_components(weights)
                yield {"ordinal": i, "components": components, "values": values}

        _write_parquet_stream(
            out_dir / self.vectors_name, rows(), schema=self._schema(), skip_if_done=False
        )
        return None

    def read(self, block_dir: Path, passages: int) -> Any:
        rows = read_parquet(block_dir / self.vectors_name)
        if len(rows) != int(passages):
            raise VectorIntegrityError(
                f"{block_dir}: {self.vectors_name} holds {len(rows)} rows but the witness "
                f"claims {passages} passages"
            )
        out: list[tuple[list[str], list[float]]] = []
        for i, row in enumerate(rows):
            if int(row["ordinal"]) != i:
                raise VectorIntegrityError(
                    f"{block_dir}: {self.vectors_name} row {i} carries ordinal "
                    f"{row['ordinal']}, the file is not in block ordinal order, so it "
                    "cannot be concatenated with its neighbours"
                )
            out.append(
                ([str(c) for c in row["components"]], [float(v) for v in row["values"]])
            )
        return out


@dataclass(frozen=True, slots=True)
class LateInteractionVectors:
    """``vectors.bin`` (flat ``dtype[total_vectors, dim]``) + a ``counts.parquet`` sidecar.

    Reconstruction is cumulative offsets over ``n_vectors``, which is why the flat file needs
    no per-passage framing at all. The two rejected alternatives, both for measured reasons:
    Arrow nested lists would carry 24-48 GB of offset buffers over 6.03e9 leaf vectors, and
    compression buys ~nothing on fp16 embeddings while costing real CPU on the 1.5 TB read
    side, the side that has to keep up with the GPUs.
    """

    leg: str
    dtype: str = DEFAULT_LATE_INTERACTION_DTYPE

    def __post_init__(self) -> None:
        _np_dtype(self.dtype)  # see DenseVectors.__post_init__

    @property
    def vectors_name(self) -> str:
        return _VECTORS_STEM + _LATE_INTERACTION_SUFFIX

    def payload_names(self) -> tuple[str, ...]:
        return (self.vectors_name, COUNTS_FILENAME)

    def _counts_schema(self) -> Any:
        import pyarrow as pa

        return pa.schema([pa.field("passage_id", pa.string()), pa.field("n_vectors", pa.int32())])

    def write(self, out_dir: Path, reps: Sequence[Any], refs: Sequence[PassageRef]) -> int | None:
        import numpy as np

        dt = np.dtype(_np_dtype(self.dtype))
        counts: list[int] = []
        dim = 0
        with open(out_dir / self.vectors_name, "wb") as f:
            for rep in reps:
                arr = np.asarray(rep, dtype=dt)
                if arr.size == 0:
                    # A passage that yielded no token vectors is kept (its count is 0), never
                    # skipped: dropping it would shift every later passage's offsets.
                    counts.append(0)
                    continue
                if arr.ndim != 2:
                    raise BlockGeometryError(
                        f"late-interaction rep has shape {arr.shape}, expected "
                        "(n_vectors, dim) for one passage"
                    )
                if dim and int(arr.shape[1]) != dim:
                    raise BlockGeometryError(
                        f"late-interaction dim changed mid-block: {arr.shape[1]} != {dim}"
                    )
                dim = int(arr.shape[1])
                f.write(arr.tobytes())
                counts.append(int(arr.shape[0]))
        _write_parquet_stream(
            out_dir / COUNTS_FILENAME,
            (
                {"passage_id": ref.passage_id, "n_vectors": n}
                for ref, n in zip(refs, counts, strict=True)
            ),
            schema=self._counts_schema(),
            skip_if_done=False,
        )
        return int(sum(counts))

    def read(self, block_dir: Path, passages: int) -> Any:
        import numpy as np

        rows = read_parquet(block_dir / COUNTS_FILENAME)
        if len(rows) != int(passages):
            raise VectorIntegrityError(
                f"{block_dir}: {COUNTS_FILENAME} holds {len(rows)} rows but the witness "
                f"claims {passages} passages"
            )
        counts = [int(r["n_vectors"]) for r in rows]
        total = sum(counts)
        dt = np.dtype(_np_dtype(self.dtype))
        flat = np.fromfile(block_dir / self.vectors_name, dtype=dt)
        if total == 0:
            if flat.size:
                raise VectorIntegrityError(
                    f"{block_dir}: {COUNTS_FILENAME} accounts for 0 vectors but "
                    f"{self.vectors_name} holds {flat.size} values"
                )
            return [np.zeros((0, 0), dtype=dt) for _ in range(int(passages))]
        if flat.size % total:
            raise VectorIntegrityError(
                f"{block_dir}: {self.vectors_name} holds {flat.size} values, which is not a "
                f"whole number of vectors over the {total} the sidecar accounts for, the "
                "flat file and its offsets are not from the same write"
            )
        dim = flat.size // total
        mat = flat.reshape(total, dim)
        # ``np.split`` returns views into the one buffer: reconstruction costs no copy, which
        # is what keeps a 9 GB part's read at one pass over the bytes.
        bounds = [int(b) for b in np.cumsum(counts)[:-1]]
        return list(np.split(mat, bounds))


def format_for(leg: str, *, dtype: str | None = None) -> VectorFormat:
    """The :class:`VectorFormat` for ``leg``, at ``dtype`` (its documented default if None).

    The one place a leg name maps to a format, and the only place in this module leg names
    appear at all, they are imported from ``preprocess.index``, which owns them, so a renamed
    leg cannot leave this module pointing at a name nobody writes.
    """
    from .index import DENSE_LEG, LATE_INTERACTION_LEG, LEGS, SPARSE_LEG

    if leg == DENSE_LEG:
        return DenseVectors(leg=leg, dtype=dtype or DEFAULT_DENSE_DTYPE)
    if leg == SPARSE_LEG:
        return SparseVectors(leg=leg, dtype=dtype or DEFAULT_SPARSE_DTYPE)
    if leg == LATE_INTERACTION_LEG:
        return LateInteractionVectors(leg=leg, dtype=dtype or DEFAULT_LATE_INTERACTION_DTYPE)
    raise BlockGeometryError(f"unknown leg {leg!r}; expected one of {LEGS}")


# --------------------------------------------------------------------------- #
# Write: one block, published as one atomic unit, its witness written last.
# --------------------------------------------------------------------------- #
def write_block(
    cell_dir: str | os.PathLike[str],
    *,
    block: int,
    plan: BlockPlan,
    fmt: VectorFormat,
    encode_hash: str,
    lang: str,
    variant: str | None,
    refs: Sequence[PassageRef],
    reps: Sequence[Any],
) -> Path:
    """Write one encode block into ``cell_dir``; return its published directory.

    ``reps[i]`` belongs to ``refs[i]``, and both are in block ordinal order, the order the
    cell's pinned ``(token_count, passage_id)`` key defines, sliced by
    :meth:`BlockPlan.block_bounds`. That ordering is the caller's to establish and this
    function's to preserve: the vectors file, the id map and the reader all address rows
    positionally, so a re-ordered ``reps`` is a silently mis-addressed index.

    Resume is a no-op: a block whose ``_SUCCESS`` marker exists is skipped untouched (its
    bytes and its mtime), so a re-launched vectorize task re-encodes only what is missing.
    Publication is ``common.io``'s temp->rename + marker contract applied to a directory, with
    the witness written last inside the temp dir, so a ``block-*`` directory either holds a
    complete block or does not exist.
    """
    cell = Path(cell_dir)
    block = int(block)
    expected = plan.block_passage_count(block)  # also range-checks ``block``
    if len(refs) != len(reps):
        raise BlockGeometryError(
            f"{len(refs)} passage refs for {len(reps)} representations, the id map and the "
            "vectors would address different rows"
        )
    if len(refs) != expected:
        raise BlockGeometryError(
            f"block {block} of this plan holds {expected} passages, got {len(refs)}: a block "
            "is a fixed ordinal slice of the cell, not whatever the worker happened to encode"
        )

    final = cell / block_dir_name(block)
    if is_done(final):
        _log.info("preprocess.vectors.block.resume", path=str(final), leg=fmt.leg, block=block)
        return final

    tmp = cell / f".{block_dir_name(block)}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    vectors = fmt.write(tmp, reps, list(refs))
    # The id map is written inside the temp dir, so it is published by the same rename as the
    # vectors: a block can never be marked done with a missing or stale id map.
    _write_parquet_stream(
        tmp / idmap_filename(),
        (
            {
                "ordinal": i,
                "passage_id": ref.passage_id,
                "document_id": ref.document_id,
                "source_lang": lang,
            }
            for i, ref in enumerate(refs)
        ),
        schema=idmap_schema(),
        skip_if_done=False,
    )
    payload_bytes = sum(
        (tmp / name).stat().st_size for name in fmt.payload_names() if (tmp / name).exists()
    )
    # The census witness, written last: everything above is on disk (or this line was never
    # reached), so a block directory carrying this file is a complete block and one without it
    # is an abandoned attempt. No timestamp: a re-built block must be
    # Byte-identical or the artifact tree stops being a checkpoint (invariant 7).
    write_jsonl(
        tmp / BLOCK_FILENAME,
        [
            {
                "block": block,
                "blocks": plan.blocks,
                "block_passages": int(plan.block_passages),
                "passages": len(refs),
                "lang": lang,
                "variant": variant,
                "leg": fmt.leg,
                "encode_hash": str(encode_hash),
                "dtype": fmt.dtype,
                "vectors": vectors,
                "bytes": int(payload_bytes),
            }
        ],
        skip_if_done=False,
    )
    published = _publish_block_dir(tmp, final)
    _log.info(
        "preprocess.vectors.block.published",
        path=str(published),
        leg=fmt.leg,
        lang=lang,
        variant=variant,
        block=block,
        blocks=plan.blocks,
        passages=len(refs),
        dtype=fmt.dtype,
        bytes=int(payload_bytes),
    )
    return published


def _publish_block_dir(tmp: Path, final: Path) -> Path:
    """``common.io``'s temp->rename + ``_SUCCESS`` contract, applied to a directory.

    Byte-for-byte the discipline ``preprocess.index._publish_dir`` applies to a leg directory
    (an unmarked leftover from a crashed attempt is removed first, it is by definition
    incomplete, and renaming onto it would mix two writes' files). It is a local copy only
    because that helper is private to a module this one must not import at module scope
    (``index -> vectors`` would be a cycle once ``assemble`` lands); the primitive belongs in
    ``common.io`` and is flagged for that.
    """
    marker = success_marker(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if marker.exists():
            # Another worker published this block while we were building it (both passed
            # ``write_block``'s skip check). Theirs stands; drop ours rather than leaving a
            # dot-prefixed temp dir behind forever: it is invisible to the census glob, so
            # nothing would ever clean it up.
            shutil.rmtree(tmp, ignore_errors=True)
            return final
        shutil.rmtree(final)
    os.replace(tmp, final)
    marker.write_bytes(b"")
    return final


# --------------------------------------------------------------------------- #
# Read: the census first, then a strictly sequential stream over a part.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CellCensus:
    """One cell's plan, reconstructed from its blocks' own witnesses and cross-checked.

    Never one writer's claim: blocks are claimed by different workers on different machines, so
    there is no writer that knows the whole cell. Unanimity among N independent witnesses is
    strictly stronger evidence than any single record would be.
    """

    blocks: int
    block_passages: int
    passages: int
    leg: str
    encode_hash: str
    lang: str
    variant: str | None
    dtype: str
    vectors: int | None
    bytes: int
    dirs: tuple[str, ...]
    block_passage_counts: tuple[int, ...]

    def plan(self, part_passages: int) -> BlockPlan:
        """This cell's :class:`BlockPlan` at the caller's (hashed) assemble part size."""
        return BlockPlan(
            passages=self.passages,
            block_passages=self.block_passages,
            part_passages=int(part_passages),
        )

    def format(self) -> VectorFormat:
        """The format this cell was written with: read off the witnesses, never assumed.

        This is what makes the dtype a parameter rather than a literal in practice: a block
        written before a default moved is still read at the dtype it was written at.
        """
        return format_for(self.leg, dtype=self.dtype)


@dataclass(frozen=True, slots=True)
class VectorBlock:
    """One block's payload, as read.

    ``ordinals`` is the ordinal range of these rows in the space the reader was asked
    about: Part-local when it came from :func:`stream_part` (what an engine is keyed on) and
    block-local from :func:`read_block_dir`, which knows about no part. It is a ``range``
    rather than a start offset so a caller feeds ``list(block.ordinals)`` straight to a leg
    writer and cannot be off by the block's length.

    ``reps`` is the leg's own type: a ``(n, dim)`` array (dense), a list of
    ``(components, values)`` pairs (sparse), or a list of ``(n_i, dim)`` matrices (late
    interaction), exactly what ``preprocess.index``'s writers already take.
    """

    block: int
    block_dir: Path
    leg: str
    dtype: str
    lang: str
    variant: str | None
    encode_hash: str
    ordinals: range
    refs: tuple[PassageRef, ...]
    reps: Any


def read_cell_census(cell_dir: str | os.PathLike[str]) -> CellCensus:
    """One cell's census, reconstructed from every block witness and cross-checked on disk.

    Mirrors ``preprocess.index.read_shard_parts`` exactly, one level down and for the same
    reason: a reader must never infer a block count. Required, all of it:

    (a) at least one ``block-*`` directory (a cell is never the bare directory);
    (b) a ``block.json`` for every one of them (a directory without it is a crashed attempt,
        not a block) and its ``_SUCCESS`` marker;
    (c) each witness sitting in the directory its own ``block`` index names;
    (d) unanimity on the whole plan, ``blocks``/``block_passages`` (a cell cut two ways is
        unassemblable) and ``leg``/``encode_hash``/``lang``/``variant``/``dtype`` (a cell
        holding two legs' or two renderings' vectors would concatenate into nonsense);
    (e) indices covering exactly ``0..blocks-1``, with exactly ``blocks`` directories;
    (f) only the last block short, blocks are a dense partition of the cell's ordinals, so a
        short block anywhere else means some ordinals were never encoded.

    Anything else raises :class:`VectorIntegrityError`. There is no "assemble what we found".
    """
    path = Path(cell_dir)
    dirs = sorted(p for p in path.glob(BLOCK_GLOB) if p.is_dir())
    if not dirs:
        raise VectorIntegrityError(
            f"{path}: no {BLOCK_GLOB} directories, a written cell holds at least one block, "
            "and a reader never falls back to treating the cell directory itself as one"
        )
    by_block: dict[int, tuple[Path, Mapping[str, Any]]] = {}
    for block_dir in dirs:
        witness = block_dir / BLOCK_FILENAME
        if not witness.exists():
            raise VectorIntegrityError(
                f"{block_dir}: no {BLOCK_FILENAME}, every complete block records the plan it "
                "was built under (as the last act of write_block), so a block directory "
                "without one is an abandoned attempt and this cell must not be assembled "
                "until it is rebuilt"
            )
        if not is_done(block_dir):
            raise VectorIntegrityError(
                f"{block_dir}: no _SUCCESS marker, the directory was never published by the "
                "atomic rename, so its contents are not a block"
            )
        record = read_jsonl(witness)[0]
        index = int(record["block"])
        if block_dir.name != block_dir_name(index):
            raise VectorIntegrityError(
                f"{block_dir}: {BLOCK_FILENAME} claims block {index}, i.e. "
                f"{block_dir_name(index)!r}, a block in the wrong directory would be read "
                "twice or not at all"
            )
        if index in by_block:  # pragma: no cover - unreachable while names are unique
            raise VectorIntegrityError(f"{path}: two directories claim block {index}")
        by_block[index] = (block_dir, record)

    plans = {
        (
            int(r["blocks"]),
            int(r["block_passages"]),
            str(r["leg"]),
            str(r["encode_hash"]),
            str(r["lang"]),
            None if r["variant"] is None else str(r["variant"]),
            str(r["dtype"]),
        )
        for _, r in by_block.values()
    }
    if len(plans) != 1:
        raise VectorIntegrityError(
            f"{path}: the blocks disagree about the cell's plan ({sorted(plans)}), they were "
            "not written from one plan, so which of them together form the cell (and whether "
            "they are even the same leg and rendering) is unknowable"
        )
    blocks, block_passages, leg, encode_hash, lang, variant, dtype = plans.pop()
    if sorted(by_block) != list(range(blocks)):
        raise VectorIntegrityError(
            f"{path}: blocks on disk are {sorted(by_block)} but the census claims {blocks} "
            f"block(s) 0..{blocks - 1}, this cell would be assembled short (a part built "
            "from a subset is a full-looking index that is silently missing passages)"
        )
    counts = tuple(int(by_block[i][1]["passages"]) for i in range(blocks))
    for i, n in enumerate(counts[:-1]):
        if n != block_passages:
            raise VectorIntegrityError(
                f"{path}: block {i} holds {n} passages, not {block_passages}. Only the last "
                "block of a cell may be short, a short block anywhere else means the cell's "
                "ordinals are not densely covered"
            )
    if counts and counts[-1] > block_passages:
        raise VectorIntegrityError(
            f"{path}: the last block holds {counts[-1]} passages, over the block size "
            f"{block_passages}"
        )
    per_block_vectors = [by_block[i][1].get("vectors") for i in range(blocks)]
    return CellCensus(
        blocks=blocks,
        block_passages=block_passages,
        passages=sum(counts),
        leg=leg,
        encode_hash=encode_hash,
        lang=lang,
        variant=variant,
        dtype=dtype,
        vectors=(
            None
            if any(v is None for v in per_block_vectors)
            else sum(int(v) for v in per_block_vectors)
        ),
        bytes=sum(int(by_block[i][1]["bytes"]) for i in range(blocks)),
        dirs=tuple(by_block[i][0].name for i in range(blocks)),
        block_passage_counts=counts,
    )


def read_block_dir(
    block_dir: str | os.PathLike[str], *, fmt: VectorFormat | None = None
) -> VectorBlock:
    """Read one block from its own directory: the primitive, not a random-access API.

    It takes a directory rather than a block index and consults no census: it
    therefore cannot tell whether the block it read is part of a complete cell, so anything
    assembled from it alone is assembled from an unverified set. :func:`stream_part` is what
    assemble calls; this is what :func:`stream_part` (and a round-trip test) is built from.

    Ordinals are block-local here (``0..n-1``, matching the block's own ``idmap.parquet``),
    because a bare block directory knows nothing about which part it belongs to.
    """
    path = Path(block_dir)
    record = read_jsonl(path / BLOCK_FILENAME)[0]
    passages = int(record["passages"])
    fmt = fmt or format_for(str(record["leg"]), dtype=str(record["dtype"]))
    rows = read_parquet(path / idmap_filename())
    if len(rows) != passages:
        raise VectorIntegrityError(
            f"{path}: {idmap_filename()} holds {len(rows)} rows but the witness claims "
            f"{passages} passages"
        )
    refs: list[PassageRef] = []
    for i, row in enumerate(rows):
        if int(row["ordinal"]) != i:
            raise VectorIntegrityError(
                f"{path}: {idmap_filename()} row {i} carries ordinal {row['ordinal']}, the "
                "id map is not in block ordinal order and would address the wrong vectors"
            )
        refs.append(PassageRef(str(row["passage_id"]), str(row["document_id"])))
    return VectorBlock(
        block=int(record["block"]),
        block_dir=path,
        leg=str(record["leg"]),
        dtype=str(record["dtype"]),
        lang=str(record["lang"]),
        variant=None if record["variant"] is None else str(record["variant"]),
        encode_hash=str(record["encode_hash"]),
        ordinals=range(passages),
        refs=tuple(refs),
        reps=fmt.read(path, passages),
    )


def stream_part(
    cell_dir: str | os.PathLike[str],
    *,
    part: int,
    part_passages: int,
    census: CellCensus | None = None,
) -> Iterator[VectorBlock]:
    """Stream one assemble part's blocks, in order, as a sequential concatenation.

    This is the module's reason for existing. A part's blocks are a contiguous run of whole
    block files; this opens each of them once, in ascending order, and yields them with
    part-local ``ordinals`` that continue exactly where the previous block's ended, which is
    asserted against a running cursor, so "the concatenation is the part" is a checked
    property rather than a hope. Assemble can therefore do::

        writer = leg.writer(tmp, ctx)
        for block in stream_part(cell, part=p, part_passages=N):
            writer.add(list(block.ordinals), block.reps)

    and never hold more than one block. Reading a part any other way, seeking, sorting,
    gathering blocks by index, is ~17x read amplification over 1.68 TB, which is the
    difference between ~20-30 min of aggregate I/O and a build that never finishes.

    The census is validated first (:func:`read_cell_census`), so a cell missing one block
    raises before a single vector is read, rather than yielding a short part.
    """
    path = Path(cell_dir)
    census = census or read_cell_census(path)
    plan = census.plan(part_passages)
    if plan.blocks != census.blocks:  # pragma: no cover - guarded by the census's own checks
        raise VectorIntegrityError(
            f"{path}: the witnesses claim {census.blocks} blocks but {census.passages} "
            f"passages at {census.block_passages}/block is {plan.blocks}"
        )
    fmt = census.format()
    cursor = plan.part_bounds(part)[0] - part * plan.part_passages  # always 0; stated, not assumed
    for block in plan.blocks_of_part(part):
        ordinals = plan.part_ordinals(block)
        if ordinals.start != cursor:
            raise VectorIntegrityError(
                f"{path}: block {block} starts at part-local ordinal {ordinals.start}, not "
                f"{cursor}, the part's blocks are not a gapless concatenation"
            )
        read = read_block_dir(path / block_dir_name(block), fmt=fmt)
        if len(read.refs) != len(ordinals):
            raise VectorIntegrityError(
                f"{path}: block {block} holds {len(read.refs)} passages but the census's plan "
                f"gives it {len(ordinals)}"
            )
        cursor = ordinals.stop
        yield VectorBlock(
            block=read.block,
            block_dir=read.block_dir,
            leg=read.leg,
            dtype=read.dtype,
            lang=read.lang,
            variant=read.variant,
            encode_hash=read.encode_hash,
            ordinals=ordinals,
            refs=read.refs,
            reps=read.reps,
        )
    if cursor != plan.part_passage_count(part):
        raise VectorIntegrityError(
            f"{path}: part {part} streamed {cursor} passages, not the "
            f"{plan.part_passage_count(part)} its plan gives it"
        )
