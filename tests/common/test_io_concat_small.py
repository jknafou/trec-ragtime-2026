"""``concat_files``: the streaming byte-concat writer, at constant memory.

The merge-side primitive: raw bytes are copied file by file in caller order with
the same durability contract as ``write_jsonl`` (temp file, rename, ``_SUCCESS``
marker, skip-if-done resume no-op).
"""

from __future__ import annotations

import pytest

from ragtime.common.io import concat_files, is_done

pytestmark = pytest.mark.small


def test_concat_streams_bytes_in_caller_order_with_success(tmp_path) -> None:
    srcs = []
    for i, data in enumerate([b"a1\n", b"b2\nb3\n", "café — тест\n".encode()]):
        p = tmp_path / f"src_{i}"
        p.write_bytes(data)
        srcs.append(p)
    out = tmp_path / "merged.jsonl"
    got = concat_files(out, srcs)
    assert got == out
    assert out.read_bytes() == b"a1\nb2\nb3\n" + "café — тест\n".encode()
    assert is_done(out)  # _SUCCESS written last
    assert not list(tmp_path.glob("merged.jsonl.tmp-*"))  # no temp left behind


def test_concat_empty_source_list_writes_an_empty_done_artifact(tmp_path) -> None:
    out = tmp_path / "empty.jsonl"
    concat_files(out, [])
    assert out.read_bytes() == b""
    assert is_done(out)


def test_concat_skip_if_done_is_a_resume_no_op(tmp_path) -> None:
    src = tmp_path / "src"
    src.write_bytes(b"x\n")
    out = tmp_path / "out.jsonl"
    concat_files(out, [src])
    src.write_bytes(b"CHANGED\n")
    concat_files(out, [src])  # already done -> bytes untouched (the resume guarantee)
    assert out.read_bytes() == b"x\n"
    assert concat_files(out, [src], skip_if_done=False).read_bytes() == b"CHANGED\n"
