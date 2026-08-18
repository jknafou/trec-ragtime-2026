"""The chunk pool under a console-script ``__main__``.

The defect this file exists for: the pool was only ever run as an imported module, because
under pytest ``__main__`` is a real module, while a real run drives it from the uv console
script ``.venv/bin/run``, which is an extension-less file. The ``spawn`` start method sends
that path to the child and re-executes it with ``runpy.run_path``, so the child re-imports
the entrypoint. A console script whose ``main()`` is not behind an
``if __name__ == "__main__"`` guard then re-enters ``main()``, trips
``spawn._check_not_importing_main()``, and the pool answers by respawning replacement
workers for ever.

So the pool runs here in real subprocesses whose ``__main__`` is an extension-less,
console-script-shaped file, in both the guarded shape that uv generates today and the
unguarded worst case that ``spawn`` cannot survive. The models come from the
``model_factory`` seam as dependency-free fakes, so this takes seconds and needs no
wtpsplit and no network. The pass through the real console script with real models is in
``test_chunk_console_script_full.py``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from ragtime.preprocess.chunk import _pool_context, segment_documents

pytestmark = pytest.mark.small

_TIMEOUT_S = 90  # a regressed spawn path storms processes instead of failing fast

# The options and documents shared by the harness and the in-process reference.
_OPTS = {"strip_boilerplate": True, "boilerplate_rules_version": "v1"}
_DOCS = [
    {
        "id": "eng-docs/0000001",
        "text": "Home › News\nFirst para one.|First para two.\n\nSecond para only.",
        "url": "u",
        "date": "d",
        "lang": "en",
    },
    {
        "id": "spa-docs/0000002",
        "text": "El café tiene dos gramos|Segunda frase corta",
        "url": "u",
        "date": "d",
        "lang": "es",
    },
    {
        "id": "eng-docs/0000003",
        "text": "|".join(" ".join(f"s{i}_{j}" for j in range(5)) for i in range(6)),
        "url": "u",
        "date": "d",
        "lang": "en",
    },
]

# A console-script-shaped `__main__`: an extension-less file whose `main()` is imported
# from a module, which is the shape uv generates in `.venv/bin/`. `{guard}` is filled with
# either the real guard or the unguarded worst case.
_HARNESS = '''\
import json
import sys

from ragtime.preprocess.chunk import _ChunkPool


class _FakeSegmenter:
    def split(self, text):
        return [s.strip() for s in text.split("|") if s.strip()]

    def split_batch(self, texts):
        return [self.split(t) for t in texts]

    def split_spans(self, text):
        spans, pos = [], 0
        for raw in text.split("|"):
            seg = raw.strip()
            if seg:
                a = pos + (len(raw) - len(raw.lstrip()))
                spans.append((a, a + len(seg)))
            pos += len(raw) + 1
        return spans

    def split_spans_batch(self, texts):
        return [self.split_spans(t) for t in texts]


class _FakeTokenizer:
    def count(self, text):
        return len(text.split())

    def num_special(self):
        return 0


def _fake_models():
    return _FakeSegmenter(), _FakeTokenizer()


def main():
    docs = json.loads(sys.argv[1])
    opts = json.loads(sys.argv[2])
    pool = _ChunkPool(2, "unused-model", "unused-tokenizer", model_factory=_fake_models)
    try:
        out = list(pool.segment_documents(docs, batch_size=1, **opts))
    finally:
        pool.close()
    print("SENTENCES " + json.dumps([s["sentence_id"] for r in out for s in r["sentences"]]))
    return 0


{guard}
'''

_GUARDED = 'if __name__ == "__main__":\n    sys.exit(main())'
_UNGUARDED = "sys.exit(main())"


def _write_console_script(tmp_path: Path, name: str, guard: str) -> Path:
    script = tmp_path / name  # no .py suffix (a console script has none)
    script.write_text(_HARNESS.format(guard=guard), encoding="utf-8")
    script.chmod(0o755)
    return script


def _run_console_script(script: Path) -> subprocess.CompletedProcess[str]:
    """Run the harness in its own session, so a process storm can be killed whole."""
    argv = [sys.executable, str(script), json.dumps(_DOCS), json.dumps(_OPTS)]
    proc = subprocess.Popen(  # fixed argv, no shell
        argv,
        cwd=str(script.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # the whole tree, not just the head
        out, err = proc.communicate()
        pytest.fail(f"console-script pool run did not finish in {_TIMEOUT_S}s\nstderr:\n{err}")
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


class _Seg:
    """The in-process twin of the harness's fake segmenter, same four-method surface."""

    def split(self, text: str) -> list[str]:
        return [s.strip() for s in text.split("|") if s.strip()]

    def split_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.split(t) for t in texts]

    def split_spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        pos = 0
        for raw in text.split("|"):
            seg = raw.strip()
            if seg:
                a = pos + (len(raw) - len(raw.lstrip()))
                spans.append((a, a + len(seg)))
            pos += len(raw) + 1
        return spans

    def split_spans_batch(self, texts: list[str]) -> list[list[tuple[int, int]]]:
        return [self.split_spans(t) for t in texts]


class _Tok:
    def count(self, text: str) -> int:
        return len(text.split())

    def num_special(self) -> int:
        return 0


def _reference_sentence_ids() -> list[str]:
    """The sequential in-process reference the subprocess output must match exactly."""
    records = segment_documents(_DOCS, _Seg(), _Tok(), batch_size=1, **_OPTS)
    return [s["sentence_id"] for r in records for s in r["sentences"]]


def _assert_clean_pool_run(res: subprocess.CompletedProcess[str]) -> None:
    assert "bootstrapping phase" not in res.stderr, res.stderr  # the spawn re-import failure
    assert res.returncode == 0, f"exit={res.returncode}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    line = next(x for x in res.stdout.splitlines() if x.startswith("SENTENCES "))
    assert json.loads(line[len("SENTENCES ") :]) == _reference_sentence_ids()


def test_pool_runs_under_guarded_console_script(tmp_path: Path) -> None:
    """The shape of the real console script: extension-less file, guarded ``main()``."""
    _assert_clean_pool_run(_run_console_script(_write_console_script(tmp_path, "run", _GUARDED)))


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="fork-only guarantee: on non-Linux the pool uses spawn, which cannot survive an "
    "unguarded __main__ (both real entrypoints there are guarded)",
)
def test_pool_runs_under_unguarded_console_script(tmp_path: Path) -> None:
    """The worst case: an unguarded extension-less main.

    Under ``spawn`` this re-enters ``main()`` in every child, raises
    ``_check_not_importing_main`` and storms replacement processes. Under ``fork`` the child
    re-imports nothing, so the entrypoint's shape does not matter. Pinning the worst case
    keeps the pool safe whatever shape uv generates for the console script later.
    """
    _assert_clean_pool_run(
        _run_console_script(_write_console_script(tmp_path, "run", _UNGUARDED))
    )


def test_pool_start_method_is_fork_on_linux() -> None:
    """``fork`` on Linux and ``spawn`` elsewhere, so Linux never re-imports the main."""
    method = _pool_context().get_start_method()
    assert method == ("fork" if sys.platform.startswith("linux") else "spawn")
