"""Shared fixtures for the config-fairness tests.

Every "mutated" fixture is an in-memory string or a ``tmp_path`` copy: the tracked
``config/*.yml`` files are read-only test data and are never opened for writing.
The two ``synthetic_*_text`` fixtures carry every required top-level block with tiny
bodies so the small tests stay independent of the real 7 files (whose blocks may grow).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CFG_DIR = _REPO_ROOT / "config"

# A minimal, fairness-compliant e2e config: kind e2e_agentic, one moved
# knob (passage_lang), and no retrieval block (retrieval.index defaults to original).
_SYNTHETIC_E2E = """\
run:
  id: e2e-synth
  kind: e2e_agentic
  status: submitted
  description: synthetic e2e fixture
llm:
  model: "synth-model"
  role: "decomposer + k loops"
  decoding: { temperature: 0.7, top_p: 0.8, top_k: 20 }
  pin: [model_id, prompt_version_hash, seed]
claim_commit:
  rule: "span-grounding verbatim quote match"
  properties: "deterministic, language-neutral"
decomposition:
  policy: "dynamic coverage loop"
  usage: "one decomposition per run"
rag_loop:
  budget:
    min_searches: 2
    max_iters: 8
    max_searches: 5
    token_budget: 20000
  search_top_k: 20
  emission: action
  abstain: gated
chunker:
  config: "one config across variants"
  hash: "<logged at runtime>"
# The blocks below are all shared blocks, and SHARED_BLOCKS doubles as the list of
# required blocks in `schema.validate`: a config omitting one raises `missing required
# block` rather than hashing to "" and passing `family_guard` vacuously. A synthetic
# fixture must carry them all; omitting them is not minimal, it is invalid.
merge:
  merge_short_sentences: true
translation:
  config: "one translation policy across variants"
  hash: "<logged at runtime>"
reconcile:
  split_back: true
packing:
  pack_length: len_max
  pack_budget: 512
index_build:
  config: "one index-build policy across variants"
  hash: "<logged at runtime>"
# Values are the shipped ones, verbatim from the real config/*.yml, rather than
# convenient round numbers: a fixture that speaks a vocabulary the real configs do not
# use is how an unemittable knob survives a green gate.
serialize:
  k_t3: 30
  answers_cap: 5
  dedup_a_cutoff: 0.85
  rrf_k: 60
  top_docs: 1000
  dedup_q_cutoff: 0.90
# `topics` is a shared block, so it is required here too. It is a mapping rather than
# a bare scalar (see `_ALLOWED["topics"]`): the sub-key is validated, so a typo such as
# `pathh:` fails at load instead of at read time on a compute node.
topics:
  path: topics/topics.all.2026.v0625-fix.jsonl
# The e2e seed count is 1. This fixture is the contract under test, so it tracks the
# shipped value.
seeds: 1
citations: "always original doc_id"
languages: [zh, en, ru, es]
passage_lang: omt
outputs:
  - task: 1
    track: report_generation
    path: "submissions/report_generation/run_synth.jsonl"
    format: "JSONL"
    run_id: "e2e-synth"
    run_desc: "synthetic report output"
team_id: "TBD (set before submission)"
"""

# A minimal, fairness-compliant mlir config. It is `kind: e2e_agentic`, exactly like every
# shipped `config/mlir-*.yml`: Task 2 runs the same agentic pipeline, and the family comes
# from `run.id`, not from `run.kind`. It has seeds 1, one moved knob (retrieval.index), and
# no passage_lang (defaults original).
_SYNTHETIC_MLIR = """\
run:
  id: mlir-synth
  kind: e2e_agentic
  status: submitted
  description: synthetic mlir fixture
llm:
  model: "synth-model"
  role: "decomposer + k loops"
  decoding: { temperature: 0.7, top_p: 0.8, top_k: 20 }
  pin: [model_id, prompt_version_hash, seed]
claim_commit:
  rule: "span-grounding verbatim quote match"
  properties: "deterministic, language-neutral"
decomposition:
  policy: "dynamic coverage loop"
  usage: "one decomposition per run"
rag_loop:
  budget:
    min_searches: 2
    max_iters: 8
    max_searches: 5
    token_budget: 20000
  search_top_k: 20
  emission: action
  abstain: gated
chunker:
  config: "one config across variants"
  hash: "<logged at runtime>"
# The blocks below are all shared blocks, and SHARED_BLOCKS doubles as the list of
# required blocks in `schema.validate`: a config omitting one raises `missing required
# block` rather than hashing to "" and passing `family_guard` vacuously. A synthetic
# fixture must carry them all; omitting them is not minimal, it is invalid.
merge:
  merge_short_sentences: true
translation:
  config: "one translation policy across variants"
  hash: "<logged at runtime>"
reconcile:
  split_back: true
packing:
  pack_length: len_max
  pack_budget: 512
index_build:
  config: "one index-build policy across variants"
  hash: "<logged at runtime>"
# Values are the shipped ones, verbatim from the real config/*.yml, rather than
# convenient round numbers: a fixture that speaks a vocabulary the real configs do not
# use is how an unemittable knob survives a green gate.
serialize:
  k_t3: 30
  answers_cap: 5
  dedup_a_cutoff: 0.85
  rrf_k: 60
  top_docs: 1000
  dedup_q_cutoff: 0.90
# `topics` is a shared block, so it is required here too. It is a mapping rather than
# a bare scalar (see `_ALLOWED["topics"]`): the sub-key is validated, so a typo such as
# `pathh:` fails at load instead of at read time on a compute node.
topics:
  path: topics/topics.all.2026.v0625-fix.jsonl
seeds: 1
citations: "always original doc_id"
languages: [zh, en, ru, es]
retrieval:
  spine: "PLAID-X"
  dense: "Qwen3-Embedding-4B"
  sparse: "BGE-M3 sparse"
  rrf: { mode: rank, k: 60, quality_gated: true }
  reranker: { model: "Qwen3-Reranker-4B", depth: 100 }
  index: omt
outputs:
  - task: 2
    track: retrieval
    path: "submissions/retrieval/run_synth.txt"
    format: "6-column TREC run"
    run_id: "mlir-synth"
    run_desc: "synthetic retrieval output"
team_id: "TBD (set before submission)"
"""


@pytest.fixture
def synthetic_e2e_text() -> str:
    return _SYNTHETIC_E2E


@pytest.fixture
def synthetic_mlir_text() -> str:
    return _SYNTHETIC_MLIR


@pytest.fixture
def write_config() -> Callable[[Path, str], Path]:
    """Write a config text to ``tmp_path/cfg.yml`` (the only way a test feeds ``load``)."""

    def _write(tmp_path: Path, text: str) -> Path:
        tmp_path.mkdir(parents=True, exist_ok=True)
        p = tmp_path / "cfg.yml"
        p.write_text(text, encoding="utf-8")
        return p

    return _write


@pytest.fixture
def mutate_block() -> Callable[[str, str, Callable[[str], str]], str]:
    """Replace a top-level block's body with ``transform(body)`` (text-level, in memory).

    Locates the column-0 ``block_name:`` line and its indented body (up to the next
    column-0 key) and rewrites the body. Used to build mutated real-config
    fixtures without ever editing a tracked file.
    """

    def _mutate(text: str, block_name: str, transform: Callable[[str], str]) -> str:
        lines = text.splitlines(keepends=True)
        start = None
        for i, line in enumerate(lines):
            if re.match(rf"^{re.escape(block_name)}:", line) and not line.startswith((" ", "\t")):
                start = i
                break
        if start is None:
            raise KeyError(f"top-level block {block_name!r} not found")
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if re.match(r"^[A-Za-z_][\w-]*:", lines[j]) and not lines[j].startswith((" ", "\t")):
                end = j
                break
        body = "".join(lines[start:end])
        return "".join(lines[:start]) + transform(body) + "".join(lines[end:])

    return _mutate


def _cfg(name: str) -> Path:
    return _CFG_DIR / name


@pytest.fixture
def real_e2e_paths() -> list[Path]:
    return [
        _cfg("e2e-omt.yml"),
        _cfg("e2e-omt-weak.yml"),
        _cfg("e2e-original.yml"),
    ]


@pytest.fixture
def real_mlir_paths() -> list[Path]:
    return [
        _cfg("mlir-omt.yml"),
        _cfg("mlir-omt-weak.yml"),
        _cfg("mlir-original.yml"),
    ]


@pytest.fixture
def readme_outputs_table() -> dict[str, list[tuple[int, str, str, str]]]:
    """Transcription of the config/README.md run -> track -> priority -> path table.

    Each value is a list of ``(task, track, path, run_id)`` tuples per outputs[i].
    """
    return {
        "e2e-omt.yml": [
            (1, "report_generation", "submissions/report_generation/run_1.jsonl", "e2e-omt"),
            (3, "nuggetization", "submissions/nuggetization/run_1.jsonl", "e2e-omt"),
        ],
        "e2e-omt-weak.yml": [
            (1, "report_generation", "submissions/report_generation/run_2.jsonl", "e2e-omt-weak"),
            (3, "nuggetization", "submissions/nuggetization/run_2.jsonl", "e2e-omt-weak"),
        ],
        "e2e-original.yml": [
            (1, "report_generation", "submissions/report_generation/run_3.jsonl", "e2e-original"),
            (3, "nuggetization", "submissions/nuggetization/run_3.jsonl", "e2e-original"),
        ],
        "mlir-original.yml": [
            (2, "retrieval", "submissions/retrieval/run_1.txt", "mlir-original"),
        ],
        "mlir-omt.yml": [
            (2, "retrieval", "submissions/retrieval/run_2.txt", "mlir-omt"),
        ],
        "mlir-omt-weak.yml": [
            (2, "retrieval", "submissions/retrieval/run_3.txt", "mlir-omt-weak"),
        ],
    }
