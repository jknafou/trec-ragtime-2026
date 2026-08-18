"""Decompose: structural guarantees, checked on the import graph rather than in a comment.

FT-B4 (round 0 is retrieval-free), FT-B8 (no model id anywhere), FT-B27 (the grounding
gate admits or refuses, it never drops a nugget on a second model's score),
FT-B33/B34/B35 (dedup's import constraints).

These are ``ast``-level checks plus one subprocess import probe: a claim like "this
module never imports sentence_transformers" is only worth making if something goes red
when it stops being true.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.small

PKG = Path(__file__).resolve().parents[2] / "src" / "ragtime" / "pipeline" / "decompose"
MODULES = sorted(PKG.glob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    """Every dotted module name this file imports (absolute and relative)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            names.add(f"{prefix}{node.module or ''}")
    return names


def test_package_has_exactly_the_decompose_module_set():
    """The package boundary is a directory-content fact, not a statement in prose.

    `coverage_audit.py` belongs here because it is decompose from round 1 on and shares the
    import constraints: every structural rule below -- no retrieval, no rag_loop, no second
    judgement model, no client construction -- covers it too.

    `round_loop.py` and `driver.py` stay one level up, in `pipeline/`. They call `run_loop`, and
    a decompose module that imported the RAG loop would break the retrieval-freeness this file
    proves on the import graph.
    """
    present = {p.name for p in MODULES}
    assert present == {
        "__init__.py",
        "bank.py",
        "coverage_audit.py",
        "dedup.py",
        "exemplars.py",
        "fairness_anchor.py",
        "grow_nuggets.py",
        "kband.py",
        "on_topic.py",
        "prompts.py",
        "saturation.py",
        "weighting.py",
    }
    for elsewhere in ("round_loop.py", "driver.py", "records.py"):
        assert elsewhere not in present, (
            f"{elsewhere} calls the RAG loop and must live in pipeline/, not decompose/"
        )


def test_ft_b4_decompose_never_imports_retrieval_or_the_rag_loop():
    """FT-B4: round 0's retrieval-freeness is a property of the import graph."""
    forbidden = ("ragtime.retrieval", "ragtime.pipeline.rag_loop")
    for path in MODULES:
        for imported in _imported_modules(path):
            for bad in forbidden:
                assert not imported.startswith(bad), f"{path.name} imports {imported}"


def test_ft_b4b_importing_decompose_pulls_in_no_retrieval_module():
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, ragtime.pipeline.decompose; "
                "print([m for m in sys.modules if 'retrieval' in m or 'rag_loop' in m])"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]", out.stdout


def test_ft_b8_no_hardcoded_model_id_anywhere_in_the_package():
    """FT-B8: model identity flows in via ``cfg``/``ClientBundle``, never a literal."""
    # a HF-style repo id: <org>/<name>, where the name carries a model-ish marker
    model_id = re.compile(
        r"\b[A-Za-z0-9][\w.\-]*/[\w.\-]*"
        r"(qwen|bge|llama|mistral|gpt|deberta|nllb|opus|embedding|milco|plaid|m3)[\w.\-]*",
        re.IGNORECASE,
    )
    for path in MODULES:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            assert not model_id.search(line), f"{path.name}:{i} looks like a model id: {line!r}"


#: Decompose's judgements -- is this nugget on topic, is it a duplicate, has the bank
#: saturated -- are made by the one shared LLM and by arithmetic, never by a second,
#: separately-loaded scoring model. These substrings are the names such a model has gone
#: by here; banning them by name is what makes the guarantee fail loudly if one is wired
#: back in, rather than being restated in a comment nothing checks.
_SECOND_JUDGE_TOKENS = ("nli", "entail")


def test_ft_b27_on_topic_gate_admits_or_refuses_and_never_scores():
    """FT-B27: the grounding gate is an admission judgement made by the shared LLM.

    It must not delegate the call to a second scoring model, whether by importing one or
    by calling one through an object handed to it.
    """
    for name in _imported_modules(PKG / "on_topic.py"):
        for token in _SECOND_JUDGE_TOKENS:
            assert token not in name.lower(), name
    # ... and no such call sneaks in through an injected object
    code = (PKG / "on_topic.py").read_text(encoding="utf-8")
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in set(_SECOND_JUDGE_TOKENS), ast.dump(node)


def test_no_decompose_module_reaches_for_a_second_judgement_model():
    for path in MODULES:
        for name in _imported_modules(path):
            for token in _SECOND_JUDGE_TOKENS:
                assert token not in name.lower(), f"{path.name} imports {name}"


def test_ft_b33_dedup_never_imports_serving_registry():
    for name in _imported_modules(PKG / "dedup.py"):
        assert "serving" not in name, f"dedup.py imports {name}"
        assert "registry" not in name, f"dedup.py imports {name}"


def test_ft_b34_dedup_never_imports_sentence_transformers_statically():
    text = (PKG / "dedup.py").read_text(encoding="utf-8")
    for name in _imported_modules(PKG / "dedup.py"):
        assert "sentence_transformers" not in name
        # numpy is not a base dependency in this repo's uv.lock (it enters only via the
        # `chunk`/`heavy` extras), so importing it here would reproduce the very
        # platform split the sentence-transformers rule exists to prevent.
        assert name != "numpy" and not name.startswith("numpy.")
    assert "import sentence_transformers" not in text
    assert "from sentence_transformers" not in text


def test_ft_b34b_importing_dedup_pulls_in_no_heavy_library_dynamically():
    """FT-B34, the dynamic half: this catches a transitive import a grep would miss."""
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, ragtime.pipeline.decompose.dedup as d; "
                "print([m for m in ('sentence_transformers','torch','vllm','numpy') "
                "if m in sys.modules])"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]", out.stdout


def test_ft_b35_dedup_imports_nothing_from_select_serialize():
    """FT-B35: the two dedup policies stay structurally separate."""
    for name in _imported_modules(PKG / "dedup.py"):
        assert "select_serialize" not in name


def test_dedup_and_select_serialize_are_separate_policies_by_location():
    """The two dedup policies live in two places, and only the arithmetic is shared.

    ``select_serialize/dedup.py`` may import :func:`~ragtime.pipeline.decompose.dedup.cosine`,
    the one dot product in this repository. It must not import the coverage-loop policy
    ``dedup_nuggets``, which is greedy-incremental and LLM-gated and would make the terminal
    projection non-deterministic.
    """
    assert (PKG / "dedup.py").exists()
    terminal = PKG.parent / "select_serialize" / "dedup.py"
    assert terminal.exists(), "select and serialize shipped; the terminal dedup policy must exist"
    # Parsed, not grepped: the module's docstring names the policy it does not reuse, so a
    # substring check would fail on the explanation rather than on an import.
    tree = ast.parse(terminal.read_text(encoding="utf-8"), filename=str(terminal))
    borrowed = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("decompose.dedup")
        for alias in node.names
    }
    assert borrowed == {"cosine"}, borrowed


def test_no_module_builds_its_own_client():
    """One shared vLLM serves every stage: no module builds its own clients or a node."""
    for path in MODULES:
        text = path.read_text(encoding="utf-8")
        assert "build_clients" not in text, path.name
        for name in _imported_modules(path):
            assert not name.startswith("ragtime.serving.registry"), path.name
            assert not name.startswith("ragtime.serving.node"), path.name
