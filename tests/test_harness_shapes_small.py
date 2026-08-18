"""Suite-wide tripwires for harness defects: four rules, in seconds, with no index or GPU.

These are the generic half of the fast-twin layer. Each rule encodes one failure that costs a
30-40 minute corpus-scale load when it is discovered at run time, and each is answerable by
reading the test source, so it is answered here instead:

1. :func:`test_no_monkeypatch_replacement_re_looks_up_the_name_it_replaced`. A replacement
   installed by ``monkeypatch.setattr(mod, "f", repl)`` must not contain ``mod.f`` in its own
   body: once the patch is applied that name is ``repl``, so the spy self-calls forever.
2. :func:`test_no_skip_guard_probes_a_private_attribute`. A ``try`` whose handler calls
   ``pytest.skip`` must not probe a private attribute in its body: private entry points
   vanish in refactors, the broad ``except`` swallows the ``AttributeError``, and the gate
   reports a skip that blames something else while the test silently does not run.
3. :func:`test_every_fixture_a_test_requests_is_reachable_from_its_directory`. A test that
   requests a fixture from a conftest pytest does not apply to its directory errors at setup,
   which looks red while executing nothing.
4. :func:`test_every_attribute_the_full_tier_patches_still_exists_on_its_target`, the same
   class one step earlier: ``monkeypatch.setattr`` does refuse a vanished target, but only
   when the line runs, which in the full tier is after the expensive fixture has been paid for.

Rules 1 to 3 are static, using ``ast`` over the test sources: nothing is imported, nothing is
collected, no fixture runs. Rule 4 imports the ``*_full.py`` modules, which are import-safe by
construction and cost about two seconds in total, and resolves their patch targets. Either way
the rules cover the full tier, which is the tier that cannot afford to discover any of this at
run time.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.small

TESTS_ROOT = Path(__file__).resolve().parent

#: pytest's own fixtures, plus this repo's plugins (there are none beyond core).
_BUILTIN_FIXTURES = frozenset(
    {
        "cache",
        "capfd",
        "capfdbinary",
        "caplog",
        "capsys",
        "capsysbinary",
        "capteesys",
        "doctest_namespace",
        "monkeypatch",
        "pytestconfig",
        "record_property",
        "record_testsuite_property",
        "record_xml_attribute",
        "recwarn",
        "request",
        "testdir",
        "tmp_path",
        "tmp_path_factory",
        "tmpdir",
        "tmpdir_factory",
    }
)


def _test_modules() -> list[Path]:
    return sorted(p for p in TESTS_ROOT.rglob("test_*.py") if "__pycache__" not in p.parts)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _where(path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(TESTS_ROOT.parent)}:{getattr(node, 'lineno', '?')}"


# --------------------------------------------------------------------------- #
# Rule 1: a spy may not re-look-up the name it replaced.
# --------------------------------------------------------------------------- #
def _patches(tree: ast.Module):
    """Yield ``(call, target_src, attr_name, [replacement nodes])`` for every attribute patch.

    Two shapes are recognised, because both install a callable under a name that already exists:

    * ``monkeypatch.setattr(target, "name", replacement)``, the raw form;
    * ``spy_through(monkeypatch, target, "name", before=..., after=...)``, the helper form,
      whose hook bodies run inside the installed spy and can recurse in the same way.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "setattr"
            and isinstance(func.value, ast.Name)
            and "monkeypatch" in func.value.id
            and len(node.args) >= 3
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            yield node, ast.unparse(node.args[0]), node.args[1].value, [node.args[2]]
        elif (
            isinstance(func, ast.Name)
            and func.id == "spy_through"
            and len(node.args) >= 3
            and isinstance(node.args[2], ast.Constant)
            and isinstance(node.args[2].value, str)
        ):
            hooks = [kw.value for kw in node.keywords if kw.arg in ("before", "after")]
            yield node, ast.unparse(node.args[1]), node.args[2].value, hooks


def _resolve_replacement(tree: ast.Module, replacement: ast.expr) -> ast.AST | None:
    """The node whose body becomes the patched attribute: a lambda, or a named def."""
    if isinstance(replacement, ast.Lambda):
        return replacement
    if isinstance(replacement, ast.Name):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name == replacement.id
            ):
                return node
    return None  # a partial, an attribute or a call: nothing static to inspect


def test_no_monkeypatch_replacement_re_looks_up_the_name_it_replaced() -> None:
    """A replacement's body must not contain ``<target>.<name>``: that name is the replacement.

    A counting spy that delegates with ``service_mod.query_lang_leg(...)`` after
    ``monkeypatch.setattr(service_mod, "query_lang_leg", counting_query)`` has rebound that
    attribute re-enters itself on every call, and the run dies with ``RecursionError`` only
    once the corpus-scale rendering has been loaded. The same defect is visible here in a
    fraction of a second with no index at all.

    The correct shape, which passes, is to capture the real function into a local before
    patching, or to use :func:`tests.harness.spy_through`, which captures it by construction,
    so the body names a local rather than the module attribute.
    """
    offenders: list[str] = []
    for path in _test_modules():
        tree = _parse(path)
        for call, target_src, name, replacements in _patches(tree):
            bodies = [b for b in (_resolve_replacement(tree, r) for r in replacements) if b]
            for node in (n for body in bodies for n in ast.walk(body)):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, ast.Load)
                    and node.attr == name
                    and ast.unparse(node.value) == target_src
                ):
                    offenders.append(
                        f"{_where(path, call)}: the replacement for {target_src}.{name} reads "
                        f"{target_src}.{name} at line {node.lineno}; after the patch that name "
                        "is the replacement, so this self-calls forever. Capture the original "
                        "into a local before patching, or use tests.harness.spy_through."
                    )
    assert offenders == [], "\n".join(offenders)


# --------------------------------------------------------------------------- #
# Rule 2: a skip guard may not probe a private attribute.
# --------------------------------------------------------------------------- #
def _handler_skips(handler: ast.ExceptHandler) -> bool:
    for node in ast.walk(handler):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "skip"
        ):
            return True
    return False


def test_no_skip_guard_probes_a_private_attribute() -> None:
    """``try: <probe> / except ...: pytest.skip(...)`` must not probe a ``_private`` attribute.

    A reranker skip guard once called the private ``reranker._cross_encoder()`` inside such a
    guard. The method had been deleted when the CrossEncoder path became
    ``AutoModelForCausalLM``, the broad ``except Exception`` swallowed the ``AttributeError``,
    and the gate reported a labelled skip blaming a config leaf that was correct. The
    integration test for the change did not run, and the gate still returned zero.

    A public entry point is not immune to deletion, but it is part of the contract the code
    owes its callers, while a private one can vanish in any refactor. So probe the public API,
    resolve the attribute outside the ``try`` so an ``AttributeError`` surfaces as the suite
    failure it is, and keep the ``except`` for genuine runtime failures.
    """
    offenders: list[str] = []
    for path in _test_modules():
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.Try) or not any(
                _handler_skips(h) for h in node.handlers
            ):
                continue
            for stmt in node.body:
                for inner in ast.walk(stmt):
                    if (
                        isinstance(inner, ast.Attribute)
                        and inner.attr.startswith("_")
                        and not inner.attr.startswith("__")
                    ):
                        offenders.append(
                            f"{_where(path, inner)}: a skip guard probes the private "
                            f"{ast.unparse(inner)}; if it is deleted, the broad except turns a "
                            "real AttributeError into a vacuous skip. Probe the public entry "
                            "point, outside the try."
                        )
    assert offenders == [], "\n".join(offenders)


# --------------------------------------------------------------------------- #
# Rule 3: every requested fixture is reachable from the requesting directory.
# --------------------------------------------------------------------------- #
def _is_fixture(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for deco in node.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        if isinstance(target, ast.Attribute) and target.attr == "fixture":
            return True
        if isinstance(target, ast.Name) and target.id == "fixture":
            return True
    return False


def _module_level_names(tree: ast.Module) -> set[str]:
    """Every name a module binds: defs, imports, assignments.

    A fixture imported into a conftest, such as the re-export in ``tests/devkit/conftest.py``,
    is registered by pytest exactly as if it were defined there, so an imported name counts.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def _parametrized(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Argnames supplied by ``@pytest.mark.parametrize``: parameters, never fixtures."""
    out: set[str] = set()
    for deco in node.decorator_list:
        if not (isinstance(deco, ast.Call) and deco.args):
            continue
        target = deco.func
        if not (isinstance(target, ast.Attribute) and target.attr == "parametrize"):
            continue
        argnames = deco.args[0]
        if isinstance(argnames, ast.Constant) and isinstance(argnames.value, str):
            out |= {n.strip() for n in argnames.value.split(",") if n.strip()}
        elif isinstance(argnames, ast.List | ast.Tuple):
            out |= {
                e.value.strip()
                for e in argnames.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
    return out


def _argnames_of_parametrize_call(call: ast.Call) -> set[str]:
    """Argnames from one ``parametrize`` call's first argument (a str or a list/tuple of str)."""
    out: set[str] = set()
    if not call.args:
        return out
    argnames = call.args[0]
    if isinstance(argnames, ast.Constant) and isinstance(argnames.value, str):
        out |= {n.strip() for n in argnames.value.split(",") if n.strip()}
    elif isinstance(argnames, ast.List | ast.Tuple):
        out |= {
            e.value.strip()
            for e in argnames.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        }
    return out


def _generated_names(tree: ast.Module) -> set[str]:
    """Argnames a module supplies via ``pytest_generate_tests`` and ``metafunc.parametrize``.

    This is the second legitimate way pytest supplies an argname. Without it, rule 3 fires on
    any argname that is neither a fixture nor a decorator parameter but is supplied
    dynamically at collection, which is a false positive: the tempting response is to rename
    the argument until the tripwire stops complaining, which leaves the next dynamic
    parameterization to trip it again and teaches the reader that the rule cries wolf.

    Module-wide and conservative: any non-decorator ``*.parametrize(...)`` call in
    the module contributes its argnames. Over-accepting here can only cause a miss, meaning an
    unreachable fixture goes unreported, never a false block, and rule 3's job is to catch the
    accident of requesting a fixture from the wrong directory rather than to police
    parameterization style.
    """
    decorators = {
        deco
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        for deco in node.decorator_list
    }
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or node in decorators:
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "parametrize":
            out |= _argnames_of_parametrize_call(node)
    return out


def _conftest_names(directory: Path) -> set[str]:
    """Fixture names visible in ``directory`` from every conftest on the path up to ``tests/``."""
    names: set[str] = set()
    current = directory
    while True:
        conftest = current / "conftest.py"
        if conftest.exists():
            names |= _module_level_names(_parse(conftest))
        if current == TESTS_ROOT:
            break
        current = current.parent
    return names


def test_every_fixture_a_test_requests_is_reachable_from_its_directory() -> None:
    """A test's parameters must resolve where the test lives, not where the fixture was written.

    pytest does not apply one package's conftest to a sibling, so a test under
    ``tests/devkit/`` that requests a fixture defined in ``tests/retrieval/conftest.py`` errors
    at setup: the run reads as red while executing nothing. The fix is a re-export in the
    requesting directory's own conftest, and this rule keeps the next such request from being
    written.

    Static and conservative: a name is reachable if the module binds it, whether by a def or an
    import, if any conftest from the test's own directory up to ``tests/`` binds it, or if it
    is a pytest builtin. ``@pytest.mark.parametrize`` argnames are excluded, since they are
    parameters rather than fixtures.
    """
    offenders: list[str] = []
    for path in _test_modules():
        tree = _parse(path)
        available = (
            _module_level_names(tree)
            | _conftest_names(path.parent)
            | _BUILTIN_FIXTURES
            | _generated_names(tree)  # pytest_generate_tests -> metafunc.parametrize
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            params = _parametrized(node)
            wanted = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
            for arg in wanted:
                if arg in params or arg in available or arg == "self":
                    continue
                offenders.append(
                    f"{_where(path, node)}: {node.name} requests fixture {arg!r}, which no "
                    f"conftest from {path.parent.name}/ up to tests/ defines and this module "
                    "does not import; every test in the file would error at setup while the "
                    "run looks merely red."
                )
    assert offenders == [], "\n".join(offenders)


# --------------------------------------------------------------------------- #
# Rule 4: every attribute the full tier patches still exists on its target.
# --------------------------------------------------------------------------- #
_UNRESOLVED = object()


def _resolve_object(module: object, dotted: str) -> object:
    """``service_mod`` or ``IndexAdapter`` resolved in the test module's own namespace."""
    current: Any = module
    for part in dotted.split("."):
        if not part.isidentifier():
            return _UNRESOLVED
        try:
            current = getattr(current, part)
        except AttributeError:
            return _UNRESOLVED
    return current


def _raising_is_off(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "raising" and isinstance(kw.value, ast.Constant):
            return kw.value.value is False
    return False


def test_every_attribute_the_full_tier_patches_still_exists_on_its_target() -> None:
    """A ``-m full`` test may not learn on the cluster that its patch target was renamed.

    ``monkeypatch.setattr`` defaults to ``raising=True``, so a renamed target is caught, but
    only when the line executes, which for the full tier is after the session fixture has built
    or opened the thing the test needs. ``tests/preprocess/test_index_full.py`` patches
    ``IndexAdapter.load_shard`` after the session-scoped ``built`` fixture has driven shards
    through the real encoder checkpoints, and ``tests/retrieval/test_retrieval_full.py``
    patches ``service_mod.query_lang_leg`` after a corpus rendering is resident.

    Resolving the same targets by import costs about two seconds, so it is done here instead.
    Targets imported inside a test body cannot be resolved from the module namespace and are
    skipped; the count of targets actually checked is asserted, so this rule cannot quietly
    degrade into checking nothing.
    """
    import importlib

    checked = 0
    offenders: list[str] = []
    for path in sorted(TESTS_ROOT.rglob("*_full.py")):
        if "__pycache__" in path.parts:
            continue
        tree = _parse(path)
        patches = [p for p in _patches(tree) if not _raising_is_off(p[0])]
        if not patches:
            continue
        module_name = ".".join(path.relative_to(TESTS_ROOT.parent).with_suffix("").parts)
        module = importlib.import_module(module_name)
        for call, target_src, name, _replacements in patches:
            target = _resolve_object(module, target_src)
            if target is _UNRESOLVED:
                continue
            checked += 1
            if not hasattr(target, name):
                offenders.append(
                    f"{_where(path, call)}: patches {target_src}.{name}, which no longer exists "
                    f"on {target!r}; monkeypatch would raise, but only once the full tier's "
                    "expensive fixture has already been paid for."
                )
    assert offenders == [], "\n".join(offenders)
    assert checked >= 2, f"only {checked} full-tier patch targets were resolvable; rule inert"


def test_the_three_rules_above_cover_every_test_module() -> None:
    """The rules are only worth anything if they actually read the files they claim to.

    A glob that silently matches nothing is its own vacuous green, so the census is asserted:
    every ``-m full`` module that loads an index or a model must be among the parsed set.
    """
    modules = {p.relative_to(TESTS_ROOT).as_posix() for p in _test_modules()}
    assert len(modules) > 40, modules
    for expected in (
        "retrieval/test_retrieval_full.py",
        "retrieval/test_query_fan_small.py",
        "preprocess/test_index_full.py",
        "serving/test_serving_full.py",
    ):
        assert expected in modules, f"{expected} is not being checked by the harness rules"
