"""Config fairness over full data: the six committed configs are the data.

Runs ``load`` and ``validate`` over each read-only ``config/*.yml``, asserts the real
e2e and mlir families pass ``family_guard``, drives three documented in-memory
mutations (never editing a tracked file) that each raise a diagnostic
``FairnessError``, and pins ``all_hashes`` determinism across repeated loads.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from ragtime.config import (
    FairnessError,
    RunConfig,
    all_hashes,
    check,
    family_guard,
    load,
    shared_block_hash,
    top_level_blocks,
)
from ragtime.config.schema import SHARED_BLOCKS, ConfigError, validate

pytestmark = pytest.mark.full


def _family(paths: list[Path]) -> list[RunConfig]:
    return [load(p) for p in paths]


def _load_family_with_mutation(
    paths: list[Path], target_name: str, mutated_text: str, tmp_path: Path
) -> list[RunConfig]:
    """Load ``paths``, substituting a ``tmp_path`` copy carrying ``mutated_text`` for
    the file named ``target_name`` (the tracked file is never opened for writing)."""
    configs: list[RunConfig] = []
    for p in paths:
        if p.name == target_name:
            dest = tmp_path / target_name
            dest.write_text(mutated_text, encoding="utf-8")
            configs.append(load(dest))
        else:
            configs.append(load(p))
    return configs


# --------------------------------------------------------------------------- #
# load/validate over the real configs, cross-checked against config/README.md
# --------------------------------------------------------------------------- #
def test_load_validate_all_6_real_configs(real_e2e_paths, real_mlir_paths):
    for p in real_e2e_paths + real_mlir_paths:
        cfg = load(p)
        assert isinstance(cfg, RunConfig)


def test_every_shipped_config_runs_the_same_agentic_kind(real_e2e_paths, real_mlir_paths):
    """All six shipped configs are ``e2e_agentic``, the `mlir-*` family included.

    Task 2 is not a different pipeline: an `mlir-*` run decomposes, fans the same RAG loops and
    runs the same coverage-audit rounds as an `e2e-*` run, and moves only ``retrieval.index``.
    The family a config belongs to comes from its ``run.id`` (``fairness._family_of``), never
    from ``run.kind``, so one kind across both families is what the fairness gate expects.

    Worth pinning because the obvious guess is wrong: ``run.kind`` is a one-value enum, so
    Task 2 is served by the same agentic kind rather than by a batched pipeline of its own.
    """
    for p in list(real_e2e_paths) + list(real_mlir_paths):
        cfg = load(p)
        assert cfg.kind == "e2e_agentic", p.name
        assert cfg.status == "submitted", p.name


def test_knob_defaults_and_moves_per_readme(real_e2e_paths, real_mlir_paths):
    # mlir-*: passage_lang defaults to original (absent); retrieval_index is explicit.
    for p in real_mlir_paths:
        cfg = load(p)
        assert cfg.passage_lang == "original"
        assert cfg.retrieval_index in {"original", "omt", "omt_opus"}
    # Controlled e2e-*: the search knob is held at original; only passage_lang moves.
    for p in real_e2e_paths:
        cfg = load(p)
        assert cfg.retrieval_index == "original"
        assert cfg.passage_lang in {"original", "omt", "omt_opus"}


def test_seeds_per_kind_all_real_configs(real_e2e_paths, real_mlir_paths):
    """Every real config's `seeds` matches the schema's per-kind contract, not a literal.

    Asserted against `_KIND_SEEDS` rather than against a number. A hardcoded expectation
    cannot distinguish "the configs drifted from the contract", which is the defect this
    guards, from "the contract changed", which is a decision.
    """
    from ragtime.config.schema import _KIND_SEEDS

    for p in real_e2e_paths + real_mlir_paths:
        cfg = load(p)
        assert cfg.seeds == _KIND_SEEDS[cfg.kind], (
            f"{p.name}: seeds={cfg.seeds} but the schema pins {_KIND_SEEDS[cfg.kind]} for "
            f"kind {cfg.kind!r}"
        )


def test_languages_tuple_all_real_configs(real_e2e_paths, real_mlir_paths):
    for p in real_e2e_paths + real_mlir_paths:
        assert load(p).languages == ("zh", "en", "ru", "es")


def test_outputs_routing_matches_readme_table(real_e2e_paths, real_mlir_paths, readme_outputs_table):
    for p in real_e2e_paths + real_mlir_paths:
        cfg = load(p)
        got = [(o["task"], o["track"], o["path"], o["run_id"]) for o in cfg.outputs]
        assert got == readme_outputs_table[p.name]


# --------------------------------------------------------------------------- #
# family_guard: the core fairness invariant, over raw text
# --------------------------------------------------------------------------- #
def test_family_guard_real_e2e_family_ok(real_e2e_paths):
    assert family_guard(_family(real_e2e_paths)) is None


def test_family_guard_real_mlir_family_ok(real_mlir_paths):
    assert family_guard(_family(real_mlir_paths)) is None


def _divergent_e2e(real_e2e_paths, tmp_path):
    """e2e family with e2e-omt-weak's ``decomposition.policy`` mutated in an in-memory copy.

    A ``str.replace`` for a substring that no longer exists is a silent no-op, which would
    leave the "divergent" family identical to the real one, so any replacement target here
    is asserted present rather than assumed.
    """
    original = (Path(p) for p in real_e2e_paths if p.name == "e2e-omt-weak.yml")
    text = next(original).read_text(encoding="utf-8")
    assert 'policy: "dynamic coverage loop' in text, (
        "the mutation target vanished from the real config: pick a field that still exists, "
        "or this test silently stops checking anything"
    )
    mutated = text.replace('policy: "dynamic coverage loop', 'policy: "CHANGED-dynamic coverage loop')
    assert mutated != text
    return _load_family_with_mutation(real_e2e_paths, "e2e-omt-weak.yml", mutated, tmp_path)


def test_family_guard_shared_block_divergence_raises(real_e2e_paths, tmp_path):
    configs = _divergent_e2e(real_e2e_paths, tmp_path)
    with pytest.raises(FairnessError) as excinfo:
        family_guard(configs)
    msg = str(excinfo.value)
    assert "e2e" in msg and "decomposition" in msg and "e2e-omt-weak" in msg


def test_family_guard_second_knob_moved_raises(real_mlir_paths, tmp_path):
    text = next(Path(p) for p in real_mlir_paths if p.name == "mlir-original.yml").read_text(
        encoding="utf-8"
    )
    mutated = text + "passage_lang: omt\n"  # move Knob 2 on a controlled mlir run
    configs = _load_family_with_mutation(real_mlir_paths, "mlir-original.yml", mutated, tmp_path)
    with pytest.raises(FairnessError) as excinfo:
        family_guard(configs)
    msg = str(excinfo.value)
    assert "mlir-original" in msg and "passage_lang" in msg


def test_family_guard_missing_optional_status_raises(real_e2e_paths, tmp_path):
    """Rule (c): a run moving BOTH knobs is admitted only when `status: optional`.

    The rule is general, and no shipped config exercises it: every run in the repository
    is controlled and moves one knob. The both-knobs branch is therefore reached from a
    synthetic fixture -- the controlled `e2e-omt` run, which already reads `omt`, mutated
    to search the `omt` index as well while keeping `status: submitted`. `retrieval` is
    not a shared block, so the mutation reaches the knob check instead of being caught
    earlier as a family divergence.
    """
    text = next(Path(p) for p in real_e2e_paths if p.name == "e2e-omt.yml").read_text(
        encoding="utf-8"
    )
    assert "\n  index: original" in text and "status: submitted" in text
    mutated = text.replace("\n  index: original", "\n  index: omt")  # move Knob 1 as well
    assert mutated != text
    configs = _load_family_with_mutation(real_e2e_paths, "e2e-omt.yml", mutated, tmp_path)
    with pytest.raises(FairnessError) as excinfo:
        family_guard(configs)
    msg = str(excinfo.value)
    assert "e2e-omt" in msg and "optional" in msg and "status" in msg


def test_family_guard_error_message_is_diagnostic(real_e2e_paths, tmp_path):
    configs = _divergent_e2e(real_e2e_paths, tmp_path)
    with pytest.raises(FairnessError) as excinfo:
        family_guard(configs)
    msg = str(excinfo.value)
    # Names both the offending run and the offending block, not a generic message.
    assert "e2e-omt-weak" in msg and "decomposition" in msg


def test_family_guard_shared_block_divergence_with_duplicate_run_id_raises(real_e2e_paths, tmp_path):
    # Two same-family members that share a run.id but genuinely diverge on a shared
    # block must still be caught; a run_id-keyed comparison would silently drop one.
    base = next(Path(p) for p in real_e2e_paths if p.name == "e2e-omt.yml").read_text(
        encoding="utf-8"
    )
    # Replacing a substring that does not exist is a silent no-op, so target a field
    # that still exists and assert it is present.
    assert 'policy: "dynamic coverage loop' in base
    mutated = base.replace('policy: "dynamic coverage loop', 'policy: "CHANGED-dynamic coverage loop')
    assert mutated != base
    a = tmp_path / "a.yml"
    b = tmp_path / "b.yml"
    a.write_text(base, encoding="utf-8")
    b.write_text(mutated, encoding="utf-8")
    cfg_a, cfg_b = load(a), load(b)
    assert cfg_a.run_id == cfg_b.run_id == "e2e-omt"  # same run.id, different shared block
    with pytest.raises(FairnessError) as excinfo:
        family_guard([cfg_a, cfg_b])
    assert "decomposition" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# The all_hashes key set is uniform across a family (retrieval is materialised)
# --------------------------------------------------------------------------- #
def test_all_hashes_key_set_uniform_across_e2e_family(real_e2e_paths):
    key_sets = {p.name: set(all_hashes(load(p))) for p in real_e2e_paths}
    first = next(iter(key_sets.values()))
    assert "retrieval" in first  # materialised, so the key exists for every member
    assert all(ks == first for ks in key_sets.values()), key_sets


# --------------------------------------------------------------------------- #
# all_hashes determinism across two independent loads of a real file
# --------------------------------------------------------------------------- #
def test_all_hashes_deterministic_across_two_loads(real_e2e_paths):
    path = next(p for p in real_e2e_paths if p.name == "e2e-omt.yml")
    assert all_hashes(load(path)) == all_hashes(load(path))


# --------------------------------------------------------------------------- #
# Drift pin between schema.py and the standalone commit-guard copies
#
# `SHARED_BLOCKS` exists in more than one copy by design: schema.py holds the
# authoritative list that feeds family_guard, and the standalone pre-commit tooling
# carries its own copy because it must not import project code. Copies drift, so this
# section parses the live text of the other copies and derives every expectation from
# `schema.SHARED_BLOCKS`; restating the names here would make this a further copy
# rather than a pin. The tooling is developer-local, so these tests skip when it is
# not present.
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOK_PATH = _REPO_ROOT / "tools" / "guard_commit.py"
_SKILL_PATH = _REPO_ROOT / "tools" / "fairness-check.md"

requires_guard_copies = pytest.mark.skipif(
    not (_HOOK_PATH.is_file() and _SKILL_PATH.is_file()),
    reason="the standalone commit-guard copies are developer-local and not distributed",
)

# Matches `SHARED_BLOCKS = [...]` (hook) or `SHARED = [...]` (skill recipe), single-line.
_LIST_ASSIGN = re.compile(r"^\s*(?:SHARED_BLOCKS|SHARED)\s*=\s*\[([^\]]*)\]", re.MULTILINE)


def _parse_shared_list(path: Path) -> list[str]:
    """Extract the single `SHARED[_BLOCKS] = [...]` literal from a file's live text.

    Raises if absent or duplicated: a renamed, removed or second declaration must fail
    loudly rather than let the pin silently match nothing.
    """
    text = path.read_text(encoding="utf-8")
    matches = _LIST_ASSIGN.findall(text)
    assert len(matches) == 1, (
        f"expected exactly one SHARED[_BLOCKS] list literal in {path}, found {len(matches)}; "
        f"the drift pin cannot locate the copy it is meant to check"
    )
    return re.findall(r"[\"']([^\"']+)[\"']", matches[0])


@requires_guard_copies
def test_shared_blocks_copies_agree_with_schema():
    """The standalone copies must equal schema.py's authoritative SHARED_BLOCKS."""
    authoritative = list(SHARED_BLOCKS)
    assert _parse_shared_list(_HOOK_PATH) == authoritative, (
        f"{_HOOK_PATH} drifted from the authoritative SHARED_BLOCKS in config/schema.py, "
        f"which feeds family_guard, the pre-launch gate"
    )
    assert _parse_shared_list(_SKILL_PATH) == authoritative, (
        f"{_SKILL_PATH} drifted from the authoritative SHARED_BLOCKS in config/schema.py, "
        f"which feeds family_guard, the pre-launch gate"
    )


@requires_guard_copies
def test_shared_block_names_appear_in_prose_of_hook_and_skill():
    """The human-facing block lists must name the same blocks as the code lists.

    A stale name in the prose a reader consults to learn the invariant misleads exactly
    when someone is debugging a fairness failure.
    """
    slashed = "/".join(SHARED_BLOCKS)
    # Every slash-joined run of names that mentions a shared block must be exactly the
    # authoritative list, which catches both a missing name and a phantom extra one;
    # a bare `slashed in text` would happily pass on ".../chunker/translation".
    run_re = re.compile(r"[A-Za-z_]+(?:/[A-Za-z_]+)+")
    for path in (_HOOK_PATH, _SKILL_PATH):
        text = path.read_text(encoding="utf-8")
        runs = [r for r in run_re.findall(text) if "claim_commit" in r.split("/")]
        assert runs, f"{path}: no human-facing shared-block list found"
        for got in runs:
            assert got == slashed, (
                f"{path}: prose lists {got!r} but the authoritative SHARED_BLOCKS is {slashed!r}"
            )


def _skill_top_level_blocks(text: str) -> dict[str, str]:
    """Verbatim reimplementation of the standalone recipe's top_level_blocks."""
    blocks: dict[str, str] = {}
    cur: list[str] = []
    key = None
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_][\w-]*):", line)
        is_top = m and not line.startswith((" ", "\t"))
        if is_top:
            if key is not None:
                blocks[key] = "\n".join(cur)
            key, cur = m.group(1), [line]
        elif key is not None:
            cur.append(line)
    if key is not None:
        blocks[key] = "\n".join(cur)
    return blocks


def _skill_shared_hash(text: str) -> str:
    blocks = _skill_top_level_blocks(text)
    h = hashlib.sha256()
    # The block list is parsed from the recipe's live text rather than restated here,
    # so this equivalence check exercises the recipe as actually written on disk.
    for name in _parse_shared_list(_SKILL_PATH):
        h.update(name.encode())
        h.update(b"\0")
        h.update(blocks.get(name, "").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


@requires_guard_copies
def test_shared_block_hash_matches_fairness_check_skill(real_e2e_paths, real_mlir_paths):
    for p in real_e2e_paths + real_mlir_paths:
        text = Path(p).read_text(encoding="utf-8")
        assert shared_block_hash(text) == _skill_shared_hash(text)
    # The recipe's per-family grouping agrees with family_guard passing:
    e2e_hashes = {_skill_shared_hash(Path(p).read_text(encoding="utf-8")) for p in real_e2e_paths}
    mlir_hashes = {_skill_shared_hash(Path(p).read_text(encoding="utf-8")) for p in real_mlir_paths}
    assert len(e2e_hashes) == 1 and len(mlir_hashes) == 1


# --------------------------------------------------------------------------- #
# `check` as a unit: load plus family_guard over both real families
# --------------------------------------------------------------------------- #
def test_check_composes_load_and_family_guard_over_real_families(
    real_e2e_paths, real_mlir_paths, tmp_path
):
    e2e_configs = check(real_e2e_paths)
    assert len(e2e_configs) == 3 and all(isinstance(c, RunConfig) for c in e2e_configs)
    mlir_configs = check(real_mlir_paths)
    assert len(mlir_configs) == 3

    # A family that includes a mutated fixture raises before any plan/GPU step.
    text = next(Path(p) for p in real_e2e_paths if p.name == "e2e-omt-weak.yml").read_text(
        encoding="utf-8"
    )
    # Without asserting the target's presence, a replace for a vanished substring
    # no-ops and the "mutated" family is byte-identical to the real one, so the test
    # would assert a raise from a perfectly valid family. Assert both the presence of
    # the target and that the mutation actually landed.
    assert 'policy: "dynamic coverage loop' in text
    mutated = text.replace('policy: "dynamic coverage loop', 'policy: "CHANGED-dynamic coverage loop')
    assert mutated != text
    dest = tmp_path / "e2e-omt-weak.yml"
    dest.write_text(mutated, encoding="utf-8")
    paths = [dest if p.name == "e2e-omt-weak.yml" else p for p in real_e2e_paths]
    with pytest.raises(FairnessError):
        check(paths)


# --------------------------------------------------------------------------- #
# The three semantic `.config` leaves are closed.
#
# A free-form leaf contradicts the rule that unknown keys are rejected so the file
# stays the complete run record, and it is precisely the blocks carrying the run's
# semantics that matter: an open `index_build.config` would accept, hash and ignore
# a key that reaches no encoder.
# --------------------------------------------------------------------------- #
@pytest.mark.full
@pytest.mark.parametrize(
    ("block", "typo"),
    [
        ("chunker", "tokn_budget"),          # typo of token_budget
        ("translation", "beem_size"),        # typo of beam_size
        ("index_build", "dense_mrl_dim"),    # a phantom knob: hashed, and read by no encoder
    ],
)
def test_semantic_config_leaves_reject_an_unknown_key(block, typo, real_e2e_paths):
    """A misspelled or phantom knob must fail at load, not be silently ignored.

    `dense_mrl_dim` is the concrete case: a key an open leaf would parse and fold into the
    dense leg's `leg_config_hash` while reaching no encoder, moving the recipe hash with no
    stored byte behind it. That is a false provenance record, and this check catches it at
    the config edit.
    """
    import copy

    from ragtime.config.loader import _YAML  # the same safe-loader `load` uses

    raw = _YAML.load(real_e2e_paths[0].read_text(encoding="utf-8"))
    mutated = copy.deepcopy(raw)
    assert typo not in mutated[block]["config"], "fixture must not already carry it"
    mutated[block]["config"][typo] = "anything"
    with pytest.raises(ConfigError, match=rf"{block}\.config\.{typo}"):
        validate(mutated)


@pytest.mark.full
def test_every_key_the_real_configs_use_is_declared(real_e2e_paths, real_mlir_paths):
    """The closed sets must cover the shipped configs, derived rather than restated.

    Guards the other direction from the test above: closing a leaf is only safe if the
    declared set is complete, so this asserts it against the six real files rather
    than against a second hand-written list that could drift from them.
    """
    from ragtime.config.loader import _YAML
    from ragtime.config.schema import (
        CHUNKER_CONFIG,
        INDEX_BUILD_CONFIG,
        TRANSLATION_CONFIG,
    )

    declared = {
        "chunker": set(CHUNKER_CONFIG),
        "translation": set(TRANSLATION_CONFIG),
        "index_build": set(INDEX_BUILD_CONFIG),
    }
    for path in list(real_e2e_paths) + list(real_mlir_paths):
        raw = _YAML.load(path.read_text(encoding="utf-8"))
        for block, allowed in declared.items():
            used = set(raw[block]["config"])
            assert used <= allowed, (
                f"{path.name}: {block}.config uses undeclared {sorted(used - allowed)}"
            )


def test_serialize_allowed_keys_equal_the_knob_fields():
    """`_ALLOWED["serialize"]` must equal `SerializeKnobs.__dataclass_fields__` exactly.

    A knob the stage can read but a config cannot write breaks the config-driven rule
    quietly: the value comes from the code's defaults, so the file stops being the
    complete run record while still looking like one.

    Both directions are asserted. A key allowed by the schema but absent from the
    dataclass is the worse failure: `SerializeKnobs.from_cfg` rejects unknown keys
    against its own fields, so such a key would load past `config` and then fail inside
    the stage, after the expensive work rather than before it.

    This lives in tests/ rather than in `schema.py` because the dependency spine forbids
    `config` from importing `pipeline`; the test is the only place both sides may be
    seen at once.
    """
    from ragtime.config.schema import _ALLOWED
    from ragtime.pipeline.select_serialize.knobs import SerializeKnobs

    allowed = set(_ALLOWED["serialize"])
    fields = set(SerializeKnobs.__dataclass_fields__)
    assert allowed == fields, (
        f"serialize schema/knob drift: writable-but-unreadable={sorted(allowed - fields)}, "
        f"readable-but-unwritable={sorted(fields - allowed)}"
    )


# --------------------------------------------------------------------------- #
# `topics`: every run names the request set, and it is one set per family
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (_REPO_ROOT / "topics").is_dir(),
    reason="the TREC topics file is organiser data and is not distributed with this repo",
)
def test_all_6_real_configs_name_the_same_existing_topics_file(real_e2e_paths, real_mlir_paths):
    """The run record must name its requests, and the named file must actually load.

    `family_guard` already proves byte-identity within each family; this asserts the
    stronger project-level property no other check covers: all six runs, across both
    families, read one request set.
    """
    from ragtime.common import load_topics

    paths = {p.name: load(p).topics_path for p in real_e2e_paths + real_mlir_paths}
    assert len(paths) == 6
    assert len(set(paths.values())) == 1, f"runs disagree on the request set: {paths}"

    topics_file = _REPO_ROOT / next(iter(set(paths.values())))
    assert topics_file.is_file(), f"{topics_file} named by every config does not exist"
    topics = load_topics(topics_file)
    assert len(topics) == 103
    assert all(t.title.strip() for t in topics)


def test_topics_block_is_byte_identical_across_both_families(real_e2e_paths, real_mlir_paths):
    """Not implied by `family_guard`, which never compares across families."""
    bodies = {
        p.name: top_level_blocks(p.read_text(encoding="utf-8"))["topics"]
        for p in real_e2e_paths + real_mlir_paths
    }
    assert len(set(bodies.values())) == 1, f"`topics` block text differs: {sorted(bodies)}"
