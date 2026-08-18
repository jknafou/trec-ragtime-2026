"""Round 0 is computed once per (topic, seed) and read by every arm of the family.

This is a correctness mechanism whose side effect happens to be large. The runtime proof of the
fairness invariant is that the round-0 seed bank is byte-identical across a run family. Round 0
is retrieval-free by construction, since `pipeline/decompose/` imports no retrieval client at
all, and reads only the report request plus the three fairness-shared blocks `decomposition`,
`llm` and `topics`, so neither translation knob can reach it and two arms cannot legitimately
differ.

Recomputing it once per arm would make that invariant a hope about sampled decoding, and the hope
is unfounded: the same seed produced a different round-0 bank once a second client shared the
instance, and the fan puts the three arms on different pairs with different batching. Computing
it once and reading it three times makes the invariant true by construction.

The saving is the side effect: 3 e2e arms x 515 cells collapse from 1545 decompositions to 515,
about 43 GPU-hours at the measured ~150 s per seed decomposition.

Not covered here:
  - that a real decompose produces a good bank (the decompose tests own that)
  - cross-node `os.link` semantics on beegfs under real contention
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragtime.common import Layout, Nugget
from ragtime.common.io import is_done
from ragtime.pipeline.decompose.bank import (
    publish_seed_bank,
    read_bank,
    seed_bank_hash,
    write_bank,
)


def _bank(*questions: str) -> tuple[Nugget, ...]:
    return tuple(
        Nugget(nugget_id=f"2000#n{i}", question=q, weight=0.5)
        for i, q in enumerate(questions)
    )


# --------------------------------------------------------------------------- #
# The cache key: what may and may not partition it
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_every_shipped_config_resolves_to_ONE_seed_bank_identity() -> None:
    """The whole saving rests on this: all six configs must agree.

    They differ in `passage_lang` and `retrieval.index`, the two translation knobs, and agree on
    `decomposition`/`llm`/`topics`, which are fairness-shared. If this ever splits, some arm is
    leaking an arm-specific value into the key, the fleet silently recomputes 1030 extra banks
    (~43 GPU-h), and nothing reports it: the run is simply slower.
    """
    from ragtime.config import load

    hashes = {p.name: seed_bank_hash(load(str(p))) for p in sorted(Path("config").glob("*.yml"))}
    assert len(hashes) >= 6, f"expected the 6 shipped configs, saw {sorted(hashes)}"
    assert len(set(hashes.values())) == 1, (
        f"the seed-bank key partitioned by arm: {hashes}. Round 0 is retrieval-free and reads only "
        "shared blocks, so every config must resolve to one identity."
    )


@pytest.mark.small
def test_a_decomposition_change_DOES_split_the_key() -> None:
    """The complement, and the reason the key is a hash and not a constant.

    A cache that never invalidates is worse than none: it would serve banks from a superseded
    decomposition prompt to a run that declares a different one, and the artifact would look
    perfectly well-formed.
    """
    from ragtime.config import load

    real = load("config/e2e-original.yml")
    before = seed_bank_hash(real)

    # `cfg.blocks` is a mappingproxy, so the loaded config is immutable and the probe builds a
    # stand-in rather than mutating the record. `seed_bank_hash` reads only `.blocks`.
    class _Cfg:
        def __init__(self, blocks):
            self.blocks = blocks

    probed = dict(real.blocks)
    probed["decomposition"] = {**probed["decomposition"], "_probe": "changed"}
    assert seed_bank_hash(_Cfg(probed)) != before

    # ... and a change outside the three keyed blocks must not split it, which is what lets the
    # three arms share, since they differ precisely in `passage_lang` and `retrieval`.
    untouched = dict(real.blocks)
    untouched["retrieval"] = {**untouched["retrieval"], "index": "omt"}
    assert seed_bank_hash(_Cfg(untouched)) == before


# --------------------------------------------------------------------------- #
# First writer wins: the property the whole mechanism turns on
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_the_SECOND_publisher_does_not_overwrite_and_is_told_the_winner(tmp_path: Path) -> None:
    """Two arms reach one (topic, seed) at once, and under concurrent decoding their banks need
    not be equal.

    Last writer wins would make the canonical bank depend on scheduling order, and an arm that
    had already read the earlier version would be working from a bank the artifact no longer
    shows: a fairness violation with no trace. So the loser discards its own work and is handed
    the winner's, and both arms proceed on identical nuggets.
    """
    path = tmp_path / "seed0.jsonl"
    first = _bank("what did the ministry announce?")
    second = _bank("an entirely different question")

    got_first = publish_seed_bank(path, first)
    got_second = publish_seed_bank(path, second)

    assert [n.question for n in got_first] == [n.question for n in first]
    assert got_second == got_first, "the loser must be handed the winner's bank, not its own"
    on_disk = read_bank(path)
    assert on_disk == first, "the artifact must still be the FIRST bank published"


@pytest.mark.small
def test_publishing_marks_the_artifact_done_so_readers_trust_it(tmp_path: Path) -> None:
    """`is_done`, the `_SUCCESS` companion, is what `_round_zero` checks. A complete file with no
    marker would be recomputed by every arm forever, so the cache would be inert and silent."""
    path = tmp_path / "seed0.jsonl"
    assert not is_done(path)
    publish_seed_bank(path, _bank("q"))
    assert is_done(path)


@pytest.mark.small
def test_the_loser_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    """515 cells x 3 arms means the loser path runs ~1030 times. A leaked temp per loss would
    litter the shared tree with files that look like artifacts."""
    path = tmp_path / "seed0.jsonl"
    publish_seed_bank(path, _bank("a"))
    publish_seed_bank(path, _bank("b"))
    strays = [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]
    assert strays == [], f"temp files left behind: {strays}"


@pytest.mark.small
def test_the_published_bank_round_trips_every_field(tmp_path: Path) -> None:
    """`publish_seed_bank` serialises independently of `write_bank`, because it needs link
    semantics, so the two writers can drift. A field lost here is one later rounds never see."""
    path_pub = tmp_path / "pub.jsonl"
    path_wri = tmp_path / "wri.jsonl"
    bank = _bank("q0", "q1")
    publish_seed_bank(path_pub, bank)
    write_bank(path_wri, bank)
    assert read_bank(path_pub) == read_bank(path_wri) == bank
    # ... and the on-disk JSON agrees key for key, so a reader of either file sees one record.
    pub_rows = [json.loads(x) for x in path_pub.read_text().splitlines() if x.strip()]
    wri_rows = [json.loads(x) for x in path_wri.read_text().splitlines() if x.strip()]
    assert [sorted(r) for r in pub_rows] == [sorted(r) for r in wri_rows]


# --------------------------------------------------------------------------- #
# The path
# --------------------------------------------------------------------------- #
@pytest.mark.small
def test_the_shared_path_is_keyed_by_content_not_by_run_or_family(tmp_path: Path) -> None:
    """Two different runs must land on the same file, which is the whole mechanism. Keying by
    family or run id would give each arm its own copy and reuse nothing, while still looking like
    a cache."""
    layout = Layout(tmp_path, base=tmp_path)
    a = layout.seed_bank("d" * 64, "2000", 0)
    b = layout.seed_bank("d" * 64, "2000", 0)
    assert a == b
    assert a != layout.seed_bank("d" * 64, "2000", 1), "seeds must not collide"
    assert a != layout.seed_bank("e" * 64, "2000", 0), "a key change must give a fresh subtree"
    assert "decompose_seeds" in a.parts and str(tmp_path) in str(a)
