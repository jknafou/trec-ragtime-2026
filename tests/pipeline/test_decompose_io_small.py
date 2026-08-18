"""The round-0 bank artifact: a write and read round-trip through ``Layout``.

The producer/consumer contract this pins: decompose writes ``decompose/round_0.jsonl`` via
``common.Layout`` and ``common.io.write_jsonl``, and both the round loop (on resume) and
select-and-serialize read exactly that shape back. The output is a set and is
order-independent, so the round-trip is asserted on the set of records rather than on the
tuple order.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from ragtime.common import Answer, Layout, Nugget, Retrieved, Support, read_jsonl
from ragtime.common.io import is_done
from ragtime.pipeline.decompose import bank as bank_ops
from ragtime.pipeline.decompose import grow_nuggets

from .conftest import SEED_QUESTIONS, make_bundle, nuggets_response

pytestmark = pytest.mark.small


def test_ft_b46_round0_bank_write_read_roundtrip_via_layout(
    tmp_path, decompose_cfg, seed_topic
):
    clients = make_bundle(
        nuggets=[
            nuggets_response(SEED_QUESTIONS),
            nuggets_response(SEED_QUESTIONS, weights=[0.9, 0.7, 0.4, 0.2]),
        ]
    )
    bank = asyncio.run(
        grow_nuggets(
            seed_topic.problem_statement,
            seed_topic.background,
            (),
            None,
            0,
            cfg=decompose_cfg,
            clients=clients,
            topic_id=seed_topic.topic_id,
            limit=seed_topic.limit,
            seed=0,
        )
    )

    layout = Layout(run_dir=tmp_path / "e2e-original" / "original" / "seed0")
    path = layout.decompose_round(0)
    assert path.name == "round_0.jsonl"
    assert path.parent.name == "decompose"

    bank_ops.write_bank(path, bank)
    assert path.exists()
    assert is_done(path), "atomic write + _SUCCESS marker (the artifact tree IS the checkpoint)"

    back = bank_ops.read_bank(path)
    assert {dataclasses.astuple(n) for n in back} == {dataclasses.astuple(n) for n in bank}
    assert bank_ops.bank_fingerprint(back) == bank_ops.bank_fingerprint(bank)


def test_written_rows_carry_the_full_io_schema_field_set(tmp_path):
    bank = (
        Nugget(
            nugget_id="2000#n0",
            question="Which agency issued the recall?",
            weight=0.9,
            vital=True,
        ),
    )
    path = Layout(run_dir=tmp_path).decompose_round(0)
    bank_ops.write_bank(path, bank)
    row = read_jsonl(path)[0]
    assert set(row) == {
        "nugget_id",
        "question",
        "weight",
        "vital",
        "aggregator_type",
        "status",
        "origin_round",
        "trigger_passage_id",
        "retrieved",
        "answers",
    }
    assert row["trigger_passage_id"] is None
    assert row["retrieved"] == [] and row["answers"] == []


def test_roundtrip_survives_the_nested_round_ge1_records(tmp_path):
    """A round-≥1-shaped nugget round-trips too: the round loop resumes from this same file."""
    grown = Nugget(
        nugget_id="2000#n3",
        question="Which chemicals were released?",
        weight=0.8,
        vital=True,
        status="answered",
        origin_round=2,
        trigger_passage_id="spa-docs/0451820#p2",
        retrieved=(Retrieved(passage_id="spa-docs/0451820#p2", score=18.4),),
        answers=(
            Answer(
                answer="chlorine",
                sentence="Chlorine was released by the fire.",
                score=0.91,
                references={"spa-docs/0451820": 0.91},
                support=(Support(passage_id="spa-docs/0451820#p2", lang="es"),),
            ),
        ),
    )
    path = Layout(run_dir=tmp_path).decompose_round(2)
    bank_ops.write_bank(path, (grown,))
    assert bank_ops.read_bank(path) == (grown,)


def test_write_bank_is_skip_if_done_idempotent(tmp_path):
    path = Layout(run_dir=tmp_path).decompose_round(0)
    bank_ops.write_bank(path, (Nugget(nugget_id="t#n0", question="Q?"),))
    first = path.read_bytes()
    # a re-launch must not rewrite a completed artifact
    bank_ops.write_bank(path, (Nugget(nugget_id="t#n9", question="different?"),))
    assert path.read_bytes() == first


def test_serialization_uses_asdict_not_a_missing_dunder_dict():
    """A slotted frozen dataclass has no ``__dict__``: ``asdict`` is the only path."""
    n = Nugget(nugget_id="t#n0", question="Q?")
    assert not hasattr(n, "__dict__")
    with pytest.raises(AttributeError):
        _ = n.__dict__
    assert dataclasses.asdict(n)["question"] == "Q?"
