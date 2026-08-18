"""The runtime fairness anchor: round-0 banks must be identical across variants.

``config.fairness.family_guard`` checks the config-time precondition, that the shared ``llm`` and
``decomposition`` blocks are byte-identical across the family and the seed counts match. It cannot
check the claim itself, because the literal round-0 bank does not exist until round 0 runs. This
module is that missing half.

The claim is that round 0 reads only ``problem_statement`` and ``background`` and no passages, so
the three renderings at one seed must produce the same question set. A difference would confound
the whole translation comparison, since the nugget bank is the dependent variable, so this raises
rather than warning and continuing.

Comparison is by ``bank.bank_fingerprint`` (question set plus ``aggregator_type``,
order-independent) rather than tuple equality: the artifact is a set, and ``weight``'s sampled
float is outside the anchor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragtime.common import Nugget, get_logger

from .bank import bank_fingerprint

if TYPE_CHECKING:
    from ragtime.common import Statistics

__all__ = ["ROUND0_BANK_HASH", "SEED_PARITY_OK", "SeedParityError", "assert_seed_parity"]

_log = get_logger("pipeline.decompose.fairness_anchor")

#: A numeric witness of the round-0 bank hash (see :func:`bank_hash_counter`).
ROUND0_BANK_HASH = "decompose.round0_bank_hash"
#: 1.0 once a variant set has been proven to share one round-0 bank hash.
SEED_PARITY_OK = "decompose.seed_parity_ok"

#: 13 hex digits is 52 bits, the exact float64 mantissa width, so the projection below is loss-free
#: into the counter bus and two variants' witnesses compare exactly.
_HASH_COUNTER_HEX = 13


class SeedParityError(RuntimeError):
    """Round-0 banks diverged across variants at one seed."""


def bank_hash_counter(fingerprint: str) -> float:
    """Project a 64-hex fingerprint onto a counter value the ``Statistics`` bus can hold.

    ``Statistics`` holds float values only, so the hash itself cannot be emitted. The leading 13
    hex digits fit float64's mantissa exactly, so the witness is an exact integer and two variants
    that emit the same number really did produce the same fingerprint prefix. The full fingerprint
    goes to the structured log.
    """
    return float(int(fingerprint[:_HASH_COUNTER_HEX], 16))


def assert_seed_parity(
    banks_by_variant: dict[str, tuple[Nugget, ...]],
    *,
    seed: int | None = None,
    stats: Statistics | None = None,
) -> str:
    """Assert every variant's round-0 bank shares one fingerprint, and return it.

    Raises :class:`SeedParityError` naming the diverging variant and the first nugget whose
    question is not in the reference variant's question set.

    A divergence does not by itself prove that variant state leaked into the seed. Sampled
    decoding is not batch-invariant, so a co-tenant on the shared vLLM is enough to make the same
    seed produce a different bank; rule that out before hunting for leaked variant state. The
    design consequence is that decompose must not share its instance with concurrent work.

    A single-variant or empty mapping is accepted and returns its fingerprint: parity over one
    variant is vacuously true, and refusing it would make the check unusable from a per-cell run
    that only holds its own bank.
    """
    if not banks_by_variant:
        raise ValueError("assert_seed_parity needs at least one variant's round-0 bank")

    fingerprints = {v: bank_fingerprint(b) for v, b in banks_by_variant.items()}
    if stats is not None:
        slices = {} if seed is None else {"seed": seed}
        for variant, fp in fingerprints.items():
            stats.emit(ROUND0_BANK_HASH, bank_hash_counter(fp), variant=variant, **slices)

    ref_variant = min(fingerprints)
    ref_fp = fingerprints[ref_variant]
    ref_questions = {n.question for n in banks_by_variant[ref_variant]}

    for variant in sorted(fingerprints):
        if fingerprints[variant] == ref_fp:
            continue
        raise SeedParityError(
            "round-0 seed banks diverged across variants at one seed; the seed reads no "
            "passages, so the translation comparison may be confounded. "
            f"{ref_variant}={ref_fp[:12]} vs "
            f"{variant}={fingerprints[variant][:12]}; "
            + _divergence(
                ref_variant,
                ref_questions,
                variant,
                banks_by_variant[variant],
            )
        )

    if stats is not None:
        slices = {} if seed is None else {"seed": seed}
        stats.emit(SEED_PARITY_OK, 1.0, **slices)
    _log.info(
        "decompose.seed_parity_ok",
        variants=sorted(fingerprints),
        bank_hash=ref_fp,
        seed=seed,
    )
    return ref_fp


def _divergence(
    ref_variant: str,
    ref_questions: set[str],
    variant: str,
    other_bank: tuple[Nugget, ...],
) -> str:
    """Name the diverging nuggets on both sides, since the reference side is arbitrary.

    Reporting only questions present in the other variant but not the reference would stay silent
    whenever the divergence is a dropped nugget rather than an added one, and which bank is the
    reference is merely alphabetical, so both directions are listed.
    """
    other_questions = {n.question for n in other_bank}
    only_other = [n for n in other_bank if n.question not in ref_questions]
    only_ref = sorted(ref_questions - other_questions)
    parts: list[str] = []
    if only_other:
        parts.append(f"only in {variant}: {only_other[0].nugget_id}={only_other[0].question!r}")
    if only_ref:
        parts.append(f"only in {ref_variant}: {only_ref[0]!r}")
    if not parts:
        parts.append("same question set but a different aggregator_type or bank size")
    return "; ".join(parts)
