"""The ranking path in ``devkit.rsvc`` must equal the one in ``preprocess.index``.

``devkit.rsvc.merge_parts`` and ``preprocess.index.query_lang_leg`` are two implementations of one
piece of ranking arithmetic: the same per-part ``search_with_rep``, the same raw-score merge across
parts, the same total ``(-score, passage_id)`` order, the same per-part top-k. The pipeline calls
``query_lang_leg``; the retrieval service calls ``merge_parts``. If the two diverge, every number
measured through the service describes ranking the pipeline never executes, and nothing downstream
would report it.

The fixture is the small published index from ``tests/retrieval/conftest.py``: three legs across
four language cells. That proves the two agree on the arithmetic, and nothing about corpus scale,
where a cell carries 14-23 parts rather than a handful. FL-R14 covers the production fan at that
size.

Both sides are handed the same ``rep``, computed the way production computes it
(``handle.parts[0].leg_impl(leg).encode_query(handle.ctx, query)``), because ``merge_parts`` takes
a pre-computed representation while ``query_lang_leg`` encodes internally. Letting each side encode
its own query would confound an encoder difference with a merge difference.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragtime.preprocess.index import LEGS, query_lang_leg

# The `built` fixture is re-exported by tests/devkit/conftest.py, not imported here: pytest does
# not apply tests/retrieval/conftest.py to this directory, and importing the fixture into this
# module would be shadowed by every test's own `built` parameter (ruff F811).
from tests.retrieval.conftest import Built, context_for, retrieval_cfg

pytestmark = pytest.mark.small

#: Queries chosen to hit real fixture text rather than a single lucky term.
_QUERIES = ("dogs", "el perro", "government policy", "the")
_TOP_K = 10


def _rep_for(handle: Any, leg: str, query: str) -> Any:
    """The two lines production runs inside ``query_lang_leg`` to encode a query."""
    impl = handle.parts[0].leg_impl(leg)
    return impl.encode_query(handle.ctx, query)


def _cells(built: Built) -> dict[str, Any]:
    ctx = context_for(built, retrieval_cfg(built.root))
    return ctx.cells


@pytest.mark.parametrize("query", _QUERIES)
def test_dedup0_merge_parts_equals_query_lang_leg_on_every_leg_and_cell(
    built: Built, query: str
) -> None:
    """The two implementations return the identical ranked list, element for element.

    Not "the same ids", not "the same set", not "the top hit agrees": the same list, with the
    same scores in the same order. A weaker assertion here would let a reordering survive, and a
    reordering is precisely what a duplicated merge is able to introduce.
    """
    from ragtime.devkit.rsvc import merge_parts

    cells = _cells(built)
    assert cells, "fixture published no language cells"

    compared = 0
    for leg in LEGS:
        for lang, handle in sorted(cells.items()):
            rep = _rep_for(handle, leg, query)
            production = query_lang_leg(handle, leg, query, _TOP_K)
            service = merge_parts(handle, leg, rep, _TOP_K, workers=1, omp=0)
            assert service == production, (
                f"RANKING DIVERGENCE leg={leg} lang={lang} query={query!r}:\n"
                f"  preprocess.index.query_lang_leg -> {production}\n"
                f"  devkit.rsvc.merge_parts         -> {service}\n"
                "The service's docstring claims these are byte-identical. They are not, so every "
                "latency/quality number measured through the service describes ranking that the "
                "pipeline does not execute."
            )
            compared += 1
    assert compared == len(LEGS) * len(cells), "not every (leg, cell) was compared"


def test_dedup0_the_part_fan_width_changes_nothing_in_the_service_path(built: Built) -> None:
    """``merge_parts`` at any ``workers`` equals ``query_lang_leg``.

    SM-R21 asserts the production fan is width-invariant against itself. This asserts the service's
    fan is width-invariant against production, which is what one implementation standing in for the
    other rests on: if the two agree only at ``workers=1``, swapping them silently changes results
    for every fanned configuration, which is every real one.
    """
    from ragtime.devkit.rsvc import merge_parts

    cells = _cells(built)
    query = _QUERIES[0]
    for leg in LEGS:
        for lang, handle in sorted(cells.items()):
            rep = _rep_for(handle, leg, query)
            expected = query_lang_leg(handle, leg, query, _TOP_K)
            for workers in (1, 2, 4, 8):
                got = merge_parts(handle, leg, rep, _TOP_K, workers=workers, omp=0)
                assert got == expected, (
                    f"leg={leg} lang={lang} workers={workers}: the service's part fan is not "
                    "result-identical to production's."
                )


def test_dedup0_the_omp_cap_does_not_change_the_result(built: Built) -> None:
    """``omp`` is a thread-count knob and must not be able to move a single score.

    ``merge_parts`` applies the FAISS cap per task, because the cap is a per-thread OpenMP variable
    and setting it on the submitter does not bind pool threads. ``query_lang_leg`` has no equivalent
    call. If the cap is ever ported into it, this test is what says the port changed CPU usage and
    not ranking.
    """
    from ragtime.devkit.rsvc import merge_parts

    cells = _cells(built)
    query = _QUERIES[0]
    for leg in LEGS:
        for lang, handle in sorted(cells.items()):
            rep = _rep_for(handle, leg, query)
            expected = query_lang_leg(handle, leg, query, _TOP_K)
            for omp in (0, 1, 2):
                got = merge_parts(handle, leg, rep, _TOP_K, workers=2, omp=omp)
                assert got == expected, f"leg={leg} lang={lang} omp={omp} moved the ranking"
