"""Shared fixtures for the common-kernel tests.

- ``passage_store``: a small synthetic three-rendering store across en/es/ru/zh,
  including a precomposed-vs-decomposed ``é`` passage and a CJK / full-width case,
  used to exercise the NFC-symmetry commit path and render totality.
- ``small_topics_path`` / ``small_topics``: a three-topic slice of the canonical
  topics file, re-concatenated onto one line with no separators, so the TREC
  single-line quirk the loader absorbs stays exercised.
- ``real_topics_path``: the full canonical request set, read by the full tier.
- ``superseded_topics_path``: the pre-fix file, kept byte-exact, used only to show
  that ``load_topics`` refuses a request set with no ``title``.

The last two resolve under ``topics/``. That file is published by the organisers and
is not carried here; ``docs/REPRODUCE.md`` says where to put it.

The command-line options for the full tier are declared here rather than in a
per-package conftest: pytest only honours ``pytest_addoption`` in an initial
conftest, and this file is an ancestor of every test in the repo.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from ragtime.common import CANONICAL_TOPICS_REL, PassageStore, load_topics

_FIXTURES = Path(__file__).parent / "fixtures"
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Real doc-ids never contain '#'; these mirror the two forms in the docs.
_DOC_EN = "eng-docs/0007421"
_DOC_ES = "spa-docs/0451820"
_DOC_RU = "rus-docs/1793347"
_DOC_ZH = "zho-docs/2288104"

# A Spanish token with the accented e: precomposed 'é' (U+00E9) in the original
# native text against its decomposed form (e + U+0301) used elsewhere. The two are
# NFC-equal, which is what the verbatim commit gate relies on.
_CAFE_PRECOMPOSED = "café"  # 'é' as U+00E9
_CAFE_DECOMPOSED = unicodedata.normalize("NFD", _CAFE_PRECOMPOSED)  # 'e' + U+0301
# A full-width digit: NFC keeps it distinct, NFKC folds it to ASCII '2'.
_FULLWIDTH = "２ g"  # U+FF12 followed by " g"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Declare the command-line options the full tier accepts.

    ``--query-encode-device`` selects the device for the full tier's dense query
    encoder. It sets ``index_build.config.query_encode_device`` on an in-memory copy
    of the shipped config, so the tier exercises one code path with a parameter
    rather than a test-only branch. Running the encoder on CPU is a different arm,
    not a cheaper way to run the shipped one: CPU and CUDA forward passes do not
    produce the same vector, and decompose's paraphrase dedup compares that vector
    against a fixed cosine cutoff.

    ``--seed-sweep`` selects how many (topic, seed) cells the decompose parity sweep
    covers.
    """
    parser.addoption(
        "--query-encode-device",
        action="store",
        default="",
        help=(
            "device for the full tier's dense query encoder (decompose dedup): '' = as "
            "configured (shipped: cuda), 'cpu' = run the tier as a CPU client. A CPU run is "
            "a different arm, not a cheaper way to run the shipped one."
        ),
    )
    parser.addoption(
        "--seed-sweep",
        action="store",
        default="gate",
        choices=("gate", "full"),
        help=(
            "How many (topic, seed) cells the decompose parity sweep covers. "
            "'gate' (default) -- the two cells that add a property the fairness anchor does "
            "not already prove: a different topic and a different seed. (0,0) is excluded "
            "because it is the anchor's own cell. About four real decompositions. "
            "'full' -- every topic x every configured seed (20 real decompositions, ~50 min), "
            "the pre-launch sweep, run once before a scored launch and never per gate. "
            "The tier prints which arm it selected."
        ),
    )


@pytest.fixture
def passage_records() -> list[dict]:
    """Raw records for the synthetic three-rendering passage store.

    ``sentence_char_spans`` mirrors the chunker's output contract: one ``(start, end)``
    per ``sentence_ids`` entry, in the same order, as offsets into the ``original``
    (passage) text. The ``#p1`` record is multi-sentence, so the store round-trip is
    exercised with more than one span.
    """
    en_p1_s0 = "Poles engage the upper body."
    en_p1_s1 = "The technique started in Finland."
    return [
        {
            "passage_id": f"{_DOC_EN}#p0",
            "document_id": _DOC_EN,
            "lang": "en",
            "original": "Nordic walking uses specially designed poles.",
            "sentence_ids": [f"{_DOC_EN}#p0#s0"],
            "sentence_char_spans": [[0, len("Nordic walking uses specially designed poles.")]],
            "token_count": 8,
        },
        {
            "passage_id": f"{_DOC_EN}#p1",
            "document_id": _DOC_EN,
            "lang": "en",
            "original": f"{en_p1_s0} {en_p1_s1}",
            "sentence_ids": [f"{_DOC_EN}#p1#s0", f"{_DOC_EN}#p1#s1"],
            # second span starts one char after the first ends (the single-space joiner)
            "sentence_char_spans": [
                [0, len(en_p1_s0)],
                [len(en_p1_s0) + 1, len(en_p1_s0) + 1 + len(en_p1_s1)],
            ],
            "token_count": 11,
        },
        {
            "passage_id": f"{_DOC_ES}#p2",
            "document_id": _DOC_ES,
            "lang": "es",
            # native (original) carries the precomposed accent
            "original": f"El {_CAFE_PRECOMPOSED} contiene {_FULLWIDTH} de cafeína.",
            "omt_opus": "The coffee contains 2 g of caffeine.",
            "omt": "Coffee has 2 g of caffeine.",
            "sentence_ids": [f"{_DOC_ES}#p2#s0"],
            "sentence_char_spans": [
                [0, len(f"El {_CAFE_PRECOMPOSED} contiene {_FULLWIDTH} de cafeína.")]
            ],
            "token_count": 9,
        },
        {
            "passage_id": f"{_DOC_RU}#p0",
            "document_id": _DOC_RU,
            "lang": "ru",
            "original": "Отчёт о законодательстве об электронных сигаретах.",
            "omt_opus": "A report on e-cigarette legislation.",
            "omt": "Report about electronic cigarette laws.",
            "sentence_ids": [f"{_DOC_RU}#p0#s0"],
            "sentence_char_spans": [
                [0, len("Отчёт о законодательстве об электронных сигаретах.")]
            ],
            "token_count": 7,
        },
        {
            "passage_id": f"{_DOC_ZH}#p5",
            "document_id": _DOC_ZH,
            "lang": "zh",
            "original": "躺平运动在中国年轻人中很流行。",
            "omt_opus": "The lying-flat movement is popular among Chinese youth.",
            "omt": "Lying flat is popular with young people in China.",
            "sentence_ids": [f"{_DOC_ZH}#p5#s0"],
            "sentence_char_spans": [[0, len("躺平运动在中国年轻人中很流行。")]],
            "token_count": 6,
        },
    ]


@pytest.fixture
def passage_store(passage_records: list[dict]) -> PassageStore:
    return PassageStore.from_records(passage_records)


@pytest.fixture
def cafe_forms() -> tuple[str, str, str]:
    """(precomposed, decomposed, fullwidth-digit) for the normalize tests."""
    return _CAFE_PRECOMPOSED, _CAFE_DECOMPOSED, _FULLWIDTH


@pytest.fixture
def small_topics_path() -> Path:
    return _FIXTURES / "topics_small.jsonl"


@pytest.fixture
def small_topics(small_topics_path: Path):
    return load_topics(small_topics_path)


@pytest.fixture
def real_topics_path() -> Path:
    """The canonical request set, named once in ``common.topics``."""
    return _REPO_ROOT / CANONICAL_TOPICS_REL


@pytest.fixture
def superseded_topics_path() -> Path:
    """The superseded request set: kept byte-exact, and no longer loadable (no ``title``)."""
    return _REPO_ROOT / "topics" / "topics.all.2026.v0625.jsonl"
