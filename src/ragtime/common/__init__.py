"""Shared kernel: primitives every other package depends on.

Unicode normalization, ID algebra, the nugget/answer record schemas, rank fusion,
the statistics counter bus, structured logging, artifact IO, the run-directory
layout and the three-rendering passage store. This package imports nothing else
from ``ragtime``; every other package imports from here.
"""

from __future__ import annotations

from .fusion import maxp_collapse, rrf
from .ids import doc_id_of, nugget_id, passage_id, sentence_id
from .io import (
    iter_parquet,
    lmdb_open,
    read_jsonl,
    read_parquet,
    write_jsonl,
    write_parquet,
    write_parquet_stream,
)
from .layout import Layout
from .logging import StructuredLogger, get_logger
from .normalize import nfc, nfkc_len
from .passage_store import LmdbPassageStore, PassageStore
from .schemas import (
    Answer,
    Document,
    MetricRecord,
    Nugget,
    Passage,
    Retrieved,
    Sentence,
    Support,
    T1ReportRow,
    T1Response,
    T2RunRow,
    T3Answer,
    T3Nugget,
    T3NuggetBankRow,
    Translation,
    TranslationRaw,
    accumulate_answers,
    document_arrow_schema,
    final_passage_arrow_schema,
    merge_map_arrow_schema,
    passage_arrow_schema,
    sentence_arrow_schema,
    sentence_remap_arrow_schema,
    translation_final_arrow_schema,
    translation_raw_arrow_schema,
)
from .stats import Statistics, flush
from .topics import CANONICAL_TOPICS_REL, Topic, TopicsError, load_topics

__all__ = [  # noqa: RUF022 - grouped by module rather than sorted
    # normalize
    "nfc",
    "nfkc_len",
    # ids
    "doc_id_of",
    "passage_id",
    "sentence_id",
    "nugget_id",
    # fusion
    "rrf",
    "maxp_collapse",
    # stats
    "Statistics",
    "flush",
    # io
    "read_jsonl",
    "write_jsonl",
    "read_parquet",
    "write_parquet",
    "iter_parquet",
    "write_parquet_stream",
    "lmdb_open",
    # layout
    "Layout",
    # passage store
    "PassageStore",
    "LmdbPassageStore",
    # logging
    "get_logger",
    "StructuredLogger",
    # topics
    "Topic",
    "TopicsError",
    "load_topics",
    "CANONICAL_TOPICS_REL",
    # schemas
    "Document",
    "Sentence",
    "Passage",
    "Translation",
    "TranslationRaw",
    "document_arrow_schema",
    "sentence_arrow_schema",
    "passage_arrow_schema",
    "merge_map_arrow_schema",
    "translation_raw_arrow_schema",
    # schemas - reconciled final corpus
    "final_passage_arrow_schema",
    "translation_final_arrow_schema",
    "sentence_remap_arrow_schema",
    "Retrieved",
    "Support",
    "Answer",
    "Nugget",
    "MetricRecord",
    "T1Response",
    "T1ReportRow",
    "T2RunRow",
    "T3Answer",
    "T3Nugget",
    "T3NuggetBankRow",
    "accumulate_answers",
]
