"""Rebuild a passage of the RAGTIME 2026 collection in any of its three renderings.

The release publishes a passage as an ordered list of sentence ids, not as text. This
module is the reference implementation of the two composition rules that turn that list
back into a string, plus a streaming reader that applies them to a whole language split
without a join.

The two rules are different, and using the wrong one produces text that is nearly right:

    native      the member sentences concatenated, with a single space inserted only
                where the source had a gap between them
    translated  the member sentences' translations joined with a single space,
                unconditionally, for every source language

Only the three functions at the top are load-bearing; everything below them is reading
convenience. They take plain Python values, so they work equally well with `datasets`,
`pandas`, `pyarrow` or hand-built rows.

Run this file to rebuild a few passages from a local copy of the release:

    passage_text.py /path/to/release --split zho --documents 3
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "LANG_OF_SPLIT",
    "SPLIT_OF_LANG",
    "document_id_of",
    "iter_passages",
    "native_text",
    "rebuild",
    "translated_text",
]

#: The parent collection names its splits with three-letter codes; the tables carry the
#: two-letter code in `lang`. The correspondence is exact and one-to-one.
LANG_OF_SPLIT = {"eng": "en", "spa": "es", "rus": "ru", "zho": "zh"}
SPLIT_OF_LANG = {v: k for k, v in LANG_OF_SPLIT.items()}

_JOINER = " "


# --------------------------------------------------------------------------- #
# The two composition rules.
# --------------------------------------------------------------------------- #
def native_text(members: Sequence[tuple[int, int, str]]) -> str:
    """The passage in its own language, from its member sentences.

    ``members`` is one ``(char_start, char_end, text)`` per member sentence, in the order
    ``passages.sentence_ids`` gives them -- that is, the `sentences` rows named by the
    passage, left in the table's own order.

    A passage is a contiguous run of its document, so the only thing between two members
    is whatever separated them in the source. ``char_end`` of one member and ``char_start``
    of the next say which case applies: equal means the sentences abut and nothing goes
    between them, greater means the source had a separator and a single space stands in
    for it. Chinese sentences abut about 40 per cent of the time, which is why a plain
    space-join is wrong here and why the offsets are in the release at all.

    The offsets are relative information only. They index our normalised document text,
    which is not published, so their absolute values mean nothing outside this comparison.
    """
    if not members:
        return ""
    parts = [members[0][2]]
    for prev, cur in zip(members, members[1:], strict=False):
        if cur[0] > prev[1]:
            parts.append(_JOINER)
        parts.append(cur[2])
    return "".join(parts)


def translated_text(segments: Sequence[str]) -> str:
    """The passage in either English rendering, from its member translations.

    A single space between every pair, for every source language, with no reference to
    the offsets. The strings being joined are English whatever the source was, so a
    per-language joiner here would be a category error.
    """
    return _JOINER.join(segments)


def rebuild(
    passage: Mapping[str, Any],
    sentences: Mapping[str, Mapping[str, Any]],
    translations: Mapping[str, Mapping[str, str]] | None = None,
    *,
    identity_lang: str = "en",
) -> dict[str, str]:
    """One passage in every rendering asked for.

    ``passage`` is a `passages` row. ``sentences`` maps ``sentence_id`` to a `sentences`
    row; it must cover every member. ``translations`` maps a rendering name (any name you
    like -- ``"nllb"``, ``"opus"``) to a mapping from ``sentence_id`` to translated text;
    pass ``None`` for the native rendering alone.

    An English passage has no translation rows, because English is not translated. Its
    English renderings are its native text, and that is what is returned. A passage some
    of whose members are missing from a translation table is a broken copy of the release,
    not a language edge case, so it raises.
    """
    ids = list(passage["sentence_ids"])
    try:
        rows = [sentences[sid] for sid in ids]
    except KeyError as exc:
        raise KeyError(
            f"{passage['passage_id']}: member sentence {exc.args[0]!r} is not in the "
            "sentence rows given for this passage"
        ) from None
    out = {
        "native": native_text(
            [(int(r["char_start"]), int(r["char_end"]), r["text"]) for r in rows]
        )
    }
    for name, table in (translations or {}).items():
        segments = [table.get(sid) for sid in ids]
        if all(s is None for s in segments):
            if passage["lang"] != identity_lang:
                raise KeyError(
                    f"{passage['passage_id']}: rendering {name!r} has no rows at all "
                    f"for this {passage['lang']!r} passage. Only {identity_lang!r} "
                    "resolves to its native text; every other language must be "
                    "covered, and falling back here would hand you untranslated "
                    "source text under a translated name"
                )
            out[name] = out["native"]  # English source: not translated
            continue
        missing = [sid for sid, s in zip(ids, segments, strict=True) if s is None]
        if missing:
            raise KeyError(
                f"{passage['passage_id']}: rendering {name!r} covers "
                f"{len(ids) - len(missing)} of {len(ids)} member sentences "
                f"(first missing: {missing[0]!r}); a partial passage must never fall "
                "back to the native text"
            )
        out[name] = translated_text([str(s) for s in segments])
    return out


def document_id_of(ref_id: str) -> str:
    """The document a passage id or a sentence id belongs to.

    Both suffixes are first-level and no document id contains a ``#``, so one split on
    the first ``#`` recovers the document id from either.
    """
    return ref_id.split("#", 1)[0]


# --------------------------------------------------------------------------- #
# Reading a whole split without a join.
# --------------------------------------------------------------------------- #
def iter_passages(
    root: str | Path,
    split: str,
    *,
    renderings: Sequence[str] = ("nllb", "opus"),
    documents: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Every passage of one split, with all its renderings composed.

    ``root`` is a directory holding the release's config folders -- ``sentences/``,
    ``passages/``, ``translations_nllb/``, ``translations_opus/`` -- as downloaded from the
    Hub. ``split`` is one of ``eng``, ``spa``, ``rus``, ``zho``.

    All four tables are written in the same document order and each document's rows are
    contiguous, so this walks them side by side and holds one document at a time. There is
    no hash join over a hundred million ids and no sort, and memory does not grow with the
    size of the split. ``documents`` stops after that many documents, which is what makes
    a first look cheap.

    Yields the `passages` row's own fields plus one string per rendering, under ``native``
    and under each name in ``renderings``.
    """
    import pyarrow.parquet as pq

    def shards(config: str) -> list[Path]:
        return sorted(Path(root, config).glob(f"{split}-*.parquet"))

    def rows(files: Sequence[Path], columns: Sequence[str]) -> Iterator[Mapping[str, Any]]:
        for path in files:
            handle = pq.ParquetFile(path)
            for batch in handle.iter_batches(batch_size=8192, columns=list(columns)):
                yield from batch.to_pylist()

    def required(config: str) -> list[Path]:
        files = shards(config)
        if not files:
            raise FileNotFoundError(f"no {split} shards under {Path(root, config)}")
        return files

    variant_config = {"nllb": "translations_nllb", "opus": "translations_opus"}
    sent = _Peek(rows(required("sentences"),
                      ("document_id", "sentence_id", "char_start", "char_end", "text")))
    pas = _Peek(rows(required("passages"),
                     ("document_id", "passage_id", "lang", "sentence_ids",
                      "token_count", "is_oversized")))
    # A rendering with no shards for this split is empty, not absent: English is not
    # translated, so the eng split of both translation configs does not exist, and its
    # English rendering is its native text. `rebuild` makes that substitution only when
    # the passage is English, so a genuinely missing non-English shard still raises.
    trs: dict[str, _Peek | None] = {}
    for name in renderings:
        files = shards(variant_config[name])
        if not files and split != "eng":
            raise FileNotFoundError(
                f"no {split} shards under {Path(root, variant_config[name])}; only the "
                "eng split of a translation config is legitimately absent"
            )
        columns = ("document_id", "sentence_id", "text")
        trs[name] = _Peek(rows(files, columns)) if files else None

    seen = 0
    while True:
        head = pas.peek()
        if head is None:
            return
        document_id = head["document_id"]
        passages = pas.take(document_id)
        sentences = {r["sentence_id"]: r for r in sent.take(document_id)}
        tables = {
            n: ({r["sentence_id"]: r["text"] for r in c.take(document_id)} if c else {})
            for n, c in trs.items()
        }
        for row in passages:
            record = dict(row)
            record.update(rebuild(row, sentences, tables))
            yield record
        seen += 1
        if documents is not None and seen >= documents:
            return


class _Peek:
    """A forward-only row stream, taken one ``document_id`` at a time."""

    __slots__ = ("_head", "_rows")

    def __init__(self, rows: Iterable[Mapping[str, Any]]) -> None:
        self._rows = iter(rows)
        self._head: Mapping[str, Any] | None = None

    def peek(self) -> Mapping[str, Any] | None:
        if self._head is None:
            self._head = next(self._rows, None)
        return self._head

    def take(self, document_id: str) -> list[Mapping[str, Any]]:
        """This table's rows for one document. Empty is normal, not an error."""
        out: list[Mapping[str, Any]] = []
        while True:
            head = self.peek()
            if head is None or head["document_id"] != document_id:
                return out
            out.append(head)
            self._head = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild passages from a local release.")
    parser.add_argument("root", help="directory holding the release's config folders")
    parser.add_argument("--split", default="zho", choices=sorted(LANG_OF_SPLIT))
    parser.add_argument("--documents", type=int, default=2)
    parser.add_argument("--chars", type=int, default=200)
    args = parser.parse_args(argv)

    for record in iter_passages(args.root, args.split, documents=args.documents):
        print(f"\n{record['passage_id']}  lang={record['lang']}  "
              f"sentences={len(record['sentence_ids'])}  "
              f"tokens={record['token_count']}  oversized={record['is_oversized']}")
        for name in ("native", "nllb", "opus"):
            if name in record:
                print(f"  {name:<7} {record[name][:args.chars]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
