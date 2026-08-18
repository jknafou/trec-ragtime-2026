"""Verbatim NFC span-commit: the loop's only grounding gate.

A claim commits only if its quoted span is an exact substring of the passage text the model was
shown, after applying ``common.nfc`` symmetrically to both sides. There is no fuzzy threshold and
no similarity score, only exact string identity: the check is deterministic and language-neutral,
which is what makes it defensible across zh, en, ru and es without a per-language calibration nobody
could validate.

Three things this module must never do:

* Repair a span. A near-miss is rejected and the claim dropped, after bounded retry at the loop
  level. Trimming whitespace, snapping to a word boundary or taking the closest match would mean
  the committed evidence is not what the model asserted, and the citation would be ours.
* Use ``nfkc_len``. That is the Task-1 character-budget normalizer, and it is lossy in exactly the
  way a grounding check must not be: NFKC folds ligatures, width variants and compatibility forms,
  so two visibly different spans compare equal. The two normalizers are never conflated.
* Let the model supply the doc-id. It is derived from the passage id via ``common.doc_id_of``, so
  a citation always resolves to the original document whatever rendering was read or searched. The
  compiled action schema carries no ``doc_id`` field at all, so the id cannot be wrong because it
  cannot be stated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ragtime.common import doc_id_of, nfc

__all__ = ["CommittedClaim", "Rejection", "commit_span"]


#: Typographic variants a model silently corrects while copying, which ``nfc`` does not unify:
#: NFC composes accents, it does not fold dashes, quotes or exotic spaces. Used only to locate a
#: near-miss so the rejection can quote the true text back, never to accept one.
_TYPO_FOLD = str.maketrans(
    {
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
        "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
        "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
        # Invisible by definition, so written as escapes: NBSP, narrow NBSP, figure space,
        # thin space, zero-width space, BOM. A literal here would be unreviewable.
        "\u00a0": " ", "\u202f": " ", "\u2007": " ", "\u2009": " ",
        "\u200b": " ", "\ufeff": " ",
        "…": ".",
    }
)


def _fold(text: str) -> tuple[str, list[int]]:
    """Fold typography and collapse whitespace, keeping a map back to ``text`` offsets.

    Returns ``(folded, index_map)`` where ``index_map[i]`` is the offset in ``text`` of the
    character that produced ``folded[i]``. The map is what makes this usable for feedback: a
    near-match found in folded space can be reported as the original passage substring, character
    for character, rather than as our normalized rewrite of it.
    """
    out: list[str] = []
    idx: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        c = ch.translate(_TYPO_FOLD)
        if c.isspace():
            if prev_space:
                continue
            c, prev_space = " ", True
        else:
            prev_space = False
        out.append(c.casefold())
        idx.append(i)
    return "".join(out), idx


def _near_miss_hint(span: str, shown_text: str, *, max_quote: int = 300) -> str:
    """Name the defect behind a near-miss, so a retry has something new to work with.

    A rejection that only repeats the rule ("quote exactly") leaves the model with no new
    information, so its retry re-decodes the same near-miss; observed live at more than a dozen
    identical attempts on one span, because a span slip does not cost a turn of
    ``max_iters``.

    When the span differs from the passage only in typography or whitespace, this locates it and
    quotes the passage's actual characters back. That is feedback, not repair: the claim is still
    rejected and the model must re-submit the correct text. Nothing here can cause a commit, since
    ``commit_span`` has already decided.
    """
    folded_text, idx = _fold(nfc(shown_text))
    folded_span, _ = _fold(nfc(span))
    if not folded_span:
        return ""
    at = folded_text.find(folded_span)
    if at < 0:
        return (
            " Your span does not appear in this passage even ignoring case, whitespace and "
            "punctuation style, so it is not a copying slip: re-read the passage above and quote "
            "a sentence that is actually in it."
        )
    start, end = idx[at], idx[at + len(folded_span) - 1] + 1
    actual = shown_text[start:end]
    if len(actual) > max_quote:  # never let a hint dwarf the passage it is about
        actual = actual[:max_quote] + "..."
    return (
        " You are quoting the right sentence but not its exact characters (case, whitespace, or "
        "a dash/quote/space variant differ). The passage reads, character for character: "
        f"{actual!r} -- submit EXACTLY that."
    )


@dataclass(frozen=True, slots=True)
class CommittedClaim:
    """One grounded claim. ``doc_id`` is derived, never taken from the model.

    ``answer``, the short value that feeds Task 3, and ``sentence``, the full citeable English
    report sentence that feeds Task 1, are carried here because they are emitted with the claim in
    the same action, so every distinct value has its own short form and its own grounding span.
    There is no second "write the answer" generation.
    """

    claim: str
    span: str
    passage_id: str
    doc_id: str
    lang: str  # the passage's native language, for the per-language support cut
    answer: str = ""  # short value, for Task 3
    sentence: str = ""  # full citeable report sentence, for Task 1
    #: True when the model omitted `answer` or `sentence` and the claim text stood in, so the
    #: substitution is countable rather than silent.
    value_fallback: bool = False


@dataclass(frozen=True, slots=True)
class Rejection:
    """Why a claim did not commit. Carried so the loop can tell the model what went wrong."""

    reason: str
    detail: str

    def feedback(self) -> str:
        return f"claim NOT committed ({self.reason}): {self.detail}"


def commit_span(
    *,
    claim: str,
    span: str,
    passage_id: str,
    shown_text: str | None,
    shown_ids: Sequence[str] = (),
    native_lang: str = "",
    answer: str = "",
    sentence: str = "",
) -> CommittedClaim | Rejection:
    """Ground one claim, or say precisely why it did not ground.

    ``shown_text`` is the text the model was shown for ``passage_id``, or ``None`` when it cited a
    passage it never received. Checking against what was shown rather than re-fetching from the
    passage store is deliberate and stricter: it makes both "cited a passage it never saw" and
    "quoted the original when it read a translation" fail, with no extra check.
    """
    if not span:
        return Rejection("empty_span", "no quoted span was supplied")
    if not claim:
        return Rejection("empty_claim", "a span was quoted but no claim was stated")
    if not passage_id:
        return Rejection("no_passage_id", "no passage was cited")
    if shown_text is None:
        # Name the format rather than only refusing. The dominant sub-case of `unseen_passage` is
        # the model citing the document id where the citable handle is the passage, `<doc>#p<N>`,
        # and a message that only restates the rule does not tell a model that quoted the right
        # document that its id is one suffix short.
        hint = ""
        if "#p" not in passage_id:
            near = [i for i in shown_ids if i.split("#")[0] == passage_id]
            hint = (
                " You cited a DOCUMENT id; a claim must cite a PASSAGE id, which ends in "
                "'#p<N>'."
                + (f" For that document you were shown: {', '.join(sorted(near)[:6])}."
                   if near else "")
            )
        elif shown_ids:
            hint = f" Passages you were shown: {', '.join(sorted(set(shown_ids))[:8])}."
        return Rejection(
            "unseen_passage",
            f"passage_id={passage_id!r} was never returned to you by a search in this loop; "
            f"you may only cite passages you have actually been shown.{hint}",
        )

    # Symmetric normalization: both sides, same function, same direction. Normalizing only the
    # span, or only the passage, is a one-way comparison that passes on composed input and fails
    # on decomposed input, which would make the check behave differently by language.
    if nfc(span) not in nfc(shown_text):
        return Rejection(
            "not_verbatim",
            f"the quoted span is not an exact substring of passage {passage_id}. Quote the "
            "passage EXACTLY as shown, character for character, or cite a different passage."
            + _near_miss_hint(span, shown_text),
        )

    return CommittedClaim(
        claim=claim,
        span=span,
        passage_id=passage_id,
        doc_id=doc_id_of(passage_id),
        lang=native_lang,
        # Fall back to the claim when the model omitted one: both fields are optional in the flat
        # union, and a missing short value must not lose the evidence. Falling back is legitimate
        # where inventing one would not be, since the claim is text the model asserted about this
        # span.
        answer=answer or claim,
        sentence=sentence or claim,
        value_fallback=not (answer and sentence),
    )
