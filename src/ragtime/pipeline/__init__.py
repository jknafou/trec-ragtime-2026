"""The online agentic system, run per ``(topic, variant, seed)``.

Wiring and algorithms only: every model call goes through ``serving``, every search through
``retrieval``, every path through ``common.Layout``. This package instantiates no model, launches
no batch job and builds no index. It is one code path that runs identically across the
``{original, omt, omt_opus}`` family, differing only by the two knobs the config supplies:
``passage_lang`` (what is read) and ``retrieval.index`` (what is searched).
"""

from __future__ import annotations

__all__: list[str] = []
