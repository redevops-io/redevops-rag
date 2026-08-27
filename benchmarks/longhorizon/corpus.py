"""Load a Wikimedia dump into plain articles for the long-horizon haystack.

Reuses the canonical MediaWiki parser from ``context-runtime-bench`` (one source of truth for the
XML/namespace + full-history handling) rather than reimplementing it here. Yields the *current* text of
each main-namespace (ns=0) article — the haystack material. Wikitext markup is left as-is: it is realistic
distractor text and the needles are inserted as clean sentences, so light markup does not affect scoring.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# The canonical parser lives in the benchmark sibling repo; import it by path (same pattern the existing
# eval_v5_context_size.py uses for the context-vs-model harness).
_BENCH = os.environ.get(
    "CRB_WIKIMEDIA",
    "/mnt/backup/projects/context-runtime-bench/benchmarks/wikimedia",
)
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

try:
    from harness.corpus import iter_pages  # type: ignore
except Exception as e:  # pragma: no cover - environment guard
    raise SystemExit(
        f"cannot import the MediaWiki parser from {_BENCH}/harness/corpus.py ({e}). "
        f"Set CRB_WIKIMEDIA to the context-runtime-bench wikimedia dir, or point WIKI_DUMP at a dump "
        f"and ensure that repo is checked out."
    )

# Default dev dump: the sha256-pinned strategywiki full history already on disk.
DEFAULT_DUMP = os.environ.get(
    "WIKI_DUMP",
    f"{_BENCH}/data/strategywiki-20260801-pages-meta-history.xml.bz2",
)

_WS = re.compile(r"\s+")


@dataclass
class Article:
    doc_id: str          # stable id: "<corpus>:<page_id>"
    title: str
    text: str            # current-revision wikitext, whitespace-normalised


def load_articles(dump_path: str | None = None, *, corpus: str = "strategywiki",
                  min_chars: int = 400, limit: int | None = None) -> list[Article]:
    """Current-revision ns=0 articles, longest-first (so a few articles can fill a large horizon).

    ``min_chars`` drops stubs/redirects that add no haystack bulk. Deterministic ordering (by length then
    page_id) so a run is reproducible and the dev/holdout split is stable.
    """
    path = dump_path or DEFAULT_DUMP
    if not Path(path).exists():
        raise SystemExit(f"dump not found: {path} (set WIKI_DUMP)")
    arts: list[Article] = []
    for page in iter_pages(path, include_text=True):
        if page.ns != 0 or not page.revisions:
            continue
        text = _WS.sub(" ", (page.revisions[-1].text or "")).strip()
        if len(text) < min_chars or text.lower().startswith("#redirect"):
            continue
        arts.append(Article(doc_id=f"{corpus}:{page.page_id}", title=page.title, text=text))
    arts.sort(key=lambda a: (-len(a.text), a.doc_id))
    if limit:
        arts = arts[:limit]
    if not arts:
        raise SystemExit(f"no articles parsed from {path}")
    return arts
