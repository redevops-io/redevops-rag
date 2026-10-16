"""File loading + paragraph-aware chunking. Defaults tuned for note/markdown vaults
(Obsidian, docs trees)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .nemo_retriever import chunk_identity, content_hash

TEXT_EXTS = {".md", ".markdown", ".mdx", ".txt", ".rst", ".org", ".text"}


def iter_files(root: str | Path, exts: set[str] = TEXT_EXTS) -> Iterator[Path]:
    root = Path(root)
    if root.is_file():
        yield root
        return
    try:
        real_root = root.resolve()
    except OSError:
        return
    for p in sorted(root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in exts):
            continue
        try:
            inside = p.resolve().is_relative_to(real_root)
        except (OSError, RuntimeError):
            inside = False
        if inside:  # a link pointing outside the ingested root is not followed
            yield p


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Pack paragraphs up to ~``size`` chars; hard-split paragraphs longer than ``size``
    with ``overlap`` so nothing is lost and context bleeds across boundaries."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(p) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            i = 0
            step = max(size - overlap, 1)
            while i < len(p):
                chunks.append(p[i : i + size])
                i += step
            continue
        if len(buf) + len(p) + 2 <= size:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def ingest(store, embedder, path: str | Path, size: int = 1000, overlap: int = 150,
           batch: int = 64, prune: bool = True, allow_empty: bool = False,
           max_prune_fraction: float = 0.5) -> dict:
    """Index a file or folder into ``store``. FTS is rebuilt once by the caller (RAG.index).

    With ``prune`` (default) documents previously indexed under ``path`` whose source no longer exists (or
    is no longer eligible, e.g. a link leaving the root) are removed from the store. Pruning only runs when
    ``path`` currently exists, so an unmounted or missing root never wipes the index. A sweep is also refused
    (``prune_refused`` in the result) when the scan found no files, or when it would remove more than
    ``max_prune_fraction`` of the documents indexed under ``path``; ``allow_empty=True`` overrides both.
    """
    files = list(iter_files(path))
    total = 0
    unreadable: set[str] = set()
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            unreadable.add(str(f))  # transient failure: keep what is indexed
            continue
        pieces = chunk_text(text, size, overlap)  # an emptied source yields none: its old chunks are swept
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        except Exception:
            mtime = datetime.now(timezone.utc)
        did = str(f)
        seen_ids: list[str] = []
        for i in range(0, len(pieces), batch):
            part = pieces[i : i + batch]
            embs = embedder.encode(part)
            rows = []
            for j, (t, e) in enumerate(zip(part, embs)):
                ch = content_hash(t)
                # Content-addressed id: re-ingesting the same content — in any order — reproduces the
                # same id (retires the old position-based `{path}::{index}`, which silently went stale
                # when a source was edited). chunk_index is kept as metadata for ordering only.
                cid = chunk_identity(did, t)
                seen_ids.append(cid)
                rows.append({
                    "id": cid, "document_id": did, "filename": f.name,
                    "chunk_index": i + j, "content_hash": ch, "text": t, "embedding": e,
                    "metadata": {"path": str(f)}, "created_at": mtime,
                })
            total += store.add_chunks(rows)
        # Re-ingest sweep: drop any prior chunk of this document whose content no longer exists (a
        # shrunk source's tail chunks, or edited content that moved to a new content-addressed id).
        sweep = getattr(store, "prune_document", None)
        if callable(sweep):
            sweep(did, seen_ids)
    removed = 0
    refused = False
    root = Path(path)
    list_ids = getattr(store, "document_ids", None)
    drop = getattr(store, "prune_document", None)
    if prune and callable(list_ids) and callable(drop) and root.exists():
        live = {str(f) for f in files}
        stale = [d for d in list_ids(str(root)) if d not in live and d not in unreadable]
        indexed = len(set(list_ids(str(root))))
        too_many = indexed > 0 and len(stale) / indexed > max_prune_fraction
        if stale and not allow_empty and (not files or too_many):
            refused = True   # looks like a failed mount or a bulk loss, not a deliberate deletion
        else:
            for did in stale:
                drop(did, [])
                removed += 1
    return {"files": len(files), "chunks": total, "removed_documents": removed, "prune_refused": refused}
