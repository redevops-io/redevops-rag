"""File loading + paragraph-aware chunking. Defaults tuned for note/markdown vaults
(Obsidian, docs trees)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

TEXT_EXTS = {".md", ".markdown", ".mdx", ".txt", ".rst", ".org", ".text"}


def iter_files(root: str | Path, exts: set[str] = TEXT_EXTS) -> Iterator[Path]:
    root = Path(root)
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
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
           batch: int = 64) -> dict:
    """Index a file or folder into ``store``. FTS is rebuilt once by the caller (RAG.index)."""
    files = list(iter_files(path))
    total = 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        pieces = chunk_text(text, size, overlap)
        if not pieces:
            continue
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        except Exception:
            mtime = datetime.now(timezone.utc)
        did = str(f)
        for i in range(0, len(pieces), batch):
            part = pieces[i : i + batch]
            embs = embedder.encode(part)
            rows = [
                {
                    "id": f"{did}::{i + j}", "document_id": did, "filename": f.name,
                    "chunk_index": i + j, "text": t, "embedding": e,
                    "metadata": {"path": str(f)}, "created_at": mtime,
                }
                for j, (t, e) in enumerate(zip(part, embs))
            ]
            total += store.add_chunks(rows)
    return {"files": len(files), "chunks": total}
