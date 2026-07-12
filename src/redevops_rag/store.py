"""DuckDB-backed chunk store with a dense (cosine) leg and a sparse (BM25/FTS) leg.

Single-namespace: no workspace/tenant coupling. Vector search uses DuckDB's native
``array_cosine_similarity`` (no VSS extension needed); BM25 uses the DuckDB ``fts``
extension and soft-fails to an empty sparse leg if it can't load.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import duckdb


class Store:
    def __init__(self, embedder, db_path: str = ":memory:"):
        self.embedder = embedder
        self.dim = int(embedder.dim)
        self.con = duckdb.connect(db_path)
        self._fts = self._load_fts()
        self._ensure_schema()

    def _load_fts(self) -> bool:
        try:
            self.con.execute("INSTALL fts; LOAD fts;")
            return True
        except Exception:
            return False

    def _ensure_schema(self) -> None:
        self.con.execute(
            f"""CREATE TABLE IF NOT EXISTS chunks (
                    id VARCHAR PRIMARY KEY,
                    document_id VARCHAR,
                    filename VARCHAR,
                    chunk_index INTEGER,
                    text VARCHAR,
                    embedding FLOAT[{self.dim}],
                    metadata VARCHAR,
                    created_at TIMESTAMP
                )"""
        )

    def add_chunks(self, chunks: list[dict[str, Any]], reindex: bool = False) -> int:
        rows = []
        for c in chunks:
            cid = c.get("id") or str(uuid.uuid4())
            rows.append((
                cid, c.get("document_id"), c.get("filename"), int(c.get("chunk_index", 0)),
                c["text"], c["embedding"], json.dumps(c.get("metadata") or {}),
                c.get("created_at") or datetime.now(timezone.utc),
            ))
        self.con.executemany(
            "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?)", rows
        )
        if reindex:
            self.reindex_fts()
        return len(rows)

    def reindex_fts(self) -> None:
        if not self._fts:
            return
        try:
            self.con.execute("PRAGMA create_fts_index('chunks', 'id', 'text', overwrite=1)")
        except Exception:
            pass

    def semantic_search(self, text: str, top_k: int = 50, threshold: float = 0.4,
                        document_ids: list | None = None) -> list[dict]:
        q = self.embedder.encode([text])[0]
        scope = ""
        params: list = [list(q), float(threshold)]
        if document_ids is not None:
            scope = "AND document_id = ANY(?::VARCHAR[]) "
            params.append(list(document_ids))
        params.append(int(top_k))
        rows = self.con.execute(
            f"""SELECT id, document_id, filename, chunk_index, text, metadata, created_at, sim
                FROM (
                    SELECT *, array_cosine_similarity(embedding, ?::FLOAT[{self.dim}]) AS sim
                    FROM chunks
                )
                WHERE sim >= ? {scope}ORDER BY sim DESC LIMIT ?""",
            params,
        ).fetchall()
        return [self._row(r, "similarity", r[7], "vector") for r in rows]

    def bm25_search(self, text: str, limit: int = 50,
                    document_ids: list | None = None) -> list[dict]:
        if not self._fts or not text.strip():
            return []
        scope = ""
        params: list = [text]
        if document_ids is not None:
            scope = "AND document_id = ANY(?::VARCHAR[]) "
            params.append(list(document_ids))
        params.append(int(limit))
        try:
            rows = self.con.execute(
                f"""SELECT id, document_id, filename, chunk_index, text, metadata, created_at, score
                   FROM (
                       SELECT *, fts_main_chunks.match_bm25(id, ?) AS score FROM chunks
                   )
                   WHERE score IS NOT NULL {scope}ORDER BY score DESC LIMIT ?""",
                params,
            ).fetchall()
        except Exception:
            return []
        return [self._row(r, "bm25_score", r[7], "bm25") for r in rows]

    @staticmethod
    def _row(r, score_key: str, score_val, source: str) -> dict:
        return {
            "chunk_id": r[0], "document_id": r[1], "filename": r[2], "chunk_index": r[3],
            "text": r[4], "metadata": json.loads(r[5]) if r[5] else {}, "created_at": r[6],
            score_key: float(score_val) if score_val is not None else 0.0,
            "source_type": source,
        }

    def count(self) -> int:
        return int(self.con.execute("SELECT count(*) FROM chunks").fetchone()[0])

    def close(self) -> None:
        self.con.close()
