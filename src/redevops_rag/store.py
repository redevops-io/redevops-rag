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
        self._stamp_encoder()

    def _stamp_encoder(self) -> None:
        """Record the building encoder's identity once (never clobber an existing stamp), so
        :func:`open_store` reopens this corpus with the SAME encoder."""
        backend = getattr(self.embedder, "backend", None)
        if backend and self.get_meta("backend") is None:
            self.set_meta(backend=backend,
                          model=getattr(self.embedder, "model_name", None)
                          or getattr(self.embedder, "model", None),
                          dim=self.dim)

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
        # key/value index metadata — records which ENCODER built the index so query-time can
        # reconstruct the same one (encoder routing is a static per-index binding; a mismatched
        # query encoder silently returns garbage against these vectors). See open_store / set_meta.
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS store_meta (key VARCHAR PRIMARY KEY, value VARCHAR)"
        )

    def set_meta(self, **kv: Any) -> None:
        """Persist index metadata (e.g. ``backend='nemotron', model=..., lang='ru', domain=...``).
        Call at build time so :func:`open_store` reopens the corpus with the matching encoder."""
        rows = [(str(k), json.dumps(v)) for k, v in kv.items()]
        if rows:
            self.con.executemany("INSERT OR REPLACE INTO store_meta VALUES (?, ?)", rows)

    def get_meta(self, key: str | None = None, default: Any = None) -> Any:
        """Read one meta value (``key`` given) or the whole dict (``key`` None)."""
        try:
            rows = self.con.execute("SELECT key, value FROM store_meta").fetchall()
        except Exception:
            return default if key is not None else {}
        meta = {k: json.loads(v) for k, v in rows}
        return meta.get(key, default) if key is not None else meta

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

    def _encode_query(self, text: str, query_mode: str) -> list[float]:
        """Encode a query, honouring an asymmetric encoder's query side.

        Reasoning/instruction-tuned encoders (Nemotron, ReasonIR) expose ``encode_queries`` — a
        query-side instruction prefix that documents don't get. ``query_mode``:
          * ``instruct`` — use ``encode_queries`` when present (the reasoning-heavy original query);
          * ``plain``    — always plain ``encode`` (expanded sub-query fragments, which the
            instruction *hurts* — this is why DIVER+reasoning-embedder regressed);
          * ``auto``     — instruct when the encoder is asymmetric, else plain.
        For a symmetric encoder (bge) all three collapse to plain ``encode`` — byte-identical."""
        instruct = query_mode == "instruct" or (query_mode == "auto")
        eq = getattr(self.embedder, "encode_queries", None)
        if instruct and callable(eq):
            return eq([text])[0]
        return self.embedder.encode([text])[0]

    def semantic_search(self, text: str, top_k: int = 50, threshold: float = 0.4,
                        document_ids: list | None = None, query_mode: str = "auto") -> list[dict]:
        q = self._encode_query(text, query_mode)
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


def open_store(db_path: str, embedder=None, **embed_kw) -> "Store":
    """Open an existing corpus with the encoder it was BUILT with (read from ``store_meta``).

    Encoder routing is a static per-index binding: a corpus embedded with Nemotron must be queried
    with Nemotron (matching space + dim), or cosine returns garbage. This reads the persisted
    ``backend``/``model`` and reconstructs the matching embedder so callers never have to remember
    which encoder a corpus used. Pass ``embedder`` to override; a legacy index with no stamp falls
    back to the default (bge). Never overwrites the existing stamp."""
    if embedder is not None:
        return Store(embedder, db_path)
    backend = model = None
    try:  # peek at the stamp without constructing the (possibly heavy) default encoder first
        con = duckdb.connect(db_path)
        try:
            rows = con.execute("SELECT key, value FROM store_meta").fetchall()
            meta = {k: json.loads(v) for k, v in rows}
            backend, model = meta.get("backend"), meta.get("model")
        finally:
            con.close()
    except Exception:
        pass
    from .embed import make_embedder
    kw = dict(embed_kw)
    if model and backend in ("nemotron", "reasonir", "colpali", "colqwen"):
        kw.setdefault("model", model)   # bge takes model_name via env, not a positional kw here
    return Store(make_embedder(backend, **kw), db_path)
