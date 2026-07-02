"""Postgres + pgvector-backed chunk store — sibling of the DuckDB :class:`Store`.

Preserves the same public interface (``add_chunks`` / ``semantic_search`` /
``bm25_search`` / ``reindex_fts`` / ``count`` / ``close``) so callers
(``redevops_rag.RAG``, ``context-runtime``'s redevops-rag plugin) can swap
storage backends by connection-URL scheme alone.

Schema (per-corpus; ``table`` is caller-configurable so one Postgres can host
several tenants side-by-side)::

    CREATE TABLE {table} (
        id           TEXT PRIMARY KEY,
        document_id  TEXT,
        filename     TEXT,
        chunk_index  INT,
        text         TEXT,
        embedding    VECTOR({dim}),
        metadata     JSONB,
        created_at   TIMESTAMPTZ,
        text_tsv     TSVECTOR GENERATED ALWAYS AS
                     (to_tsvector('english', text)) STORED
    );
    CREATE INDEX ON {table} USING GIN  (text_tsv);
    CREATE INDEX ON {table} USING ivfflat (embedding vector_cosine_ops);

Sparse leg uses Postgres tsvector (``ts_rank_cd`` + ``plainto_tsquery``)
instead of BM25. The scoring math differs but the downstream RRF fusion
only cares about ranks, so this drops in cleanly for the DuckDB fts leg.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

# Lazy import lets the base package install without psycopg for DuckDB-only users.
try:  # pragma: no cover - import guard
    import psycopg
    from pgvector.psycopg import register_vector
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Postgres store requires the `pg` extra: pip install 'redevops-rag[pg]' "
        "or add `psycopg[binary]` + `pgvector` to your environment."
    ) from e


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _valid_ident(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(f"invalid Postgres identifier: {name!r}")
    return name


class PgStore:
    """Postgres + pgvector store implementing the DuckDB Store interface."""

    def __init__(
        self,
        embedder,
        conn_url: str,
        table: str = "redevops_rag_chunks",
        schema: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.dim = int(embedder.dim)
        self.table = _valid_ident(table)
        self.schema = _valid_ident(schema) if schema else None
        self.con = psycopg.connect(conn_url, autocommit=False)
        # Register the pgvector adapters so we can bind list[float] as VECTOR
        # without manual '[...]::vector' casts.
        try:
            register_vector(self.con)
        except Exception:
            # If the extension isn't installed yet (fresh DB), ensure_schema
            # will CREATE EXTENSION and we retry.
            pass
        self._ensure_schema()

    @property
    def _qtable(self) -> str:
        # Identifiers are validated in __init__ — safe to interpolate.
        return f'"{self.schema}"."{self.table}"' if self.schema else f'"{self.table}"'

    def _ensure_schema(self) -> None:
        with self.con.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            if self.schema:
                cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._qtable} (
                    id          TEXT PRIMARY KEY,
                    document_id TEXT,
                    filename    TEXT,
                    chunk_index INT,
                    text        TEXT NOT NULL,
                    embedding   VECTOR({self.dim}),
                    metadata    JSONB,
                    created_at  TIMESTAMPTZ,
                    text_tsv    TSVECTOR GENERATED ALWAYS AS
                                (to_tsvector('english', coalesce(text, ''))) STORED
                )
                """
            )
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS "{self.table}_tsv_idx" '
                f'ON {self._qtable} USING GIN (text_tsv)'
            )
        self.con.commit()
        # Re-register in case the extension was just installed on this session.
        try:
            register_vector(self.con)
        except Exception:
            pass

    def add_chunks(
        self, chunks: list[dict[str, Any]], reindex: bool = False,
    ) -> int:
        rows: list[tuple] = []
        for c in chunks:
            cid = c.get("id") or str(uuid.uuid4())
            rows.append(
                (
                    cid,
                    c.get("document_id"),
                    c.get("filename"),
                    int(c.get("chunk_index", 0)),
                    c["text"],
                    c["embedding"],
                    json.dumps(c.get("metadata") or {}),
                    c.get("created_at") or datetime.now(timezone.utc),
                )
            )
        if not rows:
            return 0
        with self.con.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self._qtable}
                    (id, document_id, filename, chunk_index, text,
                     embedding, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    filename    = EXCLUDED.filename,
                    chunk_index = EXCLUDED.chunk_index,
                    text        = EXCLUDED.text,
                    embedding   = EXCLUDED.embedding,
                    metadata    = EXCLUDED.metadata,
                    created_at  = EXCLUDED.created_at
                """,
                rows,
            )
        self.con.commit()
        if reindex:
            self.reindex_fts()
        return len(rows)

    def reindex_fts(self) -> None:
        """Post-bulk-load maintenance. tsvector is auto-maintained by the
        generated column; this only creates the ivfflat vector index once
        the table has enough rows for it to be worth planning around, then
        ANALYZEs so the planner picks it up. Idempotent."""
        with self.con.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self._qtable}")
            row_count = int(cur.fetchone()[0])
            # ivfflat picks lists heuristic: rows / 1000, floor at 10.
            if row_count >= 1000:
                lists = max(10, row_count // 1000)
                try:
                    cur.execute(
                        f'CREATE INDEX IF NOT EXISTS '
                        f'"{self.table}_embedding_idx" '
                        f'ON {self._qtable} USING ivfflat '
                        f'(embedding vector_cosine_ops) WITH (lists = {lists})'
                    )
                except Exception:
                    # Non-fatal: sequential scan still works, just slower.
                    pass
            cur.execute(f"ANALYZE {self._qtable}")
        self.con.commit()

    def semantic_search(
        self, text: str, top_k: int = 50, threshold: float = 0.4,
    ) -> list[dict]:
        q = list(self.embedder.encode([text])[0])
        with self.con.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, document_id, filename, chunk_index, text, metadata,
                       created_at, (1 - (embedding <=> %s::vector)) AS sim
                FROM {self._qtable}
                WHERE (1 - (embedding <=> %s::vector)) >= %s
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s
                """,
                [q, q, float(threshold), q, int(top_k)],
            )
            rows = cur.fetchall()
        return [self._row(r, "similarity", r[7], "vector") for r in rows]

    def bm25_search(self, text: str, limit: int = 50) -> list[dict]:
        """Sparse-leg search using tsvector + ts_rank_cd. Keeps the name
        ``bm25_search`` and the ``bm25_score`` result key so hybrid RRF
        fusion / boosts run unchanged."""
        if not text.strip():
            return []
        try:
            with self.con.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, document_id, filename, chunk_index, text,
                           metadata, created_at,
                           ts_rank_cd(text_tsv,
                                      plainto_tsquery('english', %s)) AS score
                    FROM {self._qtable}
                    WHERE text_tsv @@ plainto_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    [text, text, int(limit)],
                )
                rows = cur.fetchall()
        except Exception:
            return []
        return [self._row(r, "bm25_score", r[7], "bm25") for r in rows]

    @staticmethod
    def _row(r, score_key: str, score_val, source: str) -> dict:
        md = r[5]
        if isinstance(md, str):
            try:
                md = json.loads(md)
            except Exception:
                md = {}
        return {
            "chunk_id": r[0],
            "document_id": r[1],
            "filename": r[2],
            "chunk_index": r[3],
            "text": r[4],
            "metadata": md or {},
            "created_at": r[6],
            score_key: float(score_val) if score_val is not None else 0.0,
            "source_type": source,
        }

    def count(self) -> int:
        with self.con.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self._qtable}")
            return int(cur.fetchone()[0])

    def close(self) -> None:
        self.con.close()
