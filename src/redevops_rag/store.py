"""DuckDB-backed chunk store with a dense (cosine) leg and a sparse (BM25/FTS) leg.

Single-namespace: no workspace/tenant coupling. Vector search uses DuckDB's native
``array_cosine_similarity`` (no VSS extension needed); BM25 uses the DuckDB ``fts``
extension and soft-fails to an empty sparse leg if it can't load.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import duckdb

_FTS_IDENT = re.compile(r"^[a-z_]+$")


def _fts_ident(name: str) -> str:
    """Validate an FTS stemmer/stopwords name before interpolating it into the create_fts_index PRAGMA
    (DuckDB doesn't parameterize PRAGMA args). Lowercase letters/underscore only — a fixed vocabulary of
    Snowball languages ('russian', 'english', …) + 'none'/'porter', never user text."""
    if not isinstance(name, str) or not _FTS_IDENT.match(name):
        raise ValueError(f"invalid FTS stemmer/stopwords identifier: {name!r}")
    return name


class Store:
    def __init__(self, embedder, db_path: str = ":memory:", *,
                 fts_stemmer: str = "porter", fts_stopwords: str = "english"):
        # BM25/FTS stemmer + stopwords. DuckDB defaults (porter/english) can't normalize non-English
        # inflection — a Russian query "Фолиновую кислоту" (accusative) misses a doc "фолиновая кислота"
        # (nominative). Pass fts_stemmer='russian', fts_stopwords='none' for an RU corpus (Snowball
        # stemmers: english/russian/german/french/…). Validated (interpolated into the PRAGMA).
        self.fts_stemmer = _fts_ident(fts_stemmer)
        self.fts_stopwords = _fts_ident(fts_stopwords)
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
                    content_hash VARCHAR,
                    text VARCHAR,
                    embedding FLOAT[{self.dim}],
                    metadata VARCHAR,
                    created_at TIMESTAMP,
                    source_ref VARCHAR,
                    source_version VARCHAR,
                    source_content_hash VARCHAR,
                    observed_at VARCHAR,
                    superseded_by VARCHAR
                )"""
        )
        # Migrate a legacy corpus in place — add columns the table may predate, so old indexes reopen
        # without a rebuild. content_hash: content-addressed identity. The source_*/observed_at/
        # superseded_by columns: canonical evidence identity + version-aware retention (v0.2.x
        # evidence-native seam). Legacy rows leave them NULL and behave exactly as before.
        for col, typ in (
            ("content_hash", "VARCHAR"),
            ("source_ref", "VARCHAR"),
            ("source_version", "VARCHAR"),
            ("source_content_hash", "VARCHAR"),
            ("observed_at", "VARCHAR"),
            ("superseded_by", "VARCHAR"),
        ):
            try:
                self.con.execute(f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS {col} {typ}")
            except Exception:
                pass
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
            # Prefer a content-addressed id (ingest mints `{document_ref}::{content_hash[:16]}`); fall
            # back to a random id only for callers that pass neither. Position-based ids are retired.
            cid = c.get("id") or str(uuid.uuid4())
            rows.append((
                cid, c.get("document_id"), c.get("filename"), int(c.get("chunk_index", 0)),
                c.get("content_hash"), c["text"], c["embedding"],
                json.dumps(c.get("metadata") or {}),
                c.get("created_at") or datetime.now(timezone.utc),
                # canonical evidence identity (NULL for legacy/base-path callers → unchanged behavior):
                c.get("source_ref"), c.get("source_version"), c.get("source_content_hash"),
                c.get("observed_at"), c.get("superseded_by"),
            ))
        self.con.executemany(
            "INSERT OR REPLACE INTO chunks "
            "(id, document_id, filename, chunk_index, content_hash, text, embedding, metadata, created_at, "
            "source_ref, source_version, source_content_hash, observed_at, superseded_by) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
        )
        if reindex:
            self.reindex_fts()
        return len(rows)

    def supersede_source(self, source_ref: str, current_version: str) -> int:
        """Mark every prior revision of ``source_ref`` as superseded by ``current_version``.

        Version-aware retention: superseded chunks are *retained and still addressable* (by their
        ``source_version`` or an as-of time) — they are not deleted — so point-in-time replay can
        recover the exact historical evidence a past decision was bound to. Only rows whose
        ``superseded_by`` is still NULL and whose version differs are updated. Returns the count.
        """
        removed = self.con.execute(
            "SELECT count(*) FROM chunks WHERE source_ref = ? AND source_version != ? "
            "AND superseded_by IS NULL", [source_ref, current_version]).fetchone()[0]
        self.con.execute(
            "UPDATE chunks SET superseded_by = ? WHERE source_ref = ? AND source_version != ? "
            "AND superseded_by IS NULL", [current_version, source_ref, current_version])
        return int(removed)

    def prune_document(self, document_id: str, keep_ids: list[str]) -> int:
        """Delete chunks of ``document_id`` whose id is NOT in ``keep_ids`` — the re-ingest sweep that
        removes now-orphaned tail chunks (a source that shrank) and content that changed to a new id.
        Content-addressed ids make re-ingest add the new chunks; this removes the stale ones, so the
        index never silently keeps a vector that no longer matches current source content."""
        keep = list(keep_ids)
        # Count first, then delete — DuckDB's DELETE rowcount isn't reliable, so count gives the caller
        # an accurate orphan count. Empty keep set ⇒ delete every chunk for the document.
        if not keep:
            removed = self.con.execute(
                "SELECT count(*) FROM chunks WHERE document_id = ?", [document_id]).fetchone()[0]
            self.con.execute("DELETE FROM chunks WHERE document_id = ?", [document_id])
            return int(removed)
        removed = self.con.execute(
            "SELECT count(*) FROM chunks WHERE document_id = ? AND id NOT IN "
            "(SELECT unnest(?::VARCHAR[]))", [document_id, keep]).fetchone()[0]
        self.con.execute(
            "DELETE FROM chunks WHERE document_id = ? AND id NOT IN "
            "(SELECT unnest(?::VARCHAR[]))", [document_id, keep])
        return int(removed)

    def reindex_fts(self) -> None:
        if not self._fts:
            return
        try:
            self.con.execute(
                f"PRAGMA create_fts_index('chunks', 'id', 'text', "
                f"stemmer='{self.fts_stemmer}', stopwords='{self.fts_stopwords}', overwrite=1)")
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

    def semantic_search(self, text: str, top_k: int = 50, threshold: float | None = None,
                        document_ids: list | None = None, query_mode: str = "auto", *,
                        current_only: bool = False, source_version: str | None = None,
                        as_of: str | None = None) -> list[dict]:
        # threshold=None → the ENCODER's own sim_floor (bge 0.4; Nemotron/ReasonIR 0.1). A single global
        # 0.4 silently discards compressed-sim encoders' vector leg — bge stays byte-identical.
        if threshold is None:
            threshold = getattr(self.embedder, "sim_floor", 0.4)
        q = self._encode_query(text, query_mode)
        scope = ""
        params: list = [list(q), float(threshold)]
        if document_ids is not None:
            scope = "AND document_id = ANY(?::VARCHAR[]) "
            params.append(list(document_ids))
        scope += self._version_scope(current_only, source_version, as_of, params)
        params.append(int(top_k))
        rows = self.con.execute(
            f"""SELECT {self._COLS}, sim
                FROM (
                    SELECT *, array_cosine_similarity(embedding, ?::FLOAT[{self.dim}]) AS sim
                    FROM chunks
                )
                WHERE sim >= ? {scope}ORDER BY sim DESC LIMIT ?""",
            params,
        ).fetchall()
        return [self._row(r, "similarity", r[13], "vector") for r in rows]

    def bm25_search(self, text: str, limit: int = 50,
                    document_ids: list | None = None, *,
                    current_only: bool = False, source_version: str | None = None,
                    as_of: str | None = None) -> list[dict]:
        if not self._fts or not text.strip():
            return []
        scope = ""
        params: list = [text]
        if document_ids is not None:
            scope = "AND document_id = ANY(?::VARCHAR[]) "
            params.append(list(document_ids))
        scope += self._version_scope(current_only, source_version, as_of, params)
        params.append(int(limit))
        try:
            rows = self.con.execute(
                f"""SELECT {self._COLS}, score
                   FROM (
                       SELECT *, fts_main_chunks.match_bm25(id, ?) AS score FROM chunks
                   )
                   WHERE score IS NOT NULL {scope}ORDER BY score DESC LIMIT ?""",
                params,
            ).fetchall()
        except Exception:
            return []
        return [self._row(r, "bm25_score", r[13], "bm25") for r in rows]

    #: SELECT column order shared by both retrieval legs; the score is appended after these.
    _COLS = ("id, document_id, filename, chunk_index, content_hash, text, metadata, created_at, "
             "source_ref, source_version, source_content_hash, observed_at, superseded_by")

    @staticmethod
    def _row(r, score_key: str, score_val, source: str) -> dict:
        return {
            "chunk_id": r[0], "document_id": r[1], "filename": r[2], "chunk_index": r[3],
            "content_hash": r[4], "text": r[5], "metadata": json.loads(r[6]) if r[6] else {},
            "created_at": r[7],
            # canonical evidence identity — None on legacy hits, populated by the evidence path:
            "source_ref": r[8], "source_version": r[9], "source_content_hash": r[10],
            "observed_at": r[11], "superseded_by": r[12],
            score_key: float(score_val) if score_val is not None else 0.0,
            "source_type": source,
        }

    @staticmethod
    def _version_scope(current_only: bool, source_version, as_of, params: list) -> str:
        """Build the optional evidence-version WHERE fragment (and append its params).

        Defaults (all off) reproduce legacy behavior byte-for-byte. ``current_only`` restricts to the
        live projection (``superseded_by IS NULL``); ``source_version`` pins an exact revision;
        ``as_of`` selects revisions observed at/before a timestamp (point-in-time). These compose.
        """
        frag = ""
        if current_only:
            frag += "AND superseded_by IS NULL "
        if source_version is not None:
            frag += "AND source_version = ? "
            params.append(source_version)
        if as_of is not None:
            frag += "AND (observed_at IS NULL OR observed_at <= ?) "
            params.append(as_of)
        return frag

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
