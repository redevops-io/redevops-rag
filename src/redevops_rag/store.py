"""DuckDB-backed chunk store with a dense (cosine) leg and a sparse (BM25/FTS) leg.

Single-namespace: no workspace/tenant coupling. Vector search uses DuckDB's native
``array_cosine_similarity`` (no VSS extension needed); BM25 uses the DuckDB ``fts``
extension and soft-fails to an empty sparse leg if it can't load.
"""
from __future__ import annotations

import json
import os
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


from .retrieval_result import BackendUnavailable, redact
from .tenancy import TenantScopeError, qualify_id, validate_tenant_scope

class Store:
    def __init__(self, embedder, db_path: str = ":memory:", *,
                 fts_stemmer: str = "porter", fts_stopwords: str = "english",
                 namespace: str | None = None, tenant_aware: bool = False):
        # Optional guard: when set, retrieval calls naming a different namespace are rejected. Tenant
        # scoping is otherwise the caller's job (see README).
        self.namespace = namespace
        # Tenant-aware mode (see tenancy.py): every read/write/admin call must carry a valid tenant_scope or it is
        # refused. False (default) = the historical unscoped behaviour, which provides no isolation.
        self.tenant_aware = bool(tenant_aware)
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
                    tenant_scope VARCHAR
                )"""
        )
        # Migrate a legacy corpus (built before content-addressed identity) in place — add the
        # content_hash column if the table predates it, so old indexes reopen without a rebuild.
        try:
            self.con.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash VARCHAR")
        except Exception:
            pass
        try:
            self.con.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tenant_scope VARCHAR")
        except Exception:
            pass
        if self.tenant_aware and not self._has_tenant_column():
            raise TenantScopeError("tenant-aware mode needs a tenant_scope column and this database could not be migrated")
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

    def _has_tenant_column(self) -> bool:
        return bool(self.con.execute(
            "SELECT count(*) FROM information_schema.columns WHERE table_name = 'chunks' AND column_name = 'tenant_scope'"
        ).fetchone()[0])

    def _scope(self, tenant_scope: str | None) -> str | None:
        """Resolve the scope for a call. Tenant-aware: required and validated. Unscoped: optional (validated if given)."""
        if tenant_scope is None:
            if self.tenant_aware:
                raise TenantScopeError("tenant-aware store: a tenant_scope is required")
            return None
        return validate_tenant_scope(tenant_scope)

    def add_chunks(self, chunks: list[dict[str, Any]], reindex: bool = False, *, tenant_scope: str | None = None) -> int:
        call_scope = self._scope(tenant_scope) if (tenant_scope is not None or not self.tenant_aware) else None
        rows = []
        for c in chunks:
            row_scope = c.get("tenant_scope")
            if row_scope is not None:
                row_scope = validate_tenant_scope(row_scope)
                if call_scope is not None and row_scope != call_scope:
                    raise TenantScopeError("chunk tenant_scope differs from the call's tenant_scope")
            else:
                row_scope = call_scope
            if self.tenant_aware and row_scope is None:
                raise TenantScopeError("tenant-aware store: every chunk needs a tenant_scope")
            # Prefer a content-addressed id (ingest mints `{document_ref}::{content_hash[:16]}`); fall
            # back to a random id only for callers that pass neither. Position-based ids are retired.
            cid = c.get("id") or str(uuid.uuid4())
            if self.tenant_aware:
                cid = qualify_id(row_scope, cid)
            rows.append((
                cid, c.get("document_id"), c.get("filename"), int(c.get("chunk_index", 0)),
                c.get("content_hash"), c["text"], c["embedding"],
                json.dumps(c.get("metadata") or {}),
                c.get("created_at") or datetime.now(timezone.utc),
                row_scope,
            ))
        if self.tenant_aware and rows:       # defence in depth: never replace a row that belongs to another scope
            owner = dict(self.con.execute("SELECT id, tenant_scope FROM chunks WHERE id = ANY(?::VARCHAR[])",
                                          [[r[0] for r in rows]]).fetchall())
            if any(r[0] in owner and owner[r[0]] != r[9] for r in rows):
                raise TenantScopeError("refusing to overwrite a chunk that belongs to a different tenant scope")
        self.con.executemany(
            "INSERT OR REPLACE INTO chunks "
            "(id, document_id, filename, chunk_index, content_hash, text, embedding, metadata, created_at, tenant_scope) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)", rows
        )
        if reindex:
            self.reindex_fts()
        return len(rows)

    def prune_document(self, document_id: str, keep_ids: list[str], *, tenant_scope: str | None = None) -> int:
        """Delete chunks of ``document_id`` whose id is NOT in ``keep_ids`` — the re-ingest sweep that
        removes now-orphaned tail chunks (a source that shrank) and content that changed to a new id.
        Content-addressed ids make re-ingest add the new chunks; this removes the stale ones, so the
        index never silently keeps a vector that no longer matches current source content.

        Tenant-aware: only rows of ``tenant_scope`` are considered, so one tenant's sweep can never delete another's."""
        scope = self._scope(tenant_scope)
        keep = [qualify_id(scope, k) for k in keep_ids] if (self.tenant_aware and scope) else list(keep_ids)
        where, base = "document_id = ?", [document_id]
        if scope is not None:
            where += " AND tenant_scope = ?"; base.append(scope)
        # Count first, then delete — DuckDB's DELETE rowcount isn't reliable, so count gives the caller
        # an accurate orphan count. Empty keep set ⇒ delete every chunk for the document (within scope).
        if not keep:
            removed = self.con.execute(f"SELECT count(*) FROM chunks WHERE {where}", base).fetchone()[0]
            self.con.execute(f"DELETE FROM chunks WHERE {where}", base)
            return int(removed)
        cond = f"{where} AND id NOT IN (SELECT unnest(?::VARCHAR[]))"
        removed = self.con.execute(f"SELECT count(*) FROM chunks WHERE {cond}", [*base, keep]).fetchone()[0]
        self.con.execute(f"DELETE FROM chunks WHERE {cond}", [*base, keep])
        return int(removed)

    def document_ids(self, root: str, *, tenant_scope: str | None = None) -> list[str]:
        """Distinct document ids equal to ``root`` or located under it (``root`` + path separator).
        Tenant-aware: only the caller's own documents are listed."""
        scope = self._scope(tenant_scope)
        sep = os.sep
        prefix = root.rstrip(sep) + sep
        sql = "SELECT DISTINCT document_id FROM chunks WHERE (document_id = ? OR starts_with(document_id, ?))"
        params: list = [root, prefix]
        if scope is not None:
            sql += " AND tenant_scope = ?"; params.append(scope)
        return [r[0] for r in self.con.execute(sql, params).fetchall()]

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
                        document_ids: list | None = None, query_mode: str = "auto",
                        tenant_scope: str | None = None) -> list[dict]:
        # tenant_scope: required when tenant-aware; filters in the WHERE clause, i.e. BEFORE ORDER BY ... LIMIT.
        scope_v = self._scope(tenant_scope)
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
        if scope_v is not None:
            scope += "AND tenant_scope = ? "
            params.append(scope_v)
        params.append(int(top_k))
        rows = self.con.execute(
            f"""SELECT id, document_id, filename, chunk_index, content_hash, text, metadata, created_at, sim
                FROM (
                    SELECT *, array_cosine_similarity(embedding, ?::FLOAT[{self.dim}]) AS sim
                    FROM chunks
                )
                WHERE sim >= ? {scope}ORDER BY sim DESC LIMIT ?""",
            params,
        ).fetchall()
        return [self._row(r, "similarity", r[8], "vector") for r in rows]

    # DuckDB's own match_bm25 defaults (k1, b); the scoped query below reproduces its formula exactly.
    BM25_K1 = 1.2
    BM25_B = 0.75

    def _scoped_bm25_supported(self) -> bool:
        """The scoped query reads DuckDB's FTS index tables (docs.field_lens, stats.num_docs). Verify the layout once, so an
        unexpected DuckDB version fails closed (leg unavailable) instead of silently falling back to corpus-wide statistics."""
        ok = getattr(self, "_scoped_bm25_ok", None)
        if ok is None:
            try:
                cols = {(r[0], r[1]) for r in self.con.execute(
                    "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'fts_main_chunks'").fetchall()}
                ok = {("docs", "docid"), ("docs", "name"), ("docs", "field_lens"), ("terms", "docid"), ("terms", "termid"),
                      ("dict", "termid"), ("dict", "term")} <= cols
            except Exception:
                ok = False
            self._scoped_bm25_ok = ok
        return ok

    def bm25_search(self, text: str, limit: int = 50,
                    document_ids: list | None = None, *, strict: bool = False,
                    tenant_scope: str | None = None, scoped_stats: bool | None = None) -> list[dict]:
        """BM25 over the FTS index. ``strict=False`` (default, legacy) returns ``[]`` when the leg cannot run.
        ``strict=True`` raises :class:`BackendUnavailable` instead, so a caller can tell "no match" from "no index".

        ``scoped_stats`` chooses where the corpus statistics (document count, per-term document frequency, average
        length) come from. ``False``: the whole index, as DuckDB's ``match_bm25`` does — a scoped query's scores then
        depend on documents outside the scope, which is an information leak across tenants and makes scores change when
        unrelated documents are ingested. ``True``: only the rows that satisfy the scope (``tenant_scope`` and/or
        ``document_ids``), so a document scores identically whether it sits in a standalone index or in a larger one.
        Default ``None`` ⇒ ``True`` when a ``tenant_scope`` is in force, else ``False`` (legacy ``document_ids`` behaviour
        is unchanged). Ties are broken on ``id`` so results are deterministic."""
        scope_v = self._scope(tenant_scope)          # validated before anything else, even for a blank query
        if not text.strip():
            return []
        if not self._fts:
            if strict:
                raise BackendUnavailable("full-text search extension is not loaded")
            return []
        if scoped_stats is None:
            scoped_stats = scope_v is not None
        try:
            if scoped_stats:
                if not self._scoped_bm25_supported():
                    raise BackendUnavailable("scoped BM25 statistics are not supported by this DuckDB full-text-search layout")
                rows = self._bm25_scoped(text, limit, document_ids, scope_v)
            else:
                rows = self._bm25_global(text, limit, document_ids, scope_v)
        except Exception as e:
            if strict:
                if isinstance(e, BackendUnavailable):
                    raise
                raise BackendUnavailable(f"bm25 query failed: {type(e).__name__}: {redact(e)}") from e
            return []
        return [self._row(r, "bm25_score", r[8], "bm25") for r in rows]

    def _bm25_global(self, text, limit, document_ids, scope_v):
        scope = ""
        params: list = [text]
        if document_ids is not None:
            scope = "AND document_id = ANY(?::VARCHAR[]) "
            params.append(list(document_ids))
        if scope_v is not None:
            scope += "AND tenant_scope = ? "
            params.append(scope_v)
        params.append(int(limit))
        return self.con.execute(
            f"""SELECT id, document_id, filename, chunk_index, content_hash, text, metadata, created_at, score
               FROM (
                   SELECT *, fts_main_chunks.match_bm25(id, ?) AS score FROM chunks
               )
               WHERE score IS NOT NULL {scope}ORDER BY score DESC, id LIMIT ?""",
            params,
        ).fetchall()

    def _bm25_scoped(self, text, limit, document_ids, scope_v):
        where, params = ["TRUE"], []
        if document_ids is not None:
            where.append("c.document_id = ANY(?::VARCHAR[])"); params.append(list(document_ids))
        if scope_v is not None:
            where.append("c.tenant_scope = ?"); params.append(scope_v)
        k1, b = float(self.BM25_K1), float(self.BM25_B)
        stem = self.fts_stemmer                                   # validated identifier (see __init__)
        sql = f"""
            WITH scoped AS (
                SELECT d.docid, d.name, d.field_lens[1] AS dl
                FROM fts_main_chunks.docs AS d JOIN chunks AS c ON c.id = d.name
                WHERE {' AND '.join(where)}),
            st AS (SELECT count(*)::DOUBLE AS n, nullif(avg(dl)::DOUBLE, 0) AS avgdl FROM scoped),
            raw AS (SELECT DISTINCT unnest(fts_main_chunks.tokenize(?)) AS t),
            toks AS (SELECT DISTINCT stem(t, '{stem}') AS t FROM raw
                     WHERE t IS NOT NULL AND t <> '' AND t NOT IN (SELECT sw FROM fts_main_chunks.stopwords)),
            qt AS (SELECT dict.termid FROM fts_main_chunks.dict AS dict JOIN toks ON dict.term = toks.t),
            tf AS (SELECT t.docid, t.termid, count(*)::DOUBLE AS tf
                   FROM fts_main_chunks.terms AS t JOIN scoped AS s ON s.docid = t.docid
                   WHERE t.termid IN (SELECT termid FROM qt) GROUP BY 1, 2),
            df AS (SELECT termid, count(*)::DOUBLE AS df FROM tf GROUP BY 1),
            sc AS (SELECT tf.docid,
                          sum(log10(((st.n - df.df + 0.5) / (df.df + 0.5)) + 1)
                              * (((tf.tf / ((1.0 - {b}) + {b} * (s.dl / st.avgdl))) * ({k1} + 1))
                                 / ((tf.tf / ((1.0 - {b}) + {b} * (s.dl / st.avgdl))) + {k1}))) AS score
                   FROM tf JOIN df USING (termid) JOIN scoped AS s USING (docid) CROSS JOIN st GROUP BY tf.docid)
            SELECT c.id, c.document_id, c.filename, c.chunk_index, c.content_hash, c.text, c.metadata, c.created_at, sc.score
            FROM sc JOIN scoped AS s ON s.docid = sc.docid JOIN chunks AS c ON c.id = s.name
            WHERE sc.score IS NOT NULL ORDER BY sc.score DESC, c.id LIMIT ?"""
        return self.con.execute(sql, [*params, text, int(limit)]).fetchall()

    @staticmethod
    def _row(r, score_key: str, score_val, source: str) -> dict:
        return {
            "chunk_id": r[0], "document_id": r[1], "filename": r[2], "chunk_index": r[3],
            "content_hash": r[4], "text": r[5], "metadata": json.loads(r[6]) if r[6] else {},
            "created_at": r[7],
            score_key: float(score_val) if score_val is not None else 0.0,
            "source_type": source,
        }

    def count(self, *, tenant_scope: str | None = None) -> int:
        scope = self._scope(tenant_scope)
        if scope is not None:
            return int(self.con.execute("SELECT count(*) FROM chunks WHERE tenant_scope = ?", [scope]).fetchone()[0])
        return int(self.con.execute("SELECT count(*) FROM chunks").fetchone()[0])

    def close(self) -> None:
        self.con.close()


def open_store(db_path: str, embedder=None, *, namespace: str | None = None, tenant_aware: bool = False,
               **embed_kw) -> "Store":
    """Open an existing corpus with the encoder it was BUILT with (read from ``store_meta``).

    Encoder routing is a static per-index binding: a corpus embedded with Nemotron must be queried
    with Nemotron (matching space + dim), or cosine returns garbage. This reads the persisted
    ``backend``/``model`` and reconstructs the matching embedder so callers never have to remember
    which encoder a corpus used. Pass ``embedder`` to override; a legacy index with no stamp falls
    back to the default (bge). Never overwrites the existing stamp."""
    if embedder is not None:
        return Store(embedder, db_path, namespace=namespace, tenant_aware=tenant_aware)
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
    return Store(make_embedder(backend, **kw), db_path, namespace=namespace, tenant_aware=tenant_aware)
