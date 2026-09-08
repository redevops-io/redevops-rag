"""Structural change-closure retrieval — the dependency-graph primitive similarity retrieval cannot represent.

A change to a provision / clause / symbol is not a passage; it is a **closure**: the unit you edit *plus
every other unit that (transitively) depends on it* — cites it, incorporates it, uses its defined term,
calls it. Content retrieval (BM25, dense, RRF-hybrid — see :mod:`redevops_rag.retrieve`) surfaces what
*resembles* the edit; it structurally cannot surface what *depends on* the edit when the dependent unit
shares no surface text (a regulation that says "as required by 15 U.S.C. 78c" looks nothing like §78c;
a covenant that *uses* "Permitted Liens" looks nothing like the *definition* of "Permitted Liens").

This module is the cross-document structural sibling of the content retrievers. It does not replace hybrid
search — it **composes** with it: :func:`graph_union_retrieve` takes the citation closure first, then fills
the remaining token budget with a content ranking you pass in. It is pure-Python and dependency-light
(no numpy/torch/duckdb at import); the optional LLM edge extractor is *injectable* and fails soft.

Measured (redevops-benchmarks/workloads/{statutory,xdoc}_change_closure, matched 4000-token budget):

    domain (corpus)                         citation_graph   best similarity   decisive
    ─────────────────────────────────────   ──────────────   ───────────────   ────────────────
    statutes    (US Code Title 11)              0.556            0.159 (dense)   GRAPH 51 / CONTENT 0
    regulations (17 CFR 240 → US Code 15)       0.484            0.066 (dense)   GRAPH 21 / CONTENT 0
    contract    (EDGAR CONMED credit agmt)      0.302            0.209 (hippo)   GRAPH 29 / CONTENT 0
    defined-terms (same contract, term graph)   1.000            0.206 (bm25)    GRAPH 52 / CONTENT 0

The graph wins or ties on every case in every domain; a HippoRAG-style *similarity*-derived PageRank graph
does not help — the win is the **citation** edges, not "a graph". The retrieval ceiling is therefore **edge
coverage**, and the lever that lifts it is *extraction* of stated-but-implicit edges (topological link
prediction fails: AP < 0.02 — legal graphs are not homophilous). :class:`PrecisionEdgeExtractor` is that
lever for the genuinely implicit edges, precision-gated so it *helps* rather than bloats the closure
(a raw "list every reference" dump lowered recall 0.230 → 0.188; confidence ≥ 0.7 + trigger-phrase
grounding flips it to 0.261 / 0.269 with gpt-5-mini / kimi-k2.7-code).

Ownership note (see the benchmark READMEs and redevops.io/blog): ReDevOps RAG owns this retrieval
*primitive*; **Context Runtime** owns the *choice* (graph vs content vs union) and the budget split for a
given task — composition is a per-task decision, not a fixed `graph_plus_hybrid` retriever, because which
representation dominates is domain-dependent (structure dominates statutes; content dominates prose).
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

__all__ = [
    "Node", "ExtractedEdge", "ChangeClosureGraph",
    "citation_closure", "citation_graph_retrieve", "graph_union_retrieve",
    "PrecisionEdgeExtractor", "openai_chat_fn",
]


# ───────────────────────────── value contracts ─────────────────────────────

@dataclass(frozen=True)
class Node:
    """A unit of a citation graph — a statute section, a contract clause, a defined term, a code symbol.

    ``id`` is the stable identity edges are expressed over. ``tokens`` is the materialization cost used by
    the budget knapsack; if left 0 it is estimated from ``text`` (whitespace words) at graph-build time.
    """
    id: str
    text: str = ""
    tokens: int = 0


@dataclass(frozen=True)
class ExtractedEdge:
    """One implicit dependency edge proposed by :class:`PrecisionEdgeExtractor`.

    ``trigger`` is the verbatim phrase (copied from the *source* node's text) that creates the dependency;
    it is what the grounding filter checks, so a hallucinated edge with no real trigger is dropped.
    """
    source: str
    target: str
    trigger: str = ""
    confidence: float = 0.0


# ───────────────────────────── functional core ─────────────────────────────

def _estimate_tokens(text: str) -> int:
    return len(text.split())


def citation_closure(cited_by: dict[str, set[str]], seed: str, hops: int = 2) -> set[str]:
    """Change-closure of ``seed``: the nodes that transitively **cite / depend on** it, up to ``hops``
    reverse-citation hops. Excludes ``seed`` itself. Pure function of the reverse-adjacency ``cited_by``.

    An edge ``a → b`` means "a depends on b"; ``cited_by[b]`` is therefore "who depends on b", and the
    closure of ``b`` is who is affected if ``b`` is edited — the retrieval target.
    """
    seen: set[str] = set()
    frontier = {seed}
    for _ in range(max(0, hops)):
        nxt: set[str] = set()
        for n in frontier:
            for a in cited_by.get(n, ()):
                if a != seed and a not in seen:
                    seen.add(a)
                    nxt.add(a)
        frontier = nxt
        if not frontier:
            break
    return seen


def _budget_fill(order: Iterable[str], tokens_of: dict[str, int], budget: int, exclude: str) -> list[str]:
    """Take ids from ``order`` until the token budget is exhausted (small-node greedy within the given
    order). ``exclude`` (the edited seed) and duplicates are skipped. Returns an ordered list."""
    out: list[str] = []
    spent = 0
    seen: set[str] = {exclude}
    for nid in order:
        if nid in seen:
            continue
        seen.add(nid)
        t = tokens_of.get(nid, 0)
        if out and spent + t > budget:
            break
        out.append(nid)
        spent += t
        if spent >= budget:
            break
    return out


def _closure_order(cited_by: dict[str, set[str]], in_degree: dict[str, int],
                   seed: str, hops: int) -> list[str]:
    """Reverse-citation traversal, hop-ordered (nearer citers first), in-degree desc as a tie-break so the
    most structurally central affected nodes come first under a tight budget."""
    order: list[str] = []
    seen = {seed}
    frontier = {seed}
    for _ in range(max(0, hops)):
        nxt: list[str] = []
        for n in frontier:
            for a in sorted(cited_by.get(n, ()), key=lambda x: -in_degree.get(x, 0)):
                if a not in seen:
                    seen.add(a)
                    nxt.append(a)
        order.extend(nxt)
        frontier = set(nxt)
        if not frontier:
            break
    return order


def citation_graph_retrieve(graph: "ChangeClosureGraph", seed: str, budget: int,
                            hops: int = 2) -> list[str]:
    """The citation-graph arm: reverse-citation closure of ``seed``, hop-ordered, budget-filled. The
    ``seed`` is never returned (you are editing it; you want the *other* affected nodes)."""
    order = _closure_order(graph.cited_by, graph._in_degree, seed, hops)
    return _budget_fill(order, graph.tokens, budget, seed)


def graph_union_retrieve(graph: "ChangeClosureGraph", seed: str, budget: int, hops: int = 2,
                         content_order: Optional[list[str]] = None) -> list[str]:
    """Closure-first union: the citation closure, then ``content_order`` (a content ranking from e.g.
    :func:`redevops_rag.retrieve.hybrid_search`, as node ids) fills the remaining budget.

    ``content_order`` keeps this decoupled from any embedder — the caller supplies whatever content ranking
    it already computed. With ``content_order=None`` this is exactly :func:`citation_graph_retrieve`.

    NB the *optimal* composition is domain-dependent (see the module docstring): where content dominates
    (prose), lead with content and let the graph *promote*; where structure dominates (statutes), lead with
    the closure as here. That policy choice belongs to Context Runtime, not to this primitive.
    """
    order = _closure_order(graph.cited_by, graph._in_degree, seed, hops)
    if content_order:
        order = list(dict.fromkeys([*order, *content_order]))
    return _budget_fill(order, graph.tokens, budget, seed)


# ───────────────────────────── imperative shell ─────────────────────────────

class ChangeClosureGraph:
    """A directed dependency graph over :class:`Node`\\ s with reverse-closure retrieval.

    Build it from nodes and edges (``{node_id: iterable of node_ids it cites/depends on}``); dangling
    targets (edges to unknown ids) are dropped and self-edges ignored. It owns the derived reverse
    adjacency and in-degree — a small store-like object, not a value — so it is a class, while the
    traversal/knapsack logic lives in the pure functions above.
    """

    def __init__(self, nodes: Iterable[Node], edges: dict[str, Iterable[str]] | None = None):
        self.nodes: list[Node] = [
            n if n.tokens else Node(n.id, n.text, _estimate_tokens(n.text)) for n in nodes
        ]
        self.by_id: dict[str, Node] = {n.id: n for n in self.nodes}
        self.tokens: dict[str, int] = {n.id: n.tokens for n in self.nodes}
        ids = set(self.by_id)
        self.cites: dict[str, set[str]] = {}
        for a, bs in (edges or {}).items():
            if a in ids:
                self.cites[a] = {b for b in bs if b in ids and b != a}
        cb: dict[str, set[str]] = {nid: set() for nid in ids}
        for a, bs in self.cites.items():
            for b in bs:
                cb[b].add(a)
        self.cited_by: dict[str, set[str]] = cb
        self._in_degree: dict[str, int] = {nid: len(v) for nid, v in cb.items()}

    # -- graph queries --
    def in_degree(self, nid: str) -> int:
        return self._in_degree.get(nid, 0)

    def closure(self, seed: str, hops: int = 2) -> set[str]:
        return citation_closure(self.cited_by, seed, hops)

    def num_edges(self) -> int:
        return sum(len(v) for v in self.cites.values())

    # -- retrieval arms (delegate to the pure core) --
    def retrieve(self, seed: str, budget: int, hops: int = 2) -> list[str]:
        return citation_graph_retrieve(self, seed, budget, hops)

    def union_retrieve(self, seed: str, budget: int, hops: int = 2,
                       content_order: Optional[list[str]] = None) -> list[str]:
        return graph_union_retrieve(self, seed, budget, hops, content_order)

    # -- enrichment --
    def with_edges(self, extra: dict[str, Iterable[str]]) -> "ChangeClosureGraph":
        """Return a NEW graph with ``extra`` edges folded in (e.g. from :class:`PrecisionEdgeExtractor`).
        The receiver is unchanged — closures/fingerprints of the base graph stay stable."""
        merged: dict[str, set[str]] = {a: set(bs) for a, bs in self.cites.items()}
        for a, bs in extra.items():
            merged.setdefault(a, set()).update(bs)
        return ChangeClosureGraph(self.nodes, merged)


# ───────────────────────── precision LLM edge extractor ─────────────────────

_SYSTEM = (
    "You are a precise dependency extractor. Read ONE unit (a statute section, contract clause, or code "
    "definition) and identify ONLY the other units it GENUINELY depends on — i.e. it conditions, excepts, "
    "limits, or incorporates a specific obligation/right/definition stated in that other unit. Do NOT list "
    "a unit merely because a similar term appears; be conservative and omit weak or generic references. "
    "For each real dependency output {\"target\":\"<unit id, exactly as given in the catalog>\","
    "\"trigger\":\"<verbatim phrase copied from THIS unit that creates the dependency>\","
    "\"confidence\":0.0-1.0}. Output ONLY a JSON array; [] if none."
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def _parse_edges(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not m:
        return []
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, list) else []
    except Exception:  # noqa: BLE001 — a malformed model reply yields no edges, never an exception
        return []


class PrecisionEdgeExtractor:
    """Recover the genuinely-IMPLICIT dependency edges no deterministic parser can reach — precision-gated
    so they help the closure instead of bloating it.

    Explicit references (``Section 7.02``, ``15 U.S.C. 78m``) and defined-term usage are recovered
    deterministically upstream. What remains are relations with no id and no number ("subject to the
    foregoing", "such Loans", "except as otherwise provided herein"), which need a reader. This asks an
    injected ``chat_fn(system, user) -> str`` per node and keeps an edge only if it clears **all three**
    precision controls that were shown to flip the LLM contribution from net-negative to net-positive:

      1. a conservative prompt (only real conditions/exceptions/incorporations);
      2. per-edge ``confidence`` ≥ ``min_confidence`` (default 0.7);
      3. **grounding** — the model's ``trigger`` phrase must actually occur (verbatim, normalized) in the
         source node's text. This is the filter that also survives implicit, number-less references, and the
         one that kills hallucinated targets.

    ``chat_fn`` is injectable so this is unit-tested offline with a stub. With ``chat_fn=None`` (no reachable
    endpoint) :meth:`extract` returns ``{}`` with a printed notice — the deterministic graph still stands.
    Results are cached per node when ``cache_dir`` is set (successful calls only). Concurrency via
    ``max_workers`` threads. No new dependencies — bring your own ``chat_fn`` (or use :func:`openai_chat_fn`).
    """

    def __init__(self, chat_fn: Optional[Callable[[str, str], str]] = None, *,
                 min_confidence: float = 0.7, max_workers: int = 4,
                 cache_dir: Optional[str] = None, system_prompt: str = _SYSTEM,
                 max_text_chars: int = 1600):
        self.chat_fn = chat_fn
        self.min_confidence = min_confidence
        self.max_workers = max_workers
        self.cache_dir = cache_dir
        self.system_prompt = system_prompt
        self.max_text_chars = max_text_chars

    def extract(self, nodes: Iterable[Node]) -> dict[str, set[str]]:
        """Return ``{source_id: {target_id, ...}}`` of implicit edges that pass all precision controls."""
        nodes = list(nodes)
        if self.chat_fn is None:
            print("PrecisionEdgeExtractor: no chat_fn — returning no implicit edges (the deterministic "
                  "graph stands). Pass a chat_fn (see openai_chat_fn) to enable.")
            return {}
        valid = {n.id for n in nodes}
        catalog = "; ".join(f"{n.id}: {n.text[:60]}" for n in nodes)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        def call_one(n: Node) -> tuple[str, list[dict]]:
            cpath = os.path.join(self.cache_dir, f"{n.id}.json") if self.cache_dir else None
            if cpath and os.path.exists(cpath):
                with open(cpath) as fh:
                    return n.id, json.load(fh)
            user = f"UNIT {n.id}:\n{n.text[:self.max_text_chars]}\n\nALL UNITS:\n{catalog}"
            try:
                raw = _parse_edges(self.chat_fn(self.system_prompt, user))
                if cpath:  # cache successful calls only — never persist a failure as "no edges"
                    with open(cpath, "w") as fh:
                        json.dump(raw, fh)
            except Exception as e:  # noqa: BLE001 — a failed call yields no edges for this node, not a crash
                print(f"  PrecisionEdgeExtractor fail {n.id}: {type(e).__name__}")
                raw = []
            return n.id, raw

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = list(ex.map(call_one, nodes))

        src_norm = {n.id: _norm(n.text) for n in nodes}
        edges: dict[str, set[str]] = {}
        for sid, raw in results:
            for e in raw:
                tgt = str(e.get("target", "")).strip()
                trig = str(e.get("trigger", "") or "")
                try:
                    conf = float(e.get("confidence", 0) or 0)
                except (TypeError, ValueError):
                    conf = 0.0
                if tgt not in valid or tgt == sid:
                    continue
                if conf < self.min_confidence:
                    continue
                if trig and _norm(trig)[:40] not in src_norm.get(sid, ""):  # grounding
                    continue
                edges.setdefault(sid, set()).add(tgt)
        return edges


def openai_chat_fn(model: str = "gpt-5-mini", *, provider: str = "openai",
                   base_url: Optional[str] = None, api_key: Optional[str] = None,
                   timeout: int = 120) -> Callable[[str, str], str]:
    """Build a ``chat_fn`` for :class:`PrecisionEdgeExtractor` over any OpenAI-compatible endpoint, using
    only the standard library (no ``openai`` dependency).

    ``provider='openai'`` reads ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``; ``provider='kimi'`` reads
    ``KIMI_API_KEY`` / ``KIMI_BASE_URL`` (default ``https://api.moonshot.ai/v1``). The key is read from the
    environment and never logged. ``temperature`` is omitted for models that only accept the default
    (gpt-5*/o3*/o4*, and Kimi), set to 0 otherwise, so extraction is as deterministic as the model allows.
    """
    import urllib.request

    if base_url is None:
        base_url = (os.environ.get("KIMI_BASE_URL", "https://api.moonshot.ai/v1") if provider == "kimi"
                    else os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    base_url = base_url.rstrip("/")
    if api_key is None:
        api_key = os.environ["KIMI_API_KEY" if provider == "kimi" else "OPENAI_API_KEY"]
    fixed_temp = provider == "kimi" or model.startswith(("gpt-5", "o3", "o4"))

    def chat(system: str, user: str) -> str:
        payload: dict = {"model": model, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": user}]}
        if not fixed_temp:
            payload["temperature"] = 0
        req = urllib.request.Request(
            f"{base_url}/chat/completions", data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"]

    return chat
