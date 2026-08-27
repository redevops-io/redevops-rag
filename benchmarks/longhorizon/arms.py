"""Context-management arms + the fixed-window answerer call.

`full` = feed the whole haystack truncated to the answerer's window (the baseline CR must beat).
`cr`   = feed only the top-k chunks retrieved from the haystack (global retrieval).
`cr-enterprise`/`cr-materialize` swap in the F4/F2 optimizations (now in the AGPL core); the answerer
never changes. The arm labels predate the open-sourcing and are kept for result continuity.

The answerer is a PLAIN fixed-window consumer (system prompt: answer only from context) — deliberately NOT
a model that manages context itself, so the measured frontier is attributable to the arm, not the model.
"""
from __future__ import annotations

from .tasks import TOK, Item

SYSTEM = ("Answer using ONLY the provided context. Reply with the exact value and nothing else. "
          "If the answer is not present in the context, reply exactly: NOT FOUND.")


def ctx_full(item: Item, window_tokens: int) -> str:
    """The whole haystack, truncated to the answerer's fixed window (chars ≈ tokens*4)."""
    return item.haystack_text()[: window_tokens * TOK]


def ctx_cr(store, item: Item, query: str, *, limit: int = 8, pool: int = 50) -> str:
    """CR context management: retrieve the top-k chunks from THIS item's haystack only."""
    from redevops_rag.retrieve import hybrid_search
    hits = hybrid_search(store, query, limit=limit, pool=pool, document_ids=item.doc_ids)
    return "\n\n".join(h["text"] for h in hits)


def ctx_cr_enterprise(retriever, query: str, *, limit: int = 8) -> str:
    """F4 arm: the sparse-region retriever narrows to top-K regions (deterministically, with a
    confidence-floor→scoped fallback) before the same hybrid retrieval runs. `retriever` is a
    context_runtime.adapters.sparse_regions.SparseRegionRetriever; its `.search` returns store Hits."""
    # F4 is identity-transparent: it returns whatever the underlying store search returned. This harness's
    # backend is hybrid_search → dicts; a live runtime store → Hit objects. Read either.
    hits = retriever.search(query, limit)
    return "\n\n".join((h["text"] if isinstance(h, dict) else h.text) for h in hits)


def _complete(client, model, ctx, question, extra_body, temperature, max_tokens):
    r = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens, extra_body=extra_body or {},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"}])
    return (r.choices[0].message.content or "").strip()


def answer_llm(client, model: str, ctx: str, question: str, *, extra_body=None,
               temperature: float = 0.0, max_tokens: int = 64) -> str:
    # Real tokenizers pack fewer chars per token than the chars/4 estimate, and the ratio varies by corpus
    # (strategywiki ~3.65, simplewiki ~3.40) — so a char-budgeted `full` context can still exceed the
    # model's window. Rather than guess a corpus-specific margin, retry on the model's context-length
    # rejection, trimming the context until it fits. Deterministic and corpus-agnostic.
    for _ in range(4):
        try:
            return _complete(client, model, ctx, question, extra_body, temperature, max_tokens)
        except Exception as e:
            if "maximum context length" in str(e) or "context length" in str(e).lower():
                ctx = ctx[: int(len(ctx) * 0.85)]        # trim 15% and retry
                continue
            raise
    return _complete(client, model, ctx, question, extra_body, temperature, max_tokens)


def answer_oracle(ctx: str, question: str) -> str:
    """--dry answerer: no LLM. Returns the context so scoring measures whether the ARM actually delivered
    the needle to the model — i.e. plumbing correctness (retrieval found it, truncation didn't drop it),
    isolated from model quality."""
    return ctx


def scored(needle_answer: str, model_output: str) -> bool:
    """Deterministic: the distinctive code appears in the output (case-insensitive substring)."""
    return needle_answer.lower() in (model_output or "").lower()


def est_tokens(text: str) -> int:
    return len(text) // TOK
