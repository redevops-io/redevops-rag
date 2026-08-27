"""Build a long-horizon needle-QA item: a real Wikimedia haystack padded to a target token budget with
K synthetic fact-needles inserted at controlled depths.

Deterministic (seeded): the same (articles, budget, needles, seed) always yields the same haystack, chunks,
and gold — so a run is reproducible and comparable against a frozen baseline. Scoring is substring match on
the needle value, which is a distinctive code absent from wiki prose (not derivable from model priors).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .corpus import Article

TOK = 4                      # ~chars per token (consistent with the rest of the benchmark suite)
CHUNK_CHARS = 2048           # ~512 tokens/chunk
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no ambiguous 0/O/1/I


def _code(rng: random.Random, n: int = 6) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(n))


@dataclass
class Needle:
    qid: str
    key: str            # a distinctive subject the question names (e.g. "GATE-7F")
    question: str
    answer: str         # the gold value (a distinctive code)
    depth: float        # relative position in the haystack [0,1]


@dataclass
class Item:
    item_id: str
    budget_tokens: int
    chunks: list[dict] = field(default_factory=list)   # store-ready: {id, document_id, filename, text}
    needles: list[Needle] = field(default_factory=list)
    est_tokens: int = 0

    @property
    def doc_ids(self) -> list[str]:
        return [c["document_id"] for c in self.chunks]

    def haystack_text(self) -> str:
        return "\n\n".join(c["text"] for c in self.chunks)


def _chunk_articles(articles: list[Article], budget_tokens: int) -> list[dict]:
    """Fill the haystack to ~budget by chunking distractor articles (cycling if the corpus is small)."""
    cap_chars = budget_tokens * TOK
    chunks, used, ai = [], 0, 0
    n = len(articles)
    while used < cap_chars:
        art = articles[ai % n]
        ai += 1
        for i in range(0, len(art.text), CHUNK_CHARS):
            piece = art.text[i:i + CHUNK_CHARS]
            chunks.append({"filename": art.title, "_body": piece})
            used += len(piece)
            if used >= cap_chars:
                break
        if ai > n * 50:          # safety: corpus far too small for the budget
            break
    return chunks


def build_item(articles: list[Article], budget_tokens: int, *, n_needles: int = 4,
               item_id: str = "lh", seed: int = 0) -> Item:
    rng = random.Random(f"{item_id}:{budget_tokens}:{seed}")
    raw = _chunk_articles(articles, budget_tokens)
    if len(raw) < n_needles + 2:
        raise SystemExit(f"haystack too small ({len(raw)} chunks) for {n_needles} needles at {budget_tokens} tok")

    needles: list[Needle] = []
    for i in range(n_needles):
        depth = (i + 1) / (n_needles + 1)
        # single unhyphenated alnum token → clean BM25/FTS tokenisation, so a missed needle reflects a
        # context-management failure (the thing under test), not a tokenizer splitting the key.
        key = f"F{_code(rng, 5)}{i}"
        val = _code(rng, 6)
        pos = min(int(depth * len(raw)), len(raw) - 1)
        # Embed the needle in the middle of a normal-looking chunk so it is not an outlier by length; the
        # question shares the KEY term, so keyword+vector retrieval must actually locate this chunk.
        body = raw[pos]["_body"]
        mid = len(body) // 2
        sentence = f" The secure access code for facility {key} is {val}. "
        raw[pos]["_body"] = body[:mid] + sentence + body[mid:]
        needles.append(Needle(
            qid=f"{item_id}-n{i}", key=key,
            question=f"What is the secure access code for facility {key}?",
            answer=val, depth=depth,
        ))

    chunks = [{"id": f"{item_id}:c{idx}", "document_id": f"{item_id}:c{idx}",
               "filename": c["filename"], "text": c["_body"]} for idx, c in enumerate(raw)]
    est = sum(len(c["text"]) for c in chunks) // TOK
    return Item(item_id=item_id, budget_tokens=budget_tokens, chunks=chunks, needles=needles, est_tokens=est)
