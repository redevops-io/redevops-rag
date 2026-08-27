# Long-horizon context-management harness (10K → 1M tokens)

Measures the **quality × cost × latency frontier** of context-management strategies as the available
context grows from 10K to 1M tokens, using a **fixed-window answerer that does not manage context itself**.
It is the measurement infra for the enterprise Context-Runtime optimizations (F4 sparse retrieval, F2
adaptive materialization, F3 pattern memory): each optimization is an arm, measured against the frozen
AGPL baseline. Lives in AGPL `redevops-rag/benchmarks` (same `Store`/`Embedder`/`hybrid_search` stack as
the cube/matrix); the enterprise overlay is injected as a treatment arm — it is **not** required to run.

## Why these two design constraints (from the program owner)

1. **Wikimedia corpus, two independent datasets.** A *dev* corpus during implementation and a separate
   *holdout* corpus for independent validation afterward — so the harness (retrieval params, chunking,
   prompt) is not tuned to the data it will be judged on.
   - **dev** = `strategywiki` (already sha256-pinned + on disk in `context-runtime-bench`).
   - **holdout** = a different Wikimedia wiki (e.g. Simple English Wikipedia), fetched at validation time.
2. **The answerer must not do its own context management.** DeepSeek-V4-Flash has built-in context
   management, so it would confound *our* CR context management with *its* internal one. The answerer is a
   **plain fixed-window consumer** — a weaker local model (Qwen3.6-35B, 32k window, thinking off) or an
   API model (Kimi / grok) used plainly. This is what makes the frontier attributable to CR, not the model.

This sharpens the whole thesis: a *small fixed-window* answerer + CR context management, measured against a
large-window reference, directly demonstrates **"better answers from less context → smaller self-hosted
models made viable"** at 1M scale — the answerer's window is fixed while the corpus grows past it, so beyond
the window the *only* way to answer is to manage context well.

## Task: real Wikimedia haystack + synthetic needles (deterministic, length-scalable)

Auto-generating good factual questions from wiki prose is itself error-prone and gameable by model priors.
Instead we use the standard long-context method: a **real Wikimedia haystack** (concatenated articles) into
which we insert **synthetic but realistic "fact" needles** with unique keys at controlled depths, then ask
for a value by its key. This gives:
- **Deterministic scoring** — substring/alias match of the needle value; no LLM judge, so it is cheap and
  reproducible at 1M tokens (a judge at that scale is cost- and noise-prohibitive).
- **Precise length control** — pad the haystack with distractor articles to hit exactly 10K/30K/100K/300K/1M.
- **Not gameable** — the needle value is not derivable from priors; the model must actually read the context.

The real-fact-QA variant (harder, judge-scored) is a later realism anchor, not part of the skeleton.

## Arms (context strategies)

| arm | what it feeds the fixed-window answerer | role |
|-----|------------------------------------------|------|
| `full` | the whole haystack, truncated to the answerer's window | the "give the model everything" baseline CR must beat |
| `cr`   | top-k chunks retrieved from the haystack (`hybrid_search`, scoped by `document_ids`) | AGPL context management |
| `cr-enterprise` | same, with the enterprise overlay injected (F4/F2/F3) | the enterprise treatment (added when each ships) |

`full` is allowed to win where it can (short horizons that fit the window). The point is the **crossover**:
the horizon past which managing context beats dumping it — and whether an optimization moves that crossover.

## Metrics (per arm × horizon × corpus)

- **accuracy** — needle recall (fraction of needles answered correctly).
- **input_tokens** — tokens actually fed to the answerer (cost proxy; ~chars/4, consistent with the rest of
  the suite).
- **latency_s** — wall time of the answerer call.

An arm only "wins" if it holds accuracy while cutting tokens/latency. Results are written as
`results/longhorizon/<corpus>__<arm>__<horizon>__<model>.json` — a directory diff vs the frozen baseline,
exactly like `matrix/` vs `matrix_control/`.

## Files

- `corpus.py` — load a Wikimedia dump → articles `(doc_id, title, text)`; reuses the canonical MediaWiki
  parser from `context-runtime-bench` (single source of truth for XML/namespace handling).
- `tasks.py` — build a needle QA item at a target token budget: haystack of distractor articles + K needles
  at controlled depths; returns the store chunks + the questions/gold.
- `arms.py` — `full` / `cr` context builders + the fixed-window answerer call (local qwen or api).
- `run_longhorizon.py` — the runner over horizons × arms; deterministic needle scoring; writes result cells
  and prints a frontier table. `--dry` uses an oracle answerer (no LLM) to validate the plumbing.

## Running

```bash
# plumbing smoke — no LLM, no GPU: proves corpus load + needle insertion + retrieval + scoring
python benchmarks/longhorizon/run_longhorizon.py --dry --horizons 10000,30000 --needles 4

# real run (local fixed-window answerer)
MODEL=qwen python benchmarks/longhorizon/run_longhorizon.py --horizons 10000,100000,1000000

# holdout / independent validation (different corpus, same harness)
WIKI_DUMP=/path/to/simplewiki-*.xml.bz2 CORPUS=simplewiki \
  MODEL=api ANSWERER_MODEL=kimi-k2.6 python benchmarks/longhorizon/run_longhorizon.py
```
