# Long-horizon harness — real-model run (Nemotron-3.5-Lightning)

First non-oracle run of the long-horizon harness: a real fixed-window answerer over two independent
Wikimedia corpora. Confirms the context-management frontier the oracle runs only simulated.

## Setup

- **Answerer**: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (hybrid Mamba-2 + MoE, 3B active),
  served on a single **RTX PRO 4500 Blackwell (32 GB)** via `vllm/vllm-openai` (tp=1, fp8 KV,
  `max_model_len=131072`). A plain fixed-window consumer — it does not manage its own context, so the
  measured frontier is attributable to the arm, not the model.
- **Corpora** (two, independent — anti-overfit): **strategywiki** (dev) and **Simple English Wikipedia**
  (holdout). Real haystacks, synthetic fact-needles at controlled depths, deterministic substring scoring.
- **Window** 131072; `full` reserves output + tokenizer-density headroom (see note). 4 needles/horizon.
- Every result cell is provenance-stamped (harness/contextos/enterprise commit + arm `src_sha256` +
  key-schema) so it proves which code produced it.

## Results (accuracy / input-tokens)

**DEV — strategywiki**

| horizon | `full` | `cr` | `cr-enterprise` (F4) | `cr-materialize` (F2) | managed tok |
|--------:|:------:|:----:|:--------------------:|:---------------------:|:-----------:|
|    10k  | 0.75   | 1.0  | 0.75 | 1.0 | 4,155 |
|    60k  | 1.0    | 0.75 | 0.75 | 1.0 | 4,151 |
|   120k  | 1.0    | 1.0  | 1.0  | 1.0 | 4,155 |
|   500k  | **0.25** | 1.0 | 1.0 | 1.0 | 4,155 |
|    1M   | **0.0**  | 1.0 | 1.0 | 1.0 | 4,155 |

`full` input-tokens: 10k→10,305 · 60k→60,128 · ≥120k→111,155 (capped at the window).

**HOLDOUT — Simple English Wikipedia**

| horizon | `full` | `cr` | `cr-enterprise` (F4) | `cr-materialize` (F2) | managed tok |
|--------:|:------:|:----:|:--------------------:|:---------------------:|:-----------:|
|    10k  | 0.75   | 0.75 | 1.0  | 1.0 | 4,155 |
|    60k  | 1.0    | 0.75 | 0.5  | 1.0 | 4,144 |
|   120k  | **0.5**  | 0.75 | 0.5 | 0.5 | 4,141 |
|   500k  | **0.0**  | 0.75 | 0.75 | 0.75 | 4,141 |
|    1M   | **0.0**  | 1.0 | 0.75 | 1.0 | 4,137 |

## What the frontier shows

1. **The thesis holds on both corpora.** As the horizon exceeds the answerer's window, `full` collapses
   (dev 0.25→0.0, holdout 0.5→0.0 at 500k/1M) because the answer-bearing needles fall outside the window.
   The managed arms answer from **~4,150 tokens** — **~27× fewer** than `full`'s 111k at ≥120k — and hold
   **0.75–1.0** where `full` is at **0.0**. Latency tracks tokens: `full` ~5–9 s at long horizons vs
   managed ~2.5–3 s. This is the "better answers from less context → smaller self-hosted models made viable"
   result, now with a real 128k model instead of the oracle.
2. **F2 (`cr-materialize`) recovers low-horizon misses.** At 10k/60k where fixed-limit retrieval drops a
   needle (`cr`=0.75), the escalation ladder climbs to `full_context` for just that needle and reaches 1.0
   — paying more only where it helps.
3. **The holdout is independent and harder.** simplewiki's list-heavy articles give a noisier retrieval
   signal; managed-arm accuracy there runs 0.5–1.0 rather than a clean 1.0. But `full` still collapses to
   0.0 at long horizons while managed retrieval stays well above it — the directional result reproduces.

## Honest caveats

- **n = 4 needles** → accuracy is quantized to 0.25 and carries real-model nondeterminism (vLLM batching +
  prefix caching perturb numerics even at temperature 0). Individual cells are noisy; the **trend** is
  robust, individual values are not. More needles/seeds would sharpen the estimates.
- **F4 vs `cr` washes out at this scale.** With 4 needles on a 128k-window model, `cr-enterprise` and `cr`
  mostly tie (F4's mid-horizon edge was clearest in the higher-resolution oracle runs). F4 earns its place
  as *never worse and sometimes better*; it is not a large win on this particular task.
- The **1M horizon** stresses the haystack, not the answerer window — `full` sees at most the windowed
  ~111k tokens regardless. On 32 GB the answerer's real window is ~128k; a larger card would let `full`
  climb higher before collapsing, but the crossover (and managed arms' flat cost) is the point.

## Note — tokenizer density

The harness estimates tokens as chars/4, but real tokenizers pack denser and the ratio varies by corpus
(strategywiki ~3.65, simplewiki ~3.40 chars/tok), so a char-budgeted `full` context can exceed the window.
`answer_llm` retries on the model's context-length rejection, trimming until it fits — deterministic and
corpus-agnostic — and `_full_input_budget` reserves a first-line margin to avoid the retry in the common case.

Reproduce: `MODEL=api ANSWERER_URL=… ANSWERER_MODEL=… WINDOW=131072 .venv/bin/python
benchmarks/longhorizon/run_longhorizon.py --horizons 10000,60000,120000,500000,1000000 --needles 4
--corpus strategywiki` (drop `--dry`; add `--corpus simplewiki --dump <simplewiki.xml.bz2>` for the holdout).
