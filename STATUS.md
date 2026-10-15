# Status

| | |
|---|---|
| Maturity | alpha (0.x library; API said to be backwards-compatible across index generations v1-v5, per README) |
| Latest git tag | `v0.4.0` (tags: v0.2.0, v0.2.1, v0.3.0, v0.4.0) |
| Package `__version__` | `0.2.0` (`src/redevops_rag/__init__.py`; pyproject reads it dynamically) |
| Last git activity | 2026-10-13 (`git log -1 --format=%cs`; clone is shallow) |
| Distribution | Install from git only; not on PyPI |

## Supported
- Hybrid search (DuckDB vector + BM25 + RRF, recency/keyword priors), optional cross-encoder rerank, CLI (`index`, `search`, `ask`), content-addressed chunk ids.
- `citation_graph` (`ChangeClosureGraph`, `Node`, `union_retrieve`): pure Python, no heavy dependencies.
- CI (`.github/workflows/tests.yml`) runs `pytest -q` with the `[pg]` extra; the 14 Postgres store tests skip without a live database; the `[rerank]` extra is not exercised in CI.

## Not supported / not claimed
- No third-party reproduction of any benchmark. `benchmarks/` scripts and `benchmarks/results/` are internally produced.
- The `citation_graph` comparison numbers in the README (e.g. 0.556 vs 0.159) are **not reproducible from this repo**: the data lives in a private repository (`redevops-benchmarks`).
- `rag-saas-platform`, the project this was carved out of, is private.
- `ask` requires you to supply an OpenAI-compatible LLM endpoint.

## Known issues
- Version mismatch: tag `v0.4.0` vs `__version__ == "0.2.0"`. Documented only; no code change made in the docs PR.
- Licence wording is inconsistent (README footer vs `LICENSE.md`); under counsel review. See `LICENSE.md`.
