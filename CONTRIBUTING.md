# Contributing

Issues and pull requests are welcome via GitHub: <https://github.com/redevops-io/redevops-rag/issues>.

## Setup and tests
```bash
python -m pip install -e ".[pg]" pytest
pytest -q
```
This is what CI runs (Python 3.12). Postgres-store tests skip unless `REDEVOPS_RAG_PG_TEST_URL` points to a live database. The `[rerank]` extra pulls torch and is not needed for the suite.

## Guidelines
- Keep changes small and focused; add or update a test in `tests/` for behaviour changes.
- Do not commit secrets (a secret-scan workflow runs on the repo).
- Documentation claims with numbers must point to a script or result file in this repo, or be labelled as not reproducible from it.
- Contributions are accepted under the licence in [LICENSE.md](LICENSE.md); questions about licensing: contact to be confirmed (use a GitHub issue meanwhile).
