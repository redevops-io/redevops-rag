# Benchmark datasets

These are processed, unified-schema (cube-schema) slices - NOT the raw upstream
corpora. The raw upstream datasets are NOT committed here; loaders/builders fetch
or rebuild them. Full source + license details are in [../DATA_SOURCES.md](../DATA_SOURCES.md).

| File | Domain | Upstream source | License | Status |
| --- | --- | --- | --- | --- |
| `musique.jsonl` | Multi-hop QA | StonyBrookNLP/musique | CC-BY-4.0 | public |
| `popqa.jsonl` | Entity / long-tail QA | AlexTMallen/adaptive-retrieval (HF `akariasai/PopQA`) | MIT | public (distractor docs synthetic) |
| `longmemeval.jsonl` | Long-term conversational memory | xiaowu0162/LongMemEval (HF `xiaowu0162/longmemeval`) | MIT | public |
| `tempo.jsonl` | Temporal reasoning | HF `tonytan48/TempReason` (DAMO-NLP-SG) | CC-BY-SA-3.0 | rebuilt by `../build_tempo.py` |
| `nutrition.jsonl` | Nutrition retrieval (RU) | internal nutribot corpus (third-party copyrighted) | none (proprietary) | PROPRIETARY - not public, not redistributed |

See [../DATA_SOURCES.md](../DATA_SOURCES.md) for citations, exact URLs, and terms.
