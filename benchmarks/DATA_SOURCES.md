# Data sources & licensing

The benchmark datasets here derive from the public sources below (plus one proprietary internal corpus). Loaders/builders fetch or rebuild the data; we do not redistribute raw upstream data. Each dataset stays under its own license.

## How this data is used

These datasets are used solely to develop and benchmark the Context Runtime software itself (internal R&D and regression testing of stack improvements). They are NOT redistributed, NOT embedded in or shipped with the product, and NOT part of any customer deployment. Context Runtime customers apply the software to their own data under their own data licenses.

## MuSiQue (multi-hop QA) - datasets/musique.jsonl
- Source: https://github.com/StonyBrookNLP/musique
- License: CC-BY-4.0.
- Citation: Trivedi et al., "MuSiQue: Multihop Questions via Single-hop Question Composition," TACL 2022.

## PopQA (entity/long-tail QA) - datasets/popqa.jsonl
- Source: https://github.com/AlexTMallen/adaptive-retrieval (also mirrored at HuggingFace `akariasai/PopQA`).
- License: MIT.
- Citation: Mallen et al., "When Not to Trust Language Models," ACL 2023.
- Note: the distractor docs around each Q/A are synthetically constructed; the questions/answers/gold are from PopQA.

## LongMemEval-S (long-term conversational memory) - datasets/longmemeval.jsonl
- Source: https://github.com/xiaowu0162/LongMemEval (HuggingFace `xiaowu0162/longmemeval`).
- License: MIT.
- Citation: Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory," ICLR 2025.

## TempReason (temporal reasoning) - datasets/tempo.jsonl (rebuilt by build_tempo.py)
- Source: HuggingFace `tonytan48/TempReason` (DAMO-NLP-SG).
- License: CC-BY-SA-3.0.
- Citation: Tan et al., "Towards Benchmarking and Improving the Temporal Reasoning Capability of Large Language Models," ACL 2023 (arXiv:2306.08952).

## tempo26/Tempo (temporal retrieval) - pulled at runtime by eval scripts
- Source: HuggingFace `tempo26/Tempo` (derived from Stack Exchange data).
- License: CC-BY-SA (inherited from Stack Exchange; no separate license field declared upstream).

## Nutrition (RU) - datasets/nutrition.jsonl - PROPRIETARY, not public
- Source: internal nutribot corpus - Russian nutrition-lecture material (third-party author "Nezhenskaya"), exported from Telegram; Q/A LLM-generated over it.
- License: none. This is third-party copyrighted course content used INTERNALLY as an anti-contamination control (no model could have trained on it). It is NOT redistributed; only aggregate recall figures are ever reported. Not an open dataset.
- Note: rebuilt 2026-07-22 (the original set was lost when systemd-tmpfiles swept `/tmp`); recall figures are **not comparable** to previously published nutrition numbers.
