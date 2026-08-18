# Submission

These are the files submitted to TREC RAGTIME 2026, exactly as filed. Each task folder also contains
the `SUBMISSION_INFO.txt` that accompanied its three runs and records the requested assessment order.

| Task | File | Run name | What the run is |
|---|---|---|---|
| 1 — Report Generation | `task1/T1_run_1.jsonl` | `T1-e2e-omt` | End-to-end agentic RAG reading our NLLB-200-3.3B translation of the corpus into English; per-sentence cited English report. |
| 1 — Report Generation | `task1/T1_run_2.jsonl` | `T1-e2e-omt-weak` | The same pipeline reading a low-tier Opus-MT translation instead, so translation quality is the only variable against run 1. |
| 1 — Report Generation | `task1/T1_run_3.jsonl` | `T1-e2e-original` | The same pipeline reading the native, untranslated passages. |
| 2 — Multilingual Retrieval | `task2/T2_run_1.txt` | `mlir-original` | Decomposition-driven retrieval over the native index, ranking documents by the accumulated importance of the claims they support. |
| 2 — Multilingual Retrieval | `task2/T2_run_2.txt` | `mlir-omt` | The same retrieval over the NLLB-200-3.3B English index. |
| 2 — Multilingual Retrieval | `task2/T2_run_3.txt` | `mlir-omt-weak` | The same retrieval over the Opus-MT English index. |
| 3 — Auto-nuggetization | `task3/T3_run_1.jsonl` | `T3-e2e-omt` | Nugget bank from the run behind `T1-e2e-omt`, grown across coverage-audit rounds to saturation and then deduplicated. |
| 3 — Auto-nuggetization | `task3/T3_run_2.jsonl` | `T3-e2e-omt-weak` | Nugget bank from the run behind `T1-e2e-omt-weak`. |
| 3 — Auto-nuggetization | `task3/T3_run_3.jsonl` | `T3-e2e-original` | Nugget bank from the run behind `T1-e2e-original`. |

Tasks 1 and 3 are two views of the same three executions: the report and the nugget bank produced by
one `e2e-*` run. Task 2 comes from two `mlir-*` runs plus a third file, `T2_run_1`, serialized from
the `e2e-original` cells rather than from a separate `mlir-original` execution — those two
configurations are identical on both knobs, so that run was not executed twice.
