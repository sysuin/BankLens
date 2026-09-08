# BankLens numbers card

The numbers I quote in an interview, with the command that produced each one. A row without a
command is not a number, it is a claim. Baseline measured **2026-09-08** on `main` at commit `39a01d4`
(branch `phase-0-baseline`), macOS, Python 3.11, `gpt-4o` + `gpt-4o-mini`, reranker off, multi-query on.

| Number | Baseline (Phase 0) | After Phase 7 | Command / source |
| --- | --- | --- | --- |
| Golden-set pass rate, deterministic layer | **65 / 65** (4 checks × 65 cases, 0 failures) | | `python -m evals.run_evals` |
| Grounded layer, sampled | **6 cases**: credit_guardrail 6/6, products_are_real 6/6, retrieval_supports_recommendation 6/6, sources_present 6/6, percentages_supported 5/6 (advisory) | | `python -m evals.run_evals --with-llm` |
| Judge groundedness | not run this pass (advisory layer; costs money) | | `--with-llm --judge` |
| Retrieval hit@4 / MRR / nDCG / precision, hybrid, no reranker | **1.000 / 0.377 / 0.879 / 0.335** at candidate k=15; **harmful 0.400** on deficit cases | | `--retrieval-ab` (2026-08 run, see memory note); BM25-only half: hit@4 0.677, hit@15 1.000 |
| Reranker effect (why it ships off) | hit 1.000 → 0.923, harmful 0.400 → 0.908 | | `--retrieval-ab` |
| p50 / p95 full analysis, hosted | **8.2 s / 14.6 s** (min 7.2, max 17.0; n = 10 runs, 5 files × 2) | | `PROFILE_CACHE_ENABLED=false python -m evals.baseline.measure_baseline --repeats 2` |
| Where the time goes (p50) | profile 5.2 s · retrieve 2.6 s (multi-query rewrites + embed + fuse) · everything else < 10 ms | | same |
| Cost per statement, hosted | **$0.0098** (p50; max $0.0101) | | same, OpenAI list prices Sep 2026 |
| Where the money goes | profile 99.0 % (≈2,375 in / 379 out tokens on gpt-4o) · retrieve 1.0 % (gpt-4o-mini rewrites) · categorize 0.0 % | | same |
| Categorizer LLM fallback | **0 rows** on all sample statements (rules matched everything) | | same, `llm_fallback_rows` |
| Cached second run | ≈18 s vs ≈40 s in production UI (older figure, includes Streamlit overhead) | | `docs/06_llmops_production_and_cost.md` |
| Vector store warm start | 1.3 s (fingerprint match, no re-embed) | | same script |
| Scanned PDF | **fails without vision OCR** (`VISION_OCR_ENABLED=false` by default; OCR sends page images out before masking) | | `data/sample_4_scanned_statement.pdf` |
| Tests | **283 passed**, 163 test functions in 12 files, 10.2 s | | `python -m pytest -q` |
| Code size | 6,304 lines across `app/`, `evals/`, `mcp_server.py`; 10 knowledge-base documents, 47 chunks | | `wc -l` |
| Tokens saved by cache | n/a (measured in Phase 4) | | |
| Guardrail block rate | n/a (suite built in Phase 5) | | |
| Bulk throughput and cost | n/a (Phase 4) | | |
| Tenant isolation test | n/a (Phase 1) | | |
| Local-model golden pass rate | n/a (Phase 4) | | |
| Minutes saved per statement | ~20 min manual (assumption, `docs/discovery.md`) | | |

## Caveats I say out loud

- Ten timed runs is enough for a baseline, not for a p99. Phase 3 tracing will give real percentiles.
- Cost uses list prices and callback token counts; embeddings are omitted (fractions of a cent).
- The sample statements are clean, so the categorizer never called the model. Real statements will.
- The retrieval numbers are from the golden queries, not from real RM queries.
- Everything here is single-tenant, single-user, no auth, no queue: the "before" picture on purpose.

Raw runs: `evals/baseline/baseline_runs.json`.
