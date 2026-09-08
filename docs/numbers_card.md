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
| Tests | **283 passed**, 163 test functions in 12 files, 10.2 s | **320 passed** in 16 files, 25 s (boots a throwaway Postgres) | `python -m pytest -q` |
| Code size | 6,304 lines across `app/`, `evals/`, `mcp_server.py`; 10 knowledge-base documents, 47 chunks | | `wc -l` |
| Tokens saved by cache | n/a (measured in Phase 4) | | |
| Guardrail block rate | n/a (suite built in Phase 5) | | |
| Bulk throughput and cost | n/a (Phase 4) | | |
| Tenant isolation test | n/a | **7 tests in `tests/test_tenancy.py` + `make prove-isolation` (RLS on → nothing; RLS off → row leaks; on → nothing)** | `make prove-isolation` |
| Local-model golden pass rate | n/a (Phase 4) | | |
| Minutes saved per statement | ~20 min manual (assumption, `docs/discovery.md`) | | |

## Phase 1 findings (2026-09-08)

- The API adds ~3 ms per read (`GET /statements` served in 3 ms from Postgres); upload of a 120-line CSV including parse, mask, categorise, compute and store ran in under 200 ms. The profile call is unchanged at ~16 s wall time, all of it the model.
- **Harbor grounding gap.** The first real Harbor profile recommended *Term Deposit* and *High-Yield Savings Account* (both valid Harbor products, validated by the tenant-aware catalogue) while retrieval had surfaced `auto_loan.md`, `cashback_credit_card.md`, `everyday_chequing.md`. Meridian's golden set does not cover Harbor; `retrieval_supports_recommendation` would have flagged this. The retrieval query and the golden set need per-tenant variants (Phase 5/6 backlog).
- The categorizer still needed no LLM call for any seeded statement.

## Phase 2 findings (2026-09-08)

- **Interrupt and resume survive a process restart.** Live run for Harbor customer H-2002: declared 40,000 vs observed 50,000 (25 %, above the 20 % threshold) paused the run; the API was killed and restarted; the reviewer approved; the graph resumed from `await_review` and finished. Same proof as `tests/test_graph.py::test_run_pauses_for_review_and_resumes_from_checkpoint`, which uses two separate app instances.
- Audit trail for that run: 8 rows, actors `rm@harbor.example` → `reviewer@harbor.example` → `system`; narrate recorded `gpt-4o`, prompt `5e372eacf2da`, 5,269 in / 801 out tokens, 10.2 s. First retrieval for Harbor took 14.4 s because its Chroma index was built on first use.
- The guardrail node's "not among retrieved sources" warning fired again on Harbor (secondary product). The Harbor grounding gap from Phase 1 is now visible in the audit trail, not only in a note.
- LangGraph's checkpoint tables are not tenant-scoped (addressed by run id only). Known limitation, recorded in migration 0002.

## Caveats I say out loud

- Ten timed runs is enough for a baseline, not for a p99. Phase 3 tracing will give real percentiles.
- Cost uses list prices and callback token counts; embeddings are omitted (fractions of a cent).
- The sample statements are clean, so the categorizer never called the model. Real statements will.
- The retrieval numbers are from the golden queries, not from real RM queries.
- Everything here is single-tenant, single-user, no auth, no queue: the "before" picture on purpose.

Raw runs: `evals/baseline/baseline_runs.json`.
