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
| p50 / p95 full analysis, hosted | **8.2 s / 14.6 s** (min 7.2, max 17.0; n = 10 runs, 5 files × 2) | **p50 8.5 s / p95 12.5 s / p99 13.6 s** on the grounded eval layer (n = 6, cache off), now printed by the eval runner | `PROFILE_CACHE_ENABLED=false python -m evals.run_evals --with-llm` |
| Where the time goes (p50) | profile 5.2 s · retrieve 2.6 s (multi-query rewrites + embed + fuse) · everything else < 10 ms | | same |
| Cost per statement, hosted | **$0.0098** (p50; max $0.0101) | **$0.0097 per query** (eval runner, mean of 6); **$0.0096** on a traced live run (2,636 in / 452 out tokens) | eval runner; `make trace` |
| Where the money goes | profile 99.0 % (≈2,375 in / 379 out tokens on gpt-4o) · retrieve 1.0 % (gpt-4o-mini rewrites) · categorize 0.0 % | | same |
| Categorizer LLM fallback | **0 rows** on all sample statements (rules matched everything) | | same, `llm_fallback_rows` |
| Cached second run | ≈18 s vs ≈40 s in production UI (older figure, includes Streamlit overhead) | | `docs/06_llmops_production_and_cost.md` |
| Vector store warm start | 1.3 s (fingerprint match, no re-embed) | | same script |
| Scanned PDF | **fails without vision OCR** (`VISION_OCR_ENABLED=false` by default; OCR sends page images out before masking) | | `data/sample_4_scanned_statement.pdf` |
| Tests | **283 passed**, 163 test functions in 12 files, 10.2 s | **342 passed** in 18 files, 40 s (boots a throwaway Postgres) | `python -m pytest -q` |
| Code size | 6,304 lines across `app/`, `evals/`, `mcp_server.py`; 10 knowledge-base documents, 47 chunks | | `wc -l` |
| Tokens saved by cache | n/a | exact-key cache now shared (Postgres, tenant-scoped); a hit skips the whole `llm.profile` span (≈2,300 in / 370 out tokens, ≈$0.0095) | `profile_cache` table, `hits` column |
| Guardrail block rate | n/a (suite built in Phase 5) | | |
| Bulk throughput and cost | n/a | **50 statements in 8.5 s worker time (356.7/min), p50 36 ms, p95 1.6 s, $0 (ingest only, concurrency 2)** | `make load` |
| Tenant isolation test | n/a | **7 tests in `tests/test_tenancy.py` + `make prove-isolation` (RLS on → nothing; RLS off → row leaks; on → nothing)** | `make prove-isolation` |
| Local-model golden pass rate | n/a | see Phase 4 findings (`make compare`) | `make compare` |
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

## Phase 3 findings (2026-09-08)

- **One trace, read aloud.** Live run for Meridian customer M-1002: 10.4 s total, of which retrieve 4.7 s (multi-query rewrites 4.3 s on the mini model, $0.0001) and narrate 5.6 s (gpt-4o, 2,307 in / 369 out tokens, $0.0095). The deterministic nodes together took under 25 ms. Retrieval is 45 % of the wall time for 1 % of the cost: the Phase 4 cache and gateway work should start there.
- The HTTP request span is 5 ms because the response streams; the run's own span (`graph.run`) is the one that closes with the last event, so a streamed run is still one complete trace.
- Span store: 11 spans per run, ~1 KB each, written by the exporter as the owner and read under RLS. Spans without a tenant (health checks) are not stored.
- Prompt registry: one row, `system_prompt` version `5e372eacf2da` × `gpt-4o`, `uses` incremented on every uncached narrate.

## Phase 4 findings (2026-09-08)

- **Zero-key path works end to end.** With `OPENAI_API_KEY` empty the gateway runs on `qwen2.5:3b` + `nomic-embed-text` (Ollama, 8 GB laptop): Meridian index built in 23 s, retrieval 6.9 s, profile 26.7 s, and the 3B model produced a profile that passed catalogue validation ("Sweep-In Fixed Deposit" / "Recurring Deposit").
- **Bulk ingest is cheap and fast**: 50 synthetic statements, worker concurrency 2, 8.5 s, p50 36 ms per statement, no model calls, so $0.
- **Budget enforcement reads the same table tracing writes.** `GET /platform/gateway` after the day's runs: spent $0.0096 of $2.00, would use `openai` (primary), both circuits closed.
- Side-by-side golden suite: table below.

### Side by side: the same golden suite on two providers (`make compare`, 2026-09-08)

| Check (grounded layer, 6 sampled cases) | OpenAI gpt-4o | Ollama qwen2.5:3b (local, 8 GB laptop) |
|---|---:|---:|
| credit_guardrail | 6/6 | 6/6 |
| products_are_real | 6/6 | 6/6 |
| retrieval_supports_recommendation | 6/6 | 6/6 |
| sources_present | 6/6 | 6/6 |
| percentages_supported (advisory) | 6/6 | **0/6** |
| cases with errors | 0 | 0 |
| p50 / p95 latency | 8.5 s / 12.0 s | **34.9 s / 50.4 s** |
| cost per query | $0.0096 | $0 (the first run mis-priced local tokens at OpenAI rates, showing $0.0115; fixed in `evals/run_evals.py`) |

What the table says, said out loud: the 3B local model passes every **blocking** check, including the credit guardrail and catalogue validation, so the zero-key demo is safe to show. It fails the advisory "quotes the exact percentages" check on every case and is four times slower. That is the honest price of running without a key, and it is a number, not a guess.

## Caveats I say out loud

- Ten timed runs is enough for a baseline, not for a p99. Phase 3 tracing will give real percentiles.
- Cost uses list prices and callback token counts; embeddings are omitted (fractions of a cent).
- The sample statements are clean, so the categorizer never called the model. Real statements will.
- The retrieval numbers are from the golden queries, not from real RM queries.
- Everything here is single-tenant, single-user, no auth, no queue: the "before" picture on purpose.

Raw runs: `evals/baseline/baseline_runs.json`.
