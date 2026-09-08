# Model card — BankLens Platform

**Version** 1.0 · **Date** 2026-09-08 · numbers from `docs/numbers_card.md`, reproducible with the commands there.

BankLens is not one model. It is a pipeline in which models do bounded jobs
behind a gateway. This card lists each job, the model that does it today,
what it may and may not decide, and how it was evaluated.

## Models in use

| Job | Hosted (default with a key) | Local (zero-key) | Decides anything? |
|---|---|---|---|
| Profile narrative + product recommendations | `gpt-4o`, temperature 0.2 | `qwen2.5:3b` via Ollama | **No.** Narrates computed numbers; product names validated against the tenant catalogue; fails closed on violations |
| Query rewrites for retrieval (multi-query) | `gpt-4o-mini`, temperature 0 | `qwen2.5:3b` | No. Retrieval candidates only, fused with anchored RRF |
| Categorizer fallback for unmatched merchants | `gpt-4o-mini`, temperature 0 | `qwen2.5:3b` | Assigns a spending category to rows the rules missed; a wrong category moves the essential/discretionary split, never income or the risk band |
| Tool-calling chat | `gpt-4o`, temperature 0.3 | `qwen2.5:3b` | No. Tools return computed metrics, catalogue passages and category totals; numeric questions bypass the model entirely (warehouse templates) |
| Embeddings | `text-embedding-3-small` | `nomic-embed-text` | No. Indexes are kept per tenant per provider |
| Reranker (shipped **off**) | `gpt-4o-mini` listwise | — | No. Disabled after an A/B measured a safety regression (harmful credit content to deficit customers 0.40 → 0.91) |
| Eval judge (advisory) | `gpt-4o-mini`, temperature 0.4 | — | No. Advisory groundedness verdicts only |

Provider selection, retries, circuit breakers, fallbacks and per-tenant
budgets live in `app/platform/gateway.py`. Prices in `app/platform/pricing.py`.

## Intended use

Internal tooling for a bank's relationship managers and credit reviewers.
Inputs: one customer's statement (CSV or PDF) and the bank's product
catalogue. Outputs are read by trained staff with the computed numbers
beside them. **Not** intended for direct customer-facing use, automated
lending decisions, or any decision without the numbers in view.

## Evaluation

Golden set: 65 deterministic cases (savings-rate sweep + edge cases), 6
sampled grounded cases, advisory judge. Retrieval: hit@4 1.000, MRR 0.377,
nDCG 0.879 with the reranker off.

| Grounded layer (6 cases) | gpt-4o | qwen2.5:3b (local) |
|---|---:|---:|
| credit_guardrail | 6/6 | 6/6 |
| products_are_real | 6/6 | 6/6 |
| retrieval_supports_recommendation | 6/6 | 6/6 |
| sources_present | 6/6 | 6/6 |
| percentages_supported (advisory) | 6/6 | 0/6 |
| p50 / p95 latency | 8.5 s / 12.0 s | 34.9 s / 50.4 s |
| cost per query | $0.0096 | $0 |

Guardrails: 80-case red-team suite, 100 % of attacks blocked, 0 % false
positives. Bias: `evals/bias_check.py` shows identical risk bands and scores
across demographic rewrites of every golden statement.

## Prompt versions

The system prompt is hashed; every profile stores its hash and model, and
`prompt_versions` records which versions ran against which models and how
often. Current: `system_prompt` `36d7cd7653ee` (Phase 8 made the product guidance bank-neutral; `5e372eacf2da` before that).

## Limitations and failure modes

- Narrative quality depends on the provider. The local model is safe (passes
  every blocking check) but imprecise with figures.
- Product recommendations are grounded in retrieved passages; the guardrail
  node warns when a recommendation was not among retrieved sources (seen on
  the Harbor catalogue).
- The categorizer fallback can mislabel unusual merchants; this moves the
  essential/discretionary split only.
- Scanned PDFs need vision OCR, which is off by default for privacy.

## Owners and change control

Prompt and code changes go through the CI gate: unit tests, the deterministic
golden layer, and the red-team suite on every push; the grounded and judged
layers nightly. A prompt change is a new hash in the registry and a new
`prompt_version` on every profile it produces.
