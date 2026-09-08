# Data handling and retention — BankLens Platform

**Version** 1.0 · **Date** 2026-09-08

## What the system stores, and where

| Data | Table | Contains PII? | Retention |
|---|---|---|---|
| Tenants, users | `tenants`, `users` | Staff emails, bcrypt password hashes | Life of the tenant |
| Customers | `customers` | Synthetic name, external reference, declared income | Life of the customer record |
| Statements and ledger lines | `statements`, `transactions` | Descriptions **after** PII masking (`app/pipeline/sanitizer.py`) and instruction neutralisation; amounts, dates, categories | Configurable per tenant; default keep |
| Computed metrics | `statement_metrics` | No | With the statement |
| Profiles | `profiles` | Model narrative about the statement; no account identifiers | With the statement |
| Runs, decisions | `runs`, `decisions` | Declared/observed income, reviewer email, note | Audit horizon |
| Audit trail | `audit_events` | Actor email, **hash** of inputs, model, prompt version, tokens | Append-only; API role cannot update or delete; audit horizon |
| Spans | `spans` | Ids and numbers only; never statement text | 30 days suggested |
| Query log | `query_log` | Template name, parameters (ids, categories), role, actor | Audit horizon |
| Jobs | `jobs` | The uploaded file bytes until processed | Purge `content` after `done`; suggested 7 days |
| Profile cache | `profile_cache` | Narrative keyed by a hash of computed inputs | 30 days suggested |
| LangGraph checkpoints | `checkpoints`, `checkpoint_blobs`, `checkpoint_writes` | Graph state, incl. metrics and the retrieved passages | Purge with the run; **not tenant-scoped** (known limitation) |

Every tenant table carries a row-level-security policy; the API role sees
one tenant per transaction; the chat role sees only tenant-filtered views.

## What leaves the bank's boundary

- **Text to a hosted model**: masked descriptions, computed metrics, catalogue
  passages, the RM's question after redaction. Never account, card, phone,
  email, PAN or SSN tokens (masked by regex before any model).
- **Page images**: only when `VISION_OCR_ENABLED=true`, for scanned PDFs, to
  the hosted vision model, **before** masking is possible. Off by default.
- **Nothing**, when the gateway runs on the local provider (`LLM_PROVIDER=ollama`
  or no API key): embeddings and generation stay on the host.

## Deletion

Deleting a customer cascades to statements, transactions, metrics, profiles,
runs, decisions, audit events, spans and query-log rows that reference them
(foreign keys with `ON DELETE CASCADE`). Checkpoint rows must be deleted by
run id in the same operation; a helper is on the Phase 8 backlog.

## Demo data

Every tenant, user, customer and statement in this repository is synthetic.
The sample statements are generated; the names are invented; no real person,
account or institution is represented. Load-test statements are perturbed
copies of the samples.

## Secrets

API keys and database URLs come from the environment (`.env`, never
committed). The JWT signing secret has a development default that the API
refuses to start with when `BANKLENS_ENV=production`.
