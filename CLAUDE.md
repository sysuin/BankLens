# CLAUDE.md — working agreement for coding agents on BankLens

You are working on a bank back-office platform. Read this before changing
anything. The doctrines below are not style: they are what the tests and
the evaluation gate enforce, and the reason the system can be trusted.

## The doctrines

1. **Deterministic first, the model on the residual.** Every number (income,
   expenses, savings rate, ratios, risk band, health score, discrepancy) is
   computed in `app/pipeline/analyzer.py` or `app/graph/nodes.py` and unit
   tested. A model narrates numbers; it never produces one that is displayed
   as a number. Do not add a field to `ProfileNarrative` that a model could use
   to set a rating.
2. **Rules before model.** Categorisation, injection scanning, scope gating
   and intent routing are pattern-based and explainable. Reach for a model
   only when the rules genuinely cannot decide, and record why.
3. **Validate before you commit.** Model output goes through Pydantic and
   the catalogue validator; chat questions go through injection → router →
   scope → PII redaction; SQL exists only as vetted templates validated at
   import. Never call `ChatOpenAI` or `OpenAIEmbeddings` directly: use
   `app.platform.gateway.chat_model()` / `embeddings()`.
4. **A regression suite is the gate.** Before you say a change is done:
   `make gate` (lint, tests, golden evals, the red-team suite and the bias
   check, in that order). All green, or explain exactly why not.
5. **Two front doors on one engine.** The Streamlit console, the REST/SSE API
   and the MCP server call the same pipeline modules. Do not fork logic into a
   front door.
6. **Tenancy is enforced in the database.** Every tenant table has a
   row-level-security policy; every transaction pins `app.tenant_id`; the
   chat's SQL role reads views only. A new tenant table needs a policy in its
   migration and a row in `TENANT_TABLES` (a test checks both).
7. **Everything is a span and an audit row.** New graph nodes, tools and model
   calls get a span (`app.platform.tracing.span`) and, if they decide or act,
   an audit row (`app.graph.audit.record`). Attributes carry ids and numbers,
   never statement text.
8. **Zero-key must keep working.** Any new model call must run on the local
   provider too. `make evals PROVIDER=ollama` is part of the gate for changes
   to prompts, retrieval or the gateway.

## Layout

```
app/pipeline/    parse, sanitize, categorize, analyze, rag, reranker, agent, chat, cache, policy
app/graph/       LangGraph decision graph: nodes, builder (checkpointer), audit
app/platform/    gateway (providers, breaker, budgets), tracing, pricing, guardrails, registry, ratelimit
app/warehouse/   semantic_layer.yaml, semantic (loader/validator), query (runner), router (intent), pilot
app/api/         FastAPI: routes/, service, decisions, traces, deps (auth, rate limit), sse
app/db/          models, session (RLS pinning), local (embedded Postgres), seed
app/worker.py    bulk job worker (SKIP LOCKED, leases)
app/main.py      Streamlit console (direct mode and API mode)
evals/           golden set, run_evals, compare_providers, redteam/, bias_check, baseline/
alembic/         migrations 0001–0009; every tenant table's RLS policy lives here
docs/            discovery, numbers_card, governance/ (tracked); the rest is a local library
scripts/         prove_isolation, show_trace, load_test, pilot_report
```

## Running things

```
make db migrate seed      # embedded Postgres (no Docker), schema, two synthetic banks
make api                  # :8000    make ui   # :8501 console in API mode
make test                 # boots a throwaway Postgres; ~75 s
make evals [PROVIDER=ollama] [TENANT=harbor]   make compare   make redteam   make load   make trace   make pilot
```

Demo users: `rm@<tenant>.example`, `reviewer@<tenant>.example`, password
`banklens-demo`, tenants `meridian` and `harbor`.

## How to add things

- **A template (numeric question):** add it to `app/warehouse/semantic_layer.yaml`
  with `roles`, `triggers`, `params` and `sql` over the views only. The loader
  rejects anything the SQL guard rejects. Add a router test.
- **A tenant:** a directory under `knowledge_base/<slug>/`, a row in
  `app/db/seed.py`, and its forbidden-credit set in `app/pipeline/policy.py`
  (the graph's guardrail node and the golden set both read it).
- **A graph node:** an async function that writes an audit row, wrapped by
  `builder._traced`; nothing with a side effect before an `interrupt()`.
- **A guardrail pattern:** add the family to `app/platform/guardrails.py`
  and both an attack and a benign case to `evals/redteam/cases.py`.

## What not to do

- No PII in fixtures, logs, spans or audit payloads. Samples are synthetic.
- No raw SQL from user input, ever. No new database role without a policy review.
- The API and the worker never connect as the owner role. Spans, budgets and
  LangGraph checkpoints run as `banklens_app` with grants from migration 0008;
  the worker claims jobs through `claim_next_job()` (0009). Deployed processes
  get no owner URL at all. A new cross-bank need is a new narrow function, not
  an owner connection.
  If a LangGraph upgrade adds checkpoint migrations, the API refuses to start:
  add a revision that calls `PostgresSaver.setup()` like 0008 does.
- A tenant slug reaches the filesystem only through
  `app.pipeline.rag.validate_tenant_slug`.
- No new dependency for something the standard library or an existing one does.
- No "fix" that raises a threshold until the numbers pass. Record the miss
  in `docs/numbers_card.md` and fix the cause.
- Do not edit `alembic/versions/*` after they have been applied; add a new
  revision.

## Open work

`TODO.md` at the repository root is the one list of what is still open.
Update it in the same commit as the work that changes it.

## Commits

One branch per phase, conventional prefixes (`feat(scope):`, `fix(scope):`,
`docs(scope):`), the body says what changed and why, and ends with the
Co-Authored-By line the repository already uses. Never push without being
asked.
