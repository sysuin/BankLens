# Challenges, decisions and the roads not taken

**Tracked, ships.** Every real problem hit while building the platform
(Phases 0 to 8, September 2026), what the symptom looked like, the approaches
that were on the table, which one was chosen and why. Written for someone new
to the codebase: each entry stands alone. Numbers come from
`docs/numbers_card.md`, where each has the command that produced it.

The format for each entry: **Symptom** (what you would have seen),
**Cause**, **Options considered**, **Chosen**, **Why**, and where relevant
**Proof**.

---

## Part A. Environment and infrastructure

### A1. Postgres would not start inside the project folder
- **Symptom:** the embedded Postgres (`pgserver`) failed at startup with a
  socket error the first time `make db` ran.
- **Cause:** the project path contains a space (`Codex Projects`). Postgres
  puts its Unix socket in the data directory, and socket paths with spaces
  are rejected.
- **Options:** (1) rename the project folder; (2) run Postgres in Docker;
  (3) keep the cluster somewhere without a space.
- **Chosen:** (3). The cluster lives in `~/.banklens/pg`
  (`LOCAL_PG_DIR`), and `app/db/local.py` refuses a directory containing a
  space with a clear message.
- **Why:** renaming the folder would break the user's other tooling; Docker
  is not installed on this machine and the zero-setup demo is a
  requirement.

### A2. No Docker, no Redis
- **Symptom:** the usual "docker compose up" story was not available, and
  the plan mentioned Redis for the rate limiter, queue and cache.
- **Options:** install Docker; require Redis; use Postgres for everything
  shared; keep single-process pieces in memory.
- **Chosen:** Postgres for everything shared (job queue with
  `SKIP LOCKED`, profile cache table, rate-limit counter rows), embedded
  Postgres for local runs. Redis remains an adapter point (`Limiter`
  protocol in `app/platform/ratelimit.py`) with no implementation.
- **Why:** doctrine "no new dependency for something an existing one does".
  Nothing measured has needed Redis; the numbers card shows the queue at
  356 statements per minute on ingest with two workers.

### A3. The `psql` shell bootstrap failed
- **Symptom:** `pgserver`'s `server.psql(...)` helper failed to create the
  roles and database.
- **Chosen:** bootstrap with `psycopg` directly from Python, and run the
  persistent cluster with `cleanup_mode=None` so tests that tear down a
  throwaway cluster do not stop the developer's cluster.
- **Also:** `_resolve_urls()` calls `ensure_local_cluster()` every time, so
  the API self-heals after a test run has stopped the cluster.

### A4. asyncpg connections cannot cross event loops
- **Symptom:** database tests raised "attached to a different loop" and
  "connection was closed" errors when a test opened an engine created
  inside FastAPI's TestClient loop.
- **Cause:** asyncpg connections belong to the loop that created them; the
  TestClient runs the app in its own loop.
- **Options:** run every test inside the app loop; use sync drivers for
  tests; give tests their own engines.
- **Chosen:** `run_db(url, fn)` in `tests/conftest.py` runs a coroutine on a
  fresh loop with a fresh engine; `session.reset_engines()` and
  `builder.reset_checkpointer()` forget pools when the loop changes; the
  lifespan shutdown is wrapped in try/finally so a failed dispose never
  hides the real error.

### A5. Black reformatting broke my own patch scripts
- **Symptom:** several edit scripts that searched for an exact code string
  failed because Black had already reformatted the target.
- **Chosen:** rewrite whole files or match tolerantly; run Black before
  matching. Not a product bug, but it cost time three or four times and is
  worth knowing.

### A6. Ollama died mid-run
- **Symptom:** a local-model eval reported "Connection error" on every
  grounded case and the circuit breaker opened.
- **Cause:** the `ollama serve` process had exited.
- **Chosen:** restart it (`nohup ollama serve &`) and re-run. The gateway's
  breaker behaved correctly: it opened after three failures instead of
  hanging six times. The eval runner reports the failure as a failed check
  (`grounded_layer_ran`) rather than a crash.

---

## Part B. Data model and tenancy

### B1. Where to enforce tenant isolation
- **Options:** (1) `WHERE tenant_id = ...` in every query; (2) one database
  per tenant; (3) one schema per tenant; (4) row-level security in a shared
  schema.
- **Chosen:** (4). Every tenant table has a policy on `tenant_id`, the API
  role is subject to the policies, and each transaction pins
  `app.tenant_id` from the signed token (`app/db/session.py`).
- **Why:** (1) is one forgotten clause away from a breach; (2) and (3) make
  migrations and analytics painful at small scale. RLS moves the rule into
  the database where application code cannot skip it.
- **Proof:** `make prove-isolation` (policy on: nothing; off: the row leaks;
  on: gone) and `tests/test_tenancy.py`, which also checks that every table
  in `TENANT_TABLES` has a policy.

### B2. LangGraph's checkpoint tables have no tenant column
- **Symptom:** the platform's isolation story had a hole: checkpoint rows
  are addressed by thread id only, and the checkpointer connects as the
  owner role, which bypasses RLS.
- **Options:** (1) fork the checkpointer to add a tenant column and policy;
  (2) one checkpoint schema per tenant; (3) namespace the thread id with the
  tenant and only ever address runs through tenant-scoped tables.
- **Chosen:** (3) in Phase 8. Thread id is `<tenant_id>:<run_id>`; the run id
  comes from an RLS-protected table, so a reviewer at another bank cannot
  construct the thread. A purge helper deletes checkpoint rows by run for
  the retention path.
- **Why:** (1) means maintaining a fork of a moving library; (2) multiplies
  connections. The namespace closes the practical hole and is documented as
  "namespaced, not policied" in the README and the retention note.

### B3. Deleting a customer through the ORM failed
- **Symptom:** `DELETE /customers/{id}` returned 500 with "null value in
  column customer_id violates not-null constraint".
- **Cause:** SQLAlchemy's `session.delete(customer)` tried to orphan the
  statements (set `customer_id` to NULL) before deleting, because the
  relationship had no cascade configured; the database's `ON DELETE CASCADE`
  never got a chance.
- **Options:** configure ORM cascades on every relationship; issue a bulk
  `DELETE` and let the database cascade.
- **Chosen:** bulk `DELETE` (`delete(Customer).where(...)`). The foreign
  keys already carry `ON DELETE CASCADE`, and RLS decides what is deletable.

### B4. `func.max()` on a UUID column
- **Symptom:** a trace listing query failed: Postgres has no `max` for
  `uuid`.
- **Chosen:** cast to text for the aggregate. Small, but the kind of thing
  that only shows up against a real database, which is why the tests boot
  one.

---

## Part C. The decision graph

### C1. Nothing with a side effect before `interrupt()`
- **The trap:** when a LangGraph run resumes, the interrupted node is
  executed again from its first line up to the `interrupt()` call. Any
  database write placed before that call would happen twice: once when the
  run paused, once when it resumed.
- **Options:** make every such write idempotent; or keep the interrupting
  node free of side effects.
- **Chosen:** the second. `await_review` does nothing before `interrupt()`;
  the decision row and the audit row are written in `verify_income`, the
  node before it. This is a rule in `CLAUDE.md` for anyone adding a node.

### C2. Resume after an API restart
- **Goal:** prove the pause is durable, not just in-process.
- **Chosen:** the checkpointer is `AsyncPostgresSaver`; the test opens two
  separate app instances, pauses in the first, decides in the second. Live
  proof: run paused, API killed and restarted, reviewer approved, run
  completed, eight audit rows across RM, reviewer and system.

---

## Part D. Tracing and cost

### D1. Cost attribution missed model calls
- **Symptom:** the cost per statement in the trace summary was lower than
  the eval runner's number.
- **Cause:** the summary summed spans stamped with the statement id, but the
  categorizer's model calls run before the statement row exists, so those
  spans carried no statement id.
- **Options:** create the statement row earlier; stamp ids later; sum over
  whole traces that touch the statement.
- **Chosen:** sum every span of every trace touching the statement. Job cost
  uses the same rule.
- **Lesson:** "where the money goes" is only true if the attribution rule is
  written down. The numbers card records both figures.

### D2. Trace listing counted only stamped spans
- Same root cause as D1 for the span count shown in the console; fixed by
  counting the whole trace.

### D3. Local tokens priced at OpenAI rates
- **Symptom:** `make compare` showed the local model costing $0.0115 per
  query.
- **Cause:** the eval runner priced tokens by the model name it expected,
  not by the provider that served the call.
- **Chosen:** price by `gateway.choose().spec`; a cost-free provider reports
  zero. The wrong number is kept in the numbers card because it is a good
  story about trusting a cost column.

---

## Part E. Guardrails

### E1. The red-team suite scored 90.2 percent on its first run
Five misses, each a distinct bug:
1. A base64 blob was not detected because the word-boundary regex failed
   against `=` padding. Fixed the boundary.
2. "run the SQL query: delete ..." scored below the block threshold because
   the tool-abuse family's weight was too low. Raised to 1.0.
3. "all customers data from other banks" likewise for the cross-tenant
   exfiltration family. Raised to 1.0.
4. "What is the capital of Australia?" was exactly half finance vocabulary
   and slipped through a `>=` scope floor. The floor is strict (`>`) at 0.5.
5. A legitimate CTE-based `SELECT` was rejected by the SQL guard because the
   CTE name was not on the relation allow-list. CTE names defined in the
   statement are now permitted.
- **Options rejected:** lowering thresholds until green. Doctrine: fix the
  cause, record the miss. Now 100 percent block, 0 percent false positives.

### E2. Neutralised rows were sent to the model anyway
- **Symptom:** the trace showed `pipeline.llm_fallback_rows: 2` on an
  injected statement.
- **Cause:** the categorizer's rules could not place a neutralised
  description, so the row went to the model fallback, which is exactly the
  place an injection wants to reach.
- **Chosen:** rows carrying the neutralised marker stay in "Others" without
  a model call. Found by a trace attribute, which is the argument for
  tracing everything.

### E3. The scope gate judged its own redaction markers
- **Symptom:** "what is the savings rate for account 1234567890" was
  abstained as out of scope.
- **Cause:** personal-data redaction ran before the scope gate, so the gate
  saw `[REDACTED_PHONE]` and counted it as an unknown word.
- **Chosen:** order is injection scan, then router, then scope gate on the
  user's own words, then redaction of what is sent onward.

### E4. Two intent-router bugs
- A digit inside a trigger phrase ("top 3 categories") failed to match:
  digits are stripped before matching.
- A single-word trigger outscored a three-word match: every matching
  trigger now contributes to the score.

---

## Part F. Retrieval and grounding

### F1. The reranker measured well and ships off
- **Symptom:** the reranker improved the usual metrics on the golden
  queries.
- **But:** the A/B measured a "harmful" rate on deficit cases (how often an
  unsecured credit product ranked in the top four for a customer in
  deficit), and it went from 0.400 to 0.908 while hit@4 fell from 1.000 to
  0.923.
- **Chosen:** ship it disabled, keep the code and the measurement.
  `app/core/config.py` says why next to the setting.
- **Lesson:** an offline metric that ignores the business rule can approve a
  regression.

### F2. Harbor's first grounded run failed, twice, for two reasons
The first run of `make evals PROVIDER=openai TENANT=harbor` failed the
catalogue check on four of six cases and the retrieval-support check on all
six.
1. **The eval's checks resolved product names outside the tenant scope.**
   "Term Deposit" (a real Harbor product) was judged unknown against
   Meridian's shelf; "Cashback Credit Card" resolved to Meridian's
   `credit_card.md`. Fix: run the checks inside `tenant_scope(tenant)`.
2. **The system prompt named Meridian's products.** For Harbor the hosted
   model proposed "Debt Consolidation Loan" and "Recurring Deposit", the
   validator rejected them, and the retry doubled the bill: $0.0186 per
   query at p50 13.7 s against $0.0096 and 8.5 s for Meridian. Fix:
   rewrite the scenario guidance as product types. Harbor fell to $0.0137;
   Meridian was unchanged; the local model still passed every blocking
   check and improved on the advisory one.

### F3. The remaining Harbor miss, the wrong fix, and the right one
After F2, two deficit or thin-saver cases still failed retrieval support:
retrieval ranked the auto loan and the credit card, the guardrail rightly
ruled them out, and the model recommended the everyday account, whose
passage was never retrieved. It knew the name because the validator's retry
message listed the whole catalogue.

- **Attempt 1 (reverted):** add a deterministic "customer need" clause to
  the retrieval query, chosen by the same flags as the prompt scenarios.
  Measured: Meridian's deficit case now retrieved the credit card and the
  mortgage (the words "credit" and "secured" are lexical matches for BM25),
  and the local model offered a personal loan to a deficit customer, a
  credit-guardrail failure. Reverted the same hour. Lesson: a hybrid
  retriever has a lexical half; negative phrasing ("avoid credit cards") is
  a positive match for the thing you are avoiding.
- **Attempt 2, recorded first:** write the miss into the numbers card and
  the README rather than tune a threshold. Doctrine.
- **Attempt 3 (kept), two cause fixes:** (a) Harbor's catalogue gained the
  product it lacked, a Fresh Start Consolidation Loan written for the
  deficit profile, and the everyday account's document now says who it is
  for; (b) inside `build_profile()` the validator is scoped to the retrieved
  shelf: the retry hint names only retrieved products, and a primary
  recommendation outside the retrieved passages is rejected. A secondary
  may be any catalogue product, because a cross-sell from the same shelf is
  not a hallucination.
- **Result:** Harbor 6 of 6 on every blocking check with zero retries at
  $0.0102 per query and p50 6.6 s; Meridian unchanged; local model passes
  every blocking check with zero retries.
- **Options not taken:** raising `RETRIEVAL_K`; validating the secondary
  against the shelf too (would force retries on ordinary cross-sells);
  removing the retry hint entirely (the hint is what lets a paraphrase be
  corrected in one round trip).

### F4. The judge that contradicts itself
- The LLM-as-judge groundedness check called a correct "88 percent"
  unsupported in the same breath as stating the ratio is 88 percent. It is
  advisory and printed, never blocking; the narrow "does the narrative
  respect the assigned risk band" question stays blocking because it holds
  at 100 percent. Full chronicle in `04_evaluation_and_llm_judge.md`.

---

## Part G. Testing and tooling

### G1. The full suite tripped the rate limiter
- **Symptom:** unrelated tests returned 429 late in a run.
- **Cause:** one very busy user (the test suite) against an in-process
  sliding window.
- **Chosen:** an autouse fixture resets the limiter before and after every
  test. When the Postgres backend was added, its test resets the table too.

### G2. The test cluster used the developer's cluster for the chat role
- **Symptom:** warehouse tests touched the wrong database.
- **Cause:** `conftest` set the app and admin URLs to the throwaway cluster
  but not the chat URL, which fell back to the default.
- **Chosen:** set all three.

### G3. The SQL guard rejected a bound `LIMIT`
- **Symptom:** every template with `LIMIT :limit` failed validation at
  import.
- **Chosen:** the guard accepts a bound parameter for the limit and the
  runner clamps the value.

### G4. The pilot report's p95 was three minutes
- **Symptom:** "seconds to profile p95 163.6 s" on a system whose model
  calls take ten.
- **Cause:** one run had waited three minutes for a human reviewer, and run
  duration included the wait.
- **Chosen:** platform time is run duration minus reviewer wait (a left
  join on the decision); reviewer wait is its own row. Now p50 5.2 s,
  p95 10.4 s.

---

## Part H. Security and operations

### H1. The silent no-op deploy (August, before the platform work)
- **Symptom:** a deploy reported success and production ran stale code.
- **Cause:** the host's 8 GB disk was full of old images, `docker pull`
  failed, and the script had no `set -e`, so it restarted the old container.
- **Chosen:** prune before pulling, fail the job on a failed pull, a 30 GB
  root volume in the Terraform stub with the date in the comment.

### H2. The security review found nothing exploitable, and one nit
- A branch-wide review of the Phase 0 to 8 changes found no high or medium
  finding after false-positive filtering. The one candidate, demo accounts
  seeded on every Compose boot, was judged a hardening gap because the
  Compose file is a local stack and the production pipeline does not run
  the seed. The guard was added anyway: `app/db/seed.py` refuses to run when
  `BANKLENS_ENV=production`, matching the existing JWT-secret guard.
- Three below-the-bar notes came with it. All three are now fixed; see H3
  and H4.

### H3. The API process connected as the database owner
- **Symptom:** none visible. The review noticed that the API held the owner
  connection in three places: the span exporter, the daily-budget lookup and
  LangGraph's checkpointer. No user input reached those connections, so it
  was not exploitable, but a bug in any of them would have run with the power
  to drop tables and bypass every row-level policy.
- **Cause:** convenience. The exporter writes spans for several banks in one
  batch, the budget reads across the day's spans, and LangGraph's `setup()`
  creates its own tables, which needs DDL rights.
- **Options:** (1) leave it and document it; (2) a third database role just
  for infrastructure writes; (3) move all three to the ordinary API role and
  grant exactly what each needs.
- **Chosen:** (3), in migration 0008. The exporter groups a batch by tenant
  and writes each group in its own transaction with the tenant pinned, so the
  spans policy checks every row. The budget lookup pins the tenant and reads
  under the policy. LangGraph's tables are created by the migration, as the
  owner, and the API role gets row access only; at startup the API checks the
  checkpoint schema version instead of running DDL, and refuses with the fix
  if a LangGraph upgrade needs a new migration.
- **Why not (2):** "no new database role without a policy review" is a rule
  in `CLAUDE.md`, and a role that exists to bypass tenancy is the one most
  worth avoiding.
- **Proof:** `test_api_runs_pause_and_resume_without_the_owner_role` points
  the owner URL at a database that does not exist, then signs in, uploads,
  pauses a run, approves it and resumes it. Spans are still stored and the
  budget is still read. The worker still uses the owner role to claim jobs
  across tenants; it is a separate process and is next on the list.

### H4. Retention missed two stores, and tenant slugs were paths
- **Spans and query-log rows survived a customer delete.** They carry
  statement and run ids but no foreign key, so the cascade never reached
  them. Adding foreign keys was considered and rejected: a span can be
  exported for a statement whose transaction later rolled back, and the
  failed insert would silently drop tracing. Chosen: the delete removes, in
  the same tenant-pinned transaction, every trace that touched the
  customer's statements or runs (whole traces, the rule cost attribution
  already uses) and the query-log rows for those statements. The response and
  the audit row now report both counts.
- **Tenant slugs became directory names unchecked.** The MCP server passed
  its `tenant` argument into `knowledge_base/<tenant>` and the index
  directory, and the index rebuild deletes that directory first. The MCP
  server is a local process driven by the operator's own agent, so this was
  not remotely reachable. Chosen: one validator in `app/pipeline/rag.py`
  (lowercase letters, digits, `_` and `-`, at most 64 characters) that every
  path, index and collection name goes through, and an MCP error that names
  the banks that exist so the calling agent can correct itself.

---

## The decisions that shaped everything

| Decision | Alternatives | Why this one |
|---|---|---|
| Numbers from code, prose from the model | Let the model rate risk; let it compute | A model's arithmetic is unreliable and unauditable; a rule is both |
| Row-level security in one schema | Per-tenant databases; WHERE clauses | Isolation in the database, provable by a script, cheap at this scale |
| LangGraph with a Postgres checkpointer | Hand-written state machine; Celery chain | Interrupt and resume are first-class, and the checkpoint survives a restart |
| Spans in Postgres, Jaeger optional | Only Jaeger; only logs | The cost query needs SQL; Jaeger is a viewer, not a store of record |
| One gateway for every model call | Direct client per call site | Retries, breaker, budget, fallback and cost live in one place |
| Ollama as the zero-key path | OpenAI only; a hosted free tier | The demo must run with no key and no spend, and the difference is measured |
| Deterministic guardrails | Model-based classifiers | Explainable blocks, testable in CI, no cost, no model to inject |
| Vetted SQL templates over views | Text-to-SQL | A model writing SQL against a bank's tables is not a risk worth taking; the router answers most numeric questions in milliseconds |
| Postgres for queue, cache, rate limit | Redis | Already deployed, transactional, nothing measured the need |
| Record misses before fixing them | Tune thresholds until green | A green suite nobody trusts is worse than a red line with a reason |
