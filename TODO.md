# TODO

The one list of what is still open. Update it in the same commit as the work
that changes it. Finished items move to "Done recently" with the date, and
drop off after a few weeks; the full history is in git and in
`docs/18_challenges_and_decisions.md`.

Last updated: 2026-10-02.

## In progress

- [ ] **Open the pull request and merge it.** The branch is pushed
  (`phase-8-backlog`, 2026-09-16). Opening a pull request is blocked for the
  assistant by this machine's permission rules, so it needs a person:
  `gh pr create --base main --head phase-8-backlog` or the link GitHub
  printed on push. CI runs lint and the full suite on the pull request;
  merging is what deploys. Checklist:
  - [x] Direct mode (what production runs) touches no database: verified on
    2026-09-16 with no database configured and a fresh home directory.
  - [ ] The Linux image builds with the new dependencies. Not verifiable
    locally (no Docker); the CI `build` job runs before `deploy`.
  - [ ] Expect the first deploy to rebuild the vector index once: indexes now
    live under `chroma_db/<tenant>/<provider>/`.

## Waiting on someone else

- [ ] **Switch on the platform API in production.** The opt-in `deploy-api`
  job is built (`docs/deploy_runbook.md`): Postgres in a container on the
  host, secrets generated on the host, no owner URL in the API, nginx route
  with rollback. It needs a host with at least 2 GB of RAM, which means
  resizing the instance, a cost decision for the AWS account owner. Then
  `gh variable set DEPLOY_API --body true` and re-run the pipeline. The first
  users are then created with `python -m scripts.create_user` on the host
  (the seed refuses production).
- [ ] **Run the pilot.** Protocol and tooling are ready
  (`docs/pilot_protocol.md`, `make pilot` reads `data/pilot/timings.csv`).
  What remains is three RMs, six statements and a stopwatch.

## Found on review, not started

Found while writing up the system in detail (2026-10-02); each has a fix in
mind but no code yet.

- [ ] **Chat history still lives with the client.** Switching statement now
  clears it, and the API scans every turn and caps its length, but a client
  can still send an invented but innocent-looking "assistant" turn. The full
  fix is server-side history keyed by tenant, user and statement: a tenant
  table with an RLS policy (new migration, `TENANT_TABLES`), included in
  deletion and retention.
- [ ] **Chat answers are rendered as Markdown with links and images intact.**
  An image URL makes the browser fetch it. Neutralise URLs in chat output on
  every door, with red-team cases.
- [ ] **`vision_ocr.py` builds `ChatOpenAI` directly**, bypassing the gateway
  (no breaker, budget, fallback or cost span). Route it through
  `gateway.chat_model()`; add a test that forbids direct clients.
- [ ] **Retrieval has no relevance floor** (the chat's product search returns
  four passages even off-catalogue), product documents aren't injection-
  scanned at index build, and `build_retrieval_query` labels the period's
  total income "monthly income".

## Done recently

- 2026-10-02: **Chat history.** The API-mode console no longer carries one
  customer's chat into the next when the RM picks another statement. The API
  scans every client-sent history turn like the question (an instruction in
  an earlier or forged turn is blocked before any model, with an audit row)
  and caps each at 4,000 characters.
- 2026-10-02: **Front-door parity.** The statement injection scan and the
  profile guardrails (catalogue, deficit-credit block, output scan) ran only
  on the API path; production's direct-mode console and the MCP server
  skipped them. Both now live in `app/pipeline/governed.py` and
  `guardrails.check_profile()`, called by the console, the API (graph and
  direct profile endpoint, which now returns 422 and an audit row when
  blocked), the worker and the MCP tools. The persona and RM talking points
  are output-scanned too.
- 2026-10-02: **Account-number masking** keeps only the last four digits at
  every length; the old pattern re-inserted the captured number, so 6-9,
  14-15 and 17+ digit accounts came out in full (10-13 were masked only by
  accident, by the phone pattern). Indian mobiles (`98765 43210`,
  `+91-9876543210`) are masked. Property tests cover every length and
  spelling.
- 2026-09-27: `docs/mcp_runbook.md` — registering the MCP server in an
  MCP host, the three tools and their tenant scoping, worked prompts and
  troubleshooting. README points at it.
- 2026-09-20: `scripts/create_user.py` and `app/db/users.py`, so a deployed
  platform can have real users without the seed.
- 2026-09-16: worker claims jobs through `claim_next_job()` (migration 0009);
  neither the API nor the worker uses the owner role, and deployed processes
  get no owner URL; opt-in API deploy job and runbook; pilot protocol, timing
  sheet and measured figures in `make pilot`.
- 2026-09-16: API process runs without the owner role (migration 0008);
  customer deletion removes spans and query-log rows; tenant slugs validated
  before any path; MCP names the available tenants; local-only files ignored.
- 2026-09-12: production seed guard; branch security review (no exploitable
  findings); beginner's tour and challenges record in `docs/`.
- 2026-09-09: Harbor grounding miss fixed at the cause.
- 2026-09-08: Phases 0 to 8.
