# TODO

The one list of what is still open. Update it in the same commit as the work
that changes it. Finished items move to "Done recently" with the date, and
drop off after a few weeks; the full history is in git and in
`docs/18_challenges_and_decisions.md`.

Last updated: 2026-09-16.

## In progress

- [ ] **Merge the branch chain and push to `main`.** Approved 2026-09-16.
  Going through a pull request so CI tests everything before the deploy job
  runs. Pre-push checklist:
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
  `gh variable set DEPLOY_API --body true` and re-run the pipeline. First
  production users need creating by hand; the seed refuses production.
- [ ] **Run the pilot.** Protocol and tooling are ready
  (`docs/pilot_protocol.md`, `make pilot` reads `data/pilot/timings.csv`).
  What remains is three RMs, six statements and a stopwatch.

## Done recently

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
