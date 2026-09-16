# TODO

The one list of what is still open. Update it in the same commit as the work
that changes it. Finished items move to "Done recently" with the date, and
drop off after a few weeks; the full history is in git and in
`docs/18_challenges_and_decisions.md`.

Last updated: 2026-09-16.

## Waiting on a decision

- [ ] **Merge the branch chain and push to `main`.** Every commit from
  `phase-0-baseline` to `phase-8-backlog` is unpushed. A push to
  `main` deploys production, so this waits for an explicit go-ahead.
  Pre-push checklist:
  - [x] Direct mode (what production runs) touches no database: verified on
    2026-09-16 with no database configured and a fresh home directory. No
    Postgres process started, a full profile in 16 s.
  - [ ] The Linux image builds with the new dependencies (`pgserver`,
    `langgraph-checkpoint-postgres`, `fastapi`). Not verifiable here because
    Docker is not installed; the CI `build` job runs before `deploy`, so a
    failed build cannot reach the host.
  - [ ] Expect the first deploy to rebuild the vector index once: indexes now
    live under `chroma_db/<tenant>/<provider>/`.
- [ ] **Deploy the API, not just the console.** Production runs only the
  Streamlit direct mode. Needs: a real Postgres (RDS or a container with a
  volume and backups), a `t3.small` host, `JWT_SECRET`,
  `BANKLENS_ENV=production` and the three database URLs as secrets, nginx
  routing `/api` to port 8000, and a deploy step that runs `alembic upgrade
  head` before starting the API. The seed already refuses production.

## Open work

- [ ] **Worker claims jobs as the owner role.** The API process no longer
  uses the owner connection anywhere; `app/worker.py` still does, to claim
  jobs across banks. Options: a narrow `SECURITY DEFINER` claim function
  owned by the owner and callable by `banklens_app`, or claim per tenant.
- [ ] **Validate a real pilot.** Three RM shadowing sessions with a stopwatch,
  to replace the twenty-minute manual assumption that `make pilot` prints.
  Cannot be done from the repository.

## Done recently

- 2026-09-16: API process runs without the owner role (migration 0008);
  customer deletion removes spans and query-log rows; tenant slugs validated
  before any path; MCP names the available tenants; local-only files ignored.
- 2026-09-12: production seed guard; branch security review (no exploitable
  findings); beginner's tour and challenges record in `docs/`.
- 2026-09-09: Harbor grounding miss fixed at the cause.
- 2026-09-08: Phases 0 to 8.
