# Deployment runbook

**Tracked, ships.** How BankLens reaches production, what runs where, and how
to switch on the platform API. Written for someone who has never touched the
pipeline.

## What runs today

| Piece | Where | Started by |
|---|---|---|
| Streamlit console, direct mode | EC2 host, container `banklens_app`, port 8501 behind the site's proxy | `deploy` job on every push to `main` |
| Vector index for the console | `/home/ec2-user/chroma_db` on the host | Built on first use |
| Platform API and Postgres | Not running until switched on | `deploy-api` job, opt-in |

Direct mode needs no database. It was checked on 2026-09-16 with no database
configured: no Postgres process starts and a full profile completes.

## The pipeline, job by job

`.github/workflows/ci_cd.yml` runs on every pull request and every push to
`main`.

1. **lint**: Black and flake8.
2. **test**: the full pytest suite (it boots its own throwaway Postgres), the
   red-team suite and the bias check.
3. **build** (push to `main` only): builds the image and pushes it to ECR.
4. **deploy** (after build): on the host, prunes old images, pulls, and
   restarts the console. `set -e` makes a failed pull stop the job with the
   old container still serving. It prints the host's CPU, memory and swap.
5. **deploy-api** (after deploy, only when the repository variable
   `DEPLOY_API` is `true`): the platform API and its database. Described
   below.

A pull request runs jobs 1 and 2 only, so nothing reaches the host until a
change is merged.

## Switching on the platform API

### Prerequisites

- **At least 2 GB of RAM on the host.** The console, the API and Postgres
  together do not fit in 1 GB. The job checks and refuses below 1.8 GB,
  leaving the console untouched. Resizing the instance (for example to a
  `t3.small`) is done in the AWS console and costs money, so it is a
  decision for the account owner.
- Nothing else. No new GitHub secret is needed.

### Switch it on

```bash
gh variable set DEPLOY_API --body true
gh workflow run "BankLens CI/CD" --ref main
```

### What the job does

1. Checks memory; stops with a clear message if the host is too small.
2. On the first run only, generates the Postgres owner password, the API and
   chat role passwords and the JWT signing secret **on the host**, in
   `/home/ec2-user/banklens-platform/secrets.env`, readable only by its owner.
   They never pass through GitHub, the image or this repository.
3. Starts `banklens_pg` (Postgres 16) on a private Docker network with a
   384 MB memory cap, if it is not already there. Its first boot creates the
   `banklens_app` and `banklens_chat` roles with the generated passwords.
4. Runs `alembic upgrade head` in a one-off container, the only place the
   owner password is used.
5. Starts `banklens_api` on `127.0.0.1:8000` with only the API and chat role
   URLs, `BANKLENS_ENV=production`, the Postgres rate limiter and its own
   vector index directory. The API never receives the owner password.
6. Waits for `/health`.
7. If the host has `/etc/nginx/default.d`, adds a `/api/` location and reloads
   nginx after `nginx -t` passes. If the test fails, it removes the snippet and
   the console keeps serving.

### After the first API deploy

- The database has no users: the seed refuses production on purpose. Create
  the first ones on the host, in a one-off container that has the owner URL.
  The password is typed at the terminal and never appears in an argument, an
  environment variable or a log:

  ```bash
  docker run --rm -it --network banklens \
    --env-file /home/ec2-user/banklens-platform/migrate.env \
    "$ECR_REPOSITORY:latest" python -m scripts.create_user \
    --tenant meridian --email asha.verma@bank.example \
    --name "Asha Verma" --role reviewer
  ```

  Passwords under 12 characters, and the demo password, are refused. Add
  `--reset-password` to change an existing user's password.
- Check `https://<site>/api/health`. If it does not answer but
  `curl http://127.0.0.1:8000/health` on the host does, the site's proxy is
  not nginx on this host; route `/api/` to port 8000 wherever the proxy is.

### Switch it off

```bash
gh variable set DEPLOY_API --body false
```

Then on the host: `docker stop banklens_api banklens_pg`. The data stays in
`/home/ec2-user/banklens-platform/pgdata`.

## Rolling back a bad deploy

Every image is pushed with its commit SHA as well as `latest`. On the host:

```bash
docker pull "$ECR_REPOSITORY:<previous-sha>"
docker stop banklens_app && docker rm banklens_app
```

Then start it with the same `docker run` line as the deploy job, using the
SHA tag. Revert the commit on `main` afterwards so the next deploy does not
bring the bad image back.
