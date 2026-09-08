# BankLens Platform — developer entry points.
#
#   make db         start the embedded local Postgres (no Docker needed)
#   make migrate    apply Alembic migrations
#   make seed       create the two synthetic banks (+ statements with SEED_STATEMENTS=1)
#   make api        run the API on :8000
#   make ui         run the Streamlit console against the API on :8501
#   make test       run the whole test suite (boots its own throwaway Postgres)
#   make evals      run the free evaluation layer
#   make prove-isolation   show the tenant guard failing, then passing
#
# PY points at the project's virtualenv; override with PY=python if you use
# another environment.

PY ?= .venv311/bin/python
UVICORN ?= .venv311/bin/uvicorn
STREAMLIT ?= .venv311/bin/streamlit

.PHONY: db db-stop migrate seed api ui test evals lint prove-isolation baseline trace worker load compare redteam bias gate pilot

db:
	$(PY) -c "from app.db.local import ensure_local_cluster, cluster_dir; ensure_local_cluster(); print('Postgres running at', cluster_dir())"

db-stop:
	$(PY) -c "from app.db.local import stop_local_cluster; stop_local_cluster(); print('stopped')"

migrate: db
	$(PY) -m alembic upgrade head

seed: migrate
	$(PY) -m app.db.seed $(if $(SEED_STATEMENTS),--with-statements,)

api: db
	$(UVICORN) app.api.app:app --host 0.0.0.0 --port 8000 --reload

ui:
	BANKLENS_API_URL=$${BANKLENS_API_URL:-http://localhost:8000} $(STREAMLIT) run app/main.py --server.port=8501

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m black --check .
	$(PY) -m flake8 .

# Free layer by default. PROVIDER=ollama runs the grounded layer on the local
# model with no API key; PROVIDER=openai on the hosted one.
#   make evals                       deterministic layer + BM25 headroom
#   make evals PROVIDER=ollama       + grounded layer on the local model
#   make evals PROVIDER=openai TENANT=harbor   the grounded layer on Harbor's catalogue
TENANT ?= meridian
evals:
ifdef PROVIDER
	LLM_PROVIDER=$(PROVIDER) LLM_FALLBACK_ENABLED=false PROFILE_CACHE_ENABLED=false $(PY) -m evals.run_evals --with-llm --tenant $(TENANT)
else
	$(PY) -m evals.run_evals --headroom --tenant $(TENANT)
endif

# What a pilot would measure: seconds to a profile, reviewer wait, throughput,
# from the run and decision timestamps already in the database.
pilot:
	$(PY) -m scripts.pilot_report --tenant $(TENANT)

# The same golden suite on both providers, side by side.
compare:
	PROFILE_CACHE_ENABLED=false $(PY) -m evals.compare_providers --providers $${PROVIDERS:-openai,ollama}

# The bulk worker (run next to the API). Drains the queue and keeps polling.
worker:
	$(PY) -m app.worker

# Enqueue 50 statements, run the worker inline, print throughput and cost.
#   make load                  ingest only (free)
#   make load KIND=ingest_and_run   also runs the graph
load:
	$(PY) -m scripts.load_test --n $${N:-50} --kind $${KIND:-ingest} --tenant $${TENANT:-meridian}

baseline:
	PROFILE_CACHE_ENABLED=false $(PY) -m evals.baseline.measure_baseline --repeats 2

# Interview demo step 8: with RLS on, a cross-tenant read returns nothing;
# disable RLS on one table and the same read leaks the row; re-enable and it
# is gone again. Runs on a throwaway cluster, never on the dev database.
prove-isolation:
	$(PY) -m scripts.prove_isolation

# Print the latest graph run as a waterfall: where the time and the money went.
#   make trace TENANT=harbor
trace:
	$(PY) -m scripts.show_trace --tenant $${TENANT:-meridian}

# The red-team suite: injection in statements (CSV and PDF), chat, SQL and
# model output. Prints the block rate and the false-positive rate; exits 1
# below the thresholds. No model calls, so it runs on every push in CI.
redteam:
	$(PY) -m evals.redteam.run

# Bias check: demographic rewrites of every golden statement must give
# identical risk bands and scores, and equal guardrail flags. Free.
bias:
	$(PY) -m evals.bias_check

# The whole gate a change must pass before it is called done.
gate: lint test evals redteam bias
