# ─────────────────────────────────────────────────────────────────────────────
# BankLens — Multi-Stage Dockerfile
#
# Stage 1 (builder): installs all Python dependencies into a virtual
#   environment. Uses the full build tools image so compiled packages
#   (like chromadb) can compile correctly.
#
# Stage 2 (runtime): copies only the installed packages and application
#   code into a slim final image. This keeps the production image small
#   (~400MB vs ~900MB for a single-stage build).
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Set the working directory for the build stage
WORKDIR /build

# Install system build dependencies needed by some Python packages
# (e.g. chromadb requires build-essential for native bindings)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first.
# Docker caches this layer — dependencies are only reinstalled when
# requirements.txt changes, not on every code change.
COPY requirements.txt pyproject.toml ./

# Install dependencies into a virtual environment inside the build stage.
# Then install the project itself in editable mode so the 'app' package
# is importable from anywhere inside the container.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Set environment variables
# - PYTHONDONTWRITEBYTECODE: prevents Python from writing .pyc files
# - PYTHONUNBUFFERED: ensures stdout/stderr are not buffered (important for logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

# Set the working directory for the runtime container
WORKDIR /app

# Copy the virtual environment from the builder stage (no build tools needed here)
COPY --from=builder /opt/venv /opt/venv

# Copy the application source code
COPY app/ ./app/
COPY knowledge_base/ ./knowledge_base/
COPY data/ ./data/
COPY prompts/ ./prompts/

# Streamlit default port
EXPOSE 8501

# Health check — Streamlit exposes a /_stcore/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
# --server.address=0.0.0.0 makes the app accessible outside the container
# --server.fileWatcherType=none disables hot reload (not needed in production)
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none"]
