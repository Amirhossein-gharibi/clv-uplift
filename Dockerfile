# Dockerfile
# =============================================================================
# Multi-stage build for the CLV Uplift FastAPI service.
#
# Stage 1 (builder): installs build tools and compiles all Python dependencies
#                    into a dedicated prefix directory /install.
# Stage 2 (runtime): starts from a clean base image, copies only the compiled
#                    packages from the builder, then adds the application code.
# =============================================================================


# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install C/C++ compilers needed to build packages like LightGBM and SHAP.
# --no-install-recommends keeps the layer small by skipping optional packages.
# Deleting the apt cache afterwards prevents it from bloating the layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy pyproject.toml first — this layer changes rarely.
# When pyproject.toml has not changed, Docker reuses the cached pip install
# layer below it, skipping the expensive multi-minute dependency installation.
COPY pyproject.toml .

# Install only the third-party dependencies declared in pyproject.toml,
# NOT the clv-uplift package itself. We use a temporary requirements extraction
# approach: install with --no-deps first won't work cleanly, so instead we
# create a minimal stub src/ directory so pip can read the package metadata
# without needing the real source code.
#
# Why a stub? pip install . needs to read pyproject.toml AND confirm that
# the src/ directory exists (because pyproject.toml declares where="src").
# The stub satisfies that check without copying all source code here,
# preserving the cache benefit — source code changes won't invalidate
# the dependency installation layer.
RUN mkdir -p src/clv_uplift && \
    echo '"""stub"""' > src/clv_uplift/__init__.py

# Now install all dependencies into /install prefix.
# This layer is cached as long as pyproject.toml and the stub don't change.
RUN pip install --upgrade pip \
    && pip install \
        --no-cache-dir \
        --prefix=/install \
        .


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
# Start from the same slim base — no compilers, no build artifacts.
FROM python:3.11-slim AS runtime

# Runtime-only system libraries. LightGBM links against OpenMP (libgomp) at load
# time; the slim base does not include it, so import fails without this. This is
# the runtime package only (~hundreds of KB), not the build-essential toolchain.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy all installed packages from the builder stage into the runtime image.
# This gives us every dependency without any build tooling.
COPY --from=builder /install /usr/local

# Set the working directory for the application.
WORKDIR /app

# Copy the real application source code.
# This layer changes on every source code edit, but because it comes after
# the dependency layer above, dependencies remain cached across source changes.
COPY src/ src/

# Create the artifacts directory where the trained model .pkl file will live.
RUN mkdir -p /app/artifacts

# Security: run as a non-root user so that a compromised container cannot
# damage the host system with root privileges.
RUN useradd --no-create-home --uid 1001 appuser \
    && chown -R appuser /app
USER appuser

# PYTHONUNBUFFERED=1 forces immediate stdout/stderr flushing so that logs
# appear in real time in `docker logs` rather than buffering silently.
# PYTHONDONTWRITEBYTECODE=1 prevents .pyc files from being written to the
# container's temporary filesystem where they serve no purpose.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV NUMBA_CACHE_DIR=/tmp/numba

# Document the port the service listens on. This is metadata only —
# the actual port mapping happens at `docker run -p 8000:8000`.
EXPOSE 8000

# Health check: Docker polls this every 30 seconds. Three consecutive
# failures mark the container unhealthy and trigger automatic restarts
# in orchestrated environments like Kubernetes or Docker Compose.
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=15s \
    --retries=3 \
    CMD python -c "\
import urllib.request, sys; \
r = urllib.request.urlopen('http://localhost:8000/health', timeout=3); \
sys.exit(0 if r.status == 200 else 1)"

# The startup command. Two workers is appropriate for a single container
# running CPU-bound ML inference. Scale by adding container replicas,
# not by increasing workers indefinitely.
CMD ["uvicorn", "clv_uplift.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]