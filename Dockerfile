# ─────────────────────────────────────────────────────────────────────────
# Space of Mind — brain-service (TRIBE v2 inference + Destrieux + nilearn)
#
# License caveat: TRIBE v2 is CC BY-NC. This image is for research /
# internal demo use only. Do NOT deploy this in a commercial flow.
#
# Build args:
#   COMPUTE=cpu (default) or gpu   — controls which PyTorch wheel index
#                                     is used. GPU build requires Railway
#                                     to provision a GPU host at deploy.
# Required runtime env vars:
#   HF_TOKEN            HuggingFace token with access granted to
#                       `facebook/tribev2` (gated repo)
#   TRIBE_MODEL_ID      Optional, defaults to "facebook/tribev2"
#   PORT                Set by Railway automatically
# Optional volume mount (highly recommended):
#   /app/cache          Persistent HuggingFace + nilearn cache so cold
#                       deploys don't re-download multi-GB weights.
# ─────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

ARG COMPUTE=cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/cache/huggingface \
    NILEARN_DATA=/app/cache/nilearn \
    TRIBE_CACHE_DIR=/app/cache/tribe \
    MPLCONFIGDIR=/app/cache/mpl \
    TOKENIZERS_PARALLELISM=false

# System deps: ffmpeg is mandatory for whisperx / torchaudio I/O.
# libgomp + libgl + libsndfile cover scientific Python wheels that link
# against OpenMP / image libs / soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libgomp1 \
        libsndfile1 \
        libgl1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch first so the build cache is reused across requirement changes.
COPY requirements.txt ./requirements.txt
RUN if [ "$COMPUTE" = "gpu" ]; then \
      pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchaudio==2.5.1 ; \
    else \
      pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchaudio==2.5.1 ; \
    fi \
 && pip install -r requirements.txt

# Install tribev2 from Meta's research repo + its transitive deps.
# tribev2 pulls in exca, einops, hydra-core, omegaconf, etc.  We let pip
# resolve these; if they clash with anything in requirements.txt, the
# build will fail loudly with a version conflict (preferable to silent
# ModuleNotFoundError at runtime).
RUN pip install "git+https://github.com/facebookresearch/tribev2.git@main"

# tribev2 shells out to `uvx whisperx` internally, but unpinned uvx pulls
# incompatible pyannote/lightning on cold workers. inference.py redirects
# those calls to the pinned `whisperx` CLI installed above. Verify the CLI
# is on PATH at build time so we fail fast if the install breaks.
RUN whisperx --help >/dev/null

COPY . /app

# Pre-create the cache dirs so the network volume mount (if any) gets the
# right perms. On RunPod Serverless, mount your network volume at /app/cache
# so the multi-GB model downloads survive worker restarts.
RUN mkdir -p /app/cache/huggingface /app/cache/nilearn /app/cache/tribe /app/cache/mpl

# RunPod Serverless entry point. The runpod package starts an internal
# worker loop that pulls events from RunPod's queue and calls handler().
# The old FastAPI app.py is preserved for local dev — run with:
#   uvicorn app:app --host 0.0.0.0 --port 8080
CMD ["python", "-u", "/app/handler.py"]
