"""Brain-service configuration. Reads from env vars with sensible defaults."""
from __future__ import annotations

import os
from pathlib import Path

# Model identifier on HuggingFace. Must have access granted to the gated repo.
TRIBE_MODEL_ID = os.getenv("TRIBE_MODEL_ID", "facebook/tribev2")

# Cache directories — mount a persistent Railway volume at /app/cache so cold
# deploys don't re-download multi-GB weights.
CACHE_ROOT = Path(os.getenv("CACHE_ROOT", "/app/cache"))
TRIBE_CACHE_DIR = Path(os.getenv("TRIBE_CACHE_DIR", str(CACHE_ROOT / "tribe")))

# Path to the bundled region label library (Destrieux → app-level regions).
LABEL_LIBRARY_PATH = Path(
    os.getenv("LABEL_LIBRARY_PATH", str(Path(__file__).parent / "label_library.yaml"))
)

# How many top regions to surface per recording.
TOP_K_REGIONS = int(os.getenv("TOP_K_REGIONS", "4"))

# Output PNG dimensions. Width × height in pixels at 180dpi.
OUTPUT_DPI = int(os.getenv("OUTPUT_DPI", "180"))

# Optional bearer token that callers must present (defense-in-depth so a
# random internet client can't burn through your Railway compute budget).
SERVICE_AUTH_TOKEN = os.getenv("SERVICE_AUTH_TOKEN", "")

# Escape hatch: if startup warmup is OOM-killing the container in a
# restart loop, set SKIP_STARTUP_WARMUP=true on Railway. Container will
# come up with only TRIBE + atlases loaded; the first real /render
# pays the full download cost. Use this once to break the loop, then
# unset it after bumping memory so future deploys auto-warm normally.
SKIP_STARTUP_WARMUP = os.getenv("SKIP_STARTUP_WARMUP", "").lower() in ("1", "true", "yes")
