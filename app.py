"""Space of Mind — brain-service.

POST /render — multipart audio in, brain PNG + region metadata out.

Architecture:
  - All heavy artifacts (TRIBE checkpoint, nilearn atlases, fsaverage5 mesh,
    optional warmup render that primes WhisperX / Llama / Wav2Vec2 caches)
    load at startup via FastAPI's `lifespan` handler. Uvicorn won't accept
    connections until startup finishes.
  - `/health` is a fast liveness probe (always 200 once the process is up).
  - `/ready` returns 200 only after startup warmup has completed
    successfully. Use this for orchestration; Vercel should consume `/ready`
    before sending real booth traffic.
  - `/warmup` triggers a full warmup-render manually (audio optional;
    without audio it just ensures the TRIBE model + atlases are loaded).
  - `/render` runs inference. Should be fast (~30–60s CPU, ~5–10s GPU)
    because everything is already in memory + on disk by this point.

Auth: optional bearer token via SERVICE_AUTH_TOKEN env var. If set, clients
(in our case the Vercel /api/analyze route) must present
`Authorization: Bearer <token>`.

License caveat: TRIBE v2 is CC BY-NC. Use for research / internal demos only.
"""
from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import inference
import regions as regions_mod
import render as render_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("brain-service")


# ─────────────────────────────────────────────────────────────────────────
# Readiness state — set by the lifespan handler. /ready reads this.
# ─────────────────────────────────────────────────────────────────────────
class _State:
    started_at: float = 0.0
    tribe_loaded: bool = False
    atlases_loaded: bool = False
    full_warmup_done: bool = False
    last_error: str | None = None


STATE = _State()


# ─────────────────────────────────────────────────────────────────────────
# Lifespan — runs once at process start and once at shutdown. Models load
# here so the first user-facing /render request doesn't pay cold-start cost.
# ─────────────────────────────────────────────────────────────────────────
def _find_warmup_audio() -> Path | None:
    """Return the first existing candidate warmup audio path, or None."""
    here = Path(__file__).parent
    candidates = [
        here / "assets" / "test.wav",
        here / "assets" / "test.webm",
        here / "assets" / "warmup-audio.wav",
        here / "warmup-audio.wav",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


WARMUP_AUDIO_PATH = _find_warmup_audio()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Block process startup until models + atlases are loaded."""
    STATE.started_at = time.time()
    logger.info("startup: warming brain-service ...")

    # ── 1. TRIBE model — multi-GB checkpoint into memory ───────────────
    try:
        t0 = time.time()
        logger.info("startup [1/3]: loading TRIBE model ...")
        inference.get_model()
        STATE.tribe_loaded = True
        logger.info("startup [1/3]: TRIBE loaded in %.1fs", time.time() - t0)
    except Exception as exc:
        STATE.last_error = f"TRIBE load failed: {exc}"
        logger.exception("startup [1/3] FAILED — service will be /health-up but /ready-down")
        # Don't raise — let the container come up so the operator can
        # inspect /ready and the logs. /render will surface the same error.

    # ── 2. nilearn atlases — fsaverage5 mesh + Destrieux parcellation ──
    try:
        t0 = time.time()
        logger.info("startup [2/3]: prefetching nilearn atlases ...")
        from nilearn import datasets
        datasets.fetch_atlas_surf_destrieux()
        datasets.fetch_surf_fsaverage("fsaverage5")
        STATE.atlases_loaded = True
        logger.info("startup [2/3]: atlases ready in %.1fs", time.time() - t0)
    except Exception as exc:
        STATE.last_error = f"atlas prefetch failed: {exc}"
        logger.exception("startup [2/3] FAILED — atlases will lazy-load on first /render")

    # ── 3. Full warmup-render — primes WhisperX uvx env, Llama, Wav2Vec2
    # Only runs if a warmup audio file exists in the image AND the
    # SKIP_STARTUP_WARMUP env var isn't set. If skipped or no file,
    # the FIRST real /render call pays the download cost.
    if config.SKIP_STARTUP_WARMUP:
        logger.info(
            "startup [3/3]: SKIPPED — SKIP_STARTUP_WARMUP=true is set. "
            "Use this to recover from an OOM restart loop after bumping "
            "container memory. Trigger a manual /render afterward to warm caches."
        )
    elif WARMUP_AUDIO_PATH is not None:
        try:
            t0 = time.time()
            logger.info(
                "startup [3/3]: running warmup-render on %s (primes WhisperX + Llama + Wav2Vec2 caches) ...",
                WARMUP_AUDIO_PATH,
            )
            preds, _ = inference.run_tribe(WARMUP_AUDIO_PATH)
            _ = inference.pick_peak_timestep(preds)
            STATE.full_warmup_done = True
            logger.info("startup [3/3]: full warmup-render complete in %.1fs", time.time() - t0)
        except Exception as exc:
            STATE.last_error = f"warmup-render failed: {exc}"
            logger.exception("startup [3/3] FAILED — first real /render will redo this work")
    else:
        logger.info(
            "startup [3/3]: SKIPPED — no warmup audio found. First real /render "
            "will pay download cost (~4 GB WhisperX env + several GB Llama + "
            "Wav2Vec2). To enable: drop a short WAV at one of: "
            "assets/test.wav, assets/test.webm, assets/warmup-audio.wav, warmup-audio.wav."
        )

    elapsed = time.time() - STATE.started_at
    logger.info(
        "startup complete in %.1fs (tribe=%s, atlases=%s, full_warmup=%s)",
        elapsed,
        STATE.tribe_loaded,
        STATE.atlases_loaded,
        STATE.full_warmup_done,
    )

    yield

    # No shutdown work needed; let the OS reclaim memory.


app = FastAPI(
    title="Space of Mind — brain-service",
    version="0.2.0",
    lifespan=lifespan,
)

# Lock CORS down to nothing by default; this service is called server→server
# from Vercel, never from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def _check_auth(authorization: str | None = Header(default=None)) -> None:
    """Bearer-token gate, only enforced when SERVICE_AUTH_TOKEN is configured."""
    if not config.SERVICE_AUTH_TOKEN:
        return
    expected = f"Bearer {config.SERVICE_AUTH_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


class RegionOut(BaseModel):
    id: str
    scientific_name: str
    anatomical_descriptor: str
    yeo_network: str
    short_function: str
    function_summary: str
    score: float


class RenderOut(BaseModel):
    """
    Response body for POST /render. `brain_image_base64` is a data-URL-safe
    PNG; caller is expected to either persist the bytes to its own object
    store (Vercel Blob in our case) or inline as a data URI.
    """
    brain_image_base64: str
    top_regions: list[RegionOut]
    dominant_yeo_network: str | None
    transcript_text: str
    peak_timestep: int


@app.get("/health")
def health() -> dict:
    """Liveness probe. Process is up — does NOT indicate readiness."""
    return {"ok": True, "service": "brain-service"}


@app.get("/ready")
def ready() -> dict:
    """
    Readiness probe. Returns 200 only when startup warmup has completed
    enough for /render to succeed quickly. Vercel / monitoring should
    poll this before routing real traffic.
    """
    is_ready = STATE.tribe_loaded and STATE.atlases_loaded
    response = {
        "ready": is_ready,
        "tribe_loaded": STATE.tribe_loaded,
        "atlases_loaded": STATE.atlases_loaded,
        "full_warmup_done": STATE.full_warmup_done,
        "uptime_seconds": round(time.time() - STATE.started_at, 1) if STATE.started_at else 0,
        "last_error": STATE.last_error,
    }
    if not is_ready:
        raise HTTPException(status_code=503, detail=response)
    return response


@app.get("/warmup")
def warmup(_: None = Depends(_check_auth)) -> dict:
    """
    Manual full warmup. Idempotent — re-loads TRIBE if not already loaded,
    re-fetches atlases. Useful if /ready reported a partial-warmup failure
    you've since fixed (e.g. HF_TOKEN updated).

    For a TRUE full warmup that also primes WhisperX/Llama/Wav2Vec2 caches,
    POST a real audio file to /render once after deploy.
    """
    if not STATE.tribe_loaded:
        inference.get_model()
        STATE.tribe_loaded = True
    if not STATE.atlases_loaded:
        from nilearn import datasets
        datasets.fetch_atlas_surf_destrieux()
        datasets.fetch_surf_fsaverage("fsaverage5")
        STATE.atlases_loaded = True
    return {
        "ok": True,
        "tribe_loaded": STATE.tribe_loaded,
        "atlases_loaded": STATE.atlases_loaded,
        "full_warmup_done": STATE.full_warmup_done,
    }


@app.post("/render", response_model=RenderOut)
async def render_endpoint(
    audio: UploadFile = File(...),
    return_b64: str = Form("true"),
    _: None = Depends(_check_auth),
) -> RenderOut:
    """
    Accept a single audio file, return the brain map + region metadata.

    Pipeline:
      1. Persist audio to a temp file (whisperx needs a real path).
      2. Run TRIBE → (T, V) activations + WhisperX transcript.
      3. Pick peak timestep → (V,) activation vector.
      4. Decode top-K regions against the Destrieux atlas.
      5. Render brand-styled cortical surface PNG.
      6. Return base64-encoded PNG + structured region data.

    On CPU this takes ~30–60s for a 60s recording. On GPU ~5–10s.
    Assumes startup warmup completed — if not, first call may pay an
    extra ~5–10 min for HF model downloads.
    """
    # The browser records in WebM/Opus by default but TRIBE only accepts
    # .flac/.mp3/.ogg/.wav. Save the upload as-is to a temp file, then
    # transcode to mono 16kHz WAV (what WhisperX inside TRIBE expects).
    # Use a SECOND distinct tempfile for the output so ffmpeg doesn't refuse
    # to overwrite its own input (happens when the upload is already .wav).
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(await audio.read())
        raw_path = Path(tmp_in.name)

    with tempfile.NamedTemporaryFile(suffix=".16k.wav", delete=False) as tmp_out:
        audio_path = Path(tmp_out.name)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(raw_path),
                "-ar", "16000",  # 16 kHz — WhisperX standard
                "-ac", "1",      # mono
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "ignore") if exc.stderr else "(empty)"
        logger.error("ffmpeg transcode failed:\n%s", stderr)
        for p in (raw_path, audio_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        # Surface the ffmpeg stderr in the response so the caller can debug
        # without digging through Railway logs.
        raise HTTPException(
            status_code=400,
            detail=f"audio transcode failed: {stderr.strip()[:400]}",
        )

    # Original upload no longer needed after transcode
    try:
        raw_path.unlink(missing_ok=True)
    except Exception:
        pass

    try:
        logger.info("running TRIBE on %s", audio_path)
        preds, transcript = inference.run_tribe(audio_path)
        peak_t, peak_vec = inference.pick_peak_timestep(preds)
        logger.info("preds shape=%s, peak_t=%d", preds.shape, peak_t)

        decoded = regions_mod.decode(peak_vec)
        png_bytes = render_mod.render_to_png(peak_vec, decoded)

        b64 = base64.b64encode(png_bytes).decode("ascii") if return_b64.lower() == "true" else ""

        # Mark full warmup as done — any successful render means all caches
        # are populated, future requests are fast.
        STATE.full_warmup_done = True

        return RenderOut(
            brain_image_base64=b64,
            top_regions=[RegionOut(**r.__dict__) for r in decoded.top_regions],
            dominant_yeo_network=decoded.dominant_yeo_network,
            transcript_text=transcript,
            peak_timestep=peak_t,
        )
    except Exception as exc:
        logger.exception("render pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass
