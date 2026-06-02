"""Space of Mind — brain-service.

POST /render — multipart audio in, brain PNG + region metadata out.

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

app = FastAPI(title="Space of Mind — brain-service", version="0.1.0")

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
    """Liveness probe. Does NOT load the model — Railway hits this often."""
    return {"ok": True, "service": "brain-service"}


@app.get("/warmup")
def warmup(_: None = Depends(_check_auth)) -> dict:
    """
    Force the TRIBE checkpoint to load. Call this once after deploy so the
    first user-facing request doesn't pay the ~30s cold start.
    """
    inference.get_model()
    return {"ok": True, "warmed": True}


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
    """
    # The browser records in WebM/Opus by default but TRIBE only accepts
    # .flac/.mp3/.ogg/.wav. Save the upload as-is to a temp file, then
    # transcode to mono 16kHz WAV (what WhisperX inside TRIBE expects).
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        raw_path = Path(tmp.name)

    audio_path = raw_path.with_suffix(".wav")
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
        logger.error("ffmpeg transcode failed: %s", exc.stderr.decode("utf-8", "ignore"))
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="audio transcode failed")

    # Original webm no longer needed after transcode
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
