"""Space of Mind brain-service — RunPod Serverless entry point.

Replaces the long-running FastAPI app.py with a request-scoped handler
function. RunPod's serverless runtime calls handler(event) once per
inference request and keeps the worker process alive across calls (until
the configured idle timeout), so we get the same warm-model benefit as
the FastAPI lifespan pattern — just expressed differently.

Worker lifecycle on RunPod Serverless:
  1. First request to a cold worker: RunPod spins up the container, this
     module is imported, the EAGER WARMUP block below runs (~30–60s),
     then handler() runs the actual inference.
  2. Subsequent requests on the same worker: handler() runs directly,
     model already in VRAM — ~20–60s per render on T4/L4.
  3. After IDLE_TIMEOUT (configured in the RunPod endpoint settings),
     RunPod shuts the worker down. Next request restarts the cycle.

For consistent latency during the event, set min active workers ≥ 1 in
the RunPod endpoint configuration. Costs a small idle fee but eliminates
the cold-start hit.

License caveat: TRIBE v2 is CC BY-NC. Research / internal demos only.
"""
from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
import time
from pathlib import Path

import runpod

import inference
import regions as regions_mod
import render as render_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("brain-handler")


# ─────────────────────────────────────────────────────────────────────────
# Worker init — runs once per cold start. Same idea as the lifespan handler
# we had in the FastAPI version; loads TRIBE + atlases so the first
# `handler()` call only pays for the actual inference.
# ─────────────────────────────────────────────────────────────────────────
def _worker_init() -> None:
    t0 = time.time()
    logger.info("worker init: loading TRIBE model ...")
    inference.get_model()
    logger.info("worker init: TRIBE loaded in %.1fs", time.time() - t0)

    t1 = time.time()
    logger.info("worker init: prefetching nilearn atlases ...")
    from nilearn import datasets
    datasets.fetch_atlas_surf_destrieux()
    datasets.fetch_surf_fsaverage("fsaverage5")
    logger.info("worker init: atlases ready in %.1fs", time.time() - t1)

    logger.info("worker init: complete in %.1fs total", time.time() - t0)


_worker_init()


# ─────────────────────────────────────────────────────────────────────────
# Handler — called per request by RunPod's serverless runtime.
# ─────────────────────────────────────────────────────────────────────────
def handler(event: dict) -> dict:
    """
    Input event shape:
      {
        "input": {
          "audio_b64": "<base64-encoded audio bytes>",
          "audio_format": "webm" | "wav" | "ogg" | "m4a"   (optional, default "webm")
        }
      }

    Output shape on success:
      {
        "brain_image_base64": "...",
        "top_regions": [ {id, scientific_name, ...}, ... ],
        "dominant_yeo_network": "Default",
        "transcript_text": "...",
        "peak_timestep": 12
      }

    Output shape on error:
      { "error": "<message>" }
    """
    try:
        payload = event.get("input") or {}
        audio_b64 = payload.get("audio_b64")
        if not audio_b64:
            return {"error": "audio_b64 required in input"}

        audio_bytes = base64.b64decode(audio_b64)
        audio_format = (payload.get("audio_format") or "webm").lstrip(".")
        suffix = f".{audio_format}"

        # Persist upload to a temp file (ffmpeg + TRIBE need real paths).
        # Use distinct in/out paths so ffmpeg doesn't refuse to overwrite
        # its own input when the upload happens to already be .wav.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            raw_path = Path(tmp_in.name)
        with tempfile.NamedTemporaryFile(suffix=".16k.wav", delete=False) as tmp_out:
            wav_path = Path(tmp_out.name)

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(raw_path),
                    "-ar", "16000",  # 16 kHz — WhisperX standard
                    "-ac", "1",      # mono
                    str(wav_path),
                ],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", "ignore") if exc.stderr else "(empty)"
            logger.error("ffmpeg transcode failed:\n%s", stderr)
            _cleanup(raw_path, wav_path)
            return {"error": f"audio transcode failed: {stderr.strip()[:400]}"}

        # ffmpeg succeeded; the raw upload is no longer needed.
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass

        try:
            logger.info("running TRIBE on %s", wav_path)
            preds, transcript = inference.run_tribe(wav_path)
            peak_t, peak_vec = inference.pick_peak_timestep(preds)
            logger.info("preds shape=%s, peak_t=%d", preds.shape, peak_t)

            decoded = regions_mod.decode(peak_vec)
            png_bytes = render_mod.render_to_png(peak_vec, decoded)

            return {
                "brain_image_base64": base64.b64encode(png_bytes).decode("ascii"),
                "top_regions": [
                    {
                        "id": r.id,
                        "scientific_name": r.scientific_name,
                        "anatomical_descriptor": r.anatomical_descriptor,
                        "yeo_network": r.yeo_network,
                        "short_function": r.short_function,
                        "function_summary": r.function_summary,
                        "score": float(r.score),
                    }
                    for r in decoded.top_regions
                ],
                "dominant_yeo_network": decoded.dominant_yeo_network,
                "transcript_text": transcript,
                "peak_timestep": int(peak_t),
            }
        except Exception as exc:
            logger.exception("inference pipeline failed")
            return {"error": f"render pipeline failed: {exc}"}
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

    except Exception as exc:
        logger.exception("handler crashed")
        return {"error": f"handler crashed: {exc}"}


def _cleanup(*paths: Path) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────
# Start the RunPod serverless loop. This blocks; RunPod's runtime feeds
# events through `handler` until the worker is killed.
# ─────────────────────────────────────────────────────────────────────────
runpod.serverless.start({"handler": handler})
