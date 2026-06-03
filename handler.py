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

import numpy as np
import runpod
import soundfile as sf

import inference
import regions as regions_mod
import render as render_mod

# Cap on frames returned to the frontend. TRIBE's native sample rate is ~1
# frame per TR (fMRI repetition time, ~1.49s) so a 3-min recording is
# already ~120 frames. We subsample only on the rare case it exceeds this.
# 150 frames * 20484 verts * 2 bytes (float16) = ~6 MB raw, ~2 MB gzipped.
MAX_ACTIVATION_FRAMES = 150

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

            # Pack the per-frame activation tensor for browser-side
            # playback. The frontend uses this to drive a three.js cortex
            # in sync with the audio scrubber on the Confirmation screen.
            audio_duration_s = float(sf.info(str(wav_path)).duration)
            activations_packed = _pack_activations(preds, audio_duration_s, peak_t)

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
                **activations_packed,
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


def _pack_activations(
    preds: np.ndarray,
    audio_duration_s: float,
    peak_t_orig: int,
) -> dict:
    """
    Convert TRIBE's (T, V) float32 tensor into a transport-friendly payload
    for the browser-side cortex viewer:
      - subsample T to <= MAX_ACTIVATION_FRAMES along a uniform time grid
      - normalize globally so the shader colormap stays consistent across
        the whole recording (per-frame norm would make every frame look
        equally intense, defeating the visualization)
      - quantize to float16 (halves payload; precision loss is irrelevant
        for a colormap with ~256 distinguishable steps)
      - generate frame_times in seconds so the frontend can look up the
        active frame from audio.currentTime

    Returns a dict whose keys are spliced into handler()'s response. Layout
    of `activations_b64` is row-major (T, V) float16, little-endian — the
    JS side reads it as Uint16Array and uploads to a half-float DataTexture.
    """
    T_orig, V = preds.shape

    # Subsample frames if needed; record which original indices we kept so
    # the peak_timestep we hand back stays consistent with the binary.
    if T_orig > MAX_ACTIVATION_FRAMES:
        kept = np.linspace(0, T_orig - 1, MAX_ACTIVATION_FRAMES, dtype=np.int64)
    else:
        kept = np.arange(T_orig, dtype=np.int64)
    subsampled = preds[kept]
    T = subsampled.shape[0]

    # Global normalization to [-1, 1] so the shader's colormap LUT can be
    # a fixed 256-step gradient. Use 99th-percentile abs to clip the long
    # tail without losing typical-frame contrast.
    abs_max = float(np.percentile(np.abs(subsampled), 99))
    if abs_max < 1e-6:
        abs_max = 1.0
    normalized = np.clip(subsampled / abs_max, -1.0, 1.0).astype(np.float16)

    # Uniform time grid over the audio duration. TRIBE outputs are
    # regression onto an fMRI TR sampling, which is approximately linear
    # in audio time — uniform spacing is the right model.
    if T == 1:
        frame_times = [0.0]
    else:
        frame_times = np.linspace(0.0, audio_duration_s, T).tolist()

    # Map the original-index peak into the subsampled space so the
    # frontend can still highlight the same moment.
    peak_t_packed = int(np.argmin(np.abs(kept - peak_t_orig)))

    return {
        "activations_b64": base64.b64encode(normalized.tobytes()).decode("ascii"),
        "activations_dtype": "float16",
        "activations_layout": "row_major_TxV",
        "frame_count": int(T),
        "vertex_count": int(V),
        "frame_times": frame_times,
        "peak_timestep_packed": peak_t_packed,
        "audio_duration_seconds": audio_duration_s,
    }


# ─────────────────────────────────────────────────────────────────────────
# Start the RunPod serverless loop. This blocks; RunPod's runtime feeds
# events through `handler` until the worker is killed.
# ─────────────────────────────────────────────────────────────────────────
runpod.serverless.start({"handler": handler})
