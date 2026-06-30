"""TRIBE v2 inference. Loads the model as a singleton on first request.

Ported from sshandhra1/self-talk-mirror/stages/tribe_inference.py with minor
adjustments: paths are config-driven and we expose the transcript text we
get from WhisperX so the Vercel side can cross-check against OpenAI Whisper.

CC BY-NC compliance: TRIBE v2 is non-commercial. See README.
"""
from __future__ import annotations

import logging
import shutil
import subprocess as _subprocess_mod
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

import config

logger = logging.getLogger(__name__)

_MODEL = None  # singleton — TribeModel instance once loaded


# ─────────────────────────────────────────────────────────────────────────
# WhisperX CPU compatibility shim.
#
# TRIBE v2 calls `uvx whisperx <audio> ...` as a subprocess for word-level
# transcription. Two problems with that default:
#
# 1. `uvx` resolves whisperx + deps at runtime with no version pins. Fresh
#    workers pull latest pyannote/lightning/torch, which breaks VAD with:
#      AttributeError: 'generator' object has no attribute 'data'
#    We redirect to the container's pinned `whisperx` CLI instead.
#
# 2. WhisperX's CLI defaults `--compute_type` to `float16`, which
#    ctranslate2 refuses on CPU hosts. We rewrite compute_type/device to
#    match the actual hardware (float16+cuda on GPU, int8+cpu on CPU).
# ─────────────────────────────────────────────────────────────────────────

_orig_subprocess_run = _subprocess_mod.run

# Pinned in requirements.txt / Dockerfile — must stay in sync with tribev2.
_WHISPERX_BIN = shutil.which("whisperx") or "whisperx"


def _force_arg(cmd: list, flag: str, value: str) -> tuple[list, str | None]:
    """Set `--flag value` in cmd, replacing the existing value if present.
    Returns (new_cmd, old_value_or_None)."""
    new_cmd = list(cmd)
    if flag in new_cmd:
        idx = new_cmd.index(flag)
        if idx + 1 < len(new_cmd):
            old = str(new_cmd[idx + 1])
            new_cmd[idx + 1] = value
            return new_cmd, old
    new_cmd.extend([flag, value])
    return new_cmd, None


def _redirect_uvx_whisperx(cmd: list) -> tuple[list, bool]:
    """Replace tribev2's `uvx whisperx ...` with our pinned container binary."""
    if len(cmd) >= 2 and Path(str(cmd[0])).name == "uvx" and str(cmd[1]).lower() == "whisperx":
        redirected = [_WHISPERX_BIN, *cmd[2:]]
        logger.info("[whisperx-shim] redirecting uvx whisperx -> %s", _WHISPERX_BIN)
        return redirected, True
    return cmd, False


def _patched_subprocess_run(cmd, *args, **kwargs):
    """
    Intercept tribev2's whisperx subprocess calls:
      - use pinned container whisperx instead of unpinned `uvx whisperx`
      - force compute_type/device to match available hardware
    """
    try:
        if isinstance(cmd, list):
            cmd_strs = [str(c) for c in cmd]
            if any("whisperx" in s.lower() for s in cmd_strs):
                cmd, _ = _redirect_uvx_whisperx(cmd)
                if torch.cuda.is_available():
                    target_ct, target_dev = "float16", "cuda"
                else:
                    # int8 is faster-whisper's recommended CPU quantization
                    # (~4x faster than float32, near-zero accuracy loss on
                    # speech transcription).
                    target_ct, target_dev = "int8", "cpu"
                cmd, old_ct = _force_arg(cmd, "--compute_type", target_ct)
                cmd, old_dev = _force_arg(cmd, "--device", target_dev)
                logger.info(
                    "[whisperx-shim] forced --compute_type %s (was %r), --device %s (was %r)",
                    target_ct, old_ct, target_dev, old_dev,
                )
    except Exception as exc:
        logger.warning("[whisperx-shim] patch failed: %s", exc)
    return _orig_subprocess_run(cmd, *args, **kwargs)


_subprocess_mod.run = _patched_subprocess_run


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model():
    """Lazy-load the TRIBE v2 checkpoint from HuggingFace. ~30s cold start."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    # Imported lazily so the module is importable even before tribev2 is
    # installed (useful for tests / local lint passes).
    from tribev2 import TribeModel  # type: ignore

    config.TRIBE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # `device: "auto"` lets the underlying feature extractors place themselves.
    #
    # batch_size override: TRIBE's default text_feature config sets
    # batch_size=4 (see facebookresearch/tribev2 grids/defaults.py), which
    # leaves a 24 GB L4 almost idle and bottlenecks the whole pipeline at
    # ~3s per 4-word batch. For a 5-take, ~3-minute booth recording that
    # adds up to ~10 minutes — past Vercel's 300s function ceiling.
    #
    # Empirical VRAM budget — RunPod endpoint runs on a 48 GB GPU
    # (upgraded from 24 GB L4 after the L4 OOMed at batch_size=32):
    #   - TRIBE + Llama-3.2-3B + Wav2Vec2 weights occupy ~20 GB.
    #   - On 24 GB L4: batch_size=32 OOMed (~800 MB short for Llama MLP
    #     forward). batch_size=16 fit.
    #   - On 48 GB: ~28 GB activation headroom. batch_size=32 fits with
    #     room to spare. Cuts embed wallclock from ~5 min (at 16) to
    #     ~2.5 min on a full 5-take recording.
    # If we ever need to go higher, 64 should fit; beyond that, trim the
    # text_feature.layers list (currently [0, .2, .4, .6, .8, 1.0]).
    config_update = {
        "data.text_feature.device": "auto",
        "data.text_feature.batch_size": 16,
        "data.audio_feature.device": "auto",
    }

    logger.info("loading TRIBE model %s into %s", config.TRIBE_MODEL_ID, config.TRIBE_CACHE_DIR)
    model = TribeModel.from_pretrained(
        config.TRIBE_MODEL_ID,
        cache_folder=str(config.TRIBE_CACHE_DIR),
        config_update=config_update,
    )

    device = _pick_device()
    logger.info("placing model on device=%s", device)
    if hasattr(model, "to"):
        try:
            model.to(device)
        except Exception:
            logger.exception("model.to(%s) failed — continuing on default device", device)

    _MODEL = model
    return model


def run_tribe(audio_path: Path) -> Tuple[np.ndarray, str]:
    """
    Run TRIBE on an audio file. Returns:
      - preds: (T, V) numpy array of cortical activations on fsaverage5,
               where V = LH_vertices + RH_vertices ≈ 20484.
      - transcript_text: WhisperX-derived word sequence joined by spaces.
    """
    model = get_model()

    # WhisperX runs inside this call — transcribes audio to word events.
    df = model.get_events_dataframe(audio_path=str(audio_path))

    word_events = df[df["type"] == "Word"].sort_values("start")
    transcript_text = " ".join(word_events["text"].astype(str).tolist())

    with torch.inference_mode():
        preds, _ = model.predict(events=df)

    if isinstance(preds, torch.Tensor):
        preds_np = preds.detach().cpu().numpy()
    else:
        preds_np = np.asarray(preds)

    return preds_np, transcript_text


def pick_peak_timestep(preds: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Reduce (T, V) → (V,) by picking the timestep with the highest L2 norm.
    That's the moment in the recording when the model predicted the most
    cortical engagement. Used as the input to region decoding + rendering.
    """
    norms = np.linalg.norm(preds, axis=1)
    peak_t = int(np.argmax(norms))
    return peak_t, preds[peak_t]
