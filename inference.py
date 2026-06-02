"""TRIBE v2 inference. Loads the model as a singleton on first request.

Ported from sshandhra1/self-talk-mirror/stages/tribe_inference.py with minor
adjustments: paths are config-driven and we expose the transcript text we
get from WhisperX so the Vercel side can cross-check against OpenAI Whisper.

CC BY-NC compliance: TRIBE v2 is non-commercial. See README.
"""
from __future__ import annotations

import logging
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
# transcription. WhisperX's CLI defaults `--compute_type` to `float16`,
# which ctranslate2 refuses on CPU hosts ("Requested float16 compute type,
# but the target device or backend do not support efficient float16
# computation."). We patch subprocess.run to inject `--compute_type int8`
# into any whisperx invocation that doesn't already specify one.
#
# int8 quantization is the faster-whisper / ctranslate2 recommended setting
# for CPU inference — ~4x faster than float32 with negligible accuracy loss
# on speech transcription.
# ─────────────────────────────────────────────────────────────────────────

_orig_subprocess_run = _subprocess_mod.run


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


def _patched_subprocess_run(cmd, *args, **kwargs):
    """
    Force whisperx invocations to CPU + int8 compute type.

    tribev2.eventstransforms._get_transcript_from_audio hard-codes the
    whisperx CLI command including --compute_type and --device with values
    that assume CUDA. On Railway's CPU host this fails with
        ValueError: Requested float16 compute type, but the target device
        or backend do not support efficient float16 computation.

    We REPLACE whatever values tribev2 supplied with `--compute_type int8`
    and `--device cpu`. int8 is faster-whisper's recommended CPU quantization
    (~4x faster than float32, negligible accuracy loss for speech).
    """
    try:
        if isinstance(cmd, list):
            cmd_strs = [str(c) for c in cmd]
            if any("whisperx" in s.lower() for s in cmd_strs):
                cmd, old_ct = _force_arg(cmd, "--compute_type", "int8")
                cmd, old_dev = _force_arg(cmd, "--device", "cpu")
                logger.info(
                    "[whisperx-shim] forced --compute_type int8 (was %r), --device cpu (was %r)",
                    old_ct, old_dev,
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
    config_update = {
        "data.text_feature.device": "auto",
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
