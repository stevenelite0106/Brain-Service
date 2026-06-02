"""TRIBE v2 inference. Loads the model as a singleton on first request.

Ported from sshandhra1/self-talk-mirror/stages/tribe_inference.py with minor
adjustments: paths are config-driven and we expose the transcript text we
get from WhisperX so the Vercel side can cross-check against OpenAI Whisper.

CC BY-NC compliance: TRIBE v2 is non-commercial. See README.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

import config

logger = logging.getLogger(__name__)

_MODEL = None  # singleton — TribeModel instance once loaded


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
