"""Destrieux atlas region decoder. Maps a (V,) activation vector on
fsaverage5 to named functional clusters using `label_library.yaml`.

Ported from sshandhra1/self-talk-mirror/stages/region_decoder.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from nilearn import datasets

import config

logger = logging.getLogger(__name__)


@dataclass
class DecodedRegion:
    id: str
    scientific_name: str
    anatomical_descriptor: str
    yeo_network: str
    short_function: str
    function_summary: str
    score: float  # mean |activation| over the region's vertices


@dataclass
class RegionDecodeResult:
    top_regions: List[DecodedRegion]
    dominant_yeo_network: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_regions": [asdict(r) for r in self.top_regions],
            "dominant_yeo_network": self.dominant_yeo_network,
        }


_LABEL_LIBRARY: Optional[List[Dict[str, Any]]] = None
_DESTRIEUX: Optional[Dict[str, Any]] = None


def _load_label_library() -> List[Dict[str, Any]]:
    global _LABEL_LIBRARY
    if _LABEL_LIBRARY is not None:
        return _LABEL_LIBRARY
    with open(config.LABEL_LIBRARY_PATH, "r", encoding="utf-8") as f:
        _LABEL_LIBRARY = yaml.safe_load(f)
    return _LABEL_LIBRARY or []


def _load_destrieux() -> Dict[str, Any]:
    global _DESTRIEUX
    if _DESTRIEUX is not None:
        return _DESTRIEUX
    _DESTRIEUX = datasets.fetch_atlas_surf_destrieux()
    return _DESTRIEUX


def decode(peak_activation: np.ndarray) -> RegionDecodeResult:
    """
    Score every region in the label library against the peak activation
    vector, return the top-K by mean |activation| over the region's vertices.

    `peak_activation` shape: (V,) where V = LH + RH vertices on fsaverage5.
    """
    destrieux = _load_destrieux()
    map_left = np.asarray(destrieux["map_left"])
    map_right = np.asarray(destrieux["map_right"])
    label_names_raw = destrieux["labels"]
    label_names = [
        n.decode() if isinstance(n, bytes) else n for n in label_names_raw
    ]
    name_to_idx = {name: i for i, name in enumerate(label_names)}

    n_left = map_left.shape[0]
    n_right = map_right.shape[0]
    n_total = n_left + n_right
    if peak_activation.shape[0] != n_total:
        logger.warning(
            "peak_activation length %d != expected %d (LH=%d + RH=%d) — proceeding anyway",
            peak_activation.shape[0], n_total, n_left, n_right,
        )

    abs_peak = np.abs(peak_activation)

    scored: List[DecodedRegion] = []
    for entry in _load_label_library():
        names = entry.get("destrieux_labels", [])
        indices = [name_to_idx[n] for n in names if n in name_to_idx]
        if not indices:
            continue

        mask_left = np.isin(map_left, indices)
        mask_right = np.isin(map_right, indices)
        mask_full = np.concatenate([mask_left, mask_right])
        if mask_full.shape[0] != abs_peak.shape[0]:
            # Length mismatch — slice or pad defensively so we don't crash
            common = min(mask_full.shape[0], abs_peak.shape[0])
            mask_full = mask_full[:common]
            slab = abs_peak[:common]
        else:
            slab = abs_peak

        if not mask_full.any():
            continue

        score = float(np.mean(slab[mask_full]))
        scored.append(
            DecodedRegion(
                id=entry["id"],
                scientific_name=entry["scientific_name"],
                anatomical_descriptor=entry["anatomical_descriptor"],
                yeo_network=entry["yeo_network"],
                short_function=entry["short_function"],
                function_summary=entry["function_summary"].strip(),
                score=score,
            )
        )

    scored.sort(key=lambda r: r.score, reverse=True)
    top = scored[: config.TOP_K_REGIONS]

    # Dominant Yeo network = most common among the top-K
    if top:
        nets = [r.yeo_network for r in top]
        dominant = max(set(nets), key=nets.count)
    else:
        dominant = None

    return RegionDecodeResult(top_regions=top, dominant_yeo_network=dominant)
