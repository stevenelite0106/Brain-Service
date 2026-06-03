"""Brain rendering — fsaverage5 PIAL surface, two lateral panels with the
matplotlib "hot" colormap. Matches the Meta TRIBE v2 demo aesthetic and
the live BrainCanvas on the Confirmation screen, so the email and the
live page read as one piece.

Pial (folded) surface rather than inflated so the gyri/sulci structure
shows through. Hot colormap (black→red→yellow→white) overlaid on the
sulc-shaded base for the classic neuroimaging look.
"""
from __future__ import annotations

import io
import logging
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets, plotting

import config
from regions import RegionDecodeResult

logger = logging.getLogger(__name__)


# Activation overlay: matplotlib's built-in "hot" colormap. We use it
# directly rather than building a custom one so the email PNG and the
# live three.js BrainCanvas (which implements "hot" in-shader) match.
_CMAP_NAME = "hot"

# Dark background — same intent as the BrainCanvas stage in
# app/globals.css: let the warm cortex glow against deep space.
_BG_COLOR = "#0A0A14"
_INK = "#FFFFFF"
_INK_DIM = "#A7A7BC"


def render_to_png(
    peak_activation: np.ndarray,
    decoded: RegionDecodeResult,
) -> bytes:
    """
    Return a PNG of the brand-styled cortical surface with the top regions
    labeled below. Caller is responsible for storing the bytes (e.g. in
    Vercel Blob).
    """
    fsavg = datasets.fetch_surf_fsaverage("fsaverage5")

    n = peak_activation.shape[0]
    lh = peak_activation[: n // 2]
    rh = peak_activation[n // 2 :]

    # Threshold at the 50th percentile of |activation| so we hide noise.
    threshold = float(np.quantile(np.abs(peak_activation), 0.5))

    fig = plt.figure(figsize=(10, 6), facecolor=_BG_COLOR)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.4, 1.0], hspace=0.05, wspace=0.05)

    ax_left = fig.add_subplot(gs[0, 0], projection="3d")
    ax_right = fig.add_subplot(gs[0, 1], projection="3d")
    ax_legend = fig.add_subplot(gs[1, :])

    plotting.plot_surf_stat_map(
        surf_mesh=fsavg["pial_left"],
        stat_map=lh,
        bg_map=fsavg["sulc_left"],
        hemi="left",
        view="lateral",
        cmap=_CMAP_NAME,
        threshold=threshold,
        colorbar=False,
        axes=ax_left,
        bg_on_data=True,
        darkness=0.6,
    )
    plotting.plot_surf_stat_map(
        surf_mesh=fsavg["pial_right"],
        stat_map=rh,
        bg_map=fsavg["sulc_right"],
        hemi="right",
        view="lateral",
        cmap=_CMAP_NAME,
        threshold=threshold,
        colorbar=False,
        axes=ax_right,
        bg_on_data=True,
        darkness=0.6,
    )

    for ax in (ax_left, ax_right):
        ax.set_facecolor(_BG_COLOR)

    # Region legend — top-K decoded regions as a calm typographic list.
    ax_legend.axis("off")
    ax_legend.set_facecolor(_BG_COLOR)

    if decoded.top_regions:
        title = "Where the patterns lit up"
        ax_legend.text(
            0.5, 0.95, title.upper(),
            ha="center", va="top",
            fontsize=8, color=_INK_DIM,
            fontfamily="DejaVu Sans Mono",
            transform=ax_legend.transAxes,
        )

        rows: List[str] = []
        for i, r in enumerate(decoded.top_regions, 1):
            rows.append(f"{i:>2}.  {r.scientific_name}   ·   {r.short_function}")
        legend_text = "\n".join(rows)
        ax_legend.text(
            0.5, 0.78, legend_text,
            ha="center", va="top",
            fontsize=10, color=_INK,
            fontfamily="DejaVu Sans",
            transform=ax_legend.transAxes,
        )

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=config.OUTPUT_DPI,
        bbox_inches="tight",
        facecolor=_BG_COLOR,
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return buf.read()
