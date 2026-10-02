"""Supplementary figure: recovery of emergence time and selection on held-out simulations.

(a) Predicted emergence time (posterior mean of the softmax head) for each true
    generation, as horizontal violins with the interquartile range and the true value.
(b) Predicted versus true selection coefficient as a hexbin, with the identity line.

Both panels use the clean (unperturbed) held-out test set at 100x depth for the
single network used throughout the simulation figures (seed 42), matching
figure2_recovery. Outputs supfigure_DL_model_ab.{pdf,svg,_preview.png}.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from .common import SIM_DIR, fig_dir
from . import figure_style as st

OUTDIR = fig_dir("Supplementary", "ECDNA_SUPP_DIR")
DEPTH, SEED = 100, 42
GMIN, GMAX = 1, 9


def load():
    d = pd.read_parquet(os.path.join(SIM_DIR, f"cnn_attn_pool_d{DEPTH}_s{SEED}.parquet"))
    return d[~d.perturbed.astype(bool)]


def panel_a(ax, d):
    rows = [(g, d.g_exp[d.g_true == g].values) for g in range(GMIN, GMAX + 1)]
    parts = ax.violinplot([v for _, v in rows], positions=[g for g, _ in rows],
                          vert=False, widths=0.95, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(st.INTRO); body.set_alpha(0.45); body.set_edgecolor("none")
    for g, v in rows:
        lo, hi = v.min(), v.max()
        ax.plot([lo, hi], [g, g], color="0.35", lw=0.5, zorder=2)
        q1, q3 = np.percentile(v, [25, 75])
        ax.plot([q1, q3], [g, g], color="0.15", lw=3.2, solid_capstyle="butt", zorder=3)
        ax.scatter(g, g, s=22, color="white", edgecolor="0.15", lw=0.9, zorder=4)
    ax.set_xlabel(r"Predicted emergence time $\hat{g}$")
    ax.set_ylabel(r"True emergence time $g$")
    ax.set_yticks(range(GMIN, GMAX + 1))
    ax.set_xticks(range(GMIN, GMAX + 1))
    ax.set_ylim(GMAX + 0.8, GMIN - 0.8)
    ax.set_xlim(GMIN - 0.4, GMAX + 0.4)


def panel_b(fig, ax, d):
    x, y = d.s_hat.values, d.s_true.values
    hb = ax.hexbin(x, y, gridsize=45, cmap="Greens", mincnt=1, linewidths=0)
    ax.plot([0, 2.5], [0, 2.5], ls="--", lw=0.8, color="0.45", zorder=3)
    r2 = 1 - ((x - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rho = spearmanr(y, x).correlation
    rmse = np.sqrt(((x - y) ** 2).mean())
    ax.text(0.03, 0.97, f"$R^2$ = {r2:.3f}\n$\\rho$ = {rho:.3f}\nRMSE = {rmse:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.2,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))
    ax.set_xlabel(r"Predicted $\hat{s}$"); ax.set_ylabel(r"True $s$")
    ax.set_xlim(0, 2.6); ax.set_ylim(0, 2.6); ax.set_aspect("equal")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("simulations", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, width=0.5); cb.outline.set_linewidth(0.5)
    return r2, rho, rmse


def build():
    st.apply()
    d = load()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.35})
    panel_a(axes[0], d)
    stats = panel_b(fig, axes[1], d)
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.18, 1.08, letter, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left")
    st.save_panel(fig, OUTDIR, "supfigure_DL_model_ab")
    print("n = %d | s: R2 %.3f, rho %.3f, RMSE %.3f" % (len(d), *stats))


if __name__ == "__main__":
    build()
