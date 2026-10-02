"""Figure 2 panels: parameter recovery on held-out simulations (depth 100).

Panels b-f, from the final-model test predictions (panel a is a schematic, panel g
is in popsize.py):
  cnn_attn_pool_d{depth}_s{seed}.parquet   (s, mu, g predictions + truth)
  phi_sim_depth{depth}.parquet             (phi predictions + truth)
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

DEPTH, SEED = 100, 42
OUTDIR = fig_dir("Figure2", "ECDNA_FIG2_DIR")


def _r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def _stats_box(ax, text):
    ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", fontsize=6.3,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))


def load():
    main = pd.read_parquet(os.path.join(SIM_DIR, f"cnn_attn_pool_d{DEPTH}_s{SEED}.parquet"))
    main = main[~main.perturbed.astype(bool)]
    phi = pd.read_parquet(os.path.join(SIM_DIR, f"phi_sim_depth{DEPTH}.parquet"))
    return main, phi


def panel_b_phi(phi):
    x, y = phi.phi_true.values, phi.phi_hat.values
    fig, ax = plt.subplots(figsize=(2.05, 1.95))
    ax.scatter(x, y, s=2, color=st.PHI, alpha=0.28, linewidths=0, rasterized=True)
    ax.plot([0.2, 1], [0.2, 1], color="black", lw=1.0)
    ax.set_xlim(0.2, 1); ax.set_ylim(0.2, 1); ax.set_aspect("equal")
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r"True $\phi$"); ax.set_ylabel(r"Predicted $\hat{\phi}$")
    _stats_box(ax, f"$R^2$ = {_r2(x, y):.3f}\n$\\rho$ = {spearmanr(x, y).correlation:.3f}\n"
                   f"RMSE = {np.sqrt(((y - x) ** 2).mean()):.3f}")
    st.save_panel(fig, OUTDIR, "panel_fig2b_phi")


def panel_c_mu(main):
    x, y = main.mu_true.values, main.mu_hat.values
    fig, ax = plt.subplots(figsize=(2.25, 1.95))
    hb = ax.hexbin(x, y, gridsize=40, cmap=st.MU_CMAP, mincnt=1, linewidths=0)
    ax.plot([0, 500], [0, 500], color="black", lw=1.0)
    ax.set_xlim(0, 500); ax.set_ylim(0, 500); ax.set_aspect("equal")
    ax.set_xlabel(r"True $\mu$"); ax.set_ylabel(r"Predicted $\hat{\mu}$")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Count", fontsize=6.5); cb.ax.tick_params(labelsize=6, width=0.5); cb.outline.set_linewidth(0.5)
    _stats_box(ax, f"$R^2$ = {_r2(x, y):.3f}\n$\\rho$ = {spearmanr(x, y).correlation:.3f}\n"
                   f"RMSE = {np.sqrt(((y - x) ** 2).mean()):.1f}")
    st.save_panel(fig, OUTDIR, "panel_fig2c_mu")


def panel_d_intro(main):
    gp = np.round(np.clip(main.g_exp, 1, 9)).astype(int)
    gt = np.clip(main.g_true, 1, 9).astype(int)
    M = np.zeros((9, 9))
    for t, p in zip(gt, gp):
        M[t - 1, p - 1] += 1
    Mn = M / M.sum(1, keepdims=True)
    exact = (np.round(main.g_exp) == main.g_true).mean()
    pm1 = (np.abs(np.round(main.g_exp) - main.g_true) <= 1).mean()
    fig, ax = plt.subplots(figsize=(2.3, 2.0))
    im = ax.imshow(Mn, cmap=st.INTRO_CMAP, vmin=0, vmax=1, origin="lower", extent=[0.5, 9.5, 0.5, 9.5])
    ax.set_xlabel("Predicted emergence time"); ax.set_ylabel("True emergence time")
    ax.set_xticks(range(1, 10, 2)); ax.set_yticks(range(1, 10, 2))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Fraction", fontsize=6.5); cb.ax.tick_params(labelsize=6, width=0.5); cb.outline.set_linewidth(0.5)
    ax.text(0.97, 0.03, f"Exact {exact:.2f}\nWithin 1  {pm1:.2f}", transform=ax.transAxes,
            va="bottom", ha="right", fontsize=6.2, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))
    st.save_panel(fig, OUTDIR, "panel_fig2d_intro_confusion")


def panel_e_stiers(main):
    sh, s_true = main.s_hat.values, main.s_true.values
    edges = [(-1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 99)]
    labs = ["Very low\n(<0.5)", "Low\n(0.5–1.0)", "Interm.\n(1.0–1.5)", "High\n(1.5–2.0)", "Very high\n(>2.0)"]
    vals = [s_true[(sh >= lo) & (sh < hi)] for lo, hi in edges]
    ns = [len(v) for v in vals]
    greens = plt.cm.Greens(np.linspace(0.28, 0.9, 5))
    fig, ax = plt.subplots(figsize=(3.0, 2.15))
    parts = ax.violinplot(vals, positions=range(1, 6), widths=0.85, showextrema=False)
    for i, (b, c) in enumerate(zip(parts["bodies"], greens)):
        b.set_facecolor("#CFCFCF" if i == 2 else c)
        b.set_alpha(0.9); b.set_edgecolor("0.35"); b.set_linewidth(0.4)
    for i, v in enumerate(vals):
        ax.hlines(np.median(v), i + 0.73, i + 1.27, color="white", lw=1.3, zorder=6)
    for yv in (1.0, 1.5):
        ax.axhline(yv, ls="--", color="0.45", lw=0.6, zorder=1)
        ax.text(0.62, yv + 0.03, f"$s$ = {yv}", fontsize=5.3, color="0.4", va="bottom")
    gbox = dict(boxstyle="round,pad=0.28", fc="#E4F1E9", ec=st.S, lw=0.5)
    greybox = dict(boxstyle="round,pad=0.28", fc="#EDEDED", ec="0.6", lw=0.5)
    p_le = lambda v, t=1.0: (v <= t).mean() * 100
    p_gt = lambda v, t: (v > t).mean() * 100
    ann = [(1, 2.28, f"{p_le(vals[0]):.0f}%\n$s$≤1", gbox),
           (2, 2.28, f"{p_le(vals[1]):.0f}%\n$s$≤1", gbox),
           (3, 2.28, "indeter-\nminate", greybox),
           (4, 0.42, f"{p_gt(vals[3], 1):.0f}%\n$s$>1", gbox),
           (5, 0.42, f"{p_gt(vals[4], 1):.0f}% $s$>1\n{p_gt(vals[4], 1.5):.0f}% $s$>1.5", gbox)]
    for x0, y0, t, bb in ann:
        ax.text(x0, y0, t, ha="center", va="center", fontsize=5.2, color="0.15", bbox=bb, zorder=7)
    ax.set_xticks(range(1, 6)); ax.set_xticklabels([f"{l}\nn={n}" for l, n in zip(labs, ns)], fontsize=5.4)
    ax.set_ylabel("True $s$"); ax.set_ylim(-0.05, 2.7); ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
    st.save_panel(fig, OUTDIR, "panel_fig2e_stiers")


def panel_f_error_vs_phi(main, phi):
    ph = phi.phi_hat.values
    series = [("Abs. error, $s$", np.abs(main.s_hat.values - main.s_true.values), st.S),
              ("Abs. error, $\\mu$", np.abs(main.mu_hat.values - main.mu_true.values), st.MU),
              ("Abs. error, $g$", np.abs(np.round(main.g_exp.values) - main.g_true.values), st.INTRO)]
    bins = np.linspace(0.2, 1.0, 9); cent = (bins[:-1] + bins[1:]) / 2; idx = np.digitize(ph, bins) - 1
    fig, axes = plt.subplots(3, 1, figsize=(2.15, 3.05), sharex=True)
    for ax, (lab, e, c) in zip(axes, series):
        m = np.array([np.nanmean(e[idx == b]) for b in range(len(cent))])
        sd = np.array([np.nanstd(e[idx == b]) for b in range(len(cent))])
        ax.plot(cent, m, "-", color=c, lw=1.3)
        ax.fill_between(cent, m - sd, m + sd, color=c, alpha=0.22, linewidth=0)
        ax.set_ylabel(lab, fontsize=6.8); ax.set_ylim(bottom=0); ax.locator_params(axis="y", nbins=4)
    axes[-1].set_xlabel(r"Estimated $\hat{\phi}$"); axes[-1].set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    fig.align_ylabels(axes)
    st.save_panel(fig, OUTDIR, "panel_fig2f_errorvsphi")


if __name__ == "__main__":
    st.apply()
    main, phi = load()
    panel_b_phi(phi)
    panel_c_mu(main)
    panel_d_intro(main)
    panel_e_stiers(main)
    panel_f_error_vs_phi(main, phi)
