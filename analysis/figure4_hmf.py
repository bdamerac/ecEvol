"""Figure 4 panels: ecDNA evolution in the HMF advanced/treated cohort.

(a) per-cancer-type PCAWG->HMF shift, (b) parameters by oncogene status,
(c) within-patient longitudinal change. Emitted as individual house-style panels.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, wilcoxon

from .common import (load_ensemble, MATCHED, HMFNAME, benjamini_hochberg,
                     significance_stars, cliffs_delta, fig_dir)
from .longitudinal import match_ecdna_across_timepoints
from . import figure_style as st

OUTDIR = fig_dir("Figure4", "ECDNA_FIG4_DIR")
PARAM_COLOR = {"phi_hat": st.PHI, "s_hat": st.S, "g_exp": st.INTRO, "log10mu": st.MU}


def load():
    pc = load_ensemble("pcawg"); hm = load_ensemble("hmf")
    for d in (pc, hm):
        d["log10mu"] = np.log10(d.mu_hat.clip(lower=1e-3))
    pc["oncogene"] = pc.oncogene_present.astype(bool)
    hm["oncogene"] = hm.agg_any_ecdna_oncogene.astype(bool)
    return pc, hm


def _fmt_p(p):
    if p >= 0.01:
        return f"$P = {p:.2f}$"
    m, e = f"{p:.1e}".split("e")
    return f"$P = {m}\\times10^{{{int(e)}}}$"


def _cohort_violin(ax, v, x, color, alpha, rng):
    body = ax.violinplot([v], positions=[x], widths=0.8, showextrema=False)["bodies"][0]
    body.set_facecolor(color); body.set_alpha(alpha); body.set_edgecolor("none")
    ax.scatter(x + rng.uniform(-0.16, 0.16, len(v)), v, s=1.2, color="0.25", alpha=0.25,
               linewidths=0, zorder=2, rasterized=True)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([x, x], [q1, q3], color="0.1", lw=1.3, zorder=3)
    ax.scatter(x, med, s=16, color="white", edgecolor="0.1", lw=0.9, zorder=4)


def panel_a_overall(pc, hm):
    """All scored features, PCAWG (light) vs HMF (dark), one violin pair per parameter."""
    rng = np.random.RandomState(0)
    params = [("g_exp", r"$\hat{g}$"), ("phi_hat", r"$\hat{\phi}$"),
              ("s_hat", r"$\hat{s}$"), ("log10mu", r"$\log_{10}\hat{\mu}$")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
    for ax, (col, lab) in zip(axes, params):
        a, b = pc[col].dropna().values, hm[col].dropna().values
        _cohort_violin(ax, a, 0, PARAM_COLOR[col], 0.35, rng)
        _cohort_violin(ax, b, 1, PARAM_COLOR[col], 0.85, rng)
        p = mannwhitneyu(a, b).pvalue
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max()); span = hi - lo
        y = hi + 0.08 * span
        ax.plot([0, 0, 1, 1], [y - 0.02 * span, y, y, y - 0.02 * span], color="0.3", lw=0.6)
        ax.text(0.5, y + 0.02 * span, _fmt_p(p), ha="center", va="bottom", fontsize=6.2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"PCAWG\nn={len(a)}", f"HMF\nn={len(b)}"], fontsize=6.5)
        ax.set_ylabel(lab); ax.set_xlim(-0.65, 1.65); ax.set_ylim(lo - 0.04 * span, hi + 0.30 * span)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    st.save_panel(fig, OUTDIR, "panel_fig4a_overall")


def panel_a_per_cancer(pc, hm):
    params = [("s_hat", r"$\hat{s}$"), ("g_exp", r"$\hat{g}$"),
              ("phi_hat", r"$\hat{\phi}$"), ("mu_hat", r"$\hat{\mu}$")]
    per = {}
    for col, _ in params:
        rows = {}
        for grp, codes in MATCHED.items():
            a = pc[pc.tumor_type.isin(codes)][col].dropna().values
            b = hm[hm.cancer_type == HMFNAME.get(grp, grp)][col].dropna().values
            rows[grp] = dict(pm=np.median(a), hm=np.median(b), p=mannwhitneyu(b, a).pvalue,
                             na=len(a), nb=len(b))
        for grp, q in zip(MATCHED, benjamini_hochberg([rows[g]["p"] for g in MATCHED])):
            rows[grp]["q"] = q
        per[col] = rows
    order = sorted(MATCHED, key=lambda g: per["g_exp"][g]["hm"])
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.6), sharey=True)
    for ax, (col, lab) in zip(axes, params):
        rows = per[col]
        for i, grp in enumerate(order):
            r = rows[grp]; sig = r["q"] < 0.05
            ax.plot([r["pm"], r["hm"]], [i, i], color="0.45" if sig else "0.8",
                    lw=1.6 if sig else 1.0, solid_capstyle="round", zorder=1)
            ax.scatter(r["pm"], i, s=16, facecolor="white", edgecolor=st.COHORT_PCAWG, lw=1.0, zorder=3)
            ax.scatter(r["hm"], i, s=16, color=st.COHORT_HMF, zorder=3)
        ax.set_title(lab, fontsize=8); ax.margins(x=0.15)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"{g} ({per['g_exp'][g]['na']}/{per['g_exp'][g]['nb']})" for g in order], fontsize=5.6)
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=st.COHORT_PCAWG, mew=1.0, ms=5, label="PCAWG"),
               Line2D([], [], marker="o", ls="", color=st.COHORT_HMF, ms=5, label="HMF")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    st.save_panel(fig, OUTDIR, "panel_fig4a_per_cancer")


def _onc_group(ax, absent, present, x_abs, x_pres, color, rng):
    """One cohort: absent (light) vs present (solid) violins + jittered points + median/IQR bar."""
    for x, v, alpha in [(x_abs, absent, 0.22), (x_pres, present, 0.5)]:
        parts = ax.violinplot([v], positions=[x], widths=0.72, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(color); b.set_alpha(alpha); b.set_edgecolor(color); b.set_linewidth(0.5)
        jx = x + (rng.rand(len(v)) * 2 - 1) * 0.14
        ax.scatter(jx, v, s=1.3, color=color, alpha=0.45, linewidths=0, rasterized=True, zorder=2)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([x, x], [q1, q3], color="0.1", lw=1.1, zorder=5)
        ax.plot([x - 0.11, x + 0.11], [med, med], color="0.1", lw=1.1, zorder=5)


def panel_b_oncogene(pc, hm):
    params = [("phi_hat", r"$\hat{\phi}$"), ("g_exp", r"$\hat{g}$"),
              ("s_hat", r"$\hat{s}$"), ("log10mu", r"$\log_{10}\hat{\mu}$")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6))
    blocks = [(pc, st.COHORT_PCAWG, "PCAWG", 1.0, 1.85), (hm, st.COHORT_HMF, "HMF", 3.15, 4.0)]
    for ax, (col, lab) in zip(axes, params):
        vals = []
        for df, color, _, xa, xp in blocks:
            a = df[~df.oncogene][col].dropna().values; b = df[df.oncogene][col].dropna().values
            _onc_group(ax, a, b, xa, xp, color, np.random.RandomState(0)); vals += [a, b]
            p = mannwhitneyu(b, a).pvalue; d = cliffs_delta(a, b)
            top = max(a.max(), b.max()); rng = top - min(a.min(), b.min()); bar = top + 0.05 * rng
            ax.plot([xa, xa, xp, xp], [bar - 0.02 * rng, bar, bar, bar - 0.02 * rng], color="0.35", lw=0.6)
            ax.text((xa + xp) / 2, bar + 0.06 * rng, f"$\\delta$={d:+.2f}", ha="center", va="bottom",
                    fontsize=5.8, color=color)
            ax.text((xa + xp) / 2, bar + 0.005 * rng, significance_stars(p), ha="center", va="bottom", fontsize=6)
        ax.axvline(2.5, color="0.85", lw=0.6, ls="--")
        ax.set_xticks([1.0, 1.85, 3.15, 4.0]); ax.set_xticklabels(["Abs.", "Pres.", "Abs.", "Pres."], fontsize=5.6)
        ax.set_ylabel(lab); ax.set_xlim(0.4, 4.7)
        lo = min(v.min() for v in vals); span = max(v.max() for v in vals) - lo
        ax.set_ylim(lo - 0.16 * span, None)
        ax.text(1.42, lo - 0.13 * span, "PCAWG", ha="center", fontsize=5.6, color=st.COHORT_PCAWG, va="top")
        ax.text(3.57, lo - 0.13 * span, "HMF", ha="center", fontsize=5.6, color=st.COHORT_HMF, va="top")
    fig.suptitle("ecDNA oncogene status  (Absent vs Present)", fontsize=8, y=1.0)
    fig.tight_layout(pad=0.5, rect=[0, 0, 1, 0.97])
    st.save_panel(fig, OUTDIR, "panel_fig4b_oncogene")


def panel_c_longitudinal():
    """One paired observation per patient: the highest-overlap matched ecDNA pair."""
    pairs = match_ecdna_across_timepoints()
    up, down, accent, grey = "#2CA089", "#E0896F", "#1F7A68", "#6B6B6B"
    panels = [("phi_hat", r"$\hat{\phi}$", accent), ("log10mu", r"$\log_{10}\hat{\mu}$", accent),
              ("s_hat", r"$\hat{s}$", grey), ("g_exp", r"$\hat{g}$", grey)]
    fig, axes = plt.subplots(1, 4, figsize=(6.4, 2.3))
    for ax, (col, lab, accent_c) in zip(axes, panels):
        v1 = pairs[f"{col}_t1"].values; v2 = pairs[f"{col}_t2"].values
        p = wilcoxon(v2, v1).pvalue
        for a, b in zip(v1, v2):
            ax.plot([0, 1], [a, b], color=(up if b >= a else down), lw=0.8, alpha=0.5, zorder=1)
        m1, m2 = np.median(v1), np.median(v2)
        ax.plot([0, 1], [m1, m2], color=accent_c, lw=2.2, marker="D", ms=5, zorder=4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["T1", "T2"]); ax.set_xlim(-0.4, 1.4)
        ax.set_title(lab, fontsize=8); ax.margins(y=0.13)
        ax.text(0.02, 0.98, f"p={p:.3f} {significance_stars(p)}", transform=ax.transAxes, va="top", fontsize=5.8)
    fig.tight_layout(pad=0.4)
    st.save_panel(fig, OUTDIR, "panel_fig4c_longitudinal")


if __name__ == "__main__":
    st.apply()
    pc, hm = load()
    panel_a_overall(pc, hm)
    panel_a_per_cancer(pc, hm)
    panel_b_oncogene(pc, hm)
    panel_c_longitudinal()
