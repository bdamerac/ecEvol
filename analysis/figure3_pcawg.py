"""Figure 3 panels: inferred ecDNA parameters across PCAWG.

(a) parameter distributions across the cohort, (b) parameters split by ecDNA
oncogene status, (c) emergence time by tumour type, (d) emergence time against
descendant fraction.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

from .common import load_ensemble, significance_stars, TUMOUR_GROUP, fig_dir
from . import figure_style as st

OUTDIR = fig_dir("Figure3", "ECDNA_FIG3_DIR")
PARAM_COLOR = {"phi_hat": st.PHI, "s_hat": st.S, "g_exp": st.INTRO, "log10mu": st.MU}
PARAM_LABEL = {"phi_hat": r"$\hat{\phi}$", "s_hat": r"$\hat{s}$",
               "g_exp": r"$\hat{g}$", "log10mu": r"$\log_{10}\hat{\mu}$"}


def load():
    pc = load_ensemble("pcawg")
    pc["log10mu"] = np.log10(pc.mu_hat.clip(lower=1e-3))
    return pc


def _violin(ax, values, x, color, rng, width=0.9, jitter=0.18):
    parts = ax.violinplot([values], positions=[x], widths=width, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(color); b.set_alpha(0.35); b.set_edgecolor(color); b.set_linewidth(0.6)
    jx = x + (rng.rand(len(values)) * 2 - 1) * jitter
    ax.scatter(jx, values, s=1.6, color=color, alpha=0.5, linewidths=0, rasterized=True)
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    ax.plot([x, x], [q1, q3], color="0.1", lw=1.4, zorder=5)
    ax.plot([x - 0.12, x + 0.12], [med, med], color="0.1", lw=1.4, zorder=5)
    return med


def panel_a_distributions(pc):
    rng = np.random.RandomState(0)
    params = ["phi_hat", "s_hat", "g_exp", "log10mu"]
    fig, axes = plt.subplots(2, 2, figsize=(3.2, 3.0))
    for ax, col in zip(axes.ravel(), params):
        v = pc[col].dropna().values
        med = _violin(ax, v, 0, PARAM_COLOR[col], rng)
        ax.annotate(f"{med:.2f}", xy=(0.14, med), xytext=(0.3, med), va="center", fontsize=6.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5))
        ax.set_xticks([]); ax.set_xlim(-0.7, 0.7); ax.set_ylabel(PARAM_LABEL[col])
    fig.tight_layout(pad=0.4)
    st.save_panel(fig, OUTDIR, "panel_fig3a_distributions")



def panel_c_oncogene(pc):
    rng = np.random.RandomState(1)
    onc = pc.oncogene_present.astype(bool)
    grey, red = "#9E9E9E", "#C0392B"
    params = ["phi_hat", "s_hat", "g_exp", "log10mu"]
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.3))
    for ax, col in zip(axes.ravel(), params):
        absent = pc[~onc][col].dropna().values
        present = pc[onc][col].dropna().values
        p = mannwhitneyu(present, absent).pvalue
        m0 = _violin(ax, absent, 1, grey, rng, width=0.8, jitter=0.14)
        m1 = _violin(ax, present, 2, red, rng, width=0.8, jitter=0.14)
        top = max(absent.max(), present.max()); rng_y = top - min(absent.min(), present.min())
        bar = top + 0.06 * rng_y
        ax.plot([1, 1, 2, 2], [bar - 0.02 * rng_y, bar, bar, bar - 0.02 * rng_y], color="0.3", lw=0.7)
        ax.text(1.5, bar + 0.01 * rng_y, f"p = {p:.2g}\n{significance_stars(p)}", ha="center", va="bottom", fontsize=6)
        for x0, m in [(1, m0), (2, m1)]:
            ax.annotate(f"{m:.2f}", xy=(x0 + 0.14, m), xytext=(x0 + 0.28, m), va="center", fontsize=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", lw=0.4))
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Onc −", "Onc +"], fontsize=6)
        ax.set_ylabel(PARAM_LABEL[col]); ax.set_ylim(top=bar + 0.22 * rng_y)
    fig.tight_layout(pad=0.4)
    st.save_panel(fig, OUTDIR, "panel_fig3b_oncogene")



def panel_c_timing_by_type(pc, min_n=15):
    """Emergence time by tumour type, for groups with at least `min_n` features."""
    pc = pc.copy(); pc["grp"] = pc.tumor_type.map(TUMOUR_GROUP).fillna("Other")
    n = pc.groupby("grp").size(); keep = n[n >= min_n].index
    d = pc[pc.grp.isin(keep)]
    order = d.groupby("grp").g_exp.median().sort_values(ascending=False).index.tolist()
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    for i, grp in enumerate(order):
        v = d[d.grp == grp].g_exp.dropna().values
        jy = i + (rng.rand(len(v)) * 2 - 1) * 0.16
        ax.scatter(v, jy, s=3.0, color=st.INTRO, alpha=0.35, linewidths=0, rasterized=True, zorder=1)
        q1, m, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [i, i], color=st.INTRO, lw=1.2, alpha=0.7, zorder=3)
        ax.scatter(m, i, marker="D", s=24, color=st.INTRO, edgecolor="0.15", lw=0.6, zorder=4)
    # significance: omnibus across groups, and each group vs CNS (BH-corrected)
    from scipy.stats import mannwhitneyu, kruskal
    vals = {g: d[d.grp == g].g_exp.dropna().values for g in order}
    kw_p = kruskal(*vals.values()).pvalue
    print(f"emergence time across tumour groups: Kruskal-Wallis p = {kw_p:.1e}")
    ref = "CNS"
    others = [g for g in order if g != ref]
    raw = np.array([mannwhitneyu(vals[ref], vals[g]).pvalue for g in others])
    rank = raw.argsort().argsort() + 1
    bh = np.minimum(1, raw * len(raw) / rank)
    bh = np.minimum.accumulate(bh[raw.argsort()[::-1]])[::-1][raw.argsort().argsort()]
    stars = {g: ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s.")
             for g, p in zip(others, bh)}
    stars[ref] = "ref."
    xr = ax.get_xlim()[1]
    for i, grp in enumerate(order):
        ax.text(xr + 0.15, i, stars[grp], fontsize=5.8, va="center", ha="left",
                color="0.25" if stars[grp] not in ("n.s.", "ref.") else "0.55")
    ax.set_xlim(right=xr + 1.1)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{g} (n={ (d.grp==g).sum() })" for g in order], fontsize=6.0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(r"$\hat{g}$  (emergence time)")
    fig.tight_layout(pad=0.4)
    st.save_panel(fig, OUTDIR, "panel_fig3c_timing_by_type")


def panel_d_g_phi(pc):
    """Earlier-emerging ecDNA reach higher descendant fraction."""
    d = pc.dropna(subset=["g_exp", "phi_hat"])
    x, y = d.g_exp.values, d.phi_hat.values
    rho, pval = spearmanr(x, y)
    if pval >= 1e-4:
        ptxt = f"$P$ = {pval:.2g}"
    else:
        mant, exp = f"{pval:.0e}".split("e")
        ptxt = f"$P = {mant}\\times10^{{{int(exp)}}}$"
    rng = np.random.RandomState(1)
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    ax.scatter(x + (rng.rand(len(x)) * 2 - 1) * 0.12, y, s=4, color=st.INTRO, alpha=0.35,
               linewidths=0, rasterized=True)
    # least-squares regression line
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, slope * xx + intercept, "-", color="0.15", lw=1.4, zorder=5)
    ax.set_xlabel(r"$\hat{g}$  (emergence time)"); ax.set_ylabel(r"$\hat{\phi}$  (descendant fraction)")
    ax.set_xlim(0.5, 9.5); ax.set_ylim(-0.02, 1.05)
    ax.text(0.96, 0.06, f"$\\rho$ = {rho:.2f}\n{ptxt}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5))
    ax.set_title("Earlier emergence → higher descendant fraction", fontsize=7.2)
    fig.tight_layout(pad=0.4)
    st.save_panel(fig, OUTDIR, "panel_fig3d_g_phi")



if __name__ == "__main__":
    st.apply()
    pc = load()
    panel_a_distributions(pc)
    panel_c_oncogene(pc)
    panel_c_timing_by_type(pc)
    panel_d_g_phi(pc)
