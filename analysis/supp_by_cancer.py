"""Supplementary figure: per-cancer-type parameter distributions in PCAWG and HMF.

For each of the ten matched cancer groups (common.MATCHED), paired violins show the four
inferred parameters in the primary (PCAWG, light) and advanced (HMF, dark) cohorts.
Groups are ordered by PCAWG median emergence time. Within each parameter the two cohorts
are compared per group by Mann-Whitney U, Benjamini-Hochberg adjusted across the ten
groups. Values are ten-seed ensemble means for scored ecDNA features.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from .common import load_ensemble, MATCHED, HMFNAME, fig_dir
from . import figure_style as st

OUTDIR = fig_dir("Supplementary", "ECDNA_SUPP_DIR")
PARAMS = [("phi_hat", r"$\hat{\phi}$", st.PHI), ("s_hat", r"$\hat{s}$", st.S),
          ("log10mu", r"$\log_{10}\hat{\mu}$", st.MU), ("g_exp", r"$\hat{g}$", st.INTRO)]
OFFSET, STEP = 0.42, 2.0


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def load():
    pc = load_ensemble("pcawg"); hm = load_ensemble("hmf")
    for d in (pc, hm):
        d["log10mu"] = np.log10(d.mu_hat.clip(lower=1))
    groups = [(grp, pc[pc.tumor_type.isin(codes)], hm[hm.cancer_type == HMFNAME.get(grp, grp)])
              for grp, codes in MATCHED.items()]
    return sorted(groups, key=lambda t: t[1].g_exp.median())


def _violin(ax, v, x, color, alpha, rng):
    v = v[np.isfinite(v)]
    if len(v) > 2 and np.ptp(v) > 0:
        body = ax.violinplot([v], positions=[x], widths=0.78, showextrema=False)["bodies"][0]
        body.set_facecolor(color); body.set_alpha(alpha); body.set_edgecolor("none")
    ax.scatter(x + rng.uniform(-0.14, 0.14, len(v)), v, s=1.5, color="0.2", alpha=0.35,
               linewidths=0, zorder=2, rasterized=True)
    ax.plot([x - 0.27, x + 0.27], [np.median(v)] * 2, color="k", lw=1.1, zorder=4)


def figure(groups):
    rng = np.random.RandomState(0)
    centres = np.arange(len(groups)) * STEP
    fig, axes = plt.subplots(len(PARAMS), 1, figsize=(7.2, 7.8), sharex=True)
    table = {g: {"nP": len(a), "nH": len(b)} for g, a, b in groups}
    for ax, (col, lab, color) in zip(axes, PARAMS):
        raw = [mannwhitneyu(a[col].dropna(), b[col].dropna()).pvalue for _, a, b in groups]
        adj = multipletests(raw, method="fdr_bh")[1]
        lo = min(min(a[col].min(), b[col].min()) for _, a, b in groups)
        hi = max(max(a[col].max(), b[col].max()) for _, a, b in groups)
        span = hi - lo
        for i, ((grp, a, b), p, c) in enumerate(zip(groups, adj, centres)):
            if i % 2 == 0:
                ax.axvspan(c - STEP / 2, c + STEP / 2, color="0.96", zorder=0, lw=0)
            _violin(ax, a[col].values, c - OFFSET, color, 0.30, rng)
            _violin(ax, b[col].values, c + OFFSET, color, 0.85, rng)
            ax.text(c, hi + 0.06 * span, stars(p), ha="center", va="bottom", fontsize=6,
                    color="#B2182B" if p < 0.05 else "0.45")
            table[grp][col] = (a[col].median(), b[col].median(), p)
        ax.set_ylabel(lab)
        ax.set_ylim(lo - 0.04 * span, hi + 0.24 * span)
        ax.set_xlim(centres[0] - STEP / 2, centres[-1] + STEP / 2)
        ax.legend(handles=[Patch(color=color, alpha=0.30, label="PCAWG"),
                           Patch(color=color, alpha=0.85, label="HMF")],
                  frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[-1].set_xticks(centres)
    axes[-1].set_xticklabels([f"{g}\nP:{len(a)}  H:{len(b)}" for g, a, b in groups],
                             fontsize=6.2, rotation=35, ha="right")
    fig.tight_layout(pad=0.4, h_pad=0.6)
    st.save_panel(fig, OUTDIR, "by_cancer_paired_hmf_pcawg")
    return table


if __name__ == "__main__":
    st.apply()
    groups = load()
    table = figure(groups)
    print("\n%-10s %4s %4s | %-19s | %-19s | %-19s | %-19s" %
          ("group", "nP", "nH", "phi P/H (BH P)", "s P/H (BH P)", "log10mu P/H (BH P)", "ghat P/H (BH P)"))
    for grp, _, _ in groups:
        r = table[grp]
        cells = ["%.2f/%.2f (%.2g)" % r[c] for c, _, _ in PARAMS]
        print("%-10s %4d %4d | %s" % (grp, r["nP"], r["nH"], " | ".join("%-19s" % x for x in cells)))
