"""Supplementary figure: robustness of the inter-cohort differences to sample quality.

A feature is high quality (HQ) if its sample has purity >= 0.6 (the lower bound of the
purity range used for training augmentation) and sequencing coverage >= 50x. PCAWG
coverage is the per-sample `depth` field. HMF coverage is tumorMeanCoverage from the
Hartwig supplementary sample table (Priestley et al. 2019), joined on patient ID.
HMF features with no recorded purity or coverage cannot be classified and are omitted
from the HQ subset. Tests compare the cohorts over all features and over the HQ
subsets.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from .common import load_ensemble, fig_dir, EXTERNAL_DIR
from . import figure_style as st

OUTDIR = fig_dir("Supplementary", "ECDNA_SUPP_DIR")
PRIESTLEY = os.environ.get(
    "ECDNA_HMF_SAMPLES",
    os.path.join(EXTERNAL_DIR, "41586_2019_1689_MOESM7_ESM.xlsx"))
MIN_PURITY, MIN_COVERAGE = 0.6, 50
PARAMS = [("g_exp", r"$\hat{g}$", st.INTRO), ("phi_hat", r"$\hat{\phi}$", st.PHI),
          ("s_hat", r"$\hat{s}$", st.S), ("log10mu", r"$\log_{10}\hat{\mu}$", st.MU)]


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def classify(purity, coverage):
    hq = (purity >= MIN_PURITY) & (coverage >= MIN_COVERAGE)
    fail = (purity < MIN_PURITY) | (coverage < MIN_COVERAGE)
    return np.where(hq, "HQ", np.where(fail, "non-HQ", "unknown"))


def load():
    pc = load_ensemble("pcawg").copy(); hm = load_ensemble("hmf").copy()
    pc["quality"] = classify(pc.purity, pc.depth)
    cov = (pd.read_excel(PRIESTLEY)[["patientId", "tumorMeanCoverage"]]
             .drop_duplicates("patientId").set_index("patientId").tumorMeanCoverage)
    hm["quality"] = classify(hm.sup7_purity_median, hm.patient_barcode.map(cov))
    for d in (pc, hm):
        d["log10mu"] = np.log10(d.mu_hat.clip(lower=1))
    return pc, hm


def figure(pc, hm):
    groups = [("PCAWG\nall", pc, 0, 0.30),
              ("PCAWG\nHQ", pc[pc.quality == "HQ"], 1, 0.85),
              ("HMF\nall", hm, 3, 0.30),
              ("HMF\nHQ", hm[hm.quality == "HQ"], 4, 0.85)]
    tests = [(1, 3), (0, 2)]   # HQ vs HQ (inner bracket), all vs all (outer bracket)
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(7.2, 3.0))
    stats = []
    for ax, (col, lab, color) in zip(axes, PARAMS):
        vals = [d[col].dropna().values for _, d, _, _ in groups]
        for (_, _, x, alpha), v in zip(groups, vals):
            body = ax.violinplot([v], positions=[x], widths=0.8, showextrema=False)["bodies"][0]
            body.set_facecolor(color); body.set_alpha(alpha); body.set_edgecolor("none")
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([x, x], [q1, q3], color="0.15", lw=1.3, zorder=3)
            ax.scatter(x, med, s=14, color="white", edgecolor="0.15", lw=0.8, zorder=4)
        lo = min(v.min() for v in vals); hi = max(v.max() for v in vals); span = hi - lo
        for k, (i, j) in enumerate(tests):
            p = mannwhitneyu(vals[i], vals[j]).pvalue
            stats.append((lab, groups[i][0].replace("\n", " "), groups[j][0].replace("\n", " "),
                          np.median(vals[i]), np.median(vals[j]), p))
            y = hi + span * (0.07 + 0.13 * k); xi, xj = groups[i][2], groups[j][2]
            ax.plot([xi, xi, xj, xj], [y - 0.02 * span, y, y, y - 0.02 * span], color="0.3", lw=0.6)
            ax.text((xi + xj) / 2, y, stars(p), ha="center", va="bottom", fontsize=6,
                    color="#B2182B" if p < 0.05 else "0.45")
        ax.axvline(2, color="0.6", ls="--", lw=0.6)
        ax.set_xticks([g[2] for g in groups])
        ax.set_xticklabels([f"{name.split(chr(10))[1]}\nn={len(v)}" for (name, _, _, _), v in zip(groups, vals)],
                           fontsize=5.6)
        for xc, cohort in [(0.5, "PCAWG"), (3.5, "HMF")]:
            ax.text(xc, -0.20, cohort, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=6.8, fontweight="bold")
        ax.set_ylabel(lab); ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(lo - 0.04 * span, hi + 0.40 * span)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    st.save_panel(fig, OUTDIR, "hq_intercohort_pvalues")
    return stats


if __name__ == "__main__":
    st.apply()
    pc, hm = load()
    for name, d in [("PCAWG", pc), ("HMF", hm)]:
        print("%-5s quality: %s" % (name, pd.Series(d.quality).value_counts().to_dict()))
    for lab, a, b, ma, mb, p in figure(pc, hm):
        print("  %-22s %-13s vs %-13s  %.2f vs %.2f  P=%.2g" % (lab, a, b, ma, mb, p))
