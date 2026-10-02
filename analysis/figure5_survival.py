"""Figure 5 panels: ecDNA emergence timing and overall survival (PCAWG).

(a) Kaplan-Meier, earliest-emerging tertile against the rest, in the single-scored-
ecDNA cohort. (b) Cox forest for the composite indices 1/g, phi/g and s*phi/g,
adjusted for age, sex and copy number and stratified by cancer type. Also prints the
adjusted hazard ratios for the earliest-emerging tertile and the nested
likelihood-ratio tests quoted in the text.
"""
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter

warnings.filterwarnings("ignore")
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

from .common import load_ensemble, fig_dir
from .survival import build_patient_table, cox_forest_terms, nested_likelihood_ratio, COVARIATES
from . import figure_style as st

OUTDIR = fig_dir("Figure5", "ECDNA_FIG5_DIR")
EARLY = st.COHORT_HMF   # red: earliest-emerging, worse survival
REST = "#6B6B6B"        # grey: the rest


def _cohort():
    t = build_patient_table(load_ensemble("pcawg"), single_ecdna=True)
    return t.assign(early=(t.g_exp <= t.g_exp.quantile(1 / 3)))


def panel_a_km(t):
    tt = t.assign(years=t.time / 365.25)
    hi, lo = tt[tt.early], tt[~tt.early]
    p = logrank_test(hi.years, lo.years, hi.event, lo.event).p_value
    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    for grp, c, lab in [(hi, EARLY, "Earliest-emerging"), (lo, REST, "Rest")]:
        KaplanMeierFitter().fit(grp.years, grp.event,
                                label=f"{lab} (n={len(grp)}, {grp.event.mean()*100:.0f}% died)")\
            .plot_survival_function(ax=ax, color=c, ci_show=True, ci_alpha=0.12, lw=1.8)
    ax.set_xlim(0, 5); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Years from diagnosis"); ax.set_ylabel("Overall survival")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
    ax.text(0.03, 0.06, f"log-rank\np = {p:.1e}  {star}", transform=ax.transAxes, va="bottom",
            fontsize=6.3, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))
    ax.legend(frameon=False, fontsize=6, loc="upper right", handlelength=1.2)
    fig.tight_layout(pad=0.4)
    st.save_panel(fig, OUTDIR, "panel_fig5a_km")


def adjusted_hazard_ratios(t):
    """Cox HR for the earliest-emerging tertile, adjusted and stratified; with and without CNS."""
    for label, d in [("all tumour types", t), ("excluding CNS", t[t.strata != "CNS"])]:
        d = d.assign(early=d.early.astype(float))
        m = CoxPHFitter().fit(d[["early"] + COVARIATES + ["time", "event", "strata"]],
                              "time", "event", strata=["strata"])
        r = m.summary.loc["early"]
        print(f"  {label:18s} n={len(d)}  HR={np.exp(r['coef']):.2f} "
              f"({np.exp(r['coef'] - 1.96 * r['se(coef)']):.2f}-{np.exp(r['coef'] + 1.96 * r['se(coef)']):.2f})"
              f"  p={r['p']:.3f}")


PRETTY = {"1/g": r"$1/\hat{g}$", "phi/g": r"$\hat{\phi}/\hat{g}$",
          "s*phi/g": r"$\hat{s}\,\hat{\phi}/\hat{g}$"}
COMPOSITE_ONLY = [("Composite indices", ["1/g", "phi/g", "s*phi/g"])]


def _forest(t, layout, name):
    f = cox_forest_terms(t).set_index("name")
    entries = []          # (kind, payload); kind in {header, row}
    for block, names in layout:
        entries.append(("header", block))
        entries += [("row", n) for n in names]
    n_slots = len(entries) + (len(layout) - 1) * 0.5   # half-gap between blocks
    fig, ax = plt.subplots(figsize=(4.9, max(1.7, 0.32 * n_slots + 0.7)))
    tx = ax.get_yaxis_transform()
    y = n_slots
    ymap = {}
    for i, (kind, payload) in enumerate(entries):
        if kind == "header" and i > 0:
            y -= 0.5
        if kind == "header":
            ax.text(0.0, y, payload, transform=tx, va="center", ha="left",
                    fontsize=6.4, style="italic", color="0.35")
        else:
            r = f.loc[payload]; sig = r.p < 0.05
            ax.plot([r.lo, r.hi], [y, y], color=EARLY, lw=1.5, zorder=2, solid_capstyle="round")
            ax.scatter(r.hr, y, s=24, color=EARLY if sig else "white", edgecolor=EARLY, lw=1.3, zorder=3)
            pstr = "p<0.001" if r.p < 0.001 else f"p={r.p:.3f}"
            ax.text(1.04, y, f"{r.hr:.2f} ({r.lo:.2f}–{r.hi:.2f})   {pstr}",
                    transform=tx, va="center", ha="left", fontsize=5.8, clip_on=False)
            ymap[payload] = y
        y -= 1
    ax.axvline(1, color="0.5", ls="--", lw=0.8, zorder=0)
    ax.set_yticks(list(ymap.values())); ax.set_yticklabels([PRETTY[n] for n in ymap], fontsize=7)
    ax.set_ylim(-0.6, n_slots + 0.6)
    ax.set_xscale("log"); ax.set_xlim(0.6, 2.5)
    ax.xaxis.set_major_locator(FixedLocator([0.7, 1.0, 1.5, 2.0]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.7", "1.0", "1.5", "2.0"]))
    ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Hazard ratio per SD (95% CI)")
    ax.text(1.02, n_slots + 0.35, "worse survival →", fontsize=5.6, color="0.45", ha="left")
    fig.tight_layout(rect=[0, 0, 0.6, 1])
    st.save_panel(fig, OUTDIR, name)


def panel_b_composite_forest(t):
    _forest(t, COMPOSITE_ONLY, "panel_fig5b_composite_forest")


if __name__ == "__main__":
    st.apply()
    t = _cohort()
    print(f"single-ecDNA cohort n={len(t)}, events={int(t.event.sum())}, "
          f"earliest n={int(t.early.sum())}")
    print("adjusted hazard ratio, earliest tertile vs rest:")
    adjusted_hazard_ratios(t)
    print("nested likelihood-ratio tests:")
    nested_likelihood_ratio(t)
    panel_a_km(t)
    panel_b_composite_forest(t)
