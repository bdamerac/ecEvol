"""Figure 2g: inferred parameters against final population size.

The same simulated trajectory is checkpointed at 10k, 20k, 30k, 40k and 50k cells and
scored at each checkpoint by the ten-seed ensemble; each parameter is expressed as a
change from that simulation's own 20k value, the size used for training. Simulations
are kept only if all five checkpoints exist with at least 4,000 ecDNA-descended cells.
Scoring is done by the population-size cell of notebooks/run_everything.ipynb.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import PRED_DIR, fig_dir
from . import figure_style as st

PRED = os.environ.get(
    "ECDNA_POPSIZE_PRED",
    os.path.join(PRED_DIR, "popsize_pred_cnn_attn_pool_d100_ensemble.tsv"))
OUTDIR = fig_dir("Figure2", "ECDNA_FIG2_DIR")
SIZES = [10000, 20000, 30000, 40000, 50000]
REF = 20000
# (delta column, axis label, colour, prior range used to express drift as % of range)
PARAMS = [("delta_s_hat_vs_20k", r"$\Delta\hat{s}$", st.S, 2.5),
          ("delta_mu_hat_vs_20k", r"$\Delta\hat{\mu}$", st.MU, 499.0),
          ("delta_g_exp_vs_20k", r"$\Delta\hat{g}$", st.INTRO, 9.0)]


def load(path=PRED):
    d = pd.read_csv(path, sep="\t")
    keep = d.groupby("simulation_id").population_size.nunique().eq(len(SIZES))
    return d[d.simulation_id.isin(keep[keep].index)]


def panel(d, name="panel_fig2g_popsize"):
    x = np.arange(len(SIZES))
    ref_x = SIZES.index(REF)
    fig, axes = plt.subplots(len(PARAMS), 1, figsize=(2.2, 2.9), sharex=True)
    for ax, (col, lab, colour, _) in zip(axes, PARAMS):
        w = d.pivot_table(index="simulation_id", columns="population_size",
                          values=col)[SIZES]
        for _, row in w.iterrows():
            ax.plot(x, row.values, color="0.75", lw=0.35, alpha=0.5,
                    zorder=1, rasterized=True)
        med = w.median().values
        ax.fill_between(x, w.quantile(0.25).values, w.quantile(0.75).values,
                        color=colour, alpha=0.22, lw=0, zorder=2)
        ax.plot(x, med, color=colour, lw=1.4, zorder=3)
        ax.axvline(ref_x, color="0.6", lw=0.6, ls=":", zorder=0)
        ax.set_ylabel(lab)
        ax.margins(x=0.05)
        ax.locator_params(axis="y", nbins=3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"{n // 1000}k" for n in SIZES], fontsize=6.5)
    axes[-1].set_xlabel("Final population size (cells)")
    fig.tight_layout(pad=0.3, h_pad=0.5)
    st.save_panel(fig, OUTDIR, name)


def report(d):
    print(f"simulations with all five checkpoints: {d.simulation_id.nunique()}")
    for col, lab, _, rng in PARAMS:
        at50 = d[d.population_size == 50000][col]
        iqr = at50.quantile(0.75) - at50.quantile(0.25)
        print(f"\n{lab}  drift {REF//1000}k->50k = {at50.median():+.3f} "
              f"({100 * abs(at50.median()) / rng:.1f}% of range; IQR across sims {iqr:.3f})")
        for n in SIZES:
            v = d[d.population_size == n][col]
            print(f"    {n // 1000:3d}k  median {v.median():+8.3f}  "
                  f"[{v.quantile(.25):+.3f}, {v.quantile(.75):+.3f}]")


if __name__ == "__main__":
    st.apply()
    d = load()
    report(d)
    panel(d)
