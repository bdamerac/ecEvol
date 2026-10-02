"""Supplementary figure: mutation burden of simulated and PCAWG spectra.

Mutation counts are the sums of the 100-bin spectra for simulations (all sequencing
depths pooled) and the lengths of the per-feature VAF lists for PCAWG, both under the
same inclusion criterion of at least ten ecDNA-region mutations. The dashed box marks
the 1st-99th percentile envelope of the simulated distribution.
"""
import os
import pickle
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .common import load_ensemble, fig_dir, DRIVE_DIR
from . import figure_style as st

DRIVE = DRIVE_DIR
VCF_PKL = os.path.join(DRIVE, "icgc_tcga_amplicon_vcfs_ecDNA_outside.pkl")
SIM_H5 = os.path.join(DRIVE, "all_simulations_20k.h5")
OUTDIR = fig_dir("Figure2", "ECDNA_FIG2_DIR")
DEPTHS = [30, 50, 100]
MIN_MUTS = 10


def load_real():
    pc = load_ensemble("pcawg")
    d = pickle.load(open(VCF_PKL, "rb"))
    pc["n_ec"] = pc.Feature_ID.map(lambda k: len(d[k][0]) if k in d else np.nan)
    pc["n_ch"] = pc.Feature_ID.map(lambda k: len(d[k][1]) if k in d else np.nan)
    return pc.dropna(subset=["n_ec"])


def load_sim(n=3000, seed=0):
    ec, ch = [], []
    with h5py.File(SIM_H5, "r") as f:
        keys = list(f.keys())
        idx = np.random.RandomState(seed).choice(len(keys), min(n, len(keys)), replace=False)
        for i in idx:
            g = f[keys[i]]
            for dep in DEPTHS:
                ec.append(g[f"r2_spectrum_{dep}"][:].sum())
                ch.append(g[f"r1_spectrum_{dep}"][:].sum())
    ec, ch = np.asarray(ec), np.asarray(ch)
    keep = ec >= MIN_MUTS
    return ec[keep], ch[keep]


def panel(real, sim_ec, sim_ch):
    re, rc = real.n_ec.values, real.n_ch.values
    (ea, eb), (ca, cb) = np.percentile(sim_ec, [1, 99]), np.percentile(sim_ch, [1, 99])
    inside = (re >= ea) & (re <= eb) & (rc >= ca) & (rc <= cb)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.scatter(sim_ec, sim_ch, s=2, color="0.62", alpha=0.18, linewidths=0,
               rasterized=True, zorder=1)
    ax.add_patch(plt.Rectangle((ea, ca), eb - ea, cb - ca, fill=False,
                               edgecolor="0.35", lw=0.8, ls="--", zorder=3))
    ax.scatter(re, rc, s=7, color=st.COHORT_HMF, alpha=0.75, linewidths=0, zorder=4)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Mutations in ecDNA region")
    ax.set_ylabel("Mutations outside ecDNA region")
    ax.text(0.03, 0.03, f"{100*inside.mean():.0f}% of PCAWG inside\nsimulated envelope",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=6.2,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))
    handles = [Line2D([], [], marker="o", ls="", color="0.62", ms=4, label="Simulated"),
               Line2D([], [], marker="o", ls="", color=st.COHORT_HMF, ms=4, label="PCAWG")]
    ax.legend(handles=handles, frameon=False, fontsize=6.2, loc="upper left",
              handlelength=1.0, borderpad=0.2)
    fig.tight_layout(pad=0.4)
    st.save_panel(fig, OUTDIR, "panel_supp_simulation_envelope")
    return inside.mean()


if __name__ == "__main__":
    st.apply()
    real = load_real()
    sim_ec, sim_ch = load_sim()
    frac = panel(real, sim_ec, sim_ch)
    print(f"simulated points: {len(sim_ec)}   PCAWG: {len(real)}")
    print(f"PCAWG inside joint simulated envelope: {100*frac:.0f}%")
