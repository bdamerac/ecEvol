"""Supplementary Table S6 and the ensemble accuracy figures quoted in Methods.

Accuracy: single networks (mean and s.d. over the ten seeds) against the ten-network
mean, on the clean held-out simulations at 100x depth. Reproducibility: intraclass
correlation across the ten seeds for the scored PCAWG features, for a single network
and for the ten-network mean.
"""
import numpy as np
from scipy.stats import spearmanr

from .common import load_ensemble, load_seeds, load_sim_seeds, intraclass_correlation, r_squared

PARAMS = [("s", "s_hat"), ("g", "g_exp"), ("mu", "mu_hat")]


def _metrics(d):
    return (spearmanr(d.s_true, d.s_hat).correlation, r_squared(d.mu_true, d.mu_hat),
            (np.abs(np.round(d.g_exp) - d.g_true) <= 1).mean())


def accuracy(depth=100):
    seeds = [d[~d.perturbed.astype(bool)].reset_index(drop=True) for d in load_sim_seeds(depth)]
    single = np.array([_metrics(d) for d in seeds])
    mean = seeds[0][["s_true", "mu_true", "g_true"]].copy()
    for col in ["s_hat", "mu_hat", "g_exp"]:
        mean[col] = np.mean([d[col].values for d in seeds], axis=0)
    ensemble = _metrics(mean)
    for name, i in [("Spearman rho, s", 0), ("R2, mu", 1), ("within-one accuracy, g", 2)]:
        print(f"  {name:24s} single {single[:, i].mean():.3f} +/- {single[:, i].std(ddof=1):.3f}"
              f"   ensemble {ensemble[i]:.3f}")


def icc_table(id_col="Feature_ID"):
    scored = load_ensemble("pcawg")[id_col].values
    frames = [d[~d.index.duplicated()].reindex(scored) for d in load_seeds("pcawg", id_col)]
    print(f"  n = {len(scored)} features")
    for name, col in PARAMS:
        _, _, single, ensemble = intraclass_correlation(frames, col)
        print(f"  {name:3s} single network {single:.3f}   ten-network mean {ensemble:.3f}")


if __name__ == "__main__":
    print("accuracy on held-out simulations (100x):")
    accuracy()
    print("intraclass correlation across seeds (PCAWG):")
    icc_table()
