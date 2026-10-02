"""Shared config, loaders and statistics helpers for the post-inference analyses.

All inputs live under one data directory (ECDNA_DATA_DIR, default ./data):

  predictions/      {cohort}_pred_{arch}_s{seed}.tsv    per-seed cohort predictions
                    {cohort}_pred_{arch}_ensemble.tsv   ensemble cohort predictions
                    popsize_pred_{arch}_d100_ensemble.tsv
  sim_predictions/  cnn_attn_pool_d{depth}_s{seed}.parquet   per-seed simulation
                    predictions (with *_true), phi_sim_depth{depth}.parquet
  simulations/      all_simulations_20k.h5 and the cohort AFS pickles
  external/         published supplementary tables used by a few panels

Figures are written under one output directory (ECDNA_OUT_DIR, default ./figures),
one subdirectory per manuscript figure. Individual directories can be overridden; see
the variables below.
"""
import os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import pearsonr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("ECDNA_DATA_DIR", os.path.join(_ROOT, "data"))
OUT_DIR = os.environ.get("ECDNA_OUT_DIR", os.environ.get("ECDNA_FIG_DIR",
                                                         os.path.join(_ROOT, "figures")))

PRED_DIR = os.environ.get("ECDNA_PRED_DIR", os.path.join(DATA_DIR, "predictions"))
SIM_DIR = os.environ.get("ECDNA_SIM_DIR", os.path.join(DATA_DIR, "sim_predictions"))
DRIVE_DIR = os.environ.get("ECDNA_DRIVE_DIR", os.path.join(DATA_DIR, "simulations"))
EXTERNAL_DIR = os.environ.get("ECDNA_EXTERNAL_DIR", os.path.join(DATA_DIR, "external"))
SEEDS = list(range(42, 52))


def fig_dir(name, env=None):
    """Output directory for one manuscript figure, e.g. fig_dir("Figure3")."""
    if env and os.environ.get(env):
        return os.environ[env]
    return os.path.join(OUT_DIR, name)

# HMF cancer_type -> PCAWG tumor_type codes, for the matched per-cancer-type comparison
MATCHED = {
    "CNS": ["GBM"], "Lung": ["LUSC", "LUAD"], "Stomach": ["STAD"], "Esophagus": ["ESAD"],
    "Skin": ["SKCM", "MELA"], "Head/Neck": ["HNSC"], "Urinary": ["BLCA"],
    "Bone/Soft": ["SARC", "BOCA"], "Breast": ["BRCA"], "Ovary": ["OV"],
}
HMFNAME = {"Head/Neck": "Head and neck", "Urinary": "Urinary tract", "Bone/Soft": "Bone/Soft tissue"}

# PCAWG 27 tumour codes -> 10 organ-system strata, used for Cox stratification
TUMOUR_GROUP = {
    "GBM": "CNS", "PBCA": "CNS", "LGG": "CNS", "LINC": "CNS", "BRCA": "Breast",
    "ESAD": "UpperGI", "STAD": "UpperGI", "GACA": "UpperGI", "COAD": "UpperGI",
    "LIRI": "HPB", "LIHC": "HPB", "PACA": "HPB", "PAEN": "HPB", "BTCA": "HPB",
    "OV": "Gyn", "UCEC": "Gyn", "CESC": "Gyn", "LUSC": "Lung", "LUAD": "Lung",
    "MELA": "Skin", "SKCM": "Skin", "SARC": "Sarcoma", "BLCA": "GU", "KIRP": "GU",
    "KICH": "GU", "PRAD": "GU", "HNSC": "HeadNeck", "MALY": "Lymph",
}

MPL_RC = {"font.size": 11, "font.family": "DejaVu Sans",
          "axes.spines.top": False, "axes.spines.right": False}

ID_COL = {"pcawg": "Feature_ID", "hmf": "feature_id_full"}
CN_COL = {"pcawg": "median_feature_cn", "hmf": "copy_number_median"}


def load_ensemble(cohort):
    df = pd.read_csv(os.path.join(PRED_DIR, f"{cohort}_pred_cnn_attn_pool_ensemble.tsv"), sep="\t")
    return df[~df.skipped.astype(bool)].copy()


def load_seeds(cohort, id_col):
    return [pd.read_csv(os.path.join(PRED_DIR, f"{cohort}_pred_cnn_attn_pool_s{s}.tsv"), sep="\t")
              .query("skipped == False").set_index(id_col) for s in SEEDS]


def load_sim_seeds(depth=100):
    return [pd.read_parquet(os.path.join(SIM_DIR, f"cnn_attn_pool_d{depth}_s{s}.parquet")) for s in SEEDS]


def zscore(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def benjamini_hochberg(pvals):
    p = np.asarray(pvals); order = np.argsort(p); n = len(p); q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[order[i]] * n / (i + 1)); q[order[i]] = prev
    return q


def significance_stars(p):
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def cliffs_delta(a, b):
    return np.sign(np.subtract.outer(b, a)).mean()


def hodges_lehmann(a, b):
    return np.median(np.subtract.outer(b, a))


def intraclass_correlation(frames, col):
    """Return (n_seeds, mean pairwise r, ICC single-seed, ICC full-ensemble) for a prediction column."""
    D = pd.DataFrame({i: f[col] for i, f in enumerate(frames)}).dropna()
    m = D.shape[1]
    pairwise_r = [pearsonr(D[i], D[j])[0] for i, j in combinations(range(m), 2)]
    var_between = D.mean(1).var()
    var_within = D.var(1).mean()
    return m, np.mean(pairwise_r), var_between / (var_between + var_within), var_between / (var_between + var_within / m)


def r_squared(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
