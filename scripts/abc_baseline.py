"""Rejection-ABC baseline for s, mu, g.

Summary per sim = r1 AFS (100) + r2 AFS (100) + log2(CN) = 201 dims, z-scored on
the training split. For a query, keep the k = floor(q*N) nearest references and
take the mean (s, mu) / mode (g) of the accepted set. q is picked on the
validation queries. Reference = train split, queries = test split, so nothing
matches itself. augmented appends 5 purity-perturbed copies per training sim.
"""

import numpy as np
import pandas as pd
from scipy import stats as spstats
from sklearn.neighbors import NearestNeighbors

from dataset import ECDNADataset, augment_purity
from train import TrainConfig, get_or_create_splits, metrics_from_preds, set_seed, SEED

Q_GRID = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]


def abc_composite(m):
    """Tolerance-selection composite: rho_s + r2_s + r2_mu + acc_pm1_g."""
    return m["spear_s"] + m["r2_s"] + m["r2_mu"] + m["acc_pm1_g"]


def _raw(ds, i):
    name = ds.sims[i]
    r1 = ds.h5f[name][f"r1_spectrum_{ds.depth}"][:].astype(np.float32)
    r2 = ds.h5f[name][f"r2_spectrum_{ds.depth}"][:].astype(np.float32)
    cn = float(ds.h5f[name].attrs["ecdna_region_cn"])
    return r1, r2, cn


def _summary(r1, r2, cn):
    return np.concatenate([r1, r2, [np.log2(max(cn, 1e-6))]]).astype(np.float32)


def build_summaries(ds, idx, augmented=False, n_aug=5, perturb=False, seed=SEED):
    """Return (X [n,201], labels{s,mu,g}). augmented adds n_aug perturbed copies
    per sim (training reference); perturb perturbs each query once (test set)."""
    if perturb:
        set_seed(seed)
    X, S, M, G = [], [], [], []
    for i in idx:
        r1, r2, cn = _raw(ds, i)
        s, mu, g = float(ds.s_values[i]), float(ds.mu_values[i]), int(ds.intro_vals[i])
        if perturb:
            r1, r2, _, _ = augment_purity(r1, r2, cn)
        X.append(_summary(r1, r2, cn)); S.append(s); M.append(mu); G.append(g)
        if augmented:
            for _ in range(n_aug):
                a1, a2, _, _ = augment_purity(r1, r2, cn)
                X.append(_summary(a1, a2, cn)); S.append(s); M.append(mu); G.append(g)
    return np.asarray(X), {"s": np.asarray(S), "mu": np.asarray(M), "g": np.asarray(G)}


def _estimates(nbr_idx, ref_lab, k):
    acc = nbr_idx[:, :k]
    s = ref_lab["s"][acc].mean(1)
    mu = ref_lab["mu"][acc].mean(1)
    g = spstats.mode(ref_lab["g"][acc], axis=1, keepdims=False).mode.astype(int)
    return s, mu, g


def _preds_df(truth, s, mu, g):
    # g is a class (the mode) so it serves as both the expected and argmax value
    return pd.DataFrame({"s_true": truth["s"], "s_hat": s,
                         "mu_true": truth["mu"], "mu_hat": mu,
                         "g_true": truth["g"].astype(int), "g_exp": g, "g_argmax": g})


def select_q(nn, ref_lab, val_X, val_truth, n_ref, q_grid=Q_GRID):
    """Pick q maximizing the validation composite; return (best_q, table)."""
    max_k = max(1, int(np.floor(max(q_grid) * n_ref)))
    _, idx = nn.kneighbors(val_X, n_neighbors=min(max_k, n_ref))
    rows, best = [], (None, -1e9)
    for q in q_grid:
        k = max(1, int(np.floor(q * n_ref)))
        s, mu, g = _estimates(idx, ref_lab, k)
        comp = abc_composite(metrics_from_preds(_preds_df(val_truth, s, mu, g)))
        rows.append({"q": q, "k": k, "val_composite": comp})
        if comp > best[1]:
            best = (q, comp)
    return best[0], pd.DataFrame(rows)


def predict(nn, ref_lab, q_X, q_truth, n_ref, q):
    """ABC predictions for a query set at tolerance q -> (df, metrics)."""
    k = max(1, int(np.floor(q * n_ref)))
    _, idx = nn.kneighbors(q_X, n_neighbors=min(k, n_ref))
    s, mu, g = _estimates(idx, ref_lab, k)
    df = _preds_df(q_truth, s, mu, g)
    return df, metrics_from_preds(df)


def run_abc(h5_path, data_dir, depth=100, augmented=False, q_grid=Q_GRID,
            n_aug=5, max_ref=None, seed=SEED):
    """End-to-end ABC at one depth. Returns dict with chosen q, q-table, and
    clean + perturbed test predictions and metrics."""
    cfg = TrainConfig(depth=depth)
    ds = ECDNADataset(h5_path, depth=depth, augment=False)
    train_idx, val_idx, test_idx, _ = get_or_create_splits(cfg, ds, data_dir)
    if max_ref is not None and len(train_idx) > max_ref:
        rng = np.random.RandomState(seed)
        train_idx = rng.choice(train_idx, size=max_ref, replace=False)

    ref_X, ref_lab = build_summaries(ds, train_idx, augmented=augmented, n_aug=n_aug, seed=seed)
    mean, std = ref_X.mean(0), ref_X.std(0) + 1e-8
    norm = lambda X: (X - mean) / std

    n_ref = len(ref_X)
    nn = NearestNeighbors(algorithm="auto").fit(norm(ref_X))

    val_X, val_truth = build_summaries(ds, val_idx)
    best_q, q_table = select_q(nn, ref_lab, norm(val_X), val_truth, n_ref, q_grid)

    test_X, test_truth = build_summaries(ds, test_idx)
    test_pert_X, _ = build_summaries(ds, test_idx, perturb=True, seed=seed)
    clean_df, clean_m = predict(nn, ref_lab, norm(test_X), test_truth, n_ref, best_q)
    pert_df, pert_m = predict(nn, ref_lab, norm(test_pert_X), test_truth, n_ref, best_q)
    clean_df.insert(0, "perturbed", False)
    pert_df.insert(0, "perturbed", True)

    return {
        "depth": depth, "augmented_reference": augmented, "n_ref": n_ref,
        "best_q": best_q, "q_table": q_table,
        "preds": pd.concat([clean_df, pert_df], ignore_index=True),
        "metrics_clean": clean_m, "metrics_perturbed": pert_m,
    }
