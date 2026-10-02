"""
ECDNADataset: per-depth AFS spectra from the simulation HDF5 file.

Each item is an r1 (diploid control) and r2 (ecDNA region) histogram plus the
copy number, six read-count summary features, and the three targets
(mu scaled, intro_generation, s) and tot_descended.

Purity augmentation is applied per-sample in `__getitem__` when enabled.
"""

import h5py
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

# dataset filters (fixed)
S_MAX = 2.5
INTRO_MAX = 9
LOWER = 4000
UPPER = 19800

BIN_EDGES = np.linspace(0.0, 1.0, 101)  # 100 bins


# ----- purity augmentation -----

def simulate_purity_misestimation(afs_counts, bin_edges, cn_tumor, purity_true, purity_est):
    """Transform an AFS histogram for purity dilution and mis-estimation."""
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Step 1: dilute to bulk space using true purity
    denom_true = purity_true * cn_tumor + (1.0 - purity_true) * 2.0
    vaf_bulk = bin_centers * (purity_true * cn_tumor) / denom_true

    # Step 2: wrong-purity correction -- re-expand as if caller used purity_est
    denom_wrong = purity_est * cn_tumor
    vaf_wrong = vaf_bulk * (purity_est * cn_tumor + (1.0 - purity_est) * 2.0) / denom_wrong

    result, _ = np.histogram(vaf_wrong, bins=bin_edges, weights=afs_counts)
    return result.astype(np.float32)


def augment_purity(X_r1, X_r2, cn, bin_edges=BIN_EDGES,
                   purity_range=(0.6, 1.0), misest_range=(0.75, 1.25)):
    """Draw purity_true / purity_est and apply mis-estimation to both spectra."""
    purity_true = np.random.uniform(*purity_range)
    purity_est = float(np.clip(purity_true * np.random.uniform(*misest_range), 0.01, 1.0))

    X_r1_aug = simulate_purity_misestimation(
        X_r1, bin_edges, cn_tumor=2.0, purity_true=purity_true, purity_est=purity_est)
    X_r2_aug = simulate_purity_misestimation(
        X_r2, bin_edges, cn_tumor=cn, purity_true=purity_true, purity_est=purity_est)
    return X_r1_aug, X_r2_aug, purity_true, purity_est


class ECDNADataset(Dataset):
    """Simulation dataset for a fixed read-depth."""

    def __init__(self, hdf5_path, depth=100, lower=LOWER, upper=UPPER, augment=True,
                 filter_descendants=True):
        super().__init__()
        self.h5f = h5py.File(hdf5_path, "r")
        self.depth = depth
        self.augment = augment
        self.filter_descendants = filter_descendants
        self.bin_edges = BIN_EDGES

        sims, mu, s, intro, tot = [], [], [], [], []
        for k in self.h5f.keys():
            a = self.h5f[k].attrs
            tot_desc = float(a["tot_descended"])
            s_val = float(a["ecdna_s"])
            intro_val = int(a["intro_generation"])
            keep_desc = (lower <= tot_desc <= upper) if filter_descendants else True
            if keep_desc and s_val < S_MAX and intro_val <= INTRO_MAX:
                if f"r1_spectrum_{depth}" in self.h5f[k] and f"r2_spectrum_{depth}" in self.h5f[k]:
                    sims.append(k)
                    mu.append(float(a["mu"]))
                    s.append(s_val)
                    intro.append(intro_val)
                    tot.append(tot_desc)

        self.sims = sims
        self.mu_values = np.array(mu, dtype=float)
        self.s_values = np.array(s, dtype=float)
        self.intro_vals = np.array(intro, dtype=int)
        self.tot_desc_vals = np.array(tot, dtype=np.float32)

        # placeholder scaling; overwritten with train-only scaler once splits are known
        self.scaler_mu = MinMaxScaler()
        self.mu_scaled = self.scaler_mu.fit_transform(self.mu_values.reshape(-1, 1)).reshape(-1)

        print(f"[ECDNADataset] depth={depth} | sims kept={len(self.sims)} | augment={augment}")

    def set_mu_scaler(self, scaler):
        """Install a scaler fit on the training mu values and rescale all targets."""
        self.scaler_mu = scaler
        self.mu_scaled = scaler.transform(self.mu_values.reshape(-1, 1)).reshape(-1)

    def __len__(self):
        return len(self.sims)

    def __getitem__(self, idx):
        sname = self.sims[idx]
        X_r1 = self.h5f[sname][f"r1_spectrum_{self.depth}"][:].astype(np.float32)
        X_r2 = self.h5f[sname][f"r2_spectrum_{self.depth}"][:].astype(np.float32)
        cn = float(self.h5f[sname].attrs["ecdna_region_cn"])

        if self.augment:
            X_r1, X_r2, _, _ = augment_purity(X_r1, X_r2, cn, self.bin_edges)

        r1_total = float(X_r1.sum())
        r2_total = float(X_r2.sum())
        total = r1_total + r2_total
        feats_tot = np.array(
            [r1_total, r2_total, total,
             np.log1p(r1_total), np.log1p(r2_total), np.log1p(total)],
            dtype=np.float32,
        )

        return (
            torch.tensor(X_r1, dtype=torch.float32),
            torch.tensor(X_r2, dtype=torch.float32),
            torch.tensor(cn, dtype=torch.float32),
            torch.tensor(feats_tot, dtype=torch.float32),
            torch.tensor(self.mu_scaled[idx], dtype=torch.float32),
            torch.tensor(self.intro_vals[idx], dtype=torch.long),
            torch.tensor(self.s_values[idx], dtype=torch.float32),
            torch.tensor(self.tot_desc_vals[idx], dtype=torch.float32),
        )
