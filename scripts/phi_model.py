"""
Descendant-fraction (phi) model: predicts the ecDNA descendant fraction phi
from the chromosomal and ecDNA allele-frequency spectra and the copy number.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as spstats
from torch.utils.data import DataLoader, Subset


class PhiModel(nn.Module):
    """Predicts phi (ecDNA descendant fraction) from the r1/r2 spectra and copy number.

    r1, r2 are raw histogram counts (B, 100) and cn is the mean ecDNA copy number (B,).
    Each spectrum is normalised to sum 1 inside the forward pass and the two are stacked
    as a 2-channel input. The output is an unbounded scalar, clipped to [0, 1] at
    inference.
    """

    def __init__(self, n_bins=100, hidden=64, p_drop=0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2), nn.GELU(), nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.GELU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.GELU(), nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),   # (B, 64, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),                           # (B, 64)
            nn.Linear(64 + 1, hidden), nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),                  # linear output, no sigmoid
        )

    def forward(self, r1, r2, cn):
        r1_n = r1 / r1.sum(dim=1, keepdim=True).clamp(min=1e-9)
        r2_n = r2 / r2.sum(dim=1, keepdim=True).clamp(min=1e-9)
        spectra = torch.stack([r1_n, r2_n], dim=1)          # (B, 2, 100)
        h = self.features(spectra).view(r1.size(0), -1)     # (B, 64)
        z = torch.cat([h, cn.view(-1, 1)], dim=1)           # (B, 65)
        return self.head(z).squeeze(-1)                      # (B,) unbounded


def _phi_metrics(phi_true, phi_hat):
    pt, ph = np.asarray(phi_true), np.asarray(phi_hat)
    r2 = 1 - np.sum((ph - pt) ** 2) / (np.sum((pt - pt.mean()) ** 2) + 1e-12)
    return {"r2": float(r2), "mae": float(np.mean(np.abs(ph - pt))),
            "pearson": float(spstats.pearsonr(pt, ph).statistic)}


def train_phi(depth, h5_path, data_dir, models_dir, ckpt_name, device=None,
              num_epochs=50, patience=6, lr=1e-3, weight_decay=1e-4,
              batch_size=128, augment=True, seed=42, verbose=True):
    """Train one phi model at a given depth; write the checkpoint, return metrics.

    Regresses phi = tot_descended / POP_SIZE from the r1/r2 spectra and copy
    number on the same train/val/test split as the evo models. Trains on
    purity-augmented spectra and reports clean-spectrum recovery.
    """
    from dataset import ECDNADataset
    from train import TrainConfig, get_or_create_splits, set_seed, POP_SIZE

    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(models_dir, exist_ok=True)
    ckpt_path = os.path.join(models_dir, ckpt_name)

    cfg = TrainConfig(depth=depth)               # reuse the evo split for this depth
    ds = ECDNADataset(h5_path, depth=depth, augment=augment,
                      filter_descendants=cfg.filter_descendants)
    train_idx, val_idx, test_idx, _ = get_or_create_splits(cfg, ds, data_dir)

    model = PhiModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    @torch.no_grad()
    def evaluate(idx, aug):
        model.eval()
        prev, ds.augment = ds.augment, aug
        loader = DataLoader(Subset(ds, idx), batch_size=512, shuffle=False, num_workers=0)
        pt, ph = [], []
        for X_r1, X_r2, X_cn, _, _, _, _, y_tot in loader:
            out = model(X_r1.to(device), X_r2.to(device), X_cn.to(device))
            ph.append(np.clip(out.cpu().numpy(), 0.0, 1.0))
            pt.append(y_tot.numpy() / POP_SIZE)
        ds.augment = prev
        return _phi_metrics(np.concatenate(pt), np.concatenate(ph))

    best, best_ep, waited, t0 = -1e9, 0, 0, time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        ds.augment = augment
        loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size,
                            shuffle=True, num_workers=2, pin_memory=True)
        run = 0.0
        for X_r1, X_r2, X_cn, _, _, _, _, y_tot in loader:
            phi_true = (y_tot / POP_SIZE).float().to(device)
            out = model(X_r1.to(device), X_r2.to(device), X_cn.to(device))
            loss = F.mse_loss(out, phi_true)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item()

        vm = evaluate(val_idx, aug=False)
        sched.step(vm["r2"])
        if verbose:
            print(f"[phi d{depth}] ep{epoch:02d} loss {run / max(1, len(loader)):.4f} | val_r2 {vm['r2']:.4f}")
        if vm["r2"] > best:
            best, best_ep, waited = vm["r2"], epoch, 0
            torch.save({"model_state": model.state_dict(),
                        "config": {"depth": depth, "augment": augment, "seed": seed}}, ckpt_path)
        else:
            waited += 1
            if waited >= patience:
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state"])
    val_m, test_m = evaluate(val_idx, aug=False), evaluate(test_idx, aug=False)
    if verbose:
        print(f"  done: val_r2 {val_m['r2']:.3f} | test_r2 {test_m['r2']:.3f} | test_mae {test_m['mae']:.4f}")
    return {"depth": depth, "best_epoch": best_ep, "best_val_r2": float(best),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "train_min": (time.time() - t0) / 60.0,
            **{f"val_{k}": v for k, v in val_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()}}
