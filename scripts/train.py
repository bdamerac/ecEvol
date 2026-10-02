"""
Model training and evaluation.

Models are trained on simulated allele-frequency spectra and selected on a
held-out validation split using a composite of the three task metrics (see
`composite`). `predict` produces per-sample predictions on a split, either
clean or with purity perturbation applied, and metrics are computed from those
predictions.
"""

import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as spstats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Subset

from dataset import ECDNADataset, S_MAX
from model import build_model, s_to_z, z_to_s, N_INTRO

SEED = 42
POP_SIZE = 20000  # fixed final population size; phi_true = tot_descended / POP_SIZE


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    backbone: str = "cnn_attn"
    depth: int = 100
    batch_size: int = 128
    num_epochs: int = 50
    patience: int = 6
    lr: float = 1e-3
    weight_decay: float = 1e-4
    huber_delta: float = 1.0
    grad_clip: float = 1.0
    augment: bool = True
    filter_descendants: bool = True
    intro_mode: str = "softmax"            # "softmax" or "ordinal"
    heads: tuple = ("s", "mu", "g")        # drop entries for single-task ablations
    hidden_dim: int = 128
    num_filters: int = 32
    num_heads: int = 4
    seed: int = SEED


# ----- ordinal g helpers -----

def _ordinal_targets(g, K=N_INTRO - 1):
    """g (B,) long -> (B,K) float where column k = 1[g > k]."""
    ks = torch.arange(K, device=g.device).unsqueeze(0)
    return (g.unsqueeze(1) > ks).float()


def _ordinal_pmf(logits):
    """Ordinal logits (N,K) -> class pmf (N,K+1) over g in {0..K}."""
    pgt = 1.0 / (1.0 + np.exp(-logits))
    pgt = np.minimum.accumulate(pgt, axis=1)      # enforce monotone P(g>k)
    K = pgt.shape[1]
    pmf = np.zeros((pgt.shape[0], K + 1), dtype=np.float32)
    pmf[:, 0] = 1.0 - pgt[:, 0]
    for k in range(1, K):
        pmf[:, k] = pgt[:, k - 1] - pgt[:, k]
    pmf[:, K] = pgt[:, K - 1]
    return pmf


# ----- loss -----

def compute_loss(out, y_mu, y_g, y_s, cfg):
    """Equal-weighted sum over the active heads."""
    loss = 0.0
    if "s_raw" in out:
        loss = loss + nn.HuberLoss(delta=cfg.huber_delta)(out["s_raw"], s_to_z(y_s))
    if "mu_raw" in out:
        loss = loss + F.mse_loss(out["mu_raw"], y_mu)
    if "intro_logits" in out:
        if cfg.intro_mode == "softmax":
            loss = loss + F.cross_entropy(out["intro_logits"], y_g)
        else:
            loss = loss + F.binary_cross_entropy_with_logits(
                out["intro_logits"], _ordinal_targets(y_g))
    return loss


# ----- data -----

def _split_path(cfg, data_dir):
    tag = "" if cfg.filter_descendants else "_nofilter"
    return os.path.join(data_dir, f"splits_depth{cfg.depth}{tag}.npz")


def get_or_create_splits(cfg, ds, data_dir):
    """Load cached stratified 80/10/10 split, else build and save it."""
    path = _split_path(cfg, data_dir)
    if os.path.exists(path):
        sp = np.load(path)
        return sp["train_idx"], sp["val_idx"], sp["test_idx"], sp["mu_train"]

    bins_s = np.linspace(0, S_MAX, num=6)
    strat_s = np.digitize(ds.s_values, bins_s) - 1
    rng = np.random.RandomState(SEED)
    # mix s-bin and g-class so both stratify across splits
    mix = np.where(rng.rand(len(strat_s)) > 0.5, ds.intro_vals.astype(int), strat_s + 100)

    idx = np.arange(len(ds))
    train_idx, temp = train_test_split(idx, test_size=0.2, stratify=mix, random_state=SEED)
    val_idx, test_idx = train_test_split(temp, test_size=0.5, stratify=mix[temp], random_state=SEED)
    mu_train = ds.mu_values[train_idx]

    os.makedirs(data_dir, exist_ok=True)
    np.savez(path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, mu_train=mu_train)
    return train_idx, val_idx, test_idx, mu_train


def make_data(cfg, h5_path, data_dir):
    """Build one dataset and Subset it into train/val/test.

    `augment` applies to the whole dataset, train and validation/test alike;
    `predict` toggles it per evaluation for the clean-versus-perturbed reporting.
    """
    ds = ECDNADataset(h5_path, depth=cfg.depth, augment=cfg.augment,
                      filter_descendants=cfg.filter_descendants)
    train_idx, val_idx, test_idx, mu_train = get_or_create_splits(cfg, ds, data_dir)
    scaler = MinMaxScaler().fit(mu_train.reshape(-1, 1))
    ds.set_mu_scaler(scaler)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=cfg.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    return ds, scaler, train_loader, splits


# ----- prediction + metrics -----

@torch.no_grad()
def predict(model, ds, idx, cfg, scaler, device, augment=False, seed=None):
    """Per-sample predictions over `idx` -> DataFrame.

    `augment` sets the dataset's purity-augmentation state for this evaluation and
    restores it afterward; pass a `seed` for a reproducible perturbed evaluation.
    """
    model.eval()
    prev = ds.augment
    ds.augment = augment
    if seed is not None:
        set_seed(seed)
    loader = DataLoader(Subset(ds, idx), batch_size=512, shuffle=False, num_workers=0)

    rows = {k: [] for k in ("s_true", "s_hat", "mu_true", "mu_hat", "g_true",
                            "g_argmax", "g_exp", "tot_descended", "ecdna_region_cn")}
    g_probs = []
    for X_r1, X_r2, X_cn, X_tot, y_mu, y_g, y_s, y_tot in loader:
        out = model(X_r1.to(device), X_r2.to(device), X_cn.to(device), X_tot.to(device))

        if "s_raw" in out:
            rows["s_true"].append(y_s.numpy())
            rows["s_hat"].append(z_to_s(out["s_raw"].cpu().numpy()))
        if "mu_raw" in out:
            mu_sc = np.clip(out["mu_raw"].cpu().numpy(), 0.0, 1.0)
            rows["mu_true"].append(scaler.inverse_transform(y_mu.numpy().reshape(-1, 1)).ravel())
            rows["mu_hat"].append(scaler.inverse_transform(mu_sc.reshape(-1, 1)).ravel())
        if "intro_logits" in out:
            logits = out["intro_logits"].cpu().numpy()
            pmf = (F.softmax(torch.from_numpy(logits), dim=1).numpy()
                   if cfg.intro_mode == "softmax" else _ordinal_pmf(logits))
            classes = np.arange(pmf.shape[1])
            rows["g_true"].append(y_g.numpy().astype(int))
            rows["g_argmax"].append(pmf.argmax(axis=1))
            rows["g_exp"].append((pmf * classes[None, :]).sum(axis=1))
            g_probs.append(pmf)
        rows["tot_descended"].append(y_tot.numpy())
        rows["ecdna_region_cn"].append(X_cn.numpy())

    ds.augment = prev
    df = pd.DataFrame({k: np.concatenate(v) for k, v in rows.items() if v})
    df["phi_true"] = df["tot_descended"] / POP_SIZE
    if g_probs:
        gp = np.concatenate(g_probs)
        for k in range(gp.shape[1]):
            df[f"g_p{k}"] = gp[:, k]
    df.insert(0, "perturbed", augment)
    return df


def metrics_from_preds(df):
    """Compute the manuscript's reported metrics from a prediction DataFrame."""
    m = {"n": len(df)}
    eps = 1e-3
    if "s_hat" in df:
        st, sp = df["s_true"].to_numpy(), df["s_hat"].to_numpy()
        m["spear_s"] = float(spstats.spearmanr(st, sp).statistic)
        m["pear_log_s"] = float(spstats.pearsonr(np.log(st + eps), np.log(sp + eps)).statistic)
        m["rmse_s"] = float(np.sqrt(np.mean((sp - st) ** 2)))
        m["mae_s"] = float(np.mean(np.abs(sp - st)))
        m["r2_s"] = float(1 - np.sum((sp - st) ** 2) / (np.sum((st - st.mean()) ** 2) + 1e-12))
        # positive predictive value of the high/low selection tiers
        hi, lo = sp > 1.25, sp < 0.75
        m["ppv_s_high"] = float((st[hi] > 1).mean()) if hi.any() else float("nan")
        m["ppv_s_low"] = float((st[lo] < 1).mean()) if lo.any() else float("nan")
    if "mu_hat" in df:
        mt, mp = df["mu_true"].to_numpy(), df["mu_hat"].to_numpy()
        m["rmse_mu"] = float(np.sqrt(np.mean((mp - mt) ** 2)))
        m["r2_mu"] = float(1 - np.sum((mp - mt) ** 2) / (np.sum((mt - mt.mean()) ** 2) + 1e-12))
        m["mae_mu"] = float(np.mean(np.abs(mp - mt)))
    if "g_exp" in df:
        gt = df["g_true"].to_numpy().astype(int)
        ge, ga = df["g_exp"].to_numpy(), df["g_argmax"].to_numpy().astype(int)
        m["mae_g"] = float(np.mean(np.abs(ge - gt)))
        m["r2_g"] = float(1 - np.sum((ge - gt) ** 2) / (np.sum((gt - gt.mean()) ** 2) + 1e-12))
        gr = np.clip(np.rint(ge).astype(int), 0, N_INTRO - 1)
        m["acc_exact_g"] = float(np.mean(ga == gt))
        m["acc_pm1_g"] = float(np.mean(np.abs(ga - gt) <= 1))
        # accuracy after rounding the expected value to the nearest class
        m["round_exact_g"] = float(np.mean(gr == gt))
        m["round_pm1_g"] = float(np.mean(np.abs(gr - gt) <= 1))
    return m


def composite(m):
    """Validation composite (higher is better): spearman_s + r2_s + r2_mu + acc_pm1_g.

    Rank and calibration terms for s, plus one term each for mu and g. The same
    composite is used to select the ABC tolerance.
    """
    return m["spear_s"] + m["r2_s"] + m["r2_mu"] + m["acc_pm1_g"]


def select_metric(m, cfg):
    """Score used for early stopping / LR scheduling (single-task falls back)."""
    if {"s", "mu", "g"} <= set(cfg.heads):
        return composite(m)
    if cfg.heads == ("s",):
        return m["spear_s"] + m["r2_s"]
    if cfg.heads == ("g",):
        return m["acc_pm1_g"]
    if cfg.heads == ("mu",):
        return m["r2_mu"]
    return m.get("pear_log_s", 0.0)


# ----- training -----

def train(cfg, h5_path, data_dir, models_dir, ckpt_name, device=None,
          run_dir=None, save_preds_fn=None, verbose=True):
    """Train one model (one seed). Returns a result dict; writes best checkpoint.

    If `run_dir` and `save_preds_fn` are given, writes clean and
    purity-perturbed test-set per-sample predictions for downstream analysis.
    """
    set_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(models_dir, exist_ok=True)
    ckpt_path = os.path.join(models_dir, ckpt_name)

    ds, scaler, train_loader, splits = make_data(cfg, h5_path, data_dir)
    model = build_model(cfg.backbone, cfg.hidden_dim, cfg.num_filters, cfg.num_heads,
                        intro_mode=cfg.intro_mode, heads=cfg.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best_score, best_epoch, waited = -1e9, 0, 0
    t0 = time.time()
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running = 0.0
        for X_r1, X_r2, X_cn, X_tot, y_mu, y_g, y_s, _ in train_loader:
            X_r1, X_r2, X_cn, X_tot = (X_r1.to(device), X_r2.to(device),
                                       X_cn.to(device), X_tot.to(device))
            y_mu, y_g, y_s = y_mu.to(device), y_g.to(device), y_s.to(device)
            out = model(X_r1, X_r2, X_cn, X_tot)
            loss = compute_loss(out, y_mu, y_g, y_s, cfg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item()

        # selection metric on the purity-perturbed val split (the imperfect-purity
        # case the model sees on real data)
        val_m = metrics_from_preds(predict(model, ds, splits["val"], cfg, scaler, device,
                                           augment=cfg.augment))
        score = select_metric(val_m, cfg)
        sched.step(score)
        if verbose:
            print(f"[{cfg.backbone} d{cfg.depth} s{cfg.seed}] ep{epoch:02d} "
                  f"loss {running / max(1, len(train_loader)):.3f} | sel {score:.4f}")

        if score > best_score:
            best_score, best_epoch, waited = score, epoch, 0
            torch.save({"model_state": model.state_dict(), "mu_scaler": scaler,
                        "config": asdict(cfg)}, ckpt_path)
        else:
            waited += 1
            if waited >= cfg.patience:
                break

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])

    # clean val/test metrics plus a deterministic purity-perturbed test
    val_df = predict(model, ds, splits["val"], cfg, scaler, device, augment=False)
    test_df = predict(model, ds, splits["test"], cfg, scaler, device, augment=False)
    test_pert_df = predict(model, ds, splits["test"], cfg, scaler, device, augment=True, seed=cfg.seed)
    val_m, test_m = metrics_from_preds(val_df), metrics_from_preds(test_df)
    test_pert_m = metrics_from_preds(test_pert_df)

    if run_dir and save_preds_fn:
        tag = f"{cfg.backbone}_d{cfg.depth}_s{cfg.seed}"
        all_test = pd.concat([test_df, test_pert_df], ignore_index=True)
        save_preds_fn(all_test, os.path.join(run_dir, "preds", f"{tag}.parquet"))

    result = {
        "backbone": cfg.backbone, "depth": cfg.depth, "seed": cfg.seed,
        "augment": cfg.augment, "intro_mode": cfg.intro_mode,
        "heads": "+".join(cfg.heads), "filter_descendants": cfg.filter_descendants,
        "lr": cfg.lr, "weight_decay": cfg.weight_decay, "batch_size": cfg.batch_size,
        "best_epoch": best_epoch, "best_val_score": float(best_score),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "train_min": (time.time() - t0) / 60.0, "ckpt": ckpt_path,
        **{f"val_{k}": v for k, v in val_m.items()},
        **{f"test_{k}": v for k, v in test_m.items()},
        **{f"testpert_{k}": v for k, v in test_pert_m.items()},
    }
    if verbose:
        print(f"  done: val_sel {best_score:.4f} | test spear_s "
              f"{test_m.get('spear_s', float('nan')):.3f} | "
              f"test_pert spear_s {test_pert_m.get('spear_s', float('nan')):.3f}")
    return result


def run_seeds(cfg, seeds, h5_path, data_dir, models_dir, ckpt_stem, **kw):
    """Train one config across seeds; return (per_seed_df, summary_row)."""
    rows = []
    for sd in seeds:
        c = TrainConfig(**{**asdict(cfg), "seed": sd})
        rows.append(train(c, h5_path, data_dir, models_dir, f"{ckpt_stem}_s{sd}.pt", **kw))
    per = pd.DataFrame(rows)
    metric_cols = [c for c in per.columns
                   if c.startswith(("val_", "test_", "testpert_")) or c == "best_val_score"]
    summary = {"backbone": cfg.backbone, "heads": "+".join(cfg.heads),
               "intro_mode": cfg.intro_mode, "augment": cfg.augment,
               "filter_descendants": cfg.filter_descendants, "n_seeds": len(seeds)}
    for c in metric_cols:
        summary[f"{c}_mean"] = float(per[c].mean())
        summary[f"{c}_std"] = float(per[c].std(ddof=0))
    return per, summary
