"""
Inference for tumour cohorts and phi on simulated samples.

Applies the evo model (s, mu, g) and the phi model to produce per-sample
predictions: phi_hat, s_hat, mu_hat, g_exp, g_argmax, and the full g softmax
(g_p0..g_p9). Input metadata columns are passed through unchanged so any
per-sample annotation in the metadata table is carried into the output.
"""

import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from model import build_model, z_to_s
from phi_model import PhiModel

BIN_EDGES = np.linspace(0.0, 1.0, 101)


# ----- checkpoint loading -----

def load_evo_model(ckpt_path, device="cpu"):
    """Rebuild the evo model from its stored config; return (model, mu_scaler)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt["config"]
    model = build_model(c["backbone"], c["hidden_dim"], c["num_filters"], c["num_heads"],
                        intro_mode=c["intro_mode"], heads=tuple(c["heads"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["mu_scaler"]


def load_phi_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PhiModel()
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


# ----- AFS loaders -----

def load_pcawg_afs(pkl_path):
    """feature_id -> (ecDNA_vafs, outside_vafs) as float32 arrays (purity-corrected)."""
    raw = pickle.load(open(pkl_path, "rb"))
    return {k: (np.asarray(v[0], dtype=np.float32), np.asarray(v[1], dtype=np.float32))
            for k, v in raw.items()}


def make_afs_loader(afs_dict, bin_edges=BIN_EDGES, min_mutations=10):
    """PCAWG loader(row) -> (r1, r2), or (None, None) if r2 has < min_mutations VAFs."""
    def loader(row):
        fid = row["Feature_ID"]
        if fid not in afs_dict:
            return None, None
        ec_vafs, out_vafs = afs_dict[fid]
        if len(ec_vafs) < min_mutations:
            return None, None
        r2 = np.histogram(ec_vafs, bins=bin_edges)[0].astype(np.float32)
        r1 = np.histogram(out_vafs, bins=bin_edges)[0].astype(np.float32)
        return r1, r2
    return loader


def load_hmf_cache(cache_path):
    df = pd.read_csv(cache_path).drop_duplicates(subset="feature_id_full", keep="first")
    return df.set_index("feature_id_full").to_dict("index")


def make_hmf_loader(cache_path, bin_edges=BIN_EDGES, min_mutations=10):
    """HMF loader(row) -> (r1, r2). r1 is zeros when the outside region is absent."""
    cache = load_hmf_cache(cache_path)
    n_bins = len(bin_edges) - 1

    def loader(row):
        entry = cache.get(row["feature_id_full"])
        if entry is None:
            return None, None
        ec_vafs = np.array(json.loads(entry["ecdna_vafs"]), dtype=np.float32)
        if len(ec_vafs) < min_mutations:
            return None, None
        out_vafs = np.array(json.loads(entry["outside_vafs"]), dtype=np.float32)
        r2 = np.histogram(ec_vafs, bins=bin_edges)[0].astype(np.float32)
        r1 = (np.histogram(out_vafs, bins=bin_edges)[0].astype(np.float32)
              if len(out_vafs) else np.zeros(n_bins, dtype=np.float32))
        return r1, r2
    return loader


# ----- prediction -----

def _predict_batch(evo, phi, mu_scaler, r1_b, r2_b, cn_b, device):
    r1_t = torch.tensor(np.stack(r1_b)).to(device)
    r2_t = torch.tensor(np.stack(r2_b)).to(device)
    cn_t = torch.tensor(cn_b, dtype=torch.float32).to(device)
    feats = [[float(r1.sum()), float(r2.sum()), float(r1.sum() + r2.sum()),
              np.log1p(r1.sum()), np.log1p(r2.sum()), np.log1p(r1.sum() + r2.sum())]
             for r1, r2 in zip(r1_b, r2_b)]
    feats_t = torch.tensor(feats, dtype=torch.float32).to(device)

    with torch.no_grad():
        out = evo(r1_t, r2_t, cn_t, feats_t)
        phi_hat = (np.clip(phi(r1_t, r2_t, cn_t).cpu().numpy(), 0.0, 1.0)
                   if phi is not None else np.full(len(r1_b), np.nan))

    s_hat = z_to_s(out["s_raw"].cpu().numpy())
    mu_hat = mu_scaler.inverse_transform(out["mu_raw"].cpu().numpy().reshape(-1, 1)).ravel()
    p = F.softmax(out["intro_logits"], dim=1).cpu().numpy()
    classes = np.arange(p.shape[1])
    return phi_hat, s_hat, mu_hat, (p * classes).sum(1), p.argmax(1), p


def run_inference(metadata_tsv, afs_loader, evo_ckpt, phi_ckpt, out_tsv,
                  cn_col="median_feature_cn", required_cols=None,
                  device="cpu", batch_size=256):
    """Run evo (and phi, if given) over metadata_tsv; write predictions TSV.

    `phi_ckpt=None` skips phi and leaves phi_hat as NaN.
    """
    evo, mu_scaler = load_evo_model(evo_ckpt, device)
    phi = load_phi_model(phi_ckpt, device) if phi_ckpt else None
    meta = (metadata_tsv if isinstance(metadata_tsv, pd.DataFrame)
            else pd.read_csv(metadata_tsv, sep="\t"))
    if required_cols:
        missing = [c for c in required_cols if c not in meta.columns]
        if missing:
            raise ValueError(f"metadata_tsv missing figure columns: {missing}")

    records, batch_rows, r1_b, r2_b, cn_b = [], [], [], [], []

    def flush():
        if not batch_rows:
            return
        phi_hat, s_hat, mu_hat, g_exp, g_arg, p = _predict_batch(
            evo, phi, mu_scaler, r1_b, r2_b, cn_b, device)
        for i, row in enumerate(batch_rows):
            rec = {**row, "phi_hat": float(phi_hat[i]), "s_hat": float(s_hat[i]),
                   "mu_hat": float(mu_hat[i]), "g_exp": float(g_exp[i]),
                   "g_argmax": int(g_arg[i]), "skipped": False}
            for k in range(p.shape[1]):
                rec[f"g_p{k}"] = float(p[i, k])
            records.append(rec)
        batch_rows.clear(); r1_b.clear(); r2_b.clear(); cn_b.clear()

    for _, row in meta.iterrows():
        r1, r2 = afs_loader(row)
        if r1 is None:
            records.append({**row.to_dict(), "phi_hat": np.nan, "s_hat": np.nan,
                            "mu_hat": np.nan, "g_exp": np.nan, "g_argmax": np.nan,
                            "skipped": True})
            continue
        r1_b.append(r1); r2_b.append(r2); cn_b.append(float(row[cn_col]))
        batch_rows.append(row.to_dict())
        if len(batch_rows) >= batch_size:
            flush()
    flush()

    out = pd.DataFrame(records)
    if out_tsv is not None:
        out.to_csv(out_tsv, sep="\t", index=False)
        print(f"wrote {out_tsv}  ({len(out)} rows, {int(out['skipped'].sum())} skipped)")
    return out


def run_inference_depth_matched(metadata_tsv, afs_loader, evo_ckpts, phi_ckpts, out_tsv,
                                depths=(30, 50, 100), coverage_col="coverage",
                                cn_col="median_feature_cn", required_cols=None,
                                device="cpu", batch_size=256):
    """Score each sample with the model whose depth is nearest its coverage.

    `evo_ckpts` / `phi_ckpts` are dicts {depth: path}. Produces one
    depth-matched predictions table.
    """
    meta = pd.read_csv(metadata_tsv, sep="\t")
    if coverage_col not in meta.columns:
        raise ValueError(f"coverage_col {coverage_col!r} not in metadata; "
                         f"use run_inference for single-depth scoring")
    deps = np.array(sorted(depths))
    nearest = deps[np.abs(meta[coverage_col].to_numpy()[:, None] - deps[None, :]).argmin(1)]
    meta = meta.assign(_matched_depth=nearest)

    parts = []
    for d in deps:
        sub = meta[meta["_matched_depth"] == d]
        if len(sub) == 0:
            continue
        phi_ckpt = phi_ckpts.get(d) if phi_ckpts else None
        parts.append(run_inference(sub, afs_loader, evo_ckpts[d], phi_ckpt, None,
                                   cn_col=cn_col, required_cols=required_cols,
                                   device=device, batch_size=batch_size))
    out = pd.concat(parts, ignore_index=True)
    if out_tsv is not None:
        out.to_csv(out_tsv, sep="\t", index=False)
        print(f"wrote {out_tsv}  ({len(out)} rows, {int(out['skipped'].sum())} skipped)")
    return out


def ensemble_predictions(pred_paths, out_path, id_col="Feature_ID",
                         param_cols=("phi_hat", "s_hat", "mu_hat", "g_exp")):
    """Average per-feature predictions across seed runs, adding a `{col}_sd` column
    for the across-run spread. Metadata is taken from the first file."""
    dfs = [pd.read_csv(p, sep="\t").drop_duplicates(subset=id_col).set_index(id_col) for p in pred_paths]
    out = dfs[0].copy()
    for col in param_cols:
        present = [d[col] for d in dfs if col in d.columns]
        if present:
            stack = pd.concat(present, axis=1)
            out[col] = stack.mean(axis=1)
            out[f"{col}_sd"] = stack.std(axis=1)
    out = out.reset_index()
    if out_path is not None:
        out.to_csv(out_path, sep="\t", index=False)
        print(f"wrote {out_path}  ({len(pred_paths)} runs averaged)")
    return out


def predict_phi_on_dataset(phi_ckpt, ds_eval, idx, device="cpu"):
    """Apply phi to a simulation subset -> DataFrame(phi_true, phi_hat)."""
    phi = load_phi_model(phi_ckpt, device)
    from train import POP_SIZE
    loader = DataLoader(Subset(ds_eval, idx), batch_size=512, shuffle=False, num_workers=0)
    phi_true, phi_hat = [], []
    with torch.no_grad():
        for X_r1, X_r2, X_cn, _, _, _, _, y_tot in loader:
            ph = phi(X_r1.to(device), X_r2.to(device), X_cn.to(device))
            phi_hat.append(np.clip(ph.cpu().numpy(), 0.0, 1.0))
            phi_true.append(y_tot.numpy() / POP_SIZE)
    return pd.DataFrame({"phi_true": np.concatenate(phi_true),
                         "phi_hat": np.concatenate(phi_hat)})
