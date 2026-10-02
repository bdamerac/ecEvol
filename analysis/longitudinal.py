"""HMF longitudinal analysis: paired ecDNA parameter change across a patient's timepoints.

ecDNA features are matched across a patient's earliest and latest biopsy by
genomic-interval Jaccard overlap, one pair per patient. Timepoints come from the Hartwig metadata
(patient_token groups a patient's biopsies; tumor_timepoint_num orders them).
"""
import ast
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .common import load_ensemble

JACCARD_THRESHOLD = 0.1
PARAMS = ["phi_hat", "log10mu", "s_hat", "g_exp"]


def _parse_intervals(text):
    out = []
    for interval in ast.literal_eval(text):
        chrom, rng = interval.split(":"); start, end = rng.split("-")
        out.append((chrom, int(start), int(end)))
    return out


def _interval_jaccard(a, b):
    total = lambda ivs: sum(end - start for _, start, end in ivs)
    overlap = 0
    for c1, s1, e1 in a:
        for c2, s2, e2 in b:
            if c1 == c2:
                overlap += max(0, min(e1, e2) - max(s1, s2))
    union = total(a) + total(b) - overlap
    return overlap / union if union > 0 else 0.0


def match_ecdna_across_timepoints(threshold=JACCARD_THRESHOLD, best_only=True):
    """Match ecDNA features between a patient's earliest and latest biopsy.

    Patients carrying several ecDNA yield several matched pairs; with `best_only`
    (the default) only the highest-overlap pair is kept, giving one paired
    observation per patient.
    """
    hmf = load_ensemble("hmf")
    hmf["log10mu"] = np.log10(hmf.mu_hat.clip(lower=1e-3))
    multi = hmf[hmf.within_patient_token_n_samples >= 2].copy()
    multi["intervals_parsed"] = multi.intervals.map(_parse_intervals)
    pairs = []
    for patient, group in multi.groupby("patient_token"):
        timepoints = sorted(group.tumor_timepoint_num.unique())
        if len(timepoints) < 2:
            continue
        earliest = group[group.tumor_timepoint_num == timepoints[0]]
        latest = group[group.tumor_timepoint_num == timepoints[-1]]
        candidates = []
        for ie, re_ in earliest.iterrows():
            for il, rl in latest.iterrows():
                j = _interval_jaccard(re_.intervals_parsed, rl.intervals_parsed)
                if j >= threshold:
                    candidates.append((j, ie, il))
        candidates.sort(reverse=True, key=lambda x: x[0])
        used_early, used_late = set(), set()
        for j, ie, il in candidates:
            if ie in used_early or il in used_late:
                continue
            used_early.add(ie); used_late.add(il)
            pairs.append(dict(patient=patient, jaccard=j,
                              **{f"{p}_t1": earliest.loc[ie, p] for p in PARAMS},
                              **{f"{p}_t2": latest.loc[il, p] for p in PARAMS}))
            if best_only:
                break
    return pd.DataFrame(pairs)


def paired_change_summary(pairs):
    print(f"n = {len(pairs)} patients, median interval overlap {pairs.jaccard.median():.2f}")
    for p in PARAMS:
        v1 = pairs[f"{p}_t1"].values; v2 = pairs[f"{p}_t2"].values
        print(f"  {p:10s} median {np.median(v1):.2f} -> {np.median(v2):.2f}   "
              f"increased in {(v2 > v1).sum()}/{len(pairs)}   p={wilcoxon(v2, v1).pvalue:.3f}")


if __name__ == "__main__":
    paired_change_summary(match_ecdna_across_timepoints())
