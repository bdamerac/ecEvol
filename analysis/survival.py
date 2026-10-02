"""PCAWG overall-survival analysis.

Deceased patients are event=1 at survival_time; alive patients are censored at
interval_of_last_followup; patients missing the relevant time or age at diagnosis are
dropped, with no imputation. The cohort keeps only patients with a single scored
ecDNA.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from lifelines import CoxPHFitter
from scipy.stats import chi2

from .common import TUMOUR_GROUP, zscore

COVARIATES = ["age_z", "male", "log2cn_z"]


def build_patient_table(predictions, single_ecdna=True):
    amp = predictions.copy(); amp["log2cn"] = np.log2(amp["median_feature_cn"])
    per_patient = amp.drop_duplicates("patient_id").set_index("patient_id")
    if single_ecdna:
        n_features = amp.groupby("patient_id").size()
        amp = amp[amp.patient_id.isin(n_features[n_features == 1].index)]
        table = amp.copy()
    else:
        table = amp.loc[amp.groupby("patient_id").phi_hat.idxmax()].copy()
    for c in ["donor_vital_status", "donor_survival_time", "donor_interval_of_last_followup",
              "donor_age_at_diagnosis", "donor_sex"]:
        if c not in table:
            table[c] = table.patient_id.map(per_patient[c])
    table["event"] = np.where(table.donor_vital_status == "deceased", 1.0,
                              np.where(table.donor_vital_status == "alive", 0.0, np.nan))
    table["time"] = np.where(table.donor_vital_status == "deceased", table.donor_survival_time,
                             table.donor_interval_of_last_followup)
    table = table[table.time.notna() & table.event.notna() & (table.time > 0)
                  & table.donor_age_at_diagnosis.notna()].copy()
    table["strata"] = table.tumor_type.map(TUMOUR_GROUP).fillna("Other")
    table["age_z"] = zscore(table.donor_age_at_diagnosis)
    table["male"] = (table.donor_sex == "male").astype(int)
    table["log2cn_z"] = zscore(table.log2cn)
    return table


def fit_cox(table, term, extra=COVARIATES, strata="strata"):
    cols = [term] + extra + ["time", "event"] + ([strata] if strata else [])
    model = CoxPHFitter().fit(table[cols], "time", "event", strata=[strata] if strata else None)
    r = model.summary.loc[term]
    return dict(hr=np.exp(r["coef"]), lo=np.exp(r["coef"] - 1.96 * r["se(coef)"]),
                hi=np.exp(r["coef"] + 1.96 * r["se(coef)"]), p=r["p"], loglik=model.log_likelihood_)


def cox_forest_terms(table):
    t = table.copy()
    t["inv_g_z"] = zscore(1 / t.g_exp)
    t["phi_over_g_z"] = zscore(t.phi_hat / t.g_exp)
    t["s_phi_over_g_z"] = zscore(t.s_hat * t.phi_hat / t.g_exp)
    rows = []
    for name, col in [("s*phi/g", "s_phi_over_g_z"), ("phi/g", "phi_over_g_z"), ("1/g", "inv_g_z")]:
        rows.append(dict(name=name, **fit_cox(t, col)))
    return pd.DataFrame(rows)


def nested_likelihood_ratio(table):
    t = table.copy()
    t["timing"] = zscore(np.exp(-t.g_exp)); t["phi_z"] = zscore(t.phi_hat); t["s_z"] = zscore(t.s_hat)
    loglik = lambda cols: CoxPHFitter().fit(
        t[cols + COVARIATES + ["time", "event", "strata"]], "time", "event", strata=["strata"]).log_likelihood_
    l0, lg, lgp, lgps = loglik([]), loglik(["timing"]), loglik(["timing", "phi_z"]), loglik(["timing", "phi_z", "s_z"])
    for label, full, reduced in [("timing vs base", lg, l0), ("+phi | timing", lgp, lg), ("+s | timing,phi", lgps, lgp)]:
        stat = 2 * (full - reduced)
        print(f"  {label:16s} chi2={stat:6.2f} p={chi2.sf(stat, 1):.4f}")
