"""
Patient-level analyses of the paper (Fig 3, Table 3, S1 Text).

Everything here uses only the deposited out-of-fold predictions (via the per-patient S1 Table) and public
patient-level tables. No model is retrained.

Analyses (keys of ``outputs/paper_extras.json``):
  cross_fitted       threshold (Youden) and Platt recalibration fitted on the other four folds, applied to the
                     held-out fold; threshold metrics against the majority class
  fold_scale         pooled AUROC after label-free standardization of the logit within each test fold
  incremental        out-of-fold AUROC, on the fixed folds, of age + subtype, the image logit, and both
  purity, egfr       ABSOLUTE tumor purity and EGFR amplification as confounders
  survival           overall survival against the image score (Kaplan-Meier, log-rank, Cox)
  cohort             sensitivity analysis without the patients whose copy number 0 comes from a recurrent-tumor
                     sample; breakdown of the eligible patients who are not in the cohort
  robustness         model AUROC under every control (rows of Fig 3A), with 95% CIs

Data fetching (optional, needs network): ``--fetch-extra`` writes ``data/extra_covariates.csv`` from cBioPortal;
``--fetch-gdc`` writes ``data/ambiguous_copy_number.csv`` and ``data/gdc_cohort_slides.csv`` from the GDC API.
Only numpy, pandas, scipy and matplotlib are needed (no scikit-learn).
"""
import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.confound_checks import _r, auroc, hanley_mcneil_ci  # noqa: E402
from analysis.metrics_report import jeffreys_interval  # noqa: E402
from analysis.supplementary_analyses import _boot_stratified_auroc, stratified_auroc  # noqa: E402

CBIO = "https://www.cbioportal.org/api"
GDC = "https://api.gdc.cancer.gov"


# ----------------------------------------------------------------------------------------------------------------
# small estimators
# ----------------------------------------------------------------------------------------------------------------
def _p(p):
    """p-value with three significant digits (so that small values are not rounded to 0)."""
    return float(f"{float(p):.3g}")


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logistic(X, y):
    """Unpenalized logistic regression with intercept (BFGS). Returns the coefficient vector (intercept first)."""
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)

    def nll(b):
        eta = X @ b
        return float(np.sum(np.logaddexp(0, eta) - y * eta))

    def grad(b):
        return X.T @ (1 / (1 + np.exp(-(X @ b))) - y)

    return optimize.minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS").x


def predict_logistic(b, X):
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    return 1 / (1 + np.exp(-(X @ b)))


def loglik_logistic(b, X, y):
    p = np.clip(predict_logistic(b, X), 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def youden_threshold(y, s):
    """Threshold maximizing sensitivity + specificity - 1; midpoint between adjacent distinct scores."""
    y, s = np.asarray(y), np.asarray(s, float)
    u = np.unique(s)
    cuts = np.concatenate([[u[0] - 1e-9], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-9]])
    best, best_j = cuts[0], -np.inf
    for c in cuts:
        pred = s > c
        sens = pred[y == 1].mean()
        spec = (~pred[y == 0]).mean()
        if sens + spec - 1 > best_j + 1e-12:
            best, best_j = c, sens + spec - 1
    return float(best)


def threshold_block(y, pred):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(bool)
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())

    def prop(x, n):
        lo, hi = jeffreys_interval(x, n) if n else (np.nan, np.nan)
        return {"value": _r(x / n) if n else None, "ci95": [_r(lo), _r(hi)], "x": x, "n": n}

    majority = max(y.mean(), 1 - y.mean())
    acc_x = tp + tn
    return {
        "confusion_matrix": {"TP": tp, "FN": fn, "TN": tn, "FP": fp},
        "sensitivity": prop(tp, tp + fn), "specificity": prop(tn, tn + fp),
        "ppv": prop(tp, tp + fp), "npv": prop(tn, tn + fn), "accuracy": prop(acc_x, len(y)),
        "balanced_accuracy": _r((tp / (tp + fn) + tn / (tn + fp)) / 2),
        "f1": _r(2 * tp / (2 * tp + fp + fn)) if tp else 0.0,
        "accuracy_vs_majority_binomial_one_sided_p": _p(stats.binomtest(acc_x, len(y), majority,
                                                                       alternative="greater").pvalue),
    }


def auroc_ci(y, s):
    y = np.asarray(y)
    a = auroc(y, s)
    lo, hi = hanley_mcneil_ci(a, int(y.sum()), int(len(y) - y.sum()))
    return {"auroc": _r(a), "ci95": [_r(lo), _r(hi)], "n_HD": int(y.sum()), "n_neutral": int(len(y) - y.sum())}


def calibration_summary(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    x = logit(p)
    a = optimize.minimize_scalar(lambda a: float(np.sum(np.logaddexp(0, a + x) - y * (a + x)))).x
    b = fit_logistic(x[:, None], y)
    return {"mean_predicted": _r(p.mean()), "prevalence": _r(y.mean()), "brier": _r(np.mean((p - y) ** 2), 4),
            "calibration_intercept": _r(a), "calibration_slope": _r(b[1])}


# ----------------------------------------------------------------------------------------------------------------
# 1. cross-fitted threshold and recalibration
# ----------------------------------------------------------------------------------------------------------------
def cross_fitted(df):
    """
    For each test fold k, a threshold (Youden) and a Platt recalibration (label ~ a + b * logit) are fitted on the
    pooled out-of-fold predictions of the other four folds and applied to fold k. Fold k's own labels are never used
    to set its threshold. The other folds' predictions come from different models, so this approximates a threshold
    chosen on held-out data.
    """
    y = df.label.to_numpy()
    x = logit(df.probability)
    pred_youden = np.zeros(len(df), bool)
    p_cal = np.zeros(len(df))
    thresholds = {}
    for k in sorted(df.fold.unique()):
        te, tr = (df.fold == k).to_numpy(), (df.fold != k).to_numpy()
        t = youden_threshold(y[tr], df.probability.to_numpy()[tr])
        thresholds[int(k)] = _r(t)
        pred_youden[te] = df.probability.to_numpy()[te] > t
        b = fit_logistic(x[tr][:, None], y[tr])
        p_cal[te] = predict_logistic(b, x[te][:, None])
    return {
        "youden_thresholds_by_test_fold": thresholds,
        "youden": threshold_block(y, pred_youden),
        "platt_at_0.5": threshold_block(y, p_cal > 0.5),
        "platt_calibration": calibration_summary(y, p_cal),
        "platt_auroc_pooled": _r(auroc(y, p_cal)),
        "published_threshold_0.5": threshold_block(y, df.probability.to_numpy() > 0.5),
        "always_HD": threshold_block(y, np.ones(len(y), bool)),
    }, p_cal


# ----------------------------------------------------------------------------------------------------------------
# 2. fold-scale check
# ----------------------------------------------------------------------------------------------------------------
def fold_scale(df):
    """Pooled AUROC after z-scoring the logit within each test fold (uses no labels)."""
    x = pd.Series(logit(df.probability), index=df.index)
    z = x.groupby(df.fold).transform(lambda v: (v - v.mean()) / v.std(ddof=1))
    per_fold = [auroc(g.label, g.probability) for _, g in df.groupby("fold")]
    return {"pooled_raw": auroc_ci(df.label, df.probability), "pooled_within_fold_z": auroc_ci(df.label, z),
            "mean_per_fold": _r(np.mean(per_fold)),
            "fold_mean_logit": {int(k): _r(v) for k, v in x.groupby(df.fold).mean().items()}}


# ----------------------------------------------------------------------------------------------------------------
# 3. incremental value on the fixed folds
# ----------------------------------------------------------------------------------------------------------------
def _design(d, cols):
    parts = []
    for c in cols:
        if c == "expression_subtype":
            parts.append(pd.get_dummies(d[c], prefix="st", drop_first=True, dtype=float))
        elif c == "egfr_amp":
            parts.append(d[[c]].astype(float))
        else:
            v = d[[c]].astype(float)
            parts.append((v - v.mean()) / v.std(ddof=0))
    return pd.concat(parts, axis=1).to_numpy()


def oof_logistic(d, cols):
    """Out-of-fold predictions on the fixed test folds; each fold predicted by a model fitted on the other four."""
    X, y = _design(d, cols), d.label.to_numpy()
    out = np.zeros(len(d))
    for k in sorted(d.fold.unique()):
        te, tr = (d.fold == k).to_numpy(), (d.fold != k).to_numpy()
        out[te] = predict_logistic(fit_logistic(X[tr], y[tr]), X[te])
    return out


def incremental(df, cov, n_boot=2000, seed=0, cols_clin=("age", "expression_subtype")):
    cols_clin = list(cols_clin)
    d = df.merge(cov, on="patient_id", how="left").dropna(subset=cols_clin).reset_index(drop=True)
    d["image_logit"] = logit(d.probability)
    y = d.label.to_numpy()
    s_clin = oof_logistic(d, cols_clin)
    s_img = oof_logistic(d, ["image_logit"])
    s_both = oof_logistic(d, cols_clin + ["image_logit"])
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    diffs = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(auroc(y[i], s_both[i]) - auroc(y[i], s_clin[i]))
    X0, X1 = _design(d, cols_clin), _design(d, cols_clin + ["image_logit"])
    lr = 2 * (loglik_logistic(fit_logistic(X1, y), X1, y) - loglik_logistic(fit_logistic(X0, y), X0, y))
    return {
        "covariates": cols_clin, "n": int(len(d)), "n_HD": int(y.sum()),
        "oof_auroc_clinical": auroc_ci(y, s_clin), "oof_auroc_image": auroc_ci(y, s_img),
        "oof_auroc_clinical_plus_image": auroc_ci(y, s_both),
        "oof_auroc_difference": _r(auroc(y, s_both) - auroc(y, s_clin)),
        "oof_auroc_difference_ci95_bootstrap": [_r(np.percentile(diffs, 2.5)), _r(np.percentile(diffs, 97.5))],
        "lr_chi2_in_sample": _r(lr, 2), "lr_p_in_sample": _p(stats.chi2.sf(lr, 1)),
    }


# ----------------------------------------------------------------------------------------------------------------
# 4. purity and EGFR
# ----------------------------------------------------------------------------------------------------------------
def purity_egfr(df, cov):
    d = df.merge(cov, on="patient_id", how="left")
    out = {}
    p = d.dropna(subset=["purity"]).reset_index(drop=True)
    p["purity_tertile"] = pd.qcut(p.purity, 3, labels=False)
    a_strat, pairs = stratified_auroc(p, "purity_tertile")
    x = logit(p.probability)
    X0, X1 = p[["purity"]].to_numpy(), np.column_stack([p.purity, x])
    lr = 2 * (loglik_logistic(fit_logistic(X1, p.label), X1, p.label)
              - loglik_logistic(fit_logistic(X0, p.label), X0, p.label))
    out["purity"] = {
        "n": int(len(p)), "median_HD": _r(p.purity[p.label == 1].median()),
        "median_neutral": _r(p.purity[p.label == 0].median()),
        "mannwhitney_p": _p(stats.mannwhitneyu(p.purity[p.label == 1], p.purity[p.label == 0]).pvalue),
        "auroc_alone": auroc_ci(p.label, p.purity),
        "spearman_purity_vs_probability": _r(stats.spearmanr(p.purity, p.probability)[0]),
        "model_auroc_purity_stratified": _r(a_strat), "pairs": pairs,
        "model_auroc_purity_stratified_ci95": _boot_stratified_auroc(p, "purity_tertile"),
        "model_auroc_overall_same_patients": auroc_ci(p.label, p.probability),
        "lr_logit_added_to_purity_chi2": _r(lr, 2), "lr_p": _p(stats.chi2.sf(lr, 1)),
    }
    e = d.dropna(subset=["egfr_amp"]).reset_index(drop=True)
    tab = pd.crosstab(e.egfr_amp, e.label)
    a_e, pairs_e = stratified_auroc(e, "egfr_amp")
    out["egfr"] = {
        "n": int(len(e)), "amplified_HD": f"{int(tab.loc[1, 1])} of {int(tab[1].sum())}",
        "amplified_neutral": f"{int(tab.loc[1, 0])} of {int(tab[0].sum())}",
        "fisher_p": _p(stats.fisher_exact(tab.to_numpy())[1]),
        "model_auroc_egfr_amplified": auroc_ci(e.label[e.egfr_amp == 1], e.probability[e.egfr_amp == 1]),
        "model_auroc_egfr_not_amplified": auroc_ci(e.label[e.egfr_amp == 0], e.probability[e.egfr_amp == 0]),
        "model_auroc_egfr_stratified": _r(a_e), "pairs": pairs_e,
        "model_auroc_egfr_stratified_ci95": _boot_stratified_auroc(e, "egfr_amp"),
    }
    return out


# ----------------------------------------------------------------------------------------------------------------
# 5. survival
# ----------------------------------------------------------------------------------------------------------------
def kaplan_meier(t, e):
    t, e = np.asarray(t, float), np.asarray(e, int)
    times = np.unique(t[e == 1])
    s, surv = 1.0, []
    for u in times:
        n = (t >= u).sum()
        d = ((t == u) & (e == 1)).sum()
        s *= 1 - d / n
        surv.append(s)
    return times, np.array(surv)


def km_median(t, e):
    times, surv = kaplan_meier(t, e)
    below = np.where(surv <= 0.5)[0]
    return float(times[below[0]]) if len(below) else np.nan


def logrank(t, e, g):
    t, e, g = np.asarray(t, float), np.asarray(e, int), np.asarray(g, int)
    o_minus_e, var = 0.0, 0.0
    for u in np.unique(t[e == 1]):
        at = t >= u
        n, n1 = at.sum(), (at & (g == 1)).sum()
        d = ((t == u) & (e == 1)).sum()
        d1 = ((t == u) & (e == 1) & (g == 1)).sum()
        o_minus_e += d1 - d * n1 / n
        if n > 1:
            var += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    chi2 = o_minus_e ** 2 / var
    return float(chi2), float(stats.chi2.sf(chi2, 1))


def cox(t, e, X):
    """Cox proportional hazards, Breslow ties, Newton-Raphson. Returns coefficients, SEs and p-values."""
    t, e, X = np.asarray(t, float), np.asarray(e, int), np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    order = np.argsort(-t, kind="stable")
    t, e, X = t[order], e[order], X[order]
    b = np.zeros(X.shape[1])
    for _ in range(50):
        w = np.exp(X @ b)
        grad, hess = np.zeros_like(b), np.zeros((len(b), len(b)))
        for u in np.unique(t[e == 1]):
            risk = t >= u
            dead = (t == u) & (e == 1)
            s0 = w[risk].sum()
            s1 = (w[risk, None] * X[risk]).sum(0)
            s2 = (w[risk, None, None] * X[risk, :, None] * X[risk, None, :]).sum(0)
            d = dead.sum()
            grad += X[dead].sum(0) - d * s1 / s0
            hess -= d * (s2 / s0 - np.outer(s1, s1) / s0 ** 2)
        step = np.linalg.solve(hess, grad)
        b = b - step
        if np.max(np.abs(step)) < 1e-9:
            break
    se = np.sqrt(np.diag(np.linalg.inv(-hess)))
    z = b / se
    return b, se, 2 * stats.norm.sf(np.abs(z))


def survival(df, cov, idh_csv=None, clinical=None):
    d = df.merge(cov, on="patient_id", how="left").dropna(subset=["os_months", "os_event"]).reset_index(drop=True)
    if clinical is not None:
        d = d.merge(clinical[["patient_id", "age"]], on="patient_id", how="left")
    d["score_z"] = (logit(d.probability) - logit(d.probability).mean()) / logit(d.probability).std(ddof=0)
    d["high_score"] = (d.probability > d.probability.median()).astype(int)
    t, ev = d.os_months.to_numpy(), d.os_event.to_numpy().astype(int)
    out = {"n": int(len(d)), "events": int(ev.sum()),
           "median_os_months_all": _r(km_median(t, ev), 1)}

    def hr(cols, sub=None):
        dd = d if sub is None else d[sub]
        dd = dd.dropna(subset=cols)
        b, se, p = cox(dd.os_months, dd.os_event, dd[cols].to_numpy())
        return {c: {"hr": _r(np.exp(b[i])), "ci95": [_r(np.exp(b[i] - 1.96 * se[i])), _r(np.exp(b[i] + 1.96 * se[i]))],
                    "p": _p(p[i])} for i, c in enumerate(cols)} | {"n": int(len(dd)), "events": int(dd.os_event.sum())}

    chi_s, p_s = logrank(t, ev, d.high_score)
    chi_l, p_l = logrank(t, ev, d.label)
    out["km_by_score_median_split"] = {
        "median_os_high_score": _r(km_median(t[d.high_score == 1], ev[d.high_score == 1]), 1),
        "median_os_low_score": _r(km_median(t[d.high_score == 0], ev[d.high_score == 0]), 1),
        "logrank_chi2": _r(chi_s, 2), "logrank_p": _p(p_s)}
    out["km_by_label"] = {
        "median_os_HD": _r(km_median(t[d.label == 1], ev[d.label == 1]), 1),
        "median_os_neutral": _r(km_median(t[d.label == 0], ev[d.label == 0]), 1),
        "logrank_chi2": _r(chi_l, 2), "logrank_p": _p(p_l)}
    out["cox_score_per_sd"] = hr(["score_z"])
    out["cox_label"] = hr(["label"])
    if "age" in d:
        d["age_per_10y"] = d.age / 10
        out["cox_score_label_age"] = hr(["score_z", "label", "age_per_10y"])
        if idh_csv is not None and "idh_mutant" in df:
            out["cox_score_label_age_idh_wt_only"] = hr(["score_z", "label", "age_per_10y"],
                                                        sub=(d.idh_mutant == False).to_numpy())  # noqa: E712
    return out, d


# ----------------------------------------------------------------------------------------------------------------
# 6. cohort items
# ----------------------------------------------------------------------------------------------------------------
def cohort_items(df, copy_number_csv, ambiguous_csv=None, gdc_slides_csv=None):
    out = {}
    if ambiguous_csv is not None and Path(ambiguous_csv).exists():
        amb = pd.read_csv(ambiguous_csv)
        out["ambiguous_copy_number_files"] = amb.to_dict(orient="records")
        recurrent_only = sorted(amb.groupby("patient_id").filter(
            lambda g: (g.loc[g.sample_type == "Primary Tumor", "mtap_copy_number"] != 0).all()
            and (g.loc[g.sample_type == "Recurrent Tumor", "mtap_copy_number"] == 0).any()).patient_id.unique())
    else:
        recurrent_only = ["TCGA-06-0152", "TCGA-06-0190"]
    keep = ~df.patient_id.isin(recurrent_only)
    out["excluded_for_sensitivity"] = recurrent_only
    out["sensitivity_without_recurrent_only_hd"] = {
        **auroc_ci(df.label[keep], df.probability[keep]),
        "threshold_0.5": threshold_block(df.label[keep], df.probability[keep] > 0.5)}
    mc = pd.read_csv(copy_number_csv)
    e = mc[mc.has_diagnostic_slide & mc.copy_number.isin([0, 2]) & ~mc.in_cohort]
    out["eligible_not_in_cohort"] = {
        "n": int(len(e)),
        "by_copy_number_and_slide_table": {f"cn{int(k[0])}_{'tiled' if k[1] else 'not_tiled'}": int(v)
                                            for k, v in e.groupby(["copy_number", "in_slide_table"]).size().items()},
        "fisher_p_vs_cohort": _p(stats.fisher_exact([[int((e.copy_number == 0).sum()), int((e.copy_number == 2).sum())],
                                                     [int(df.label.sum()), int((df.label == 0).sum())]])[1])}
    if gdc_slides_csv is not None and Path(gdc_slides_csv).exists():
        s = pd.read_csv(gdc_slides_csv)
        s = s[s.patient_id.isin(df.patient_id)]
        out["gdc_dx_slides_of_cohort_patients"] = int(len(s))
    return out


# ----------------------------------------------------------------------------------------------------------------
# 7. robustness rows (Fig 3A)
# ----------------------------------------------------------------------------------------------------------------
def robustness_rows(df, clinical, cov, recurrent_only):
    d = df.merge(clinical[["patient_id", "age", "expression_subtype"]], on="patient_id", how="left") \
          .merge(cov, on="patient_id", how="left")
    rows = []

    def add(group, label, sub=None, strat=None):
        dd = d if sub is None else d[sub(d)]
        if strat is None:
            r = auroc_ci(dd.label, dd.probability)
            rows.append({"group": group, "row": label, "auroc": r["auroc"], "lo": r["ci95"][0], "hi": r["ci95"][1],
                         "n": int(len(dd)), "kind": "subset", "ci": "Hanley-McNeil"})
        else:
            dd = dd.dropna(subset=[strat]).reset_index(drop=True)
            a, pairs = stratified_auroc(dd, strat)
            lo, hi = _boot_stratified_auroc(dd, strat)
            rows.append({"group": group, "row": label, "auroc": _r(a), "lo": lo, "hi": hi, "n": int(len(dd)),
                         "kind": f"within strata ({pairs} pairs)", "ci": "bootstrap"})

    d["tile_tertile"] = pd.qcut(d.n_tiles, 3, labels=False)
    d["age_tertile"] = pd.qcut(d.age, 3, labels=False)
    d["purity_tertile"] = pd.qcut(d.purity, 3, labels=False)
    add("All patients", "All patients")
    add("Technical", "Same tissue source site", strat="tss")
    add("Technical", "Same tile-count tertile", strat="tile_tertile")
    add("Technical", "One slide only", sub=lambda x: x.n_slides == 1)
    add("Technical", "Primary-tumor copy number only", sub=lambda x: ~x.patient_id.isin(recurrent_only))
    add("Clinical and molecular", "Same age tertile", strat="age_tertile")
    add("Clinical and molecular", "IDH-wild-type", sub=lambda x: x.idh_mutant == False)  # noqa: E712
    add("Clinical and molecular", "Excluding IDH-mutant or G-CIMP",
        sub=lambda x: ~((x.idh_mutant == True) | (x.g_cimp == True)))  # noqa: E712
    add("Clinical and molecular", "Same purity tertile", strat="purity_tertile")
    add("Clinical and molecular", "Same EGFR status", strat="egfr_amp")
    add("Expression program", "Same transcriptional subtype", strat="expression_subtype")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# data fetching (network)
# ----------------------------------------------------------------------------------------------------------------
def _get_json(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def _cbio_attribute(study, attribute, level):
    d = pd.DataFrame(_get_json(f"{CBIO}/studies/{study}/clinical-data?attributeId={attribute}"
                               f"&clinicalDataType={level}&projection=SUMMARY&pageSize=50000"))
    if d.empty:
        return pd.Series(dtype=object, name=attribute)
    if level == "SAMPLE":
        d = d[d.sampleId.str[13:15] == "01"]          # primary tumor sample
    return d.groupby("patientId").value.first().rename(attribute)


def fetch_extra(out_csv, patient_ids):
    """
    ABSOLUTE purity and ploidy of the primary tumor (TCGA pan-glioma study, ``lgggbm_tcga_pub``); GISTIC calls for
    EGFR, MTAP and CDKN2A and overall survival (TCGA PanCanAtlas, ``gbm_tcga_pan_can_atlas_2018``).
    """
    ids = sorted(set(patient_ids))
    out = pd.DataFrame(index=pd.Index(ids, name="patient_id"))
    out["purity"] = pd.to_numeric(_cbio_attribute("lgggbm_tcga_pub", "PURITY_ABSOLUTE", "SAMPLE"), errors="coerce")
    out["ploidy"] = pd.to_numeric(_cbio_attribute("lgggbm_tcga_pub", "ABSOLUTE_EXTRACT_PLOIDY", "SAMPLE"),
                                  errors="coerce")
    g = pd.DataFrame(_get_json(f"{CBIO}/molecular-profiles/gbm_tcga_pan_can_atlas_2018_gistic/molecular-data/fetch"
                               "?projection=SUMMARY",
                               {"sampleListId": "gbm_tcga_pan_can_atlas_2018_cna", "entrezGeneIds": [1956, 4507, 1029]}))
    g = g[g.sampleId.str[13:15] == "01"]
    g["gene"] = g.entrezGeneId.map({1956: "egfr_gistic", 4507: "mtap_gistic", 1029: "cdkn2a_gistic"})
    out = out.join(g.pivot_table(index="patientId", columns="gene", values="value", aggfunc="first"))
    out["egfr_amp"] = np.where(out.egfr_gistic.isna(), np.nan, (out.egfr_gistic == 2).astype(float))
    os_m = pd.to_numeric(_cbio_attribute("gbm_tcga_pan_can_atlas_2018", "OS_MONTHS", "PATIENT"), errors="coerce")
    os_s = _cbio_attribute("gbm_tcga_pan_can_atlas_2018", "OS_STATUS", "PATIENT")
    out["os_months"] = os_m
    out["os_event"] = os_s.reindex(out.index).map(lambda v: np.nan if pd.isna(v) else float(str(v).startswith("1")))
    out.reset_index().to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: purity {out.purity.notna().sum()}, EGFR {out.egfr_amp.notna().sum()}, "
          f"OS {out.os_months.notna().sum()} of {len(out)}")
    return out.reset_index()


def fetch_gdc(ambiguous_csv, slides_csv, patient_ids, ambiguous=("TCGA-06-0152", "TCGA-06-0190")):
    """MTAP and CDKN2A copy number per ABSOLUTE LiftOver file with its sample type; GDC diagnostic slides."""
    def files(filters, fields):
        return _get_json(f"{GDC}/files", {"filters": filters, "fields": ",".join(fields), "format": "JSON",
                                          "size": 10000})["data"]["hits"]

    hits = files({"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.submitter_id", "value": list(ambiguous)}},
        {"op": "=", "content": {"field": "analysis.workflow_type", "value": "ABSOLUTE LiftOver"}},
        {"op": "=", "content": {"field": "data_type", "value": "Gene Level Copy Number"}}]},
        ["file_id", "cases.submitter_id", "cases.samples.submitter_id", "cases.samples.sample_type"])
    rows = []
    for h in hits:
        with urllib.request.urlopen(f"{GDC}/data/{h['file_id']}", timeout=600) as r:
            t = pd.read_csv(io.BytesIO(r.read()), sep="\t")
        s = h["cases"][0]["samples"][0]
        rows.append({"patient_id": h["cases"][0]["submitter_id"], "sample": s["submitter_id"],
                     "sample_type": s["sample_type"], "file_id": h["file_id"],
                     "mtap_copy_number": float(t.loc[t.gene_name == "MTAP", "copy_number"].iloc[0]),
                     "cdkn2a_copy_number": float(t.loc[t.gene_name == "CDKN2A", "copy_number"].iloc[0])})
    pd.DataFrame(rows).sort_values(["patient_id", "sample"]).to_csv(ambiguous_csv, index=False)
    sl = files({"op": "and", "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-GBM"}},
        {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
        {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}}]},
        ["file_id", "file_name", "file_size", "cases.submitter_id"])
    s = pd.DataFrame([{"patient_id": h["cases"][0]["submitter_id"], "file_name": h["file_name"],
                       "file_id": h["file_id"], "file_size": h["file_size"]} for h in sl])
    s[s.patient_id.isin(set(patient_ids))].sort_values("file_name").to_csv(slides_csv, index=False)
    print(f"wrote {ambiguous_csv} ({len(rows)} files) and {slides_csv}")


# ----------------------------------------------------------------------------------------------------------------
def run(root, out_dir, n_boot=2000):
    root, out_dir = Path(root), Path(out_dir)
    df = pd.read_csv(out_dir / "confound_S1_Table.csv", dtype={"tss": str})
    clinical = pd.read_csv(root / "data" / "clinical.csv")
    cov = pd.read_csv(root / "data" / "extra_covariates.csv")
    cf, p_cal = cross_fitted(df)
    coh = cohort_items(df, root / "data" / "mtap_copy_number.csv", root / "data" / "ambiguous_copy_number.csv",
                       root / "data" / "gdc_cohort_slides.csv")
    surv, surv_df = survival(df, cov, idh_csv=True, clinical=clinical)
    summary = {
        "n_patients": int(len(df)),
        "cross_fitted": cf,
        "fold_scale": fold_scale(df),
        "incremental": incremental(df, clinical.merge(cov, on="patient_id", how="left"), n_boot=n_boot),
        "incremental_age": incremental(df, clinical.merge(cov, on="patient_id", how="left"), n_boot=n_boot,
                                       cols_clin=("age",)),
        "incremental_extended": incremental(df, clinical.merge(cov, on="patient_id", how="left"), n_boot=n_boot,
                                            cols_clin=("age", "expression_subtype", "purity", "egfr_amp")),
        **purity_egfr(df, cov),
        "survival": surv,
        "cohort": coh,
    }
    rob = robustness_rows(df, clinical, cov, coh["excluded_for_sensitivity"])
    rob.to_csv(out_dir / "paper_robustness.csv", index=False)
    summary["robustness"] = rob.to_dict(orient="records")
    (out_dir / "paper_extras.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary, rob, surv_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fetch-extra", action="store_true")
    ap.add_argument("--fetch-gdc", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    root = Path(a.root)
    out_dir = Path(a.out_dir) if a.out_dir else root / "outputs"
    ids = pd.read_csv(root / "data" / "cohort.csv").patient_id
    if a.fetch_extra:
        fetch_extra(root / "data" / "extra_covariates.csv", ids)
    if a.fetch_gdc:
        fetch_gdc(root / "data" / "ambiguous_copy_number.csv", root / "data" / "gdc_cohort_slides.csv", ids)
    run(root, out_dir, n_boot=a.n_boot)
    print(f"wrote {out_dir / 'paper_extras.json'} and {out_dir / 'paper_robustness.csv'}")


if __name__ == "__main__":
    main()
