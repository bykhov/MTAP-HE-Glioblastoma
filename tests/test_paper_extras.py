"""Tests of analysis/paper_extras.py on small synthetic data (CPU, no network, no real data)."""
import numpy as np
import pandas as pd

from analysis import paper_extras as pe
from analysis.confound_checks import auroc


def _cohort(n=200, seed=0, signal=1.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.6).astype(int)
    x = rng.normal(0, 1, n) + signal * y - 0.3
    return pd.DataFrame({"patient_id": [f"TCGA-00-{i:04d}" for i in range(n)], "label": y,
                         "probability": 1 / (1 + np.exp(-x)), "fold": np.arange(n) % 5,
                         "n_tiles": rng.integers(100, 20000, n), "n_slides": rng.integers(1, 4, n),
                         "tss": rng.choice(["02", "06", "08", "12"], n),
                         "idh_mutant": rng.choice([0.0, 1.0, np.nan], n, p=[0.85, 0.05, 0.1]),
                         "g_cimp": rng.choice([0.0, 1.0], n, p=[0.95, 0.05])})


def test_youden_threshold_separates_perfectly():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = pe.youden_threshold(y, s)
    assert 0.3 < t < 0.7


def test_fit_logistic_recovers_coefficients():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-(0.5 + 1.5 * x)))).astype(int)
    b = pe.fit_logistic(x[:, None], y)
    assert abs(b[0] - 0.5) < 0.15 and abs(b[1] - 1.5) < 0.15


def test_cross_fitted_uses_other_folds_and_recalibrates():
    df = _cohort()
    out, p_cal = pe.cross_fitted(df)
    assert set(out["youden_thresholds_by_test_fold"]) == {0, 1, 2, 3, 4}
    cm = out["youden"]["confusion_matrix"]
    assert sum(cm.values()) == len(df)
    # recalibrated mean prediction is close to the prevalence
    assert abs(out["platt_calibration"]["mean_predicted"] - df.label.mean()) < 0.05
    # the published-threshold block equals probability > 0.5
    assert out["published_threshold_0.5"]["confusion_matrix"]["TP"] == int(((df.probability > 0.5) & (df.label == 1)).sum())


def test_fold_scale_matches_raw_when_folds_share_scale():
    df = _cohort(seed=2)
    out = pe.fold_scale(df)
    assert abs(out["pooled_within_fold_z"]["auroc"] - out["pooled_raw"]["auroc"]) < 0.03


def test_incremental_image_adds_to_uninformative_covariate():
    df = _cohort(seed=3, signal=1.5)
    rng = np.random.default_rng(3)
    cov = pd.DataFrame({"patient_id": df.patient_id, "age": rng.normal(60, 10, len(df)),
                        "expression_subtype": rng.choice(["Classical", "Neural", "Proneural"], len(df))})
    out = pe.incremental(df, cov, n_boot=200, cols_clin=("age", "expression_subtype"))
    assert out["oof_auroc_clinical_plus_image"]["auroc"] > out["oof_auroc_clinical"]["auroc"] + 0.1
    assert out["oof_auroc_difference_ci95_bootstrap"][0] > 0


def test_cox_and_logrank_detect_a_hazard_difference():
    rng = np.random.default_rng(4)
    g = np.repeat([0, 1], 300)
    t = rng.exponential(np.where(g == 1, 0.5, 1.0))
    e = np.ones_like(g)
    b, se, p = pe.cox(t, e, g)
    assert abs(np.exp(b[0]) - 2.0) < 0.4 and p[0] < 1e-6
    chi2, pl = pe.logrank(t, e, g)
    assert pl < 1e-6
    assert pe.km_median(t[g == 0], e[g == 0]) > pe.km_median(t[g == 1], e[g == 1])


def test_robustness_rows_and_purity_egfr():
    df = _cohort(seed=5)
    rng = np.random.default_rng(5)
    clinical = pd.DataFrame({"patient_id": df.patient_id, "age": rng.normal(60, 10, len(df)),
                             "expression_subtype": rng.choice(["Classical", "Neural"], len(df))})
    cov = pd.DataFrame({"patient_id": df.patient_id, "purity": rng.uniform(0.3, 1, len(df)),
                        "egfr_amp": rng.choice([0.0, 1.0], len(df))})
    rows = pe.robustness_rows(df, clinical, cov, recurrent_only=[df.patient_id[0]])
    assert rows.row.iloc[0] == "All patients"
    assert abs(rows.auroc.iloc[0] - round(auroc(df.label, df.probability), 3)) < 1e-9
    assert (rows.lo <= rows.auroc).all() and (rows.auroc <= rows.hi).all()
    pe_out = pe.purity_egfr(df, cov)
    assert {"purity", "egfr"} <= set(pe_out)
