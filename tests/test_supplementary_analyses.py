"""Supplementary analyses and combined IDH calls on synthetic data (no real data, CPU)."""
import importlib.util

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_s1(n=60, seed=0):
    rng = np.random.default_rng(seed)
    label = np.array([i % 3 != 0 for i in range(n)], dtype=int)  # 40 HD, 20 neutral
    prob = np.clip(0.4 + 0.2 * label + rng.normal(0, 0.15, n), 0.01, 0.99)
    return pd.DataFrame({"patient_id": [f"TCGA-{['02', '06'][i % 2]}-{i:04d}" for i in range(n)], "label": label,
                         "n_tiles": rng.integers(100, 9000, n).astype(float), "n_slides": 1 + (np.arange(n) % 4 == 0),
                         "tss": [["02", "06"][i % 2] for i in range(n)], "idh_mutant": np.where(np.arange(n) < 50, 0.0, np.nan),
                         "fold": np.arange(n) % 5, "probability": prob})


def make_clinical(s1, seed=1):
    rng = np.random.default_rng(seed)
    subtypes = np.array(["Classical", "Proneural", "Mesenchymal"])
    return pd.DataFrame({"patient_id": s1.patient_id, "age": rng.integers(30, 80, len(s1)).astype(float),
                         "sex": np.where(np.arange(len(s1)) % 3 == 0, "Female", "Male"),
                         "expression_subtype": subtypes[np.arange(len(s1)) % 3]})


def test_operating_points_reach_their_targets():
    sa = load("supplementary_analyses")
    s1 = make_s1()
    ops = sa.operating_points(s1)
    for row in ops["rule_out"]:
        assert row["sensitivity"] >= row["target_sensitivity"]
        assert row["TP"] + row["FN"] == 40 and row["TN"] + row["FP"] == 20
    for row in ops["rule_in"]:
        assert row["specificity"] >= row["target_specificity"]


def test_stratified_auroc_with_one_stratum_equals_pooled_auroc():
    sa = load("supplementary_analyses")
    s1 = make_s1().assign(stratum="one")
    value, pairs = sa.stratified_auroc(s1, "stratum")
    assert value == pytest.approx(sa.auroc(s1.label, s1.probability))
    assert pairs == 40 * 20


def test_calibration_of_a_perfectly_calibrated_forecast():
    sa = load("supplementary_analyses")
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    df = pd.DataFrame({"label": (rng.uniform(size=p.size) < p).astype(int), "probability": p})
    c = sa.calibration(df)
    assert c["calibration_slope"] == pytest.approx(1.0, abs=0.05)
    assert c["calibration_intercept"] == pytest.approx(0.0, abs=0.05)
    assert c["brier"] < c["brier_prevalence_only"]


def test_run_writes_outputs(tmp_path):
    sa = load("supplementary_analyses")
    s1 = make_s1()
    s1.to_csv(tmp_path / "s1.csv", index=False)
    make_clinical(s1).to_csv(tmp_path / "clinical.csv", index=False)
    summary = sa.run(tmp_path / "s1.csv", tmp_path / "clinical.csv", tmp_path, make_fig=False)
    assert summary["slides_per_patient"]["total_slides"] == int(s1.n_slides.sum())
    assert len(pd.read_csv(tmp_path / "supplementary_S2_Table.csv")) == 5
    table = pd.read_csv(tmp_path / "supplementary_cohort_table.csv")
    assert str(table.iloc[0]["all"]) == "60"
    assert summary["covariate_checks"]["expression_subtype"]["n_known"] == 60


def test_combine_idh_keeps_pancan_call_and_fills_only_missing():
    cc = load("confound_checks")
    pancan = pd.DataFrame({"patient_id": ["A", "B", "C"], "idh_mutant": [0.0, np.nan, np.nan], "idh_detail": ["", "", ""]})
    first = pd.DataFrame({"patient_id": ["A", "B", "X"], "value": [1, 1, 0]})   # A disagrees; X is not in pancan
    second = pd.DataFrame({"patient_id": ["B", "C"], "value": [0, 0]})          # B already filled by `first`
    g = pd.DataFrame({"patient_id": ["C"], "value": [1]})
    out = cc.combine_idh(pancan, [("first", first), ("second", second)], g).set_index("patient_id")
    assert list(out.index) == ["A", "B", "C"]                       # no patients added from the annotations
    assert out.loc["A", "idh_mutant"] == 0 and out.loc["A", "idh_source"] == "pancan_atlas_mutations"
    assert out.loc["A", "idh_disagreement"] == "first=1;"
    assert out.loc["B", "idh_mutant"] == 1 and out.loc["B", "idh_source"] == "first"
    assert out.loc["C", "idh_mutant"] == 0 and out.loc["C", "idh_source"] == "second"
    assert out.loc["C", "g_cimp"] == 1 and np.isnan(out.loc["A", "g_cimp"])
