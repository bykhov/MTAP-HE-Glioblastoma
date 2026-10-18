"""Confounder checks on a synthetic 50-patient cohort (no real data, CPU)."""
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT


def load_module():
    spec = importlib.util.spec_from_file_location("confound_checks", ROOT / "analysis" / "confound_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_cohort(sites, seed=0, tile_effect=False):
    """50 patients; `sites` is a list of tissue source site codes cycled over the patients."""
    rng = np.random.default_rng(seed)
    labels = np.array([i % 2 for i in range(50)])
    tiles = rng.integers(500, 5000, 50)
    if tile_effect:
        tiles = tiles + labels * 20000  # tile count separates the classes perfectly
    ids = [f"TCGA-{sites[i % len(sites)]}-{i:04d}" for i in range(50)]
    df = pd.DataFrame({"patient_id": ids, "patient_label": labels, "n_patches": tiles, "n_slides": 1})
    scores = np.clip(0.35 + 0.3 * labels + rng.normal(0, 0.2, 50), 0.01, 0.99)
    pred = pd.DataFrame({"patient_id": ids, "true_label": labels, "probability": scores, "fold": np.arange(50) % 5})
    return df, pred


def write_inputs(tmp_path, df, pred):
    manifest = tmp_path / "manifest.csv"
    df.to_csv(manifest, index=False)
    for k, part in pred.groupby("fold"):
        part.to_csv(tmp_path / f"fold{k}_test_predictions.csv", index=False)
    return str(manifest), str(tmp_path / "fold*_test_predictions.csv")


def test_single_site_stratified_auroc_equals_pooled_auroc(tmp_path):
    cc = load_module()
    df, pred = make_cohort(["06"])
    manifest, pattern = write_inputs(tmp_path, df, pred)
    loaded, predictions = cc.load_manifest(manifest), cc.load_predictions(pattern)
    summary, _ = cc.check_site(loaded, predictions)
    pooled = cc.overall(predictions)["auroc_pooled"]
    assert summary["n_sites"] == 1
    assert summary["model_auroc_site_stratified"] == pytest.approx(pooled, abs=1e-3)


def test_site_stratified_auroc_uses_only_same_site_pairs():
    cc = load_module()
    # site A: model ranks perfectly (AUROC 1); site B: model ranks backwards (AUROC 0); equal numbers of pairs
    df = pd.DataFrame({"patient_id": ["TCGA-AA-0001", "TCGA-AA-0002", "TCGA-BB-0003", "TCGA-BB-0004"],
                       "label": [1, 0, 1, 0], "n_tiles": [10.0] * 4})
    df["tss"] = df.patient_id.map(cc.tss_from_barcode)
    pred = df.rename(columns={"label": "true_label"}).assign(probability=[0.9, 0.1, 0.1, 0.9])[
        ["patient_id", "true_label", "probability"]]
    summary, _ = cc.check_site(df, pred)
    assert summary["model_auroc_site_stratified"] == 0.5 and summary["site_stratified_pairs"] == 2
    # pooled AUROC ignores the sites: 2 of 4 positive-negative pairs are ordered correctly -> 0.5 as well,
    # but the within-site values show the structure
    assert summary["model_auroc_within_site"] == {"AA": 1.0, "BB": 0.0}


def test_tile_count_alone_detects_a_tile_count_confound():
    cc = load_module()
    df, pred = make_cohort(["06", "12"], tile_effect=True)
    result = cc.check_tiles(_manifest(cc, df), None)
    assert result["auroc_tilecount_alone"] == 1.0
    assert result["auroc_tilecount_alone_95ci"][0] > 0.9


def _manifest(cc, df, tmp=None):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.csv"
        df.to_csv(path, index=False)
        return cc.load_manifest(str(path))


def test_tile_count_independent_of_label_is_near_chance():
    cc = load_module()
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({"patient_id": [f"TCGA-06-{i:04d}" for i in range(n)], "patient_label": rng.integers(0, 2, n),
                       "n_patches": rng.integers(100, 9000, n)})
    result = cc.check_tiles(_manifest(cc, df), None)
    assert 0.44 < result["auroc_tilecount_alone"] < 0.56
    assert result["mannwhitney_p"] > 0.01


def test_site_alone_leave_one_out_has_no_self_leakage():
    cc = load_module()
    # one patient per site: the leave-one-out score is the prior for everybody, so AUROC is exactly 0.5
    n = 20
    df = pd.DataFrame({"patient_id": [f"TCGA-{i:02d}-0001" for i in range(n)], "label": [i % 2 for i in range(n)],
                       "n_tiles": 1.0})
    df["tss"] = df.patient_id.map(cc.tss_from_barcode)
    summary, _ = cc.check_site(df, None)
    assert summary["auroc_site_alone_loo"] == 0.5


def test_idh_counts_missing_calls_and_fisher(tmp_path):
    cc = load_module()
    df, _ = make_cohort(["06"])
    manifest = cc.load_manifest(_write(tmp_path / "m.csv", df))
    idh = pd.DataFrame({"patient_id": manifest.patient_id, "idh_mutant": np.nan})
    idh.loc[:9, "idh_mutant"] = 0.0           # 10 wild-type
    idh.loc[10:11, "idh_mutant"] = 1.0        # 2 mutant
    # rows 12 onward: no call (not sequenced)
    path = tmp_path / "idh.csv"
    idh.to_csv(path, index=False)
    result, table = cc.check_idh(manifest, None, str(path))
    assert result["n_idh_known"] == 12 and result["n_idh_unknown"] == 38 and result["n_idh_mutant"] == 2
    assert 0 <= result["fisher_p"] <= 1
    assert table["idh_mutant"].isna().sum() == 38  # unknown stays unknown, it is never treated as wild-type


def _write(path, df):
    df.to_csv(path, index=False)
    return str(path)


def test_cli_without_predictions_writes_summary_sites_and_s1(tmp_path, monkeypatch):
    cc = load_module()
    df, pred = make_cohort(["06", "12", "02"])
    manifest = _write(tmp_path / "manifest.csv", df)
    idh = pd.DataFrame({"patient_id": df.patient_id, "idh_mutant": 0.0})
    idh_path = _write(tmp_path / "idh.csv", idh)
    out = str(tmp_path / "confound")
    monkeypatch.setattr("sys.argv", ["confound_checks.py", "--manifest", manifest, "--idh", idh_path, "--out", out])
    cc.main()
    summary = json.loads((tmp_path / "confound_summary.json").read_text())
    assert summary["n_patients"] == 50 and summary["idh"]["n_idh_unknown"] == 0
    assert "overall" not in summary and "model_auroc_by_tertile" not in summary["tiles"]
    assert (tmp_path / "confound_sites.csv").exists() and (tmp_path / "confound_S1_Table.csv").exists()


def test_cli_with_predictions_adds_model_auroc_under_control(tmp_path, monkeypatch):
    cc = load_module()
    df, pred = make_cohort(["06", "12", "02"])
    manifest, pattern = write_inputs(tmp_path, df, pred)
    idh_path = _write(tmp_path / "idh.csv", pd.DataFrame({"patient_id": df.patient_id, "idh_mutant": 0.0}))
    out = str(tmp_path / "confound")
    monkeypatch.setattr("sys.argv", ["confound_checks.py", "--manifest", manifest, "--predictions", pattern,
                                     "--idh", idh_path, "--out", out])
    cc.main()
    summary = json.loads((tmp_path / "confound_summary.json").read_text())
    assert set(summary["tiles"]["model_auroc_by_tertile"]) == {"low", "mid", "high"}
    assert summary["site"]["model_auroc_site_stratified"] is not None
    # all patients are IDH wild type, so the IDH-wt AUROC is the pooled AUROC
    assert summary["idh"]["model_auroc_idh_wt"] == summary["overall"]["auroc_pooled"]


def test_fetch_idh_marks_unsequenced_patients_as_unknown(tmp_path, monkeypatch):
    """cBioPortal is mocked: 0001 mutant, 0002 sequenced wild type, 0003 not sequenced, 0004 only a silent mutation."""
    cc = load_module()
    import requests

    class Reply:
        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    def fake_get(url, **kwargs):
        if url.endswith("/samples"):
            return Reply([{"patientId": f"TCGA-AA-000{i}"} for i in (1, 2, 3, 4)])
        return Reply(["TCGA-AA-0001-01", "TCGA-AA-0002-01", "TCGA-AA-0004-01"])

    def fake_post(url, **kwargs):
        return Reply([{"patientId": "TCGA-AA-0001", "mutationType": "Missense_Mutation", "proteinChange": "R132H",
                       "gene": {"hugoGeneSymbol": "IDH1"}},
                      {"patientId": "TCGA-AA-0004", "mutationType": "Silent", "proteinChange": "", "gene": {"hugoGeneSymbol": "IDH2"}}])

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    out = tmp_path / "idh.csv"
    cc.fetch_idh(str(out))
    table = pd.read_csv(out).set_index("patient_id")
    assert table.loc["TCGA-AA-0001", "idh_mutant"] == 1 and table.loc["TCGA-AA-0002", "idh_mutant"] == 0
    assert np.isnan(table.loc["TCGA-AA-0003", "idh_mutant"]) and table.loc["TCGA-AA-0004", "idh_mutant"] == 0
