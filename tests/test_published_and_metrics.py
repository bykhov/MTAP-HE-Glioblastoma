"""scripts/validate_published_predictions.py and analysis/metrics_report.py on a synthetic 233-patient cohort
built to have exactly the confusion matrix of the paper: TP 90, FN 56, TN 59, FP 28."""
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy import stats
from sklearn.metrics import roc_auc_score

from conftest import ROOT
from dataset.mil_dataset import folds_table


def load(relative):
    spec = importlib.util.spec_from_file_location(relative.replace("/", "_").removesuffix(".py"), ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cohort():
    """Manifest, folds table and out-of-fold predictions with TP 90 / FN 56 / TN 59 / FP 28."""
    rng = np.random.default_rng(0)
    ids = [f"TCGA-{(i % 9) + 10:02d}-{i:04d}" for i in range(233)]
    labels = np.array([1] * 146 + [0] * 87)
    manifest = pd.DataFrame({"patient_id": ids, "patient_label": labels,
                             "n_patches": rng.integers(100, 9000, 233), "patient_h5_path": "x"})
    folds = folds_table(manifest, "patient_id", "patient_label", 5, 42)
    prob = np.empty(233)
    prob[:90] = rng.uniform(0.55, 0.95, 90)       # TP
    prob[90:146] = rng.uniform(0.05, 0.45, 56)    # FN
    prob[146:146 + 59] = rng.uniform(0.05, 0.45, 59)  # TN
    prob[146 + 59:] = rng.uniform(0.55, 0.95, 28)     # FP
    pred = pd.DataFrame({"patient_id": ids, "true_label": labels, "prediction": (prob > 0.5).astype(int),
                         "probability": prob, "logit": np.log(prob / (1 - prob))})
    pred["fold"] = folds.set_index("patient_id").loc[pred["patient_id"], "fold"].to_numpy()
    return manifest, folds, pred


def write_folds(pred, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for k, part in pred.groupby("fold"):
        part.drop(columns="fold").to_csv(directory / f"fold{k}_test_predictions.csv", index=False)


# ---------------------------------------------------------------- validation of the deposited predictions
def test_validation_passes_on_predictions_with_the_paper_confusion_matrix(cohort, tmp_path):
    script = load("scripts/validate_published_predictions.py")
    manifest, folds, pred = cohort
    write_folds(pred, tmp_path)
    loaded, files = script.load_published(tmp_path)
    expected = dict(script.EXPECTED, auroc=round(roc_auc_score(pred.true_label, pred.probability), 3))
    report, passed = script.validate(loaded, files, manifest, folds, expected=expected)
    assert passed, report["checks"]
    assert report["confusion_matrix"] == {"TP": 90, "FN": 56, "TN": 59, "FP": 28}


def test_validation_reports_an_auroc_mismatch(cohort, tmp_path):
    script = load("scripts/validate_published_predictions.py")
    manifest, folds, pred = cohort
    write_folds(pred, tmp_path)
    loaded, files = script.load_published(tmp_path)
    expected = dict(script.EXPECTED, auroc=0.5)
    report, passed = script.validate(loaded, files, manifest, folds, expected=expected)
    assert not passed and not report["checks"]["pooled_auroc_matches_paper"]["passed"]
    assert report["checks"]["confusion_matrix_matches_paper"]["passed"]


@pytest.mark.parametrize("damage, failing", [
    ("duplicate", "unique_patients"),
    ("drop", "unique_patients"),
    ("label", "labels_equal_manifest"),
    ("fold", "patients_in_the_fold_of_folds_csv"),
    ("flip", "confusion_matrix_matches_paper"),
    ("range", "probabilities_in_unit_interval"),
])
def test_validation_fails_loudly_on_each_defect(cohort, damage, failing):
    script = load("scripts/validate_published_predictions.py")
    manifest, folds, pred = cohort
    pred = pred.copy()
    files = [f"fold{k}_test_predictions.csv" for k in range(5)]
    if damage == "duplicate":
        pred = pd.concat([pred, pred.iloc[:1]], ignore_index=True)
    elif damage == "drop":
        pred = pred.iloc[1:]
    elif damage == "label":
        pred.loc[0, "true_label"] = 0
    elif damage == "fold":
        pred.loc[0, "fold"] = (pred.loc[0, "fold"] + 1) % 5
    elif damage == "flip":
        pred.loc[0, "probability"] = 0.2
    elif damage == "range":
        pred.loc[0, "probability"] = 1.5
    expected = dict(script.EXPECTED, auroc=round(roc_auc_score(cohort[2].true_label, cohort[2].probability), 3))
    report, passed = script.validate(pred, files, manifest, folds, expected=expected)
    assert not passed and report["checks"][failing]["passed"] is False


def test_columns_are_normalized_and_slide_barcodes_reduced_to_patients(tmp_path):
    script = load("scripts/validate_published_predictions.py")
    raw = pd.DataFrame({"Case_ID": ["TCGA-02-0001-01Z-00-DX1", "TCGA-02-0002-01Z-00-DX1"], "Label": [1, 0],
                        "Prob": [0.8, 0.2]})
    raw.to_csv(tmp_path / "fold3_test_predictions.csv", index=False)
    loaded, files = script.load_published(tmp_path)
    assert loaded["patient_id"].tolist() == ["TCGA-02-0001", "TCGA-02-0002"]
    assert loaded["fold"].tolist() == [3, 3] and loaded["true_label"].tolist() == [1, 0]
    with pytest.raises(ValueError, match="probability"):
        script.normalize(pd.DataFrame({"patient_id": ["a"], "label": [1]}), "x.csv")


def test_cli_exit_codes(cohort, tmp_path, monkeypatch):
    script = load("scripts/validate_published_predictions.py")
    manifest, folds, pred = cohort
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    folds.to_csv(tmp_path / "folds.csv", index=False)
    common = ["v", "--manifest", str(tmp_path / "manifest.csv"), "--folds", str(tmp_path / "folds.csv"),
              "--out", str(tmp_path / "report.json")]
    monkeypatch.setattr("sys.argv", [*common, "--published-dir", str(tmp_path / "empty")])
    with pytest.raises(SystemExit) as nothing:
        script.main()
    assert nothing.value.code == 2  # no prediction files
    write_folds(pred, tmp_path / "published")
    monkeypatch.setattr("sys.argv", [*common, "--published-dir", str(tmp_path / "published")])
    with pytest.raises(SystemExit) as result:
        script.main()
    assert result.value.code == 1  # the confusion matrix matches, but the synthetic AUROC is not 0.704
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["confusion_matrix"]["TP"] == 90 and report["passed"] is False


# ---------------------------------------------------------------- metrics report
@pytest.fixture(scope="module")
def summary(cohort):
    metrics = load("analysis/metrics_report.py")
    result, curves = metrics.summarize(cohort[2], 0.5, n_boot=200, seed=0, fpr_grid=np.linspace(0, 1, 101))
    return metrics, result, curves


def test_threshold_metrics_equal_hand_computed_values(summary):
    _, s, _ = summary
    assert s["confusion_matrix"] == {"TP": 90, "FN": 56, "TN": 59, "FP": 28}
    assert s["sensitivity"]["value"] == pytest.approx(90 / 146) and s["specificity"]["value"] == pytest.approx(59 / 87)
    assert s["ppv"]["value"] == pytest.approx(90 / 118) and s["npv"]["value"] == pytest.approx(59 / 115)
    assert s["accuracy"]["value"] == pytest.approx(149 / 233) and s["f1"] == pytest.approx(180 / 264)
    assert [round(v, 3) for v in (s["sensitivity"]["value"], s["specificity"]["value"], s["npv"]["value"])] == [0.616, 0.678, 0.513]


def test_jeffreys_intervals_match_the_beta_quantiles(summary):
    metrics = summary[0]
    lo, hi = metrics.jeffreys_interval(59, 115)
    assert lo == pytest.approx(stats.beta.ppf(0.025, 59.5, 56.5)) and hi == pytest.approx(stats.beta.ppf(0.975, 59.5, 56.5))
    assert round(lo, 3) == 0.422 and round(hi, 3) == 0.603  # the NPV interval of the paper
    assert metrics.jeffreys_interval(0, 10)[0] == 0.0 and metrics.jeffreys_interval(10, 10)[1] == 1.0


def test_majority_rate_binomial_test_and_always_hd_classifier(summary):
    _, s, _ = summary
    m = s["accuracy_vs_majority_rate"]
    assert m["majority_class"] == "HD" and m["majority_rate"] == pytest.approx(146 / 233)
    assert m["binomial_one_sided_p"] == pytest.approx(stats.binomtest(149, 233, 146 / 233, alternative="greater").pvalue)
    assert round(m["majority_rate"], 3) == 0.627 and round(m["accuracy"], 3) == 0.639
    always = s["always_HD_classifier"]
    assert always["sensitivity"]["value"] == 1.0 and always["specificity"]["value"] == 0.0
    assert always["accuracy"]["value"] == pytest.approx(146 / 233) and always["npv"]["value"] is None


def test_auroc_ci_and_per_fold_values(summary, cohort):
    _, s, curves = summary
    pred = cohort[2]
    assert s["auroc_pooled"] == pytest.approx(roc_auc_score(pred.true_label, pred.probability))
    lo, hi = s["auroc_pooled_ci95_hanley_mcneil"]
    assert 0 < lo < s["auroc_pooled"] < hi < 1
    blo, bhi = s["auroc_pooled_ci95_bootstrap"]
    assert blo < s["auroc_pooled"] < bhi
    per_fold = [roc_auc_score(g.true_label, g.probability) for _, g in pred.groupby("fold")]
    assert s["auroc_per_fold_mean"] == pytest.approx(np.mean(per_fold)) and s["auroc_per_fold_sd"] == pytest.approx(np.std(per_fold, ddof=1))
    z = s["z_test_vs_0.5_hanley_mcneil_se"]
    assert z["z"] == pytest.approx((s["auroc_pooled"] - 0.5) / s["auroc_se_hanley_mcneil"])
    assert z["p_two_sided"] == pytest.approx(2 * stats.norm.sf(abs(z["z"]))) and z["p_two_sided"] < 0.001
    assert s["z_test_vs_0.5_null_se"]["p_two_sided"] < 0.001
    assert curves.shape == (200, 101)


def test_bootstrap_is_reproducible_for_a_seed(cohort):
    metrics = load("analysis/metrics_report.py")
    pred = cohort[2]
    a, _ = metrics.stratified_bootstrap(pred.true_label, pred.probability, 100, seed=3)
    b, _ = metrics.stratified_bootstrap(pred.true_label, pred.probability, 100, seed=3)
    assert a == b


def test_cli_writes_summary_and_a_plos_compliant_single_file_figure(cohort, tmp_path, monkeypatch):
    metrics = load("analysis/metrics_report.py")
    write_folds(cohort[2], tmp_path / "published")
    monkeypatch.setattr("sys.argv", ["m", "--predictions", str(tmp_path / "published" / "fold*_test_predictions.csv"),
                                     "--n-boot", "50", "--out-dir", str(tmp_path / "out")])
    metrics.main()
    data = json.loads((tmp_path / "out" / "metrics_summary.json").read_text())
    assert data["n_patients"] == 233 and data["confusion_matrix"]["TP"] == 90
    tif = tmp_path / "out" / "figures" / "Fig2.tif"
    assert (tmp_path / "out" / "figures" / "Fig2.png").exists()
    with Image.open(tif) as img:
        assert img.mode == "RGB" and img.info["compression"] == "tiff_lzw"
        dpi = img.info["dpi"][0]
        assert 300 <= dpi <= 600
        assert img.width / dpi <= 7.5 + 1e-6 and img.height / dpi <= 8.75
    assert b"\r\n" not in (tmp_path / "out" / "metrics_summary.json").read_bytes()


def test_duplicate_patient_rows_are_rejected(cohort, tmp_path):
    metrics = load("analysis/metrics_report.py")
    pred = pd.concat([cohort[2], cohort[2].iloc[:1]]).drop(columns="fold")
    pred.to_csv(tmp_path / "fold0_test_predictions.csv", index=False)
    with pytest.raises(ValueError, match="more than one"):
        metrics.load_predictions(str(tmp_path / "fold*_test_predictions.csv"))
