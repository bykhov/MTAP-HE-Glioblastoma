"""
Validate the deposited out-of-fold predictions of the reported run.

Reads ``<published-dir>/fold{k}_test_predictions.csv`` (k = 0..4), normalizes the column names
(``patient_id, true_label, prediction, probability, logit, fold``; ``fold`` comes from the file name) and
checks the table against the patient manifest and ``outputs/folds.csv``:

  1. 233 unique patients, none duplicated across folds, none missing, none unknown;
  2. the label of every patient equals the manifest label;
  3. every patient is in the fold that ``folds.csv`` assigns (``fold`` = the run in which the patient is tested);
  4. probabilities lie in [0, 1];
  5. confusion matrix at the threshold and pooled AUROC equal the paper (TP 90, FN 56, TN 59, FP 28; AUROC 0.704).

The report lists every check; the exit code is 1 if any check fails.

    python scripts/validate_published_predictions.py \
        --manifest source/precomputed_features/uni-patient-level/feature_manifest.parquet \
        [--published-dir outputs/published] [--folds outputs/folds.csv] [--out outputs/published_validation.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ALIASES = {
    "patient_id": ["patient_id", "case_id", "patient", "case_submitter_id", "id"],
    "true_label": ["true_label", "label", "target", "y_true", "true"],
    "prediction": ["prediction", "pred", "predicted_label", "predicted", "y_pred"],
    "probability": ["probability", "prob", "probs", "y_prob", "score", "probability_hd"],
    "logit": ["logit", "logits"],
}
EXPECTED = {"n_patients": 233, "TP": 90, "FN": 56, "TN": 59, "FP": 28, "auroc": 0.704}


def patient_barcode(x):
    """TCGA-02-0001-01Z-00-DX1 -> TCGA-02-0001."""
    m = re.match(r"^(TCGA-[0-9A-Za-z]{2}-[0-9A-Za-z]{4})", str(x))
    return m.group(1) if m else str(x)


def normalize(df, source):
    """Rename columns to the standard schema; raise if patient_id, true_label or probability cannot be found."""
    lower = {c.lower().strip(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for target, names in ALIASES.items():
        found = next((lower[n] for n in names if n in lower), None)
        if found is not None:
            out[target] = df[found]
        elif target in ("patient_id", "true_label", "probability"):
            raise ValueError(f"{source}: no column for {target}; looked for {names}, have {list(df.columns)}")
    out["patient_id"] = out["patient_id"].map(patient_barcode)
    out["true_label"] = out["true_label"].astype(int)
    out["probability"] = out["probability"].astype(float)
    return out


def load_published(published_dir):
    files = sorted(Path(published_dir).glob("fold*_test_predictions.csv"))
    frames = []
    for path in files:
        k = int(re.search(r"fold[_-]?(\d+)", path.name).group(1))
        frame = normalize(pd.read_csv(path), path.name)
        frame["fold"] = k
        frame["_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None, [p.name for p in files]


def auroc(y, s):
    from scipy import stats

    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    return float(stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))


def validate(pred, files, manifest, folds, threshold=0.5, expected=EXPECTED):
    """Returns ``(report dict, passed bool)``."""
    checks = {}

    def add(name, passed, detail):
        checks[name] = {"passed": bool(passed), "detail": detail}

    add("five_fold_files", files == [f"fold{k}_test_predictions.csv" for k in range(5)], files)
    n_unique = int(pred["patient_id"].nunique())
    duplicated = sorted(pred.loc[pred["patient_id"].duplicated(keep=False), "patient_id"].unique())
    add("unique_patients", n_unique == expected["n_patients"] and not duplicated,
        {"unique": n_unique, "rows": int(len(pred)), "duplicated_across_or_within_folds": duplicated[:10]})
    man = manifest.set_index("patient_id")["patient_label"].astype(int)
    unknown = sorted(set(pred["patient_id"]) - set(man.index))
    missing = sorted(set(man.index) - set(pred["patient_id"]))
    add("same_patients_as_manifest", not unknown and not missing,
        {"not_in_manifest": unknown[:10], "missing_from_predictions": missing[:10]})
    known = pred[pred["patient_id"].isin(man.index)]
    wrong = known[known["true_label"].to_numpy() != man.reindex(known["patient_id"]).to_numpy()]
    add("labels_equal_manifest", len(wrong) == 0, {"mismatches": wrong["patient_id"].tolist()[:10]})
    assigned = folds.set_index("patient_id")["fold"]
    comparable = pred[pred["patient_id"].isin(assigned.index)]
    off = comparable[comparable["fold"].to_numpy() != assigned.reindex(comparable["patient_id"]).to_numpy()]
    add("patients_in_the_fold_of_folds_csv", len(off) == 0 and len(comparable) == len(pred),
        {"in_a_different_fold": len(off), "examples": off[["patient_id", "fold"]].head(10).to_dict("records"),
         "not_in_folds_csv": int(len(pred) - len(comparable))})
    add("probabilities_in_unit_interval", bool(pred["probability"].between(0, 1).all()), None)

    predicted = (pred["probability"] > threshold).astype(int)
    y = pred["true_label"]
    confusion = {"TP": int(((predicted == 1) & (y == 1)).sum()), "FN": int(((predicted == 0) & (y == 1)).sum()),
                 "TN": int(((predicted == 0) & (y == 0)).sum()), "FP": int(((predicted == 1) & (y == 0)).sum())}
    pooled = auroc(y, pred["probability"]) if y.nunique() == 2 else float("nan")
    add("confusion_matrix_matches_paper", all(confusion[k] == expected[k] for k in ("TP", "FN", "TN", "FP")),
        {"observed": confusion, "paper": {k: expected[k] for k in ("TP", "FN", "TN", "FP")}})
    add("pooled_auroc_matches_paper", round(pooled, 3) == expected["auroc"],
        {"observed": round(pooled, 4), "paper": expected["auroc"]})
    if "prediction" in pred:  # informational: the stored hard prediction versus probability > threshold
        disagree = int((pred["prediction"].astype(int) != predicted).sum())
        add("stored_prediction_equals_probability_above_threshold", disagree == 0, {"disagreements": disagree})
    report = {"threshold": threshold, "n_rows": int(len(pred)), "confusion_matrix": confusion,
              "pooled_auroc": pooled, "checks": checks, "passed": all(c["passed"] for c in checks.values())}
    return report, report["passed"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="patient-level feature_manifest.parquet or CSV")
    ap.add_argument("--published-dir", default=str(ROOT / "outputs" / "published"))
    ap.add_argument("--folds", default=str(ROOT / "outputs" / "folds.csv"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "published_validation.json"))
    args = ap.parse_args()

    pred, files = load_published(args.published_dir)
    if pred is None:
        print(f"No fold*_test_predictions.csv files in {args.published_dir}.")
        sys.exit(2)
    manifest = pd.read_parquet(args.manifest) if args.manifest.endswith(".parquet") else pd.read_csv(args.manifest)
    report, passed = validate(pred, files, manifest, pd.read_csv(args.folds), args.threshold)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    for name, check in report["checks"].items():
        print(f"[{'ok  ' if check['passed'] else 'FAIL'}] {name}: {check['detail']}")
    print("confusion matrix", report["confusion_matrix"], "pooled AUROC", round(report["pooled_auroc"], 4))
    print("ALL CHECKS PASSED" if passed else "CHECKS FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
