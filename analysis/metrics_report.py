"""
Metrics report: the discrimination and threshold metrics of the paper (Table 1) and Fig 2 from the out-of-fold
predictions.

Input: ``fold{k}_test_predictions.csv`` files (columns ``patient_id, true_label, probability`` and optionally
``fold``), by default ``outputs/published/``. Class 1 (HD) is the positive class.

Writes
  ``<out-dir>/metrics_summary.json``   pooled AUROC with Hanley-McNeil 95% CI and z-tests against 0.5, a stratified
                                        patient bootstrap CI (2,000 resamples), per-fold AUROC mean and SD, and at the
                                        threshold: confusion matrix, sensitivity, specificity, PPV, NPV, accuracy
                                        (Jeffreys 95% intervals), F1, a one-sided exact binomial test of accuracy against
                                        the majority-class rate, and the same metrics for the "always HD" classifier
  ``<out-dir>/figures/Fig2.tif``       one combined figure (PLOS ONE): A pooled ROC with the five per-fold curves,
                                        bootstrap band and chance line; B threshold
                                        metrics against the always-HD classifier, and PPV / NPV against prevalence
  ``<out-dir>/figures/Fig2.png``       preview

    python analysis/metrics_report.py --predictions "outputs/published/fold*_test_predictions.csv" \
        [--threshold 0.5] [--out-dir outputs]

The order of the keys in ``metrics_summary.json`` follows Table 1.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.confound_checks import auroc, hanley_mcneil_ci  # noqa: E402  (same estimators as the confounder checks)


def load_predictions(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no prediction files match {pattern!r}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if "fold" not in frame.columns:
            frame["fold"] = int(re.search(r"fold[_-]?(\d+)", Path(path).name).group(1))
        frames.append(frame[["patient_id", "true_label", "probability", "fold"]])
    pred = pd.concat(frames, ignore_index=True)
    if not pred["patient_id"].is_unique:
        raise ValueError("a patient occurs in more than one prediction row; expected one out-of-fold row per patient")
    pred["true_label"] = pred["true_label"].astype(int)
    return pred


def jeffreys_interval(x, n, level=0.95):
    """Jeffreys interval for a proportion x / n (Beta(x + 1/2, n - x + 1/2) quantiles; end points fixed at 0 and 1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    a = (1 - level) / 2
    lo = 0.0 if x == 0 else float(stats.beta.ppf(a, x + 0.5, n - x + 0.5))
    hi = 1.0 if x == n else float(stats.beta.ppf(1 - a, x + 0.5, n - x + 0.5))
    return (lo, hi)


def _ratio(x, n):
    """Proportion with its Jeffreys interval; None if the denominator is zero (the metric is undefined)."""
    if n == 0:
        return {"value": None, "ci95": None, "x": int(x), "n": 0}
    lo, hi = jeffreys_interval(x, n)
    return {"value": float(x / n), "ci95": [lo, hi], "x": int(x), "n": int(n)}


def confusion(y, predicted):
    y, predicted = np.asarray(y).astype(int), np.asarray(predicted).astype(int)
    return {"TP": int(((predicted == 1) & (y == 1)).sum()), "FN": int(((predicted == 0) & (y == 1)).sum()),
            "TN": int(((predicted == 0) & (y == 0)).sum()), "FP": int(((predicted == 1) & (y == 0)).sum())}


def threshold_metrics(y, predicted):
    c = confusion(y, predicted)
    tp, fn, tn, fp = c["TP"], c["FN"], c["TN"], c["FP"]
    n = tp + fn + tn + fp
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return {"confusion_matrix": c,
            "sensitivity": _ratio(tp, tp + fn), "specificity": _ratio(tn, tn + fp),
            "ppv": _ratio(tp, tp + fp), "npv": _ratio(tn, tn + fn), "accuracy": _ratio(tp + tn, n),
            "f1": f1}


def stratified_bootstrap(y, s, n_boot=2000, seed=0, fpr_grid=None):
    """Resample positives and negatives separately. Returns AUROC percentile CI and the TPR curves on ``fpr_grid``."""
    from sklearn.metrics import roc_curve

    rng = np.random.default_rng(seed)
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs, curves = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs.append(auroc(y[idx], s[idx]))
        if fpr_grid is not None:
            fpr, tpr, _ = roc_curve(y[idx], s[idx])
            curves.append(np.interp(fpr_grid, fpr, tpr))
    ci = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    return ci, (np.array(curves) if curves else None)


def summarize(pred, threshold=0.5, n_boot=2000, seed=0, fpr_grid=None):
    """Returns ``(summary dict, bootstrap ROC curves or None)``; keys follow the order of Table 1."""
    y, s = pred["true_label"].to_numpy(), pred["probability"].to_numpy()
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    a = auroc(y, s)
    lo, hi = hanley_mcneil_ci(a, n1, n0)
    q1, q2 = a / (2 - a), 2 * a ** 2 / (1 + a)
    se = float(np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a ** 2) + (n0 - 1) * (q2 - a ** 2)) / (n1 * n0)))
    se_null = float(np.sqrt((n1 + n0 + 1) / (12 * n1 * n0)))
    boot_ci, curves = stratified_bootstrap(y, s, n_boot, seed, fpr_grid)
    per_fold = {int(k): float(auroc(g["true_label"], g["probability"])) for k, g in pred.groupby("fold")}
    values = np.array(list(per_fold.values()))

    summary = {
        "n_patients": int(len(pred)), "n_HD": n1, "n_neutral": n0, "threshold": threshold,
        "auroc_pooled": float(a), "auroc_pooled_ci95_hanley_mcneil": [lo, hi], "auroc_se_hanley_mcneil": se,
        "z_test_vs_0.5_hanley_mcneil_se": {"z": (a - 0.5) / se, "p_two_sided": float(2 * stats.norm.sf(abs(a - 0.5) / se))},
        "z_test_vs_0.5_null_se": {"z": (a - 0.5) / se_null, "p_two_sided": float(2 * stats.norm.sf(abs(a - 0.5) / se_null))},
        "auroc_pooled_ci95_bootstrap": boot_ci, "bootstrap_resamples": n_boot, "bootstrap_seed": seed,
        "auroc_per_fold": per_fold,
        "auroc_per_fold_mean": float(values.mean()), "auroc_per_fold_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
    }
    predicted = (s > threshold).astype(int)
    model = threshold_metrics(y, predicted)
    summary.update(model)

    majority = max(n1, n0) / len(pred)
    correct = model["confusion_matrix"]["TP"] + model["confusion_matrix"]["TN"]
    test = stats.binomtest(correct, len(pred), p=majority, alternative="greater")
    summary["accuracy_vs_majority_rate"] = {
        "majority_class": "HD" if n1 >= n0 else "neutral", "majority_rate": float(majority),
        "accuracy": float(correct / len(pred)), "binomial_one_sided_p": float(test.pvalue)}
    summary["always_HD_classifier"] = threshold_metrics(y, np.ones_like(y))
    return summary, curves


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def make_figure(pred, summary, curves, fpr_grid, out_dir, dpi=300):
    """Fig 2 as one RGB LZW TIFF (7.5 x 3.6 in) plus a PNG preview."""
    from sklearn.metrics import roc_curve

    plt = _setup_matplotlib()
    HD, GR, OR = "#1f4e79", "#8a8a8a", "#c47a1f"
    fig = plt.figure(figsize=(7.5, 3.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.15])
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b, ax_c = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])

    y, s = pred["true_label"].to_numpy(), pred["probability"].to_numpy()
    if curves is not None:
        ax_a.fill_between(fpr_grid, np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0),
                          color=HD, alpha=0.15, lw=0, label="Bootstrap 95% band")
    for k, g in pred.groupby("fold"):
        fpr, tpr, _ = roc_curve(g["true_label"], g["probability"])
        ax_a.plot(fpr, tpr, color=GR, lw=0.7, alpha=0.8, label="Folds" if k == pred["fold"].min() else None)
    fpr, tpr, _ = roc_curve(y, s)
    lo, hi = summary["auroc_pooled_ci95_hanley_mcneil"]
    ax_a.plot(fpr, tpr, color=HD, lw=1.6, label=f"Pooled, AUROC {summary['auroc_pooled']:.3f} ({lo:.3f}–{hi:.3f})")
    ax_a.plot([0, 1], [0, 1], color=GR, lw=0.8, ls=":", label="Chance")
    ax_a.set_xlabel("1 − specificity"); ax_a.set_ylabel("Sensitivity"); ax_a.set_aspect("equal")
    ax_a.legend(frameon=False, fontsize=6, loc="lower right")

    names = ["Sensitivity", "Specificity", "PPV", "NPV", "Accuracy"]
    keys = ["sensitivity", "specificity", "ppv", "npv", "accuracy"]
    x = np.arange(len(keys))
    model = [summary[k]["value"] for k in keys]
    err = np.array([[summary[k]["value"] - summary[k]["ci95"][0], summary[k]["ci95"][1] - summary[k]["value"]] for k in keys]).T
    always = [(summary["always_HD_classifier"][k]["value"] or 0.0) for k in keys]
    ax_b.bar(x - 0.2, model, 0.38, yerr=err, color=HD, capsize=2, error_kw={"lw": 0.7}, label=f"Model (threshold {summary['threshold']})")
    ax_b.bar(x + 0.2, always, 0.38, color=GR, alpha=0.6, label="Always HD")
    ax_b.set_xticks(x, names, fontsize=6.5); ax_b.set_ylim(0, 1.05); ax_b.set_ylabel("Value")
    ax_b.legend(frameon=False, fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))

    sens, spec = summary["sensitivity"]["value"], summary["specificity"]["value"]
    prevalence = np.linspace(0.02, 0.98, 97)
    ppv = sens * prevalence / (sens * prevalence + (1 - spec) * (1 - prevalence))
    npv = spec * (1 - prevalence) / (spec * (1 - prevalence) + (1 - sens) * prevalence)
    ax_c.plot(prevalence, ppv, color=HD, lw=1.2, label="PPV"); ax_c.plot(prevalence, npv, color=OR, lw=1.2, label="NPV")
    observed = summary["n_HD"] / summary["n_patients"]
    ax_c.axvline(observed, color=GR, lw=0.8, ls=":"); ax_c.text(observed + 0.01, 0.05, "this cohort", fontsize=6, color=GR)
    ax_c.set_xlabel("HD prevalence"); ax_c.set_ylabel("Predictive value"); ax_c.set_ylim(0, 1.02)
    ax_c.legend(frameon=False, fontsize=6, loc="center right")

    ax_a.text(-0.12, 1.02, "A", transform=ax_a.transAxes, fontweight="bold", fontsize=10)
    ax_b.text(-0.12, 1.12, "B", transform=ax_b.transAxes, fontweight="bold", fontsize=10)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "Fig2.png", dpi=dpi)
    plt.close(fig)
    save_plos_tiff(out_dir / "Fig2.png", out_dir / "Fig2.tif", dpi)


def save_plos_tiff(png_path, tif_path, dpi=300):
    """PLOS ONE: RGB (no alpha channel), LZW compression, 300 to 600 dpi."""
    from PIL import Image

    with Image.open(png_path) as img:
        img.convert("RGB").save(tif_path, compression="tiff_lzw", dpi=(dpi, dpi))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default=str(ROOT / "outputs" / "published" / "fold*_test_predictions.csv"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    pred = load_predictions(args.predictions)
    fpr_grid = np.linspace(0, 1, 101)
    summary, curves = summarize(pred, args.threshold, args.n_boot, args.seed, fpr_grid)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_summary.json", "w", newline="\n") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    make_figure(pred, summary, curves, fpr_grid, out_dir / "figures")
    print(json.dumps({k: summary[k] for k in ("n_patients", "auroc_pooled", "auroc_pooled_ci95_hanley_mcneil",
                                              "auroc_pooled_ci95_bootstrap", "confusion_matrix")}, indent=2))
    print("wrote", out_dir / "metrics_summary.json", "and", out_dir / "figures" / "Fig2.tif")


if __name__ == "__main__":
    main()
