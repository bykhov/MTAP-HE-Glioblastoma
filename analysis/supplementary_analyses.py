"""
Supplementary analyses (S1 Fig, S2 Table, S1 Text) from the per-patient out-of-fold table only.

CPU only. Input is S1 Table (``outputs/confound_S1_Table.csv``: patient_id, label, n_tiles, n_slides, tss,
idh_mutant, fold, probability) and, for the cohort table, ``data/clinical.csv`` (age, sex, expression subtype).

Writes
  ``<out-dir>/supplementary_analyses.json`` slides per patient; model AUROC in patients with one slide and with
                                             more than one; operating points read off the pooled ROC; per-fold
                                             AUROC; Brier score, calibration intercept and slope, binned calibration
  ``<out-dir>/supplementary_S2_Table.csv`` per-fold table (S2 Table)
  ``<out-dir>/supplementary_cohort_table.csv`` cohort characteristics by class
                                             and, in the JSON, age and expression subtype as confounders
  ``<out-dir>/figures/S1_Fig.tif`` / .png   calibration curve and predicted probabilities by class (S1 Fig)

    python analysis/supplementary_analyses.py --s1 outputs/confound_S1_Table.csv --clinical data/clinical.csv --out-dir outputs

    # to rebuild data/clinical.csv from cBioPortal (needs internet access):
    python analysis/supplementary_analyses.py --fetch-clinical data/clinical.csv --s1 outputs/confound_S1_Table.csv

The operating points are descriptive: their thresholds are read from the same out-of-fold predictions they are
evaluated on, so they are not estimates of performance at a pre-specified threshold.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.confound_checks import _r, auroc, hanley_mcneil_ci, hanley_mcneil_se  # noqa: E402
from analysis.metrics_report import _setup_matplotlib, jeffreys_interval, save_plos_tiff  # noqa: E402

CBIO = "https://www.cbioportal.org/api"


# ----------------------------------------------------------------------------
# clinical data
# ----------------------------------------------------------------------------
def _clinical(study, data_type):
    import requests

    r = requests.get(f"{CBIO}/studies/{study}/clinical-data",
                     params={"clinicalDataType": data_type, "projection": "SUMMARY", "pageSize": 100000}, timeout=120)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def fetch_clinical(out_csv, patient_ids):
    """
    Age and sex from the TCGA GBM study of cBioPortal (``gbm_tcga``, complete for the cohort); expression subtype
    (Verhaak classes plus G-CIMP) of the primary tumor sample from Brennan et al. 2013 (``gbm_tcga_pub2013``).
    Sex from the 2013 study is kept in ``sex_pub2013`` so that disagreements stay visible.
    """
    ids = set(patient_ids)

    def pivot(df, attrs, level):
        df = df[df.patientId.isin(ids) & df.clinicalAttributeId.isin(attrs)]
        if level == "SAMPLE":
            df = df[df.sampleId.str[13:15] == "01"]
        return df.pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first")

    base = pivot(_clinical("gbm_tcga", "PATIENT"), ["AGE", "SEX"], "PATIENT")
    pub_pat = pivot(_clinical("gbm_tcga_pub2013", "PATIENT"), ["SEX"], "PATIENT")
    pub_smp = pivot(_clinical("gbm_tcga_pub2013", "SAMPLE"), ["EXPRESSION_SUBTYPE"], "SAMPLE")
    out = pd.DataFrame({"patient_id": sorted(ids)}).set_index("patient_id")
    out["age"] = pd.to_numeric(base.get("AGE"), errors="coerce")
    out["sex"] = base.get("SEX")
    out["sex_pub2013"] = pub_pat.get("SEX")
    out["expression_subtype"] = pub_smp.get("EXPRESSION_SUBTYPE")
    out.reset_index().to_csv(out_csv, index=False)
    disagree = out.dropna(subset=["sex", "sex_pub2013"])
    disagree = disagree[disagree.sex.str.lower() != disagree.sex_pub2013.str.lower()]
    print(f"wrote {out_csv}: {len(out)} patients, age for {out.age.notna().sum()}, sex for {out.sex.notna().sum()}, "
          f"subtype for {out.expression_subtype.notna().sum()}; sex disagrees with gbm_tcga_pub2013 for "
          f"{list(disagree.index)}")


# ----------------------------------------------------------------------------
# analyses
# ----------------------------------------------------------------------------
def _describe(x):
    x = pd.Series(x, dtype=float).dropna()
    return {"n": int(len(x)), "mean": _r(x.mean(), 2), "sd": _r(x.std(ddof=1), 2), "median": _r(x.median(), 1),
            "q1": _r(x.quantile(0.25), 1), "q3": _r(x.quantile(0.75), 1), "min": _r(x.min(), 1), "max": _r(x.max(), 1)}


def slides_per_patient(df):
    hd, ne = df.loc[df.label == 1, "n_slides"], df.loc[df.label == 0, "n_slides"]
    return {"total_slides": int(df.n_slides.sum()), "all": _describe(df.n_slides), "HD": _describe(hd),
            "neutral": _describe(ne), "mann_whitney_p": _r(stats.mannwhitneyu(hd, ne).pvalue, 3),
            "counts": {int(k): int(v) for k, v in df.n_slides.value_counts().sort_index().items()}}


def _auroc_block(g):
    y, s = g.label.to_numpy(), g.probability.to_numpy()
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    a = auroc(y, s)
    return {"n": int(len(y)), "n_HD": n1, "n_neutral": n0, "auroc": _r(a), "ci95": [_r(c) for c in hanley_mcneil_ci(a, n1, n0)],
            "median_tiles": _r(g.n_tiles.median(), 0)}


def auroc_by_slide_count(df):
    """Model AUROC in patients with one slide and with more than one; z-test for two independent AUROCs."""
    one, many = df[df.n_slides == 1], df[df.n_slides > 1]
    a, b = _auroc_block(one), _auroc_block(many)
    se = np.hypot(hanley_mcneil_se(a["auroc"], a["n_HD"], a["n_neutral"]),
                  hanley_mcneil_se(b["auroc"], b["n_HD"], b["n_neutral"]))
    z = (b["auroc"] - a["auroc"]) / se
    return {"one_slide": a, "more_than_one_slide": b, "difference": _r(b["auroc"] - a["auroc"]),
            "z": _r(z, 2), "p_two_sided": _r(2 * stats.norm.sf(abs(z)), 3)}


def operating_points(df, sensitivities=(0.8, 0.9, 0.95), specificities=(0.9, 0.95)):
    """
    Points of the pooled ROC (a patient is called HD if probability >= threshold). For each target sensitivity the
    highest threshold that reaches it; for each target specificity the lowest threshold that reaches it. Reports what a
    rule-out (or rule-in) step before confirmatory testing would do in this cohort.
    """
    y, s = df.label.to_numpy(), df.probability.to_numpy()
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    thresholds = np.unique(s)[::-1]

    def at(t):
        pos = s >= t
        tp, fp = int((pos & (y == 1)).sum()), int((pos & (y == 0)).sum())
        fn, tn = n1 - tp, n0 - fp
        return {"threshold": _r(t, 4), "sensitivity": _r(tp / n1), "specificity": _r(tn / n0),
                "TP": tp, "FN": fn, "TN": tn, "FP": fp, "called_neutral": fn + tn,
                "called_neutral_fraction": _r((fn + tn) / len(y)),
                "npv": _r(tn / (tn + fn)) if tn + fn else None, "ppv": _r(tp / (tp + fp)) if tp + fp else None}

    out = {"rule_out": [], "rule_in": []}
    for target in sensitivities:
        t = next(t for t in thresholds if (s[y == 1] >= t).mean() >= target)
        out["rule_out"].append({"target_sensitivity": target, **at(t)})
    for target in specificities:
        t = next(t for t in thresholds[::-1] if (s[y == 0] < t).mean() >= target)
        out["rule_in"].append({"target_specificity": target, **at(t)})
    return out


def per_fold(df):
    rows = []
    for k, g in df.groupby("fold"):
        b = _auroc_block(g)
        rows.append({"fold": int(k), "n": b["n"], "n_HD": b["n_HD"], "n_neutral": b["n_neutral"],
                     "auroc": b["auroc"], "ci95_low": b["ci95"][0], "ci95_high": b["ci95"][1]})
    return pd.DataFrame(rows)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration(df, n_bins=5):
    """
    Brier score against a forecast of the cohort prevalence for every patient; calibration intercept (slope fixed at
    1, i.e. calibration in the large) and slope by logistic regression of the label on the logit of the prediction;
    observed HD fraction in equal-count bins of predicted probability, with Jeffreys intervals.
    """
    y, p = df.label.to_numpy().astype(float), df.probability.to_numpy().astype(float)
    x = _logit(p)
    prevalence = y.mean()

    def nll(a, b):
        eta = a + b * x
        return float(np.sum(np.logaddexp(0, eta) - y * eta))

    intercept = optimize.minimize_scalar(lambda a: nll(a, 1.0)).x
    a_fit, b_fit = optimize.minimize(lambda v: nll(v[0], v[1]), x0=[0.0, 1.0], method="BFGS").x
    brier = float(np.mean((p - y) ** 2))
    brier_ref = float(np.mean((prevalence - y) ** 2))
    bins = pd.qcut(p, n_bins, duplicates="drop")
    table = []
    for interval, g in df.assign(_bin=bins).groupby("_bin", observed=True):
        k, n = int(g.label.sum()), int(len(g))
        lo, hi = jeffreys_interval(k, n)
        table.append({"bin": str(interval), "n": n, "mean_predicted": _r(g.probability.mean()),
                      "observed_HD": _r(k / n), "ci95": [_r(lo), _r(hi)]})
    return {"prevalence": _r(prevalence), "mean_predicted": _r(p.mean()), "brier": _r(brier, 4),
            "brier_prevalence_only": _r(brier_ref, 4), "brier_skill": _r(1 - brier / brier_ref),
            "calibration_intercept": _r(intercept), "calibration_slope": _r(b_fit),
            "calibration_slope_intercept": _r(a_fit), "bins": table}


def stratified_auroc(df, col):
    """Model AUROC over HD-neutral pairs from the same stratum (pair-weighted mean of within-stratum AUROCs)."""
    num = den = 0.0
    for _, g in df.dropna(subset=[col]).groupby(col):
        n1, n0 = int(g.label.sum()), int(len(g) - g.label.sum())
        if n1 and n0:
            num += auroc(g.label, g.probability) * n1 * n0
            den += n1 * n0
    return (num / den if den else np.nan), int(den)


def _boot_stratified_auroc(df, col, n_boot=2000, seed=0):
    """Percentile CI from a patient bootstrap stratified by label."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(df.label == 1)[0], np.where(df.label == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(stratified_auroc(df.iloc[idx], col)[0])
    return [_r(np.nanpercentile(vals, 2.5)), _r(np.nanpercentile(vals, 97.5))]


def _loo_rate(df, col):
    """Each patient scored by the HD rate of the other patients in the same stratum (as for site in confound_checks)."""
    hd, n = df.groupby(col).label.transform("sum"), df.groupby(col).label.transform("count")
    return np.where(n > 1, (hd - df.label) / (n - 1).clip(lower=1), df.label.mean())


def _loglik(model, X, y):
    p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def covariate_checks(df, clinical, n_boot=2000):
    """
    Age and expression subtype as confounders. For each: its association with the label (AUROC of the covariate
    alone), and the model AUROC computed only over HD-neutral pairs within the same stratum (age tertile, subtype).
    Then, in patients with a known subtype, a likelihood-ratio test of whether the model's logit adds to a logistic
    regression of the label on age and subtype (unpenalized, in-sample).
    """
    from sklearn.linear_model import LogisticRegression

    d = df.merge(clinical[["patient_id", "age", "expression_subtype"]], on="patient_id", how="left")
    y = d.label.to_numpy()
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    a_age = auroc(y, d.age)
    d["age_tertile"] = pd.qcut(d.age, 3, labels=False)
    a_age_strat, pairs_age = stratified_auroc(d, "age_tertile")

    k = d.dropna(subset=["expression_subtype"]).reset_index(drop=True)
    yk = k.label.to_numpy()
    a_sub = auroc(yk, _loo_rate(k, "expression_subtype"))
    a_sub_strat, pairs_sub = stratified_auroc(k, "expression_subtype")
    no_gcimp = k[k.expression_subtype != "G-CIMP"]
    a_ng = auroc(no_gcimp.label, no_gcimp.probability)
    m1, m0 = int(no_gcimp.label.sum()), int(len(no_gcimp) - no_gcimp.label.sum())

    X_base = pd.get_dummies(k[["age", "expression_subtype"]], columns=["expression_subtype"], drop_first=True, dtype=float)
    X_full = X_base.assign(model_logit=_logit(k.probability))
    base = LogisticRegression(penalty=None, max_iter=5000).fit(X_base, yk)
    full = LogisticRegression(penalty=None, max_iter=5000).fit(X_full, yk)
    lr = 2 * (_loglik(full, X_full, yk) - _loglik(base, X_base, yk))
    return {
        "age": {"auroc_alone": _r(a_age), "auroc_alone_ci95": [_r(c) for c in hanley_mcneil_ci(a_age, n1, n0)],
                "tertile_cuts": [_r(c, 1) for c in d.age.quantile([1 / 3, 2 / 3])],
                "model_auroc_age_stratified": _r(a_age_strat), "pairs": pairs_age,
                "model_auroc_age_stratified_ci95": _boot_stratified_auroc(d, "age_tertile", n_boot)},
        "expression_subtype": {
            "n_known": int(len(k)), "auroc_alone_loo": _r(a_sub),
            "model_auroc_subtype_stratified": _r(a_sub_strat), "pairs": pairs_sub,
            "model_auroc_subtype_stratified_ci95": _boot_stratified_auroc(k, "expression_subtype", n_boot),
            "model_auroc_within_subtype": {s: _r(auroc(g.label, g.probability)) for s, g in k.groupby("expression_subtype")},
            "model_auroc_without_gcimp": _r(a_ng), "without_gcimp_n_HD": m1, "without_gcimp_n_neutral": m0,
            "model_auroc_without_gcimp_ci95": [_r(c) for c in hanley_mcneil_ci(a_ng, m1, m0)]},
        "incremental": {
            "n": int(len(k)), "auroc_age_subtype_in_sample": _r(auroc(yk, base.predict_proba(X_base)[:, 1])),
            "auroc_age_subtype_model_in_sample": _r(auroc(yk, full.predict_proba(X_full)[:, 1])),
            "lr_chi2": _r(lr, 2), "lr_p": _p(stats.chi2.sf(lr, 1)),
            "model_logit_coef": _r(full.coef_[0][-1])},
    }


def _p(p):
    """p-value to two significant digits."""
    return float(f"{p:.2g}")


def cohort_characteristics(df, clinical):
    """Cohort characteristics by class, with the test used for each row."""
    d = df.merge(clinical, on="patient_id", how="left")
    hd, ne = d[d.label == 1], d[d.label == 0]
    rows = []

    def continuous(name, col, digits=0):
        def fmt(x):
            x = x.dropna()
            return f"{x.median():.{digits}f} ({x.quantile(.25):.{digits}f}-{x.quantile(.75):.{digits}f})"
        p = stats.mannwhitneyu(hd[col].dropna(), ne[col].dropna()).pvalue
        rows.append({"characteristic": f"{name}, median (IQR)", "all": fmt(d[col]), "HD": fmt(hd[col]),
                     "neutral": fmt(ne[col]), "p": _p(p), "test": "Mann-Whitney U",
                     "missing": int(d[col].isna().sum())})

    def count(g, mask):
        n = int(g.loc[mask.loc[g.index]].shape[0])
        return f"{n} ({100 * n / len(g):.0f}%)"

    continuous("Age at diagnosis, years", "age")
    male = d.sex.str.lower() == "male"
    sex_tab = pd.crosstab(d.label, male)
    rows.append({"characteristic": "Male sex, n (%)", "all": count(d, male), "HD": count(hd, male),
                 "neutral": count(ne, male), "p": _p(stats.fisher_exact(sex_tab.to_numpy())[1]),
                 "test": "Fisher exact", "missing": int(d.sex.isna().sum())})
    known = d.dropna(subset=["expression_subtype"])
    sub_tab = pd.crosstab(known.expression_subtype, known.label)
    chi2, p, dof, _ = stats.chi2_contingency(sub_tab.to_numpy())
    rows.append({"characteristic": "Expression subtype, n (% of known)", "all": "", "HD": "", "neutral": "",
                 "p": _p(p), "test": f"chi-square, {dof} df", "missing": int(d.expression_subtype.isna().sum())})
    for subtype in sub_tab.index:
        def frac(g):
            k = g.dropna(subset=["expression_subtype"])
            n = int((k.expression_subtype == subtype).sum())
            return f"{n} ({100 * n / len(k):.0f}%)"
        rows.append({"characteristic": f"  {subtype}", "all": frac(d), "HD": frac(hd), "neutral": frac(ne),
                     "p": None, "test": "", "missing": None})
    idh = d.dropna(subset=["idh_mutant"])
    idh_tab = pd.crosstab(idh.label, idh.idh_mutant)
    def idh_frac(g):
        k = g.dropna(subset=["idh_mutant"])
        return f"{int(k.idh_mutant.sum())} of {len(k)}"
    rows.append({"characteristic": "IDH-mutant, n of tested", "all": idh_frac(d), "HD": idh_frac(hd),
                 "neutral": idh_frac(ne), "p": _p(stats.fisher_exact(idh_tab.to_numpy())[1]),
                 "test": "Fisher exact", "missing": int(d.idh_mutant.isna().sum())})
    continuous("Slides per patient", "n_slides")
    continuous("Tiles per patient", "n_tiles")
    rows.append({"characteristic": "Tissue source sites, n", "all": str(d.tss.nunique()), "HD": str(hd.tss.nunique()),
                 "neutral": str(ne.tss.nunique()), "p": None, "test": "", "missing": 0})
    header = {"characteristic": "Patients, n", "all": str(len(d)), "HD": str(len(hd)), "neutral": str(len(ne)),
              "p": None, "test": "", "missing": 0}
    return pd.DataFrame([header, *rows])


# ----------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------
def calibration_figure(df, calib, out_dir, dpi=300):
    """S1 Fig: A binned calibration with Jeffreys intervals; B predicted probabilities by class."""
    plt = _setup_matplotlib()
    HD, GR, OR = "#1f4e79", "#8a8a8a", "#c47a1f"  # same colors as Fig 2
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.5, 3.2), constrained_layout=True)
    pred = [b["mean_predicted"] for b in calib["bins"]]
    obs = [b["observed_HD"] for b in calib["bins"]]
    err = np.array([[o - b["ci95"][0], b["ci95"][1] - o] for o, b in zip(obs, calib["bins"])]).T
    ax_a.plot([0, 1], [0, 1], color=GR, lw=0.8, ls=":", label="Perfect calibration")
    ax_a.axhline(calib["prevalence"], color=GR, lw=0.8, ls="--", label=f"HD prevalence ({calib['prevalence']:.3f})")
    ax_a.errorbar(pred, obs, yerr=err, color=HD, marker="o", ms=4, lw=1.2, capsize=2, elinewidth=0.7,
                  label="Model, equal-count bins")
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1.02); ax_a.set_aspect("equal")
    ax_a.set_xlabel("Mean predicted probability"); ax_a.set_ylabel("Observed fraction HD")
    ax_a.legend(frameon=False, fontsize=6, loc="upper left")
    ax_a.text(0.98, 0.04, f"Brier {calib['brier']:.3f} (prevalence only {calib['brier_prevalence_only']:.3f})\n"
              f"Calibration slope {calib['calibration_slope']:.2f}", transform=ax_a.transAxes, ha="right", fontsize=6)

    edges = np.linspace(0, 1, 21)
    ax_b.hist(df.loc[df.label == 1, "probability"], bins=edges, color=HD, alpha=0.75, label="HD", edgecolor="white", lw=0.5)
    ax_b.hist(df.loc[df.label == 0, "probability"], bins=edges, color=OR, alpha=0.65, label="Neutral", edgecolor="white", lw=0.5)
    ax_b.axvline(0.5, color=GR, lw=0.8, ls=":")
    ax_b.text(0.5, 1.0, "threshold 0.5", transform=ax_b.get_xaxis_transform(), fontsize=6, color=GR,
              ha="center", va="bottom")
    ax_b.set_xlabel("Out-of-fold predicted probability"); ax_b.set_ylabel("Patients")
    ax_b.legend(frameon=False, fontsize=6, loc="upper left")

    ax_a.text(-0.16, 1.02, "A", transform=ax_a.transAxes, fontweight="bold", fontsize=10)
    ax_b.text(-0.16, 1.02, "B", transform=ax_b.transAxes, fontweight="bold", fontsize=10)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "S1_Fig.png", dpi=dpi)
    plt.close(fig)
    save_plos_tiff(out_dir / "S1_Fig.png", out_dir / "S1_Fig.tif", dpi)


# ----------------------------------------------------------------------------
def run(s1_csv, clinical_csv, out_dir, make_fig=True):
    df = pd.read_csv(s1_csv, dtype={"tss": str})
    assert df.patient_id.is_unique
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = per_fold(df)
    calib = calibration(df)
    summary = {
        "n_patients": int(len(df)), "n_HD": int(df.label.sum()),
        "slides_per_patient": slides_per_patient(df),
        "auroc_by_slide_count": auroc_by_slide_count(df),
        "operating_points": operating_points(df),
        "per_fold_auroc": folds.to_dict(orient="records"),
        "per_fold_auroc_mean": _r(folds.auroc.mean()), "per_fold_auroc_sd": _r(folds.auroc.std(ddof=1), 4),
        "calibration": calib,
    }
    folds.to_csv(out_dir / "supplementary_S2_Table.csv", index=False)
    if clinical_csv:
        clinical = pd.read_csv(clinical_csv)
        table = cohort_characteristics(df, clinical)
        table.to_csv(out_dir / "supplementary_cohort_table.csv", index=False)
        summary["cohort_table"] = table.to_dict(orient="records")
        summary["covariate_checks"] = covariate_checks(df, clinical)
    (out_dir / "supplementary_analyses.json").write_text(json.dumps(summary, indent=2))
    if make_fig:
        calibration_figure(df, calib, out_dir / "figures")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s1", default="outputs/confound_S1_Table.csv")
    ap.add_argument("--clinical", default=None, help="CSV from --fetch-clinical (patient_id, age, sex, expression_subtype)")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--fetch-clinical", metavar="CSV", help="download age, sex and expression subtype and exit")
    args = ap.parse_args()
    if args.fetch_clinical:
        fetch_clinical(args.fetch_clinical, pd.read_csv(args.s1).patient_id)
        return
    summary = run(args.s1, args.clinical, args.out_dir)
    print(json.dumps({k: summary[k] for k in ("auroc_by_slide_count", "covariate_checks") if k in summary}, indent=1)[:4000])


if __name__ == "__main__":
    main()
