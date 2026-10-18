"""
Confounder checks: tile count, tissue source site and IDH status against the label and the model score.

CPU only. Needs the patient-level manifest and the pooled out-of-fold test
predictions (one row per patient). Writes the confounder table of the paper and S1 Table.

Checks
------
1. Tiles per patient vs label
   - Mann-Whitney U, AUROC of tile count used alone (bootstrap 95% CI)
   - model AUROC within tile-count tertiles
2. Tissue source site (TCGA-<TSS>-<participant>) vs label
   - site x label table, chi-square test
   - AUROC of site alone: each patient scored by the HD rate of the OTHER
     patients from the same site (leave-one-out, so no self-leakage)
   - site-stratified model AUROC: within-site AUROCs over sites holding both
     classes, weighted by number of HD-neutral pairs (= pooled AUROC computed
     only over same-site pairs)
3. IDH1/IDH2 mutation status vs label (needs --idh CSV, see fetch_idh below)
   - Fisher exact test, model AUROC in IDH-wild-type patients only

Usage
-----
    python analysis/confound_checks.py --manifest data/cohort.csv \
        --predictions "outputs/published/fold*_test_predictions.csv" \
        --idh data/idh_status.csv --out outputs/confound

    # to rebuild data/idh_status.csv (needs internet access; PanCanAtlas mutation calls,
    # completed from the Ceccarelli 2016 and Brennan 2013 annotations; G-CIMP column):
    python analysis/confound_checks.py --fetch-idh data/idh_status.csv

Prediction CSVs need columns: patient_id, true_label (0/1), probability
(optionally fold). Manifest may be .parquet (needs pyarrow) or .csv and needs a
patient id, a 0/1 label and a tile-count column.

Outputs
-------
    <out>_summary.json     all results of the checks
    <out>_sites.csv        per-site table
    <out>_S1_Table.csv     per-patient table (S1 Table)

Dependencies: numpy, pandas, scipy (+ pyarrow for parquet, requests for --fetch-idh).
"""

from __future__ import annotations

import argparse
import glob
import json
import re

import numpy as np
import pandas as pd
from scipy import stats

PATIENT_COL_CANDIDATES = ["patient_id", "case_id", "case_submitter_id"]
LABEL_COL_CANDIDATES = ["patient_label", "slide_label", "label", "is_mtap_mutation", "true_label"]
NPATCH_COL_CANDIDATES = ["n_patches", "bag_size", "num_patches", "n_tiles"]
NSLIDE_COL_CANDIDATES = ["n_slides", "num_slides"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _pick(df, candidates, what, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"no {what} column; looked for {candidates}, have {list(df.columns)}")
    return None


def patient_barcode(x: str) -> str:
    """TCGA-02-0001-01Z-00-DX1 -> TCGA-02-0001."""
    m = re.match(r"^(TCGA-[0-9A-Za-z]{2}-[0-9A-Za-z]{4})", str(x))
    return m.group(1) if m else str(x)


def tss_from_barcode(barcode: str) -> str:
    m = re.match(r"^TCGA-([0-9A-Za-z]{2})-", str(barcode))
    return m.group(1) if m else "UNK"


def auroc(y, s):
    """Mann-Whitney AUROC; ties count 1/2. Returns nan if a class is missing."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg))


def hanley_mcneil_se(a, n1, n0):
    """Standard error of an AUROC (Hanley & McNeil 1982, exponential approximation)."""
    q1, q2 = a / (2 - a), 2 * a ** 2 / (1 + a)
    return float(np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a ** 2) + (n0 - 1) * (q2 - a ** 2)) / (n1 * n0)))


def hanley_mcneil_ci(a, n1, n0, alpha=0.05):
    if not np.isfinite(a) or n1 == 0 or n0 == 0:
        return (np.nan, np.nan)
    se = hanley_mcneil_se(a, n1, n0)
    z = stats.norm.ppf(1 - alpha / 2)
    return (float(max(0.0, a - z * se)), float(min(1.0, a + z * se)))


def boot_auroc_ci(y, s, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pi, len(pi)), rng.choice(ni, len(ni))])
        vals.append(auroc(y[idx], s[idx]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def _r(x, k=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_manifest(path):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    pcol = _pick(df, PATIENT_COL_CANDIDATES, "patient id")
    lcol = _pick(df, LABEL_COL_CANDIDATES, "label")
    ncol = _pick(df, NPATCH_COL_CANDIDATES, "tile count")
    scol = _pick(df, NSLIDE_COL_CANDIDATES, "slide count", required=False)
    out = pd.DataFrame({
        "patient_id": df[pcol].map(patient_barcode),
        "label": df[lcol].astype(int),
        "n_tiles": df[ncol].astype(float),
    })
    if scol:
        out["n_slides"] = df[scol].astype(int)
    # a slide-level manifest is collapsed to one row per patient
    agg = {"label": "max", "n_tiles": "sum"}
    if "n_slides" in out:
        agg["n_slides"] = "sum"
    g = out.groupby("patient_id")
    if g.size().max() > 1:
        if "n_slides" not in out:
            out["n_slides"] = 1
            agg["n_slides"] = "sum"
        assert (g["label"].nunique() == 1).all(), "patients with conflicting labels"
        out = out.groupby("patient_id", as_index=False).agg(agg)
    out["tss"] = out["patient_id"].map(tss_from_barcode)
    return out


def load_predictions(pattern):
    files = sorted(glob.glob(pattern)) if pattern else []
    if not files:
        return None
    p = pd.concat([pd.read_csv(f).assign(_file=f) for f in files], ignore_index=True)
    key = "patient_id" if "patient_id" in p.columns else p.columns[0]
    p["patient_id"] = p[key].map(patient_barcode)
    if "fold" not in p.columns:
        p["fold"] = p["_file"].str.extract(r"fold[_-]?(\d+)", expand=False)
    # slide-level rows -> one per patient (all slides of a patient share a bag)
    p = p.groupby("patient_id", as_index=False).agg(
        true_label=("true_label", "first"), probability=("probability", "mean"), fold=("fold", "first"))
    assert p["patient_id"].is_unique
    return p


def fetch_idh(out_csv, study="gbm_tcga_pan_can_atlas_2018"):
    """IDH1/IDH2 non-synonymous mutation calls per patient from the cBioPortal REST API."""
    import requests
    base = "https://www.cbioportal.org/api"
    prof = f"{study}_mutations"
    samples = requests.get(f"{base}/studies/{study}/samples", params={"projection": "SUMMARY"}, timeout=60).json()
    patients = sorted({s["patientId"] for s in samples})
    body = {"entrezGeneIds": [3417, 3418], "sampleListId": f"{study}_sequenced"}  # IDH1, IDH2
    muts = requests.post(f"{base}/molecular-profiles/{prof}/mutations/fetch",
                         params={"projection": "DETAILED"}, json=body, timeout=120).json()
    seq = requests.get(f"{base}/sample-lists/{study}_sequenced/sample-ids", timeout=60).json()
    seq_pat = {s[:12] for s in seq}
    mut_pat = {}
    for m in muts:
        if m.get("mutationType") in ("Silent", "Intron", "3'UTR", "5'UTR"):
            continue
        sym = m["gene"]["hugoGeneSymbol"]
        mut_pat.setdefault(m["patientId"], []).append(f"{sym} {m.get('proteinChange', '')}")
    rows = [{"patient_id": p,
             "idh_mutant": (1 if p in mut_pat else 0) if p in seq_pat else np.nan,
             "idh_detail": "; ".join(mut_pat.get(p, []))} for p in patients]
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(rows)} patients, {len(mut_pat)} IDH-mutant, {len(seq_pat)} sequenced")


# Annotations used where the PanCanAtlas study has no mutation call, in this order of precedence:
# (study, sample attribute, value -> 0/1). G-CIMP methylation is kept as a separate column, not as an IDH call.
IDH_ANNOTATIONS = [
    ("lgggbm_tcga_pub", "IDH_STATUS", {"WT": 0, "Mutant": 1}),           # Ceccarelli et al., Cell 2016
    ("gbm_tcga_pub2013", "IDH1_MUTATION", {"WT": 0, "R132H": 1}),        # Brennan et al., Cell 2013
]
G_CIMP_ANNOTATION = ("gbm_tcga_pub2013", "G_CIMP_METHYLATION", {"non-G-CIMP": 0, "G-CIMP": 1})


def combine_idh(pancan, annotations, g_cimp=None):
    """
    One IDH call per patient. ``pancan``: patient_id, idh_mutant (NaN if not sequenced), idh_detail. ``annotations``:
    list of (source name, DataFrame patient_id, value 0/1) in order of precedence. The PanCanAtlas call is kept where
    it exists; later sources only fill patients still without a call. Disagreements with the kept call are listed in
    ``idh_disagreement``. ``g_cimp`` (patient_id, value) adds a ``g_cimp`` column.
    """
    out = pancan[["patient_id", "idh_mutant", "idh_detail"]].copy()
    out["idh_source"] = np.where(out.idh_mutant.notna(), "pancan_atlas_mutations", None)
    out["idh_disagreement"] = ""
    for name, ann in annotations:
        ann = ann.dropna().drop_duplicates("patient_id").set_index("patient_id").value
        out = out.merge(ann.rename("_v"), left_on="patient_id", right_index=True, how="left")
        out["idh_disagreement"] = out.idh_disagreement.fillna("")
        differ = out.idh_mutant.notna() & out._v.notna() & (out.idh_mutant != out._v)
        out.loc[differ, "idh_disagreement"] += out.loc[differ, "_v"].map(lambda v: f"{name}={int(v)};")
        fill = out.idh_mutant.isna() & out._v.notna()
        out.loc[fill, "idh_mutant"] = out.loc[fill, "_v"]
        out.loc[fill, "idh_source"] = name
        out = out.drop(columns="_v")
    if g_cimp is not None:
        g = g_cimp.dropna().drop_duplicates("patient_id").set_index("patient_id").value.rename("g_cimp")
        out = out.merge(g, left_on="patient_id", right_index=True, how="left")
    return out.sort_values("patient_id").reset_index(drop=True)


def _sample_annotation(study, attribute, mapping):
    """Primary-tumor (-01) sample attribute of a cBioPortal study as patient_id, value (0/1)."""
    import requests
    r = requests.get(f"https://www.cbioportal.org/api/studies/{study}/clinical-data",
                     params={"clinicalDataType": "SAMPLE", "projection": "SUMMARY", "pageSize": 100000}, timeout=120)
    r.raise_for_status()
    d = pd.DataFrame(r.json())
    d = d[(d.clinicalAttributeId == attribute) & (d.sampleId.str[13:15] == "01")]
    return pd.DataFrame({"patient_id": d.patientId, "value": d.value.map(mapping)})


def fetch_idh_combined(out_csv):
    """PanCanAtlas IDH1/IDH2 mutation calls, completed from IDH_ANNOTATIONS; G-CIMP status as a separate column."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        fetch_idh(f"{tmp}/pancan.csv")
        pancan = pd.read_csv(f"{tmp}/pancan.csv")
    anns = [(f"{s}:{a}", _sample_annotation(s, a, m)) for s, a, m in IDH_ANNOTATIONS]
    out = combine_idh(pancan, anns, _sample_annotation(*G_CIMP_ANNOTATION))
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {out.idh_mutant.notna().sum()} with an IDH call "
          f"({out.idh_source.value_counts().to_dict()}), {int(out.idh_mutant.sum())} mutant; "
          f"disagreements: {out.loc[out.idh_disagreement != '', 'patient_id'].tolist()}")


# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------
def check_tiles(df, pred):
    pos = df.loc[df.label == 1, "n_tiles"].to_numpy()
    neg = df.loc[df.label == 0, "n_tiles"].to_numpy()
    mw = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    a = auroc(df.label, df.n_tiles)
    res = {
        "median_tiles_all": _r(np.median(df.n_tiles), 0),
        "iqr_tiles_all": [_r(np.percentile(df.n_tiles, 25), 0), _r(np.percentile(df.n_tiles, 75), 0)],
        "median_tiles_HD": _r(np.median(pos), 0), "median_tiles_neutral": _r(np.median(neg), 0),
        "mannwhitney_p": float(mw.pvalue),
        "auroc_tilecount_alone": _r(a), "auroc_tilecount_alone_95ci": [_r(x) for x in boot_auroc_ci(df.label, df.n_tiles)],
    }
    if pred is not None:
        m = pred.merge(df[["patient_id", "n_tiles"]], on="patient_id", how="left")
        m["tertile"] = pd.qcut(m.n_tiles, 3, labels=["low", "mid", "high"])
        terts = {}
        for t, g in m.groupby("tertile", observed=True):
            a_t = auroc(g.true_label, g.probability)
            n1, n0 = int((g.true_label == 1).sum()), int((g.true_label == 0).sum())
            terts[str(t)] = {"n_HD": n1, "n_neutral": n0, "auroc": _r(a_t),
                             "ci95": [_r(x) for x in hanley_mcneil_ci(a_t, n1, n0)],
                             "tile_range": [_r(g.n_tiles.min(), 0), _r(g.n_tiles.max(), 0)]}
        res["model_auroc_by_tertile"] = terts
        # does tile count add anything to / explain the model score?
        res["spearman_prob_vs_tiles"] = _r(stats.spearmanr(m.probability, m.n_tiles).correlation)
    return res


def check_site(df, pred):
    tab = pd.crosstab(df.tss, df.label).rename(columns={0: "n_neutral", 1: "n_HD"})
    for c in ("n_neutral", "n_HD"):
        if c not in tab:
            tab[c] = 0
    tab["n_total"] = tab.n_HD + tab.n_neutral
    tab["HD_rate"] = tab.n_HD / tab.n_total
    tab = tab.sort_values("n_total", ascending=False)
    chi2, p, dof, _ = stats.chi2_contingency(tab[["n_neutral", "n_HD"]].to_numpy())

    # leave-one-out site HD rate as a "site-only" classifier
    s_hd = df.groupby("tss").label.transform("sum")
    s_n = df.groupby("tss").label.transform("count")
    prior = df.label.mean()
    loo = np.where(s_n > 1, (s_hd - df.label) / (s_n - 1).clip(lower=1), prior)
    a_site = auroc(df.label, loo)
    res = {
        "n_sites": int(len(tab)),
        "n_sites_both_classes": int(((tab.n_HD > 0) & (tab.n_neutral > 0)).sum()),
        "largest_site": tab.index[0], "largest_site_share": _r(tab.n_total.iloc[0] / tab.n_total.sum()),
        "chi2": _r(chi2), "dof": int(dof), "chi2_p": float(p),
        "auroc_site_alone_loo": _r(a_site),
        "auroc_site_alone_loo_95ci": [_r(x) for x in boot_auroc_ci(df.label, loo)],
    }
    if pred is not None:
        m = pred.merge(df[["patient_id", "tss"]], on="patient_id", how="left")
        num = den = 0.0
        per_site = {}
        for s, g in m.groupby("tss"):
            n1, n0 = int((g.true_label == 1).sum()), int((g.true_label == 0).sum())
            if n1 and n0:
                a_s = auroc(g.true_label, g.probability)
                num += a_s * n1 * n0
                den += n1 * n0
                per_site[s] = _r(a_s)
        res["model_auroc_site_stratified"] = _r(num / den) if den else None
        res["site_stratified_pairs"] = int(den)
        res["model_auroc_within_site"] = per_site
        tab = tab.join(m.groupby("tss").probability.mean().rename("mean_pred_prob"))
        r = stats.spearmanr(tab.HD_rate, tab.mean_pred_prob, nan_policy="omit")
        res["spearman_site_HDrate_vs_meanprob"] = _r(r.correlation)
    return res, tab


def check_idh(df, pred, idh_csv):
    idh = pd.read_csv(idh_csv)
    idh["patient_id"] = idh["patient_id"].map(patient_barcode)
    m = df.merge(idh[["patient_id", "idh_mutant"]], on="patient_id", how="left")
    known = m.dropna(subset=["idh_mutant"])
    ct = pd.crosstab(known.idh_mutant.astype(int), known.label).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    res = {
        "n_idh_known": int(len(known)), "n_idh_unknown": int(m.idh_mutant.isna().sum()),
        "n_idh_mutant": int(ct.loc[1].sum()),
        "idh_mutant_HD": int(ct.loc[1, 1]), "idh_mutant_neutral": int(ct.loc[1, 0]),
        "fisher_p": float(stats.fisher_exact(ct.to_numpy())[1]),
    }
    if pred is not None:
        mp = pred.merge(m[["patient_id", "idh_mutant"]], on="patient_id", how="left")
        wt = mp[mp.idh_mutant == 0]
        a = auroc(wt.true_label, wt.probability)
        n1, n0 = int((wt.true_label == 1).sum()), int((wt.true_label == 0).sum())
        res.update({"model_auroc_idh_wt": _r(a), "model_auroc_idh_wt_95ci": [_r(x) for x in hanley_mcneil_ci(a, n1, n0)],
                    "idh_wt_n_HD": n1, "idh_wt_n_neutral": n0})
    extra = [c for c in ("idh_source", "g_cimp") if c in idh.columns]
    if extra:
        m = m.merge(idh[["patient_id", *extra]], on="patient_id", how="left")
    if "idh_source" in m:
        res["idh_source_counts"] = {str(k): int(v) for k, v in m.idh_source.value_counts().items()}
    if "g_cimp" in m:
        # G-CIMP is a methylation surrogate of IDH mutation: exclude patients that are IDH-mutant OR G-CIMP
        res["n_g_cimp"] = int((m.g_cimp == 1).sum())
        res["n_g_cimp_without_idh_call"] = int(((m.g_cimp == 1) & m.idh_mutant.isna()).sum())
        res["n_g_cimp_idh_wt"] = int(((m.g_cimp == 1) & (m.idh_mutant == 0)).sum())
        res["n_idh_mutant_or_g_cimp"] = int(((m.idh_mutant == 1) | (m.g_cimp == 1)).sum())
        if pred is not None:
            mp = pred.merge(m[["patient_id", "idh_mutant", "g_cimp"]], on="patient_id", how="left")
            keep = mp[(mp.idh_mutant != 1) & (mp.g_cimp != 1)]
            a = auroc(keep.true_label, keep.probability)
            n1, n0 = int((keep.true_label == 1).sum()), int((keep.true_label == 0).sum())
            res.update({"model_auroc_excl_idh_mut_or_g_cimp": _r(a),
                        "model_auroc_excl_idh_mut_or_g_cimp_95ci": [_r(x) for x in hanley_mcneil_ci(a, n1, n0)],
                        "excl_idh_mut_or_g_cimp_n_HD": n1, "excl_idh_mut_or_g_cimp_n_neutral": n0})
    return res, m[["patient_id", "idh_mutant", *extra]]


def overall(pred):
    a = auroc(pred.true_label, pred.probability)
    n1, n0 = int((pred.true_label == 1).sum()), int((pred.true_label == 0).sum())
    return {"n_HD": n1, "n_neutral": n0, "auroc_pooled": _r(a),
            "auroc_pooled_95ci": [_r(x) for x in hanley_mcneil_ci(a, n1, n0)]}


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--predictions", default="", help="glob for per-fold test prediction CSVs")
    ap.add_argument("--idh", default="", help="CSV with patient_id, idh_mutant (0/1)")
    ap.add_argument("--fetch-idh", default="", help="write IDH status CSV from cBioPortal and exit")
    ap.add_argument("--out", default="confound_report")
    args = ap.parse_args()

    if args.fetch_idh:
        fetch_idh_combined(args.fetch_idh)
        return
    if not args.manifest:
        ap.error("--manifest is required")

    df = load_manifest(args.manifest)
    pred = load_predictions(args.predictions)
    if pred is not None:
        missing = set(pred.patient_id) ^ set(df.patient_id)
        assert not missing, f"{len(missing)} patients differ between manifest and predictions"
        assert (pred.set_index("patient_id").true_label.astype(int)
                == df.set_index("patient_id").label.reindex(pred.patient_id).values).all(), "label mismatch"
    summ = {"n_patients": int(len(df)), "n_HD": int(df.label.sum()), "n_neutral": int((df.label == 0).sum())}
    if pred is not None:
        summ["overall"] = overall(pred)
    summ["tiles"] = check_tiles(df, pred)
    summ["site"], site_tab = check_site(df, pred)
    s1 = df.copy()
    if args.idh:
        summ["idh"], idh = check_idh(df, pred, args.idh)
        s1 = s1.merge(idh, on="patient_id", how="left")
    if pred is not None:
        s1 = s1.merge(pred[["patient_id", "fold", "probability"]], on="patient_id", how="left")

    with open(f"{args.out}_summary.json", "w") as f:
        json.dump(summ, f, indent=2, default=str)
    site_tab.to_csv(f"{args.out}_sites.csv")
    s1.sort_values("patient_id").to_csv(f"{args.out}_S1_Table.csv", index=False)
    print(json.dumps(summ, indent=2, default=str))


if __name__ == "__main__":
    main()
