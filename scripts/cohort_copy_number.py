"""
Cohort selection from the GDC: MTAP copy number of every TCGA-GBM case, checked against the cohort labels.

For every TCGA-GBM case that has an ABSOLUTE LiftOver gene-level copy-number file (open access), the MTAP row is
read from the GDC by streaming the TSV (the connection is closed at the MTAP row). The labelling rule
(``create_data_summary.py``) is: label 1 (homozygous deletion, HD) when ``copy_number`` is 0, otherwise 0. The paper
analyses copy number 0 (HD) against 2 (neutral).

Writes
  ``data/mtap_copy_number.csv``    one row per case: copy numbers, whether it is in the slide table / the cohort,
                                    the cohort label and whether it agrees with the copy number
  ``outputs/cohort_flow.json``     counts at each selection step (Fig 1B), the reasons the patients of the slide
                                    table are outside the cohort, and every disagreement between a cohort label and
                                    its copy number (the exit code is 1 if there is one)

    python scripts/cohort_copy_number.py --slide-table source/dataset_v1.parquet --cohort data/cohort.csv
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gdc_utils  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = "ABSOLUTE LiftOver"


def list_copy_number_files(project="TCGA-GBM", api=gdc_utils.GDC_API, session=gdc_utils.requests):
    filters = gdc_utils.and_filter(
        gdc_utils.equals_filter("cases.project.project_id", project),
        gdc_utils.equals_filter("data_type", "Gene Level Copy Number"),
        gdc_utils.equals_filter("analysis.workflow_type", WORKFLOW),
        gdc_utils.equals_filter("access", "open"))
    return gdc_utils.query_files(filters, api=api, session=session)


def list_diagnostic_slide_cases(project="TCGA-GBM", api=gdc_utils.GDC_API, session=gdc_utils.requests):
    filters = gdc_utils.and_filter(
        gdc_utils.equals_filter("cases.project.project_id", project),
        gdc_utils.equals_filter("data_type", "Slide Image"),
        gdc_utils.equals_filter("experimental_strategy", "Diagnostic Slide"))
    return {c["submitter_id"] for hit in gdc_utils.query_files(filters, api=api, session=session) for c in hit["cases"]}


def mtap_copy_numbers(files, gene="MTAP", workers=4, api=gdc_utils.GDC_API, session=gdc_utils.requests):
    """One row per copy-number file: case, file id, copy number and min copy number of ``gene``."""
    def one(hit):
        row = gdc_utils.with_retries(lambda: gdc_utils.stream_tsv_row(hit["file_id"], "gene_name", gene, api, session))
        return {"patient_id": hit["cases"][0]["submitter_id"], "file_id": hit["file_id"],
                "copy_number": None if row is None else int(float(row["copy_number"])),
                "min_copy_number": None if row is None else int(float(row["min_copy_number"]))}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return pd.DataFrame(list(pool.map(one, files)))


def build_tables(per_file, diagnostic_cases, slide_table, cohort):
    """Per-case table and the flow dictionary. Pure function of its inputs (tested without network access)."""
    slide_cases = set(slide_table["case_id"])
    slide_label = slide_table.groupby("case_id")["is_mtap_mutation"].max()
    cohort_label = cohort.set_index("patient_id")["label"]

    grouped = per_file.dropna(subset=["copy_number"]).sort_values("file_id").groupby("patient_id")
    cases = pd.DataFrame({
        "n_cn_files": grouped.size(),
        "copy_number": grouped["copy_number"].first().astype(int),
        "copy_numbers_all": grouped["copy_number"].agg(lambda s: ";".join(str(int(v)) for v in s)),
        "min_copy_number": grouped["min_copy_number"].first().astype(int),
    })
    cases.index.name = "patient_id"
    everyone = sorted(set(cases.index) | slide_cases | set(cohort_label.index))
    cases = cases.reindex(everyone)
    cases["has_diagnostic_slide"] = cases.index.isin(diagnostic_cases)
    cases["in_slide_table"] = cases.index.isin(slide_cases)
    cases["slide_table_label"] = slide_label.reindex(cases.index)
    cases["in_cohort"] = cases.index.isin(cohort_label.index)
    cases["cohort_label"] = cohort_label.reindex(cases.index)
    # the cohort contains copy number 0 (label 1) and copy number 2 (label 0) only
    expected = cases["copy_number"].map({0: 1, 2: 0})
    cases["label_expected_from_copy_number"] = expected
    cases["cohort_label_agrees"] = (cases["cohort_label"] == expected).where(cases["in_cohort"])
    cases["ambiguous_copy_number"] = cases["copy_numbers_all"].fillna("").str.contains(";") & (
        cases["copy_numbers_all"].fillna("").map(lambda s: len(set(s.split(";"))) > 1))

    in_cn = cases["copy_number"].notna()
    is_02 = cases["copy_number"].isin([0, 2])
    outside = cases[cases["in_slide_table"] & ~cases["in_cohort"]]
    disagree = cases[cases["in_cohort"] & (cases["cohort_label_agrees"] != True)]  # noqa: E712
    flow = {
        "steps": [
            {"step": "TCGA-GBM cases with an ABSOLUTE LiftOver gene-level copy-number file (MTAP row found)",
             "patients": int(in_cn.sum())},
            {"step": "of these, MTAP copy number 0 or 2", "patients": int(is_02.sum()),
             "copy_number_0_HD": int((cases["copy_number"] == 0).sum()), "copy_number_2_neutral": int((cases["copy_number"] == 2).sum())},
            {"step": "of these, with a diagnostic (DX) slide on the GDC", "patients": int((is_02 & cases["has_diagnostic_slide"]).sum()),
             "copy_number_0_HD": int(((cases["copy_number"] == 0) & cases["has_diagnostic_slide"]).sum()),
             "copy_number_2_neutral": int(((cases["copy_number"] == 2) & cases["has_diagnostic_slide"]).sum())},
            {"step": "patients in the slide table (any copy number)", "patients": int(cases["in_slide_table"].sum())},
            {"step": "analysed cohort (patient manifest)", "patients": int(cases["in_cohort"].sum()),
             "HD": int((cases["cohort_label"] == 1).sum()), "neutral": int((cases["cohort_label"] == 0).sum())},
        ],
        "cases_with_diagnostic_slide_and_cn_0_or_2_not_in_slide_table": int((is_02 & cases["has_diagnostic_slide"] & ~cases["in_slide_table"]).sum()),
        "slide_table_patients_outside_the_cohort": {
            "n": int(len(outside)),
            "by_mtap_copy_number": {str(k): int(v) for k, v in outside["copy_number"].fillna(-1).astype(int).value_counts().sort_index().items()},
            "copy_number_0_HD": outside.index[outside["copy_number"] == 0].tolist(),
            "copy_number_2": int((outside["copy_number"] == 2).sum()),
            "copy_number_not_0_or_2": int((~outside["copy_number"].isin([0, 2])).sum()),
            "slide_table_label_1": outside.index[outside["slide_table_label"] == 1].tolist(),
        },
        "cohort_label_disagreements": {
            "n": int(len(disagree)),
            "patients": disagree[["copy_number", "cohort_label"]].reset_index().to_dict("records"),
        },
        "cohort_patients_without_copy_number_file": cases.index[cases["in_cohort"] & ~in_cn].tolist(),
        "cohort_patients_with_ambiguous_copy_number": cases.index[cases["in_cohort"] & cases["ambiguous_copy_number"]].tolist(),
        "workflow": WORKFLOW,
        "labelling_rule": "label 1 if MTAP copy_number == 0, else 0 (create_data_summary.py)",
    }
    return cases, flow


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slide-table", required=True, help="dataset_v1.parquet")
    ap.add_argument("--cohort", default=str(ROOT / "data" / "cohort.csv"))
    ap.add_argument("--out-csv", default=str(ROOT / "data" / "mtap_copy_number.csv"))
    ap.add_argument("--out-json", default=str(ROOT / "outputs" / "cohort_flow.json"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    files = list_copy_number_files()
    print(f"{len(files)} ABSOLUTE LiftOver gene-level files; reading the MTAP row of each")
    per_file = mtap_copy_numbers(files, workers=args.workers)
    cases, flow = build_tables(per_file, list_diagnostic_slide_cases(),
                               pd.read_parquet(args.slide_table, columns=["case_id", "is_mtap_mutation"]),
                               pd.read_csv(args.cohort))
    for path in (args.out_csv, args.out_json):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(args.out_csv, lineterminator="\n")
    with open(args.out_json, "w", newline="\n") as f:
        json.dump(flow, f, indent=2)
        f.write("\n")
    for step in flow["steps"]:
        print(step)
    print("slide-table patients outside the cohort:", flow["slide_table_patients_outside_the_cohort"])
    n = flow["cohort_label_disagreements"]["n"]
    print("COHORT LABEL DISAGREEMENTS:" if n else "all cohort labels agree with the GDC copy number", flow["cohort_label_disagreements"] if n else "")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
