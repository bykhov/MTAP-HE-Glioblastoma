# MTAP/9p21 homozygous deletion in glioblastoma H&E whole-slide images

Code and patient-level results for:

> Geva R, Sobolev D, Kidron D, Tabach Y, Hoffer O, Bykhovsky D. Routine H&E whole-slide images of glioblastoma carry a detectable signal of MTAP/9p21 homozygous deletion.

TCGA-GBM, 233 patients (146 MTAP homozygous deletion, 87 copy-number neutral), 515 diagnostic slides. UNI tile embeddings are pooled per patient by a gated attention MIL model (CLAM_SB architecture trained with the bag-level loss) and evaluated by 5-fold patient-level cross-validation. Pooled out-of-fold AUROC 0.704.

## Contents

| Path | |
|---|---|
| `outputs/published/` | out-of-fold predictions of all 233 patients (`fold{k}_test_predictions.csv`) |
| `outputs/folds.csv` | fold assignment (seed 42) |
| `outputs/*.json`, `outputs/*.csv` | results behind the tables and figures |
| `data/` | cohort, copy-number, IDH and clinical tables (GDC and cBioPortal) |
| `source/`, `utils/*.parquet` | slide/tile table, patient manifest and tile coordinates |
| `analysis/`, `scripts/` | statistics, tables and figures |
| `data_preparation_scripts/`, `extract_features.py`, `create_patient_h5s.py`, `run.py` | pipeline from slides to predictions (see `docs/PIPELINE.md`) |
| `tests/` | unit and end-to-end tests on synthetic data (CPU) |

## Rebuild the tables and figures

CPU only, from the repository root, a few minutes (`pip install -r requirements-analysis.txt`):

```bash
python scripts/validate_published_predictions.py --manifest source/precomputed_features/uni-patient-level/feature_manifest.parquet
python analysis/metrics_report.py                     # Table 1, Fig 2
python analysis/confound_checks.py --manifest data/cohort.csv \
    --predictions "outputs/published/fold*_test_predictions.csv" --idh data/idh_status.csv --out outputs/confound
python analysis/supplementary_analyses.py --clinical data/clinical.csv   # S1 Fig, S2 Table
python analysis/paper_extras.py                       # Table 3, S1 Text
python analysis/paper_figures.py                      # Figs 1 and 3
```

Figures are written to `outputs/figures/`.

## Run the pipeline

```bash
conda env create -f environment.yml    # or: pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
huggingface-cli login                  # UNI weights are gated (MahmoodLab/uni); set HF_HUB_CACHE to use a local copy
python run.py                          # 5-fold training and testing, configs/train_config.yaml
```

The slides and the ABSOLUTE gene-level copy-number files are open access on the [GDC portal](https://portal.gdc.cancer.gov/projects/TCGA-GBM). Preprocessing and feature extraction are described in `docs/PIPELINE.md`.

## Tests

```bash
python -m pytest -q
```

## License

MIT (see `LICENSE`). TCGA data are subject to the [GDC data use policies](https://gdc.cancer.gov/access-data/data-access-policies); UNI is subject to its own license on Hugging Face.
