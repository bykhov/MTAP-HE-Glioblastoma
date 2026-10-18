# Pipeline: from slides to predictions

Run from the repository root unless noted.

| Step | Command | Output |
|---|---|---|
| 1. Download and arrange | `data_preparation_scripts/data_collection/` (see its README) | `raw_data/<case_id>/<slide_name>/<slide_name>.svs` and the copy-number TSV per case |
| 2. Tiling and tissue selection | `data_preparation_scripts/data_preprocessing/` stages 1.2 to 1.5 (see its README) | 512 px tiles at 20x, tissue clusters per slide |
| 3. Stain normalization | `1.6-color_normalization.py --ref-case-dir <TCGA-12-0819-01Z-00-DX1 folder> --base-dir <raw_data>` | `20xNormalizedPatches/` |
| 4. Patch table and labels | `data_preparation_scripts/data_preprocessing/utils/create_data_summary.py --data_dir <raw_data> --output <dir>` | `slides_summary.parquet` (label 1 if MTAP copy number is 0) |
| 5. UNI embeddings | `python extract_features.py source_parquet=<slides_summary.parquet>` | one H5 per slide, slide manifest (`configs/extract_config.yaml`) |
| 6. Patient bags | `python create_patient_h5s.py --slide-manifest source/precomputed_features/uni/feature_manifest.parquet --out-dir source/precomputed_features/uni-patient-level --manifest-out source/precomputed_features/uni-patient-level/feature_manifest.parquet` | one H5 per patient, patient manifest |
| 7. Training and testing | `python run.py` | `outputs/<run_name>/fold{k}_test_predictions.csv`, checkpoints, attention outputs; `outputs/folds.csv` |
| 8. Attention maps | `python utils/plot_attention_maps.py --base_dir <slide folder> --attention_scores_path <attention CSV>` | `<slide>_attention.png` |

The cohort is restricted to MTAP copy number 0 (label 1) or 2 (label 0); `scripts/cohort_copy_number.py` rebuilds the copy-number table and the cohort flow from the GDC, and `scripts/cohort_summary.py` the cohort table.

Cross-validation: `StratifiedKFold(5, shuffle=True, random_state=42)` on the label and a tile-count bucket (above or below the cohort median). In run *i*, fold *i* is the validation set, fold *(i + 1) mod 5* the test set, and the other three folds train. Class balance comes from a weighted sampler plus `pos_weight = 0.5959`. The checkpoint with the best validation AUROC is tested, and one threshold (`lit_module.threshold`, 0.5) is used for all metrics and predictions.
