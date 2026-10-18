# Data collection

1. Download the TCGA-GBM diagnostic slides with `gdc-client` into a temporary folder, together with the GDC portal `metadata.json`.
2. `python rearrange_files.py --download-dir <download folder> --raw-data <raw_data>` moves the files into `raw_data/<case_id>/`.
3. Repeat steps 1 and 2 for the ABSOLUTE LiftOver gene-level copy-number TSV files.
4. `python svs_organizer.py --base-dir <raw_data>` moves each slide into its own folder `raw_data/<case_id>/<slide_name>/`.
