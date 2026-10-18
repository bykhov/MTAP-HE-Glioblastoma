# Preprocessing

Scripts in the order they are run. Each takes `--base-dir <raw_data>` (the folder with `<case_id>/<slide_name>/`) and `--workers N`.
`1.2` to `1.4` and `1.3.1` run from inside this folder; `1.5` and `1.6` also need the repository root on `PYTHONPATH`.
OpenSlide is required for `1.2` to `1.4`.

1. `1.2-create thumbnail.py`: slide thumbnail.
2. `1.3-extract_patches.py`: 512 x 512 px tiles at 20x (level 0, resized for 40x slides); mostly white tiles (mean of the 1-bit image at least 220/255) are dropped.
3. `1.3.1-extract_patch_coordinates.py`: tile coordinates of all slides in one table (`patch_coordinates.parquet`), used for attention maps.
4. `1.4-get_size_info.py`: magnification and slide size.
5. `1.5-kmeans_clustering.py`: per slide, k-means (k = 3) on HSV tile features; the lowest-saturation cluster is dropped as background.
6. `1.6-color_normalization.py --ref-case-dir <reference slide folder>`: Reinhard normalization of the tissue tiles to the reference slide TCGA-12-0819-01Z-00-DX1.
7. `utils/create_data_summary.py --data_dir <raw_data> --output <dir>`: patch table with labels (MTAP copy number 0 = 1, else 0), the input of `extract_features.py`.

Other files: `stainNorm_Reinhard.py` (Reinhard normalizer), `utils.py` (slide folder helpers).

Directory layout per slide after all stages:

```
<raw_data>/<case_id>/<slide_name>/
|-- <slide_name>.svs
|-- <slide_name>_thumbnail.png
|-- slide_size_info.pickle
|-- patches/<n>.jpg
|-- mask/<slide_name>_mask.png
|-- clustering_results/npy_files/cluster.npy, png_files/cluster.png
|-- 20xNormalizedPatches/<n>.jpg
```
