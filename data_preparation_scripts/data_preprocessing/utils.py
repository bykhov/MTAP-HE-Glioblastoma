import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def get_svs_path(case_dir):
    svs_files = [os.path.join(case_dir, f)
             for f in os.listdir(case_dir)
             if f.lower().endswith(".svs")]
    return svs_files[0]


def iter_slide_dirs(base_dir):
    """Slide folders ``<base_dir>/<case_id>/<slide_name>/`` in sorted order (the layout the stages expect)."""
    for case_dir in sorted(Path(base_dir).iterdir()):
        if not case_dir.is_dir():
            continue
        for svs_dir in sorted(case_dir.iterdir()):
            if svs_dir.is_dir():
                yield svs_dir


def run_over_slides(func, base_dir, workers=1):
    """
    Call ``func(svs_dir)`` for every slide folder. ``workers`` > 1 uses processes (``func`` must be a
    module-level function); with 1 (the default) the slides are processed one after the other as before.
    """
    slide_dirs = list(iter_slide_dirs(base_dir))
    if workers <= 1:
        return [func(d) for d in slide_dirs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, slide_dirs))
