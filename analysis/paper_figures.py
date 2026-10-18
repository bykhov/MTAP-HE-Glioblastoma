"""
Fig 1 (design and cohort) and Fig 3 (robustness and added value) of the paper.

Inputs: outputs/cohort_flow.json, outputs/cohort_summary.json, data/cohort.csv, outputs/paper_robustness.csv and
outputs/paper_extras.json (written by analysis/paper_extras.py). Outputs: outputs/figures/Fig1.{png,tif} and
outputs/figures/Fig3.{png,tif} (RGB, LZW, 300 dpi, 7.5 in wide, Arial). CPU only; numpy, pandas, matplotlib, Pillow.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.metrics_report import save_plos_tiff  # noqa: E402

HD, NE, GR = "#1f4e79", "#c47a1f", "#8a8a8a"   # same colors as Fig 2 and the earlier Fig 3
BASE, SMALL = 8, 7


def _style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": BASE, "axes.labelsize": BASE, "axes.titlesize": BASE,
                         "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "legend.fontsize": SMALL,
                         "axes.spines.top": False, "axes.spines.right": False, "xtick.direction": "out",
                         "ytick.direction": "out", "legend.frameon": False, "savefig.dpi": 300})


def _letter(fig, ax, s, dx=-0.01, dy=0.0):
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, s, fontsize=10, fontweight="bold", ha="right", va="bottom")


def _box(ax, x, y, w, h, title, body, fc, ec, title_pad=0.03, body_gap=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.015", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h - title_pad, title, ha="center", va="top", fontsize=BASE, fontweight="bold")
    ax.text(x + w / 2, y + h - title_pad - body_gap, body, ha="center", va="top", fontsize=SMALL, linespacing=1.3)


def _arrow(ax, x0, y0, x1, y1, color="#444444", style="-|>", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=0.9, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=8))


# ------------------------------------------------------------------------------------------------------------
def fig1(root, out_dir):
    flow = json.loads((root / "outputs" / "cohort_flow.json").read_text())
    summ = json.loads((root / "outputs" / "cohort_summary.json").read_text())
    coh = pd.read_csv(root / "data" / "cohort.csv")
    steps = {s["patients"]: s for s in flow["steps"]}
    n562, s397, s265 = 562, steps[397], steps[265]
    n_slides, n_tiles = summ["n_slides"], summ["n_tiles"]

    fig = plt.figure(figsize=(7.5, 5.6))
    # A: pipeline
    axA = fig.add_axes([0.015, 0.76, 0.97, 0.20])
    axA.set_axis_off()
    axA.set_xlim(0, 1)
    axA.set_ylim(0, 1)
    boxes = [
        ("Diagnostic slides", f"{n_slides} H&E WSIs\n{len(coh)} patients\nTCGA-GBM", "#e8eef5", HD),
        ("Tiles", "512 px tiles at 20x\ntissue kept by HSV\nk-means per slide;\nReinhard normalization",
         "#eef3e8", "#5b7f3a"),
        ("UNI embedding", "tile resized to 224 px\n(about 1.14 \u00b5m/px)\n1,024-d vector per tile", "#f7f0e2", NE),
        ("Gated attention MIL", "all tiles of a patient\nattention-weighted mean\none logit per patient", "#e8eef5",
         HD),
        ("Evaluation", "5-fold patient-level CV\none out-of-fold\nprobability per patient", "#f0f0f0", GR),
    ]
    w, gap = 0.166, 0.037
    for i, (t, b, fc, ec) in enumerate(boxes):
        x = 0.01 + i * (w + gap)
        _box(axA, x, 0.04, w, 0.92, t, b, fc, ec, title_pad=0.07, body_gap=0.2)
        if i < len(boxes) - 1:
            _arrow(axA, x + w + 0.004, 0.5, x + w + gap - 0.004, 0.5)
    _letter(fig, axA, "A", dx=0.012, dy=0.01)

    # B: cohort flow
    axB = fig.add_axes([0.035, 0.03, 0.55, 0.66])
    axB.set_axis_off()
    axB.set_xlim(0, 1)
    axB.set_ylim(0, 1)
    main_x, main_w, side_x, side_w, h = 0.01, 0.50, 0.60, 0.39, 0.17
    ys = [0.82, 0.565, 0.31, 0.02]
    n_cn_excl = n562 - 397
    n_noslide = 397 - 265
    n_notincl = 265 - 232       # 145 of the 146 cohort HD patients and all 87 neutral are among the 265
    texts = [
        (f"{n562} TCGA-GBM patients", "ABSOLUTE MTAP copy number"),
        ("397 patients", f"copy number 0 ({s397['copy_number_0_HD']}) or 2 ({s397['copy_number_2_neutral']})"),
        ("265 patients", f"with a diagnostic slide\n({s265['copy_number_0_HD']} HD, {s265['copy_number_2_neutral']} neutral)"),
        (f"{len(coh)} patients analyzed", f"{int(coh.label.sum())} HD*, {int((coh.label == 0).sum())} neutral\n"
                                          f"{n_slides} slides, {n_tiles / 1e6:.2f} M tiles"),
    ]
    for (t, b), y in zip(texts, ys):
        fc, ec = ("#e8eef5", HD) if y != ys[-1] else ("#dfe8f2", HD)
        _box(axB, main_x, y, main_w, h, t, b, fc, ec, title_pad=0.02, body_gap=0.05)
    sides = [(f"{n_cn_excl} excluded", "copy number 1 or > 2"),
             (f"{n_noslide} excluded", "no diagnostic slide"),
             (f"{n_notincl} not analyzed", "23 HD, 10 neutral\n(slide quality review)")]
    for (t, b), y0, y1 in zip(sides, ys[:-1], ys[1:]):
        ym = (y0 + y1 + h) / 2
        _arrow(axB, main_x + main_w / 2, y0, main_x + main_w / 2, y1 + h)
        axB.plot([main_x + main_w / 2, side_x], [ym, ym], color=GR, lw=0.8)
        _box(axB, side_x, ym - h / 2, side_w, h, t, b, "#f6f6f6", GR, title_pad=0.02, body_gap=0.05)
    _letter(fig, axB, "B", dx=0.0, dy=-0.03)

    # C: tiles per patient
    axC = fig.add_axes([0.67, 0.10, 0.31, 0.52])
    bins = np.logspace(2, np.log10(30000), 22)
    for lab, col, name in [(1, HD, "HD"), (0, NE, "Neutral")]:
        v = coh.n_tiles[coh.label == lab]
        axC.hist(v, bins=bins, histtype="step", lw=1.2, color=col, label=f"{name} (n = {len(v)})")
    med = coh.n_tiles.median()
    axC.axvline(med, color=GR, ls=":", lw=0.9)
    axC.text(med * 1.08, axC.get_ylim()[1] * 0.97, f"median {med:,.0f}", color=GR, fontsize=SMALL, va="top")
    axC.set_xscale("log")
    axC.set_xticks([100, 1000, 10000])
    axC.set_xticklabels(["100", "1k", "10k"])
    axC.set_xlabel("Tissue tiles per patient")
    axC.set_ylabel("Patients")
    axC.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), handlelength=1.2)
    _letter(fig, axC, "C", dx=-0.04, dy=0.02)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "Fig1.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    save_plos_tiff(png, out_dir / "Fig1.tif")
    return png


# ------------------------------------------------------------------------------------------------------------
def fig3(root, out_dir):
    rob = pd.read_csv(root / "outputs" / "paper_robustness.csv")
    ex = json.loads((root / "outputs" / "paper_extras.json").read_text())
    allp = rob[rob.row == "All patients"].iloc[0]
    order = [
        ("Technical", ["Same tissue source site", "Same tile-count tertile", "One slide only",
                       "Primary-tumor copy number only"]),
        ("Clinical and molecular", ["Same age tertile", "IDH-wild-type", "Excluding IDH-mutant or G-CIMP",
                                    "Same purity tertile"]),
        ("Classical program", ["Same EGFR status", "Same transcriptional subtype"]),
    ]
    fig = plt.figure(figsize=(7.5, 3.8))
    axA = fig.add_axes([0.30, 0.13, 0.33, 0.82])
    y, ticks, labels, heads = 0, [], [], []
    for g, rows in order:
        heads.append((y, g))
        y -= 1
        for r in rows:
            row = rob[rob.row == r].iloc[0]
            open_ = g == "Classical program"
            axA.plot([row.lo, row.hi], [y, y], color=HD, lw=1.2, solid_capstyle="butt")
            axA.plot(row.auroc, y, "o", color=HD, ms=4.5, mfc="white" if open_ else HD, mew=1.2)
            axA.text(0.955, y, f"{row.auroc:.3f} ({row.lo:.3f}\u2013{row.hi:.3f})", fontsize=SMALL, va="center",
                     ha="left", transform=axA.get_yaxis_transform())
            ticks.append(y)
            labels.append(f"{r} (n = {int(row.n)})")
            y -= 1
        y -= 0.4
    axA.axvspan(allp.lo, allp.hi, color=HD, alpha=0.08, lw=0)
    axA.axvline(allp.auroc, color=HD, ls="--", lw=0.8)
    axA.axvline(0.5, color=GR, ls=":", lw=0.8)
    axA.text(allp.auroc, 0.6, f"all patients {allp.auroc:.3f}", color=HD, fontsize=SMALL, ha="center",
             va="bottom")
    axA.set_yticks(ticks)
    axA.set_yticklabels(labels)
    for yy, g in heads:
        axA.text(-0.02, yy, g, transform=axA.get_yaxis_transform(), ha="right", va="center", fontsize=BASE,
                 fontweight="bold")
    axA.set_ylim(y + 0.6, 1.2)
    axA.set_xlim(0.40, 0.95)
    axA.set_xlabel("Model AUROC (95% CI)")
    axA.tick_params(axis="y", length=0)
    axA.spines["left"].set_visible(False)
    _letter(fig, axA, "A", dx=-0.255, dy=-0.02)

    axB = fig.add_axes([0.80, 0.13, 0.18, 0.82])
    pairs = [("Age", ex["incremental_age"]), ("Age +\nsubtype", ex["incremental"])]
    for i, (name, d) in enumerate(pairs):
        yy = -i
        a0, a1 = d["oof_auroc_clinical"], d["oof_auroc_clinical_plus_image"]
        axB.plot([a0["ci95"][0], a0["ci95"][1]], [yy + 0.12] * 2, color=GR, lw=1.2)
        axB.plot(a0["auroc"], yy + 0.12, "o", color=GR, ms=4.5)
        axB.plot([a1["ci95"][0], a1["ci95"][1]], [yy - 0.12] * 2, color=HD, lw=1.2)
        axB.plot(a1["auroc"], yy - 0.12, "o", color=HD, ms=4.5)
        lo, hi = d["oof_auroc_difference_ci95_bootstrap"]
        fmt = lambda v: f"{v:+.3f}".replace("-", "\u2212")  # noqa: E731
        axB.text(0.505, yy - 0.42, f"gain {fmt(d['oof_auroc_difference'])}\n({fmt(lo)} to {fmt(hi)})",
                 fontsize=SMALL, va="center", ha="left")
        axB.text(0.49, yy, f"{name}\n(n = {d['n']})", ha="right", va="center", fontsize=SMALL)
    axB.plot([], [], "o-", color=GR, ms=4, lw=1.2, label="covariates")
    axB.plot([], [], "o-", color=HD, ms=4, lw=1.2, label="covariates + image")
    axB.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=1, handlelength=1.4)
    axB.set_ylim(-2.25, 0.45)
    axB.set_xlim(0.5, 0.9)
    axB.set_yticks([])
    axB.spines["left"].set_visible(False)
    axB.set_xlabel("Out-of-fold AUROC")
    _letter(fig, axB, "B", dx=-0.075, dy=-0.02)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "Fig3.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    save_plos_tiff(png, out_dir / "Fig3.tif")
    return png


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args()
    root = Path(a.root)
    _style()
    print(fig1(root, root / "outputs" / "figures"))
    print(fig3(root, root / "outputs" / "figures"))


if __name__ == "__main__":
    main()
