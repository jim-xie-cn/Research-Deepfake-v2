#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_figure6.py
===============
Render **Figure 6 - Performance ratio after vs. before post-processing** from the
CSV produced by ``eval_figure6_postprocessing.py``.

Four panels, one per model type (paper Sec. 4):

    Naive            : xception, efficientnet, vit
    Frequency        : f3net, spsl, srm
    Spatial          : ucf, UnivFD, core
    Fairness-enhanced: daw_fdd, dag_fdd, fair_df_detector

For every detector two curves are drawn against the post-processing axis
(JC, GB, HSV, BC, RT, RC):

    solid  + marker : ratio of F_EO   (fairness)
    dashed          : ratio of AUC    (utility)

A grey line at y = 1.0 marks "no change".  Points near 1.0 => robust.

Usage
-----
    python plot_figure6.py --in ../results/figure6_metrics.csv \
                           --out ../results/figure6.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PANELS = [
    ("Naive",             ["xception", "efficientnet", "vit"]),
    ("Frequency",         ["f3net", "spsl", "srm"]),
    ("Spatial",           ["ucf", "UnivFD", "core"]),
    ("Fairness-enhanced", ["daw_fdd", "dag_fdd", "fair_df_detector"]),
]
X_ORDER = ["JC", "GB", "HSV", "BC", "RT", "RC"]
PRETTY = {
    "xception": "Xception", "efficientnet": "EfficientB4", "vit": "ViT-B/16",
    "f3net": "F3Net", "spsl": "SPSL", "srm": "SRM",
    "ucf": "UCF", "UnivFD": "UnivFD", "core": "CORE",
    "daw_fdd": "DAW-FDD", "dag_fdd": "DAG-FDD", "fair_df_detector": "PG-FDD",
}
COLORS = plt.cm.tab10.colors


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="../results/figure6_metrics.csv")
    ap.add_argument("--out", default="../results/figure6.png")
    ap.add_argument("--ymax", type=float, default=3.5, help="y-axis upper clip")
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise SystemExit(f"[plot_figure6] input CSV not found: {args.inp}\n"
                         f"Run eval_figure6_postprocessing.py for each detector first.")
    df = pd.read_csv(args.inp)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2), sharey=True)
    for ax, (panel_name, dets) in zip(axes, PANELS):
        ax.axhline(1.0, color="grey", lw=1, ls=":", zorder=0)
        for ci, det in enumerate(dets):
            sub = df[df["detector"] == det]
            if sub.empty:
                continue
            sub = sub.set_index("method").reindex(X_ORDER)
            xs = range(len(X_ORDER))
            color = COLORS[ci % len(COLORS)]
            ax.plot(xs, sub["ratio_F_EO"].values, marker="o", color=color,
                    lw=1.8, label=f"$F_{{EO}}$ ({PRETTY.get(det, det)})")
            ax.plot(xs, sub["ratio_AUC"].values, marker="^", color=color,
                    lw=1.4, ls="--", alpha=0.8,
                    label=f"AUC ({PRETTY.get(det, det)})")
        ax.set_title(panel_name, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(X_ORDER)))
        ax.set_xticklabels(X_ORDER)
        ax.set_xlabel("Post-Processing Methods")
        ax.set_ylim(0, args.ymax)
        ax.legend(fontsize=6, ncol=1, loc="upper left")
    axes[0].set_ylabel("Ratio (after / before)")

    fig.suptitle("Figure 6. Performance ratio after vs. before post-processing "
                 "(closer to 1.0 = more robust)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"[plot_figure6] saved {out}")


if __name__ == "__main__":
    main()
