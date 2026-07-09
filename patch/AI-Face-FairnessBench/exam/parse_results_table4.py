#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parse_results_table4.py
=======================
Turn the console / log output of ``training/fairness_metrics.py::acc_fairness``
into the grid of **Table 4 - Overall performance comparison of different methods
on the AI-Face dataset**.

``acc_fairness`` (called at the end of every ``train_test*.py`` run) prints all
the numbers Table 4 needs, but scattered across several lines and with a
different scaling convention.  This script scrapes the *last* (final-epoch)
occurrence of each line from one or more training logs and assembles the table.

Line -> Table-4 mapping (group_1=Gender, group_2=Skin Tone, group_3=Age, inter=Intersection)
--------------------------------------------------------------------------------------------
  "... with group_1 : AUC .. fpr .. acc .. precision .. EER .."   -> Utility (x100)
  "inter_F_meo,g1_F_meo,g2_F_meo, g3_F_meo: .."                    -> F_MEO  (already x100)
  "inter_F_DP,g1_F_DP,g2_F_DP,g3_F_DP: .."                         -> F_DP   (already x100)
  "inter_foae, g1_foae, g2_foae, g3_foae: .."                      -> F_OAE  (already x100)
  "F_G,F_EFPR,F_EO,F_A,F_COV of group_k: a | efpr | b | .."        -> F_EO = b (x100 here)
  "F_G_inter,F_EFPR_inter,F_EO_inter,F_A_inter: .. | .. | b | .."   -> F_EO(inter) = b (x100)

Note the scaling quirk: F_MEO / F_DP / F_OAE are printed already multiplied by
100, but F_EO ('b') is printed as a raw fraction, so we multiply it by 100 to be
consistent with the paper. Utility (AUC/ACC/AP/EER/FPR) is printed as a fraction
and multiplied by 100.

Usage
-----
    python parse_results_table4.py \
        --log xception=../training/checkpoints/xception/log_training.txt \
        --log efficientnet=../training/checkpoints/efficientnet/log_training.txt \
        --find-csv ../results/individual_fairness.csv \
        --out-md  ../results/table4.md \
        --out-csv ../results/table4_long.csv
"""

import argparse
import os
import re

NUM = r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"


def _last(pattern, text, groups):
    m = list(re.finditer(pattern, text))
    if not m:
        return None
    g = m[-1].groups()
    return [float(g[i]) for i in groups]


def parse_log(text: str) -> dict:
    """Extract all Table-4 quantities for one detector from its log text."""
    d = {}

    # ---- overall utility from group_1 -----------------------------------
    util = _last(
        r"with group_1 : AUC:\s*" + NUM + r"\s*\|\s*ErrorRate:\s*" + NUM +
        r"\s*\|\s*fpr:\s*" + NUM + r"\s*\|\s*tpr:\s*" + NUM +
        r"\s*\|\s*acc:\s*" + NUM + r"\s*\|\s*precision:\s*" + NUM +
        r"\s*\|\s*EER:\s*" + NUM, text, [0, 2, 4, 5, 6])
    if util:
        auc, fpr, acc, ap, eer = util
        d["AUC"], d["FPR"], d["ACC"], d["AP"], d["EER"] = (
            100 * auc, 100 * fpr, 100 * acc, 100 * ap, 100 * eer)

    # ---- F_MEO / F_DP / F_OAE (already x100): inter | g1 | g2 | g3 --------
    for key, pat in (("MEO", r"inter_F_meo.*?:\s*"),
                     ("DP",  r"inter_F_DP.*?:\s*"),
                     ("OAE", r"inter_foae.*?:\s*")):
        vals = _last(pat + NUM + r"\s*\|\s*" + NUM + r"\s*\|\s*" + NUM +
                     r"\s*\|\s*" + NUM, text, [0, 1, 2, 3])
        if vals:
            inter, g1, g2, g3 = vals
            d[f"{key}_inter"], d[f"{key}_gender"] = inter, g1
            d[f"{key}_skintone"], d[f"{key}_age"] = g2, g3

    # ---- F_EO per group ('b', raw fraction -> x100) ----------------------
    for m in re.finditer(
            r"F_EO, F_A, F_COV of group_(\d):\s*" + NUM + r"\s*\|\s*" + NUM +
            r"\s*\|\s*" + NUM, text):
        gi = int(m.group(1)); b = float(m.group(4))
        name = {1: "gender", 2: "skintone", 3: "age"}.get(gi)
        if name:
            d[f"EO_{name}"] = 100 * b
    inter_eo = _last(
        r"F_EO_inter, F_A_inter:\s*" + NUM + r"\s*\|\s*" + NUM + r"\s*\|\s*" + NUM,
        text, [2])
    if inter_eo:
        d["EO_inter"] = 100 * inter_eo[0]
    return d


ATTRS = [("Skin Tone", "skintone"), ("Gender", "gender"),
         ("Age", "age"), ("Intersection", "inter")]
FAIR = [("F_MEO", "MEO"), ("F_DP", "DP"), ("F_OAE", "OAE"), ("F_EO", "EO")]
UTIL = ["AUC", "ACC", "AP", "EER", "FPR"]


def build_markdown(results: dict, find: dict) -> str:
    dets = list(results.keys())
    out = ["# Table 4 - Overall performance comparison on AI-Face\n",
           "Fairness (%) lower is better; Utility (%) higher is better "
           "(EER/FPR lower).\n"]
    header = "| Measure | Attribute | Metric | " + " | ".join(dets) + " |"
    sep = "|" + "---|" * (3 + len(dets))
    out += [header, sep]
    for attr_label, attr_key in ATTRS:
        for metric_label, metric_key in FAIR:
            cells = []
            for det in dets:
                v = results[det].get(f"{metric_key}_{attr_key}")
                cells.append("-" if v is None else f"{v:.3f}")
            out.append(f"| Fairness | {attr_label} | {metric_label} | " +
                       " | ".join(cells) + " |")
    # Individual / F_IND
    cells = [f"{find.get(det, float('nan')):.3f}" if det in find else "-"
             for det in dets]
    out.append("| Fairness | Individual | F_IND | " + " | ".join(cells) + " |")
    # Utility
    for u in UTIL:
        cells = []
        for det in dets:
            v = results[det].get(u)
            cells.append("-" if v is None else f"{v:.3f}")
        out.append(f"| Utility | - | {u} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", required=True, metavar="MODEL=PATH",
                    help="training log for a detector (repeatable)")
    ap.add_argument("--find-csv", default=None,
                    help="optional individual_fairness.py output to fill F_IND")
    ap.add_argument("--out-md", default="../results/table4.md")
    ap.add_argument("--out-csv", default="../results/table4_long.csv")
    args = ap.parse_args()

    results = {}
    for spec in args.log:
        if "=" not in spec:
            raise SystemExit(f"--log must be MODEL=PATH, got '{spec}'")
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"[table4] WARNING: log not found, skipping: {path}")
            continue
        with open(path, "r", errors="ignore") as fh:
            results[name] = parse_log(fh.read())
        print(f"[table4] parsed {name}: {len(results[name])} values")

    find = {}
    if args.find_csv and os.path.exists(args.find_csv):
        import pandas as pd
        fdf = pd.read_csv(args.find_csv)
        find = dict(zip(fdf["detector"], fdf["F_IND"]))

    if not results:
        raise SystemExit("[table4] no logs parsed; nothing to write")

    md = build_markdown(results, find)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    with open(args.out_md, "w") as fh:
        fh.write(md)
    print(f"[table4] wrote {args.out_md}")

    # long-format CSV
    import csv
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["detector", "measure", "attribute", "metric", "value"])
        for det, vals in results.items():
            for attr_label, attr_key in ATTRS:
                for metric_label, metric_key in FAIR:
                    v = vals.get(f"{metric_key}_{attr_key}")
                    if v is not None:
                        w.writerow([det, "Fairness", attr_label, metric_label, f"{v:.5f}"])
            if det in find:
                w.writerow([det, "Fairness", "Individual", "F_IND", f"{find[det]:.5f}"])
            for u in UTIL:
                if u in vals:
                    w.writerow([det, "Utility", "-", u, f"{vals[u]:.5f}"])
    print(f"[table4] wrote {args.out_csv}")
    print("\n" + md)


if __name__ == "__main__":
    main()
