#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_figure6_postprocessing.py
==============================
Reproduce the raw numbers behind **Figure 6 - Performance ratio after vs. before
post-processing** (fairness-robustness evaluation, paper Sec. 5.2).

Each trained detector is evaluated on the AI-Face **test set** under 6 image
post-processing operations plus a clean baseline.  For every operation we record
two quantities on the *intersectional* (Gender x Skin-Tone, 6 groups) split:

    F_EO(inter)   -- Equal-Odds fairness gap (lower is fairer)
    AUC           -- utility

The figure plots the *ratio*  metric_after / metric_before ; a ratio of 1.0
means "post-processing changed nothing" (perfect robustness).  Because it is a
ratio, the usual x100 scaling of F_EO cancels out, so we keep the raw values.

Post-processing operations (implemented in training/transform.py)
-----------------------------------------------------------------
    JC  jpeg_compression            (quality 60 / 80 for vit-clip)
    GB  gaussian_blur               (kernel 5 / 3 for vit-clip)
    HSV hue_saturation_value        (shift limit 50)
    BC  random_brightness_contrast  (limit 0.8 / 0.4 for vit-clip)
    RT  rotation                    (limit 45 / 30 for vit-clip)
    RC  random_crop                 (224 crop)

The clean baseline uses only Resize+Normalize (methods=['']).

Test CSV columns read
---------------------
    Image Path , Target , Intersection
where Intersection in {0..5} encodes  0=F-Light 1=F-Medium 2=F-Dark
3=M-Light 4=M-Medium 5=M-Dark  (the repo's skin-tone x gender convention).

Example
-------
    python eval_figure6_postprocessing.py \
        --model xception \
        --checkpoint ../training/checkpoints/xception/xception9.pth \
        --test-csv   ../dataset/test.csv \
        --out        ../results/figure6_metrics.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from detector_io import build_detector, ALL_MODELS
from gender_fairness_metrics import to_prob

# label -> (albumentations method key). Order matches the paper's x-axis.
POSTPROCESS = [
    ("baseline", ""),
    ("JC",  "jpeg_compression"),
    ("GB",  "gaussian_blur"),
    ("HSV", "hue_saturation_value"),
    ("BC",  "random_brightness_contrast"),
    ("RT",  "rotation"),
    ("RC",  "random_crop"),
]


class TestCSV(Dataset):
    def __init__(self, csv_file, transform, path_col, target_col, inter_col):
        df = pd.read_csv(csv_file)
        for c in (path_col, target_col, inter_col):
            if c not in df.columns:
                raise ValueError(f"'{c}' not in {csv_file}; columns={list(df.columns)}")
        self.paths = df[path_col].astype(str).tolist()
        self.labels = df[target_col].astype(int).tolist()
        self.inter = df[inter_col].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = np.array(Image.open(self.paths[i]).convert("RGB"))
        img = self.transform(image=img)["image"]
        return {"image": img, "label": self.labels[i], "inter": self.inter[i]}


def _rates(label, pred_bin):
    n_pos = (label == 1).sum()
    n_neg = (label == 0).sum()
    fp = ((pred_bin == 1) & (label == 0)).sum()
    tp = ((pred_bin == 1) & (label == 1)).sum()
    fpr = fp / n_neg if n_neg > 0 else np.nan
    tpr = tp / n_pos if n_pos > 0 else np.nan
    return fpr, tpr


def intersectional_feo_auc(labels, probs, inter, threshold=0.5):
    """F_EO over the 6 intersectional groups (vs. overall) and overall AUC.

    Mirrors ``fairness_metrics.acc_fairness`` -> b_inter and group_1 AUC.
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    inter = np.asarray(inter).astype(int)
    pred_bin = (probs >= threshold).astype(int)

    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else np.nan
    fpr_all, tpr_all = _rates(labels, pred_bin)

    f_eo = 0.0
    for g in range(6):
        m = inter == g
        if m.sum() == 0:
            continue
        fpr_g, tpr_g = _rates(labels[m], pred_bin[m])
        term = 0.0
        if not np.isnan(fpr_g) and not np.isnan(fpr_all):
            term += abs(fpr_g - fpr_all)
        if not np.isnan(tpr_g) and not np.isnan(tpr_all):
            term += abs(tpr_g - tpr_all)
        f_eo += term
    return f_eo, auc


def infer(det, csv_file, method_key, args, device):
    transform = det.make_transform([method_key] if method_key else [""])
    ds = TestCSV(csv_file, transform, args.path_col, args.target_col, args.inter_col)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    logits, labels, inter = [], [], []
    for batch in tqdm(dl, desc=method_key or "baseline", leave=False):
        out = det.logits(batch["image"].to(device))
        logits.append(np.atleast_1d(out) if out.ndim == 1 else out)
        labels.extend(batch["label"].numpy().tolist())
        inter.extend(batch["inter"].numpy().tolist())
    logits = np.concatenate(logits, axis=0)
    probs = to_prob(logits, det.activation)
    return intersectional_feo_auc(labels, probs, inter, args.threshold)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-csv", default="../dataset/test.csv")
    ap.add_argument("--path-col", default="Image Path")
    ap.add_argument("--target-col", default="Target")
    ap.add_argument("--inter-col", default="Intersection")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--methods", default="all",
                    help="comma list of labels to run (JC,GB,HSV,BC,RT,RC) or 'all'")
    ap.add_argument("--out", default="../results/figure6_metrics.csv")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    det = build_detector(args.model, args.checkpoint, device)

    if args.methods == "all":
        plan = POSTPROCESS
    else:
        want = {"baseline"} | set(x.strip() for x in args.methods.split(","))
        plan = [(lab, key) for lab, key in POSTPROCESS if lab in want]
        if not any(lab == "baseline" for lab, _ in plan):
            plan = [("baseline", "")] + plan

    results = {}
    for label, key in plan:
        feo, auc = infer(det, args.test_csv, key, args, device)
        results[label] = (feo, auc)
        print(f"[figure6] {args.model:16s} {label:9s}  F_EO(inter)={feo:.5f}  AUC={auc:.5f}")

    base_feo, base_auc = results.get("baseline", (np.nan, np.nan))

    def _safe_ratio(x, base):
        # ratio after/before; guard div-by-zero and NaN baselines explicitly
        if base is None or np.isnan(base) or base == 0:
            return np.nan
        return x / base

    rows = []
    for label, key in plan:
        if label == "baseline":
            continue
        feo, auc = results[label]
        rows.append({
            "detector": args.model, "method": label,
            "F_EO": feo, "AUC": auc,
            "F_EO_baseline": base_feo, "AUC_baseline": base_auc,
            "ratio_F_EO": _safe_ratio(feo, base_feo),
            "ratio_AUC": _safe_ratio(auc, base_auc),
        })

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.exists(out) and not args.overwrite:
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\n[figure6] metrics written to {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
