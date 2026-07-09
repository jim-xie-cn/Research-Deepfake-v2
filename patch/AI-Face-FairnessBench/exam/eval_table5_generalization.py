#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_table5_generalization.py
=============================
Reproduce **Table 5 - Fairness generalization results based on the gender
attribute**.

A detector trained on AI-Face (80 %% train split) is evaluated *zero-shot* on an
external face dataset that carries a self-reported / annotated gender label.
The paper uses three such datasets:

    CCv2       (Casual Conversations v2)  -- REAL images only  -> F_OAE, ACC
    DF-Platter                            -- real + fake        -> F_OAE, F_EO, AUC
    GenData                               -- real + fake        -> F_OAE, F_EO, AUC

Only ``F_OAE`` and ``ACC`` are meaningful for CCv2 because every image is real
(no positive class), exactly as noted in the paper's Table-5 caption.

External CSV format (one file per dataset)
------------------------------------------
    Image Path , Target , Gender
    /abs/img1.png , 0 , 0        # 0 = real,   Gender 0 = female
    /abs/img2.png , 1 , 1        # 1 = fake,   Gender 1 = male

* ``--path-col``/``--target-col``/``--gender-col`` let you rename these.
* If the gender column uses strings ('male'/'female') they are mapped
  automatically (male/m/1 -> 1, female/f/0 -> 0).

Example
-------
    python eval_table5_generalization.py \
        --model xception \
        --checkpoint ../training/checkpoints/xception/xception9.pth \
        --dataset CCv2=/data/ccv2_gender.csv \
        --dataset DF-Platter=/data/dfplatter_gender.csv \
        --dataset GenData=/data/gendata_gender.csv \
        --out ../results/table5_xception.csv

Run once per detector; ``append`` mode (default) accumulates all detectors into
one CSV so it can be pivoted into the paper's table layout.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from detector_io import build_detector, ALL_MODELS
from gender_fairness_metrics import compute_gender_metrics, to_prob, format_report

_GENDER_MAP = {"male": 1, "m": 1, "1": 1, "man": 1,
               "female": 0, "f": 0, "0": 0, "woman": 0}


def _coerce_gender(v):
    s = str(v).strip().lower()
    if s in _GENDER_MAP:
        return _GENDER_MAP[s]
    try:
        return int(float(s))
    except ValueError:
        return -1


class ExternalCSV(Dataset):
    """Minimal albumentations-based reader for an external gender-labelled CSV."""

    def __init__(self, csv_file, transform, path_col, target_col, gender_col):
        df = pd.read_csv(csv_file)
        for c in (path_col, target_col, gender_col):
            if c not in df.columns:
                raise ValueError(f"'{c}' not in {csv_file}; columns={list(df.columns)}")
        self.paths = df[path_col].astype(str).tolist()
        self.labels = df[target_col].astype(int).tolist()
        self.genders = [_coerce_gender(g) for g in df[gender_col].tolist()]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = np.array(Image.open(self.paths[i]).convert("RGB"))
        img = self.transform(image=img)["image"]
        return {"image": img, "label": self.labels[i], "gender": self.genders[i]}


def run_one(det, csv_file, args, device):
    transform = det.make_transform([""])          # no post-processing here
    ds = ExternalCSV(csv_file, transform, args.path_col, args.target_col, args.gender_col)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    logits, labels, genders = [], [], []
    for batch in tqdm(dl, desc=os.path.basename(csv_file)):
        out = det.logits(batch["image"].to(device))
        logits.append(np.atleast_1d(out) if out.ndim == 1 else out)
        labels.extend(batch["label"].numpy().tolist())
        genders.extend(batch["gender"].numpy().tolist())
    logits = np.concatenate(logits, axis=0)
    probs = to_prob(logits, det.activation)
    return compute_gender_metrics(np.array(labels), probs, np.array(genders),
                                  threshold=args.threshold)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--checkpoint", required=True, help="trained .pth for --model")
    ap.add_argument("--dataset", action="append", required=True,
                    metavar="NAME=CSV",
                    help="external dataset as NAME=path/to.csv (repeatable)")
    ap.add_argument("--path-col", default="Image Path")
    ap.add_argument("--target-col", default="Target")
    ap.add_argument("--gender-col", default="Gender")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="../results/table5_results.csv")
    ap.add_argument("--overwrite", action="store_true",
                    help="start a fresh --out instead of appending")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[table5] building {args.model} from {args.checkpoint} on {device}")
    det = build_detector(args.model, args.checkpoint, device)

    rows = []
    for spec in args.dataset:
        if "=" not in spec:
            sys.exit(f"--dataset must be NAME=CSV, got '{spec}'")
        name, csv_file = spec.split("=", 1)
        if not os.path.exists(csv_file):
            sys.exit(f"[table5] dataset CSV not found: {csv_file}")
        m = run_one(det, csv_file, args, device)
        print(format_report(name, args.model, m))
        rows.append({"detector": args.model, "dataset": name,
                     "N": m["N"], "F_OAE": m["F_OAE"], "F_EO": m["F_EO"],
                     "ACC": m["ACC"], "AUC": m["AUC"]})

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.exists(out) and not args.overwrite:
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\n[table5] results written to {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
