#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_pair_dataset.py
====================
Generate the *paired* training CSVs required by the disentanglement-based
detectors (``ucf`` and ``fair_df_detector`` / PG-FDD) from a single flat
``train.csv`` of the AI-Face dataset.

Why this is needed
------------------
`training/dataset/pair_dataset.py` (``pairDataset``) expects **two** CSV files
instead of the single ``train.csv`` used by every other detector::

    train_fake_spe.csv   # ONLY fake rows, MUST contain a 'Specific' column
    train_real.csv       # ONLY real rows

At every step it pairs one fake image (index `idx`) with one *randomly drawn*
real image, so the batch that reaches the network is always 50 % real / 50 %
fake.  This is what the reconstruction / contrastive branches of UCF and PG-FDD
need.

The 'Specific' column
---------------------
`UCFDetector` and `FairDetector` both build a *specific-forgery* head with
``specific_task_number = 4`` and optimise it with plain ``CrossEntropyLoss``.
Inside ``pairDataset`` the **real** samples receive ``spe_label = Target = 0``.
To avoid a label collision the fakes must therefore use classes ``1..3``.  The
AI-Face dataset groups its 37 generators into three families
(Deepfake-video, GAN, Diffusion-Model), which maps cleanly onto::

    0 -> Real          (assigned automatically to the real CSV by pairDataset)
    1 -> Deepfakes     (FF++, DFDC, DFD, Celeb-DF-v2 ...)
    2 -> GANs          (AttGAN, StarGAN, StyleGAN, ProGAN, STGAN, VQGAN ...)
    3 -> DMs           (StableDiffusion, Palette, DALLE2, Midjourney ...)

This is the DeepfakeBench convention the repository is adapted from.  The family
of each fake row is inferred from its image path (see ``FAMILY_RULES``).  If your
annotation CSV already carries an explicit method/family column you can point
``--specific-column`` at it and skip the path heuristic entirely.

Usage
-----
    python make_pair_dataset.py \
        --train-csv   ../dataset/train.csv \
        --out-dir     ../dataset

    # -> writes ../dataset/train_fake_spe.csv and ../dataset/train_real.csv

Then train, e.g.::

    cd ../training
    python train_test.py --model ucf            --dataset_type pair
    python train_test.py --model fair_df_detector --dataset_type pair
"""
import argparse
import os
import re
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Path -> forgery-family heuristics.  First matching rule wins.
# Keys are the 'Specific' class id (1=Deepfakes, 2=GANs, 3=DMs).
# Matching is case-insensitive against the raw image-path string.
# ---------------------------------------------------------------------------
FAMILY_RULES = {
    1: [  # Deepfake video
        r"deepfake", r"faceswap", r"face2face", r"neuraltextures",
        r"faceshifter", r"ff\+\+", r"faceforensics", r"dfdc", r"dfd",
        r"celeb-?df", r"deeperforensics",
    ],
    2: [  # GAN
        r"gan", r"attgan", r"stgan", r"stargan", r"stylegan", r"msggan",
        r"mmdgan", r"progan", r"vqgan",
    ],
    3: [  # Diffusion Model
        r"\bdm\b", r"diffusion", r"stable[_-]?diffusion", r"latent",
        r"palette", r"dalle", r"midjourney", r"dcface", r"\bif\b", r"inpaint",
    ],
}

FAMILY_NAME = {0: "Real", 1: "Deepfakes", 2: "GANs", 3: "DMs"}


def infer_specific(path: str, default: int = 1) -> int:
    """Infer the specific-forgery family id (1/2/3) from an image path."""
    p = str(path).lower()
    for class_id, patterns in FAMILY_RULES.items():
        for pat in patterns:
            if re.search(pat, p):
                return class_id
    return default


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-csv", default="../dataset/train.csv",
                    help="flat training CSV with an 'Image Path' and 'Target' column")
    ap.add_argument("--out-dir", default="../dataset",
                    help="directory to write train_fake_spe.csv / train_real.csv")
    ap.add_argument("--path-column", default="Image Path",
                    help="name of the image-path column in --train-csv")
    ap.add_argument("--target-column", default="Target",
                    help="name of the real(0)/fake(1) column")
    ap.add_argument("--specific-column", default=None,
                    help="OPTIONAL: existing column that already holds the "
                         "specific-family id for fakes (1..3). If given, the "
                         "path heuristic is skipped.")
    ap.add_argument("--default-specific", type=int, default=1,
                    help="family id assigned when the path cannot be classified")
    args = ap.parse_args()

    if not os.path.exists(args.train_csv):
        sys.exit(f"[make_pair_dataset] train CSV not found: {args.train_csv}")

    df = pd.read_csv(args.train_csv)
    for col in (args.path_column, args.target_column):
        if col not in df.columns:
            sys.exit(f"[make_pair_dataset] column '{col}' missing from "
                     f"{args.train_csv}. Found: {list(df.columns)}")

    os.makedirs(args.out_dir, exist_ok=True)

    real_df = df[df[args.target_column] == 0].reset_index(drop=True)
    fake_df = df[df[args.target_column] == 1].reset_index(drop=True)

    if len(fake_df) == 0 or len(real_df) == 0:
        sys.exit("[make_pair_dataset] need both real (Target=0) and fake "
                 f"(Target=1) rows; got real={len(real_df)} fake={len(fake_df)}")

    # ---- assign the 'Specific' family label to the fake rows --------------
    if args.specific_column and args.specific_column in df.columns:
        fake_df = fake_df.copy()
        fake_df["Specific"] = fake_df[args.specific_column].astype(int)
        source = f"column '{args.specific_column}'"
    else:
        fake_df = fake_df.copy()
        fake_df["Specific"] = fake_df[args.path_column].apply(
            lambda p: infer_specific(p, args.default_specific)).astype(int)
        source = "path heuristic (FAMILY_RULES)"

    # 'pair_dataset.py' also reads 'Intersection' from both CSVs; keep every
    # original column so any downstream reader still works.
    fake_path = os.path.join(args.out_dir, "train_fake_spe.csv")
    real_path = os.path.join(args.out_dir, "train_real.csv")
    fake_df.to_csv(fake_path, index=False)
    real_df.to_csv(real_path, index=False)

    # ---- report -----------------------------------------------------------
    print(f"[make_pair_dataset] Specific labels via {source}")
    print(f"[make_pair_dataset] wrote {fake_path}  ({len(fake_df)} fake rows)")
    print(f"[make_pair_dataset] wrote {real_path}  ({len(real_df)} real rows)")
    dist = fake_df["Specific"].value_counts().sort_index()
    print("[make_pair_dataset] fake 'Specific' distribution:")
    for k, v in dist.items():
        print(f"    {k} ({FAMILY_NAME.get(int(k), '?'):9s}): {v}")
    if "Intersection" not in df.columns:
        print("[make_pair_dataset] WARNING: no 'Intersection' column found; "
              "UCF/PG-FDD read it during training and will fail without it.")


if __name__ == "__main__":
    main()
