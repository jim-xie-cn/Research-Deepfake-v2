#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
individual_fairness.py
======================
Compute the **F_IND** (individual fairness) value that fills the *Individual*
row of Table 4.  The AI-Face repository ships code for every other Table-4
fairness metric (F_MEO / F_DP / F_OAE / F_EO inside
``training/fairness_metrics.py``) **but not for F_IND**, so this script supplies
it.

Definition
----------
Individual fairness [paper refs 94, 100] states that *similar individuals should
receive similar predicted outcomes*.  We operationalise it with the widely used
**consistency / inconsistency** measure (Zemel et al., "Learning Fair
Representations", 2013):

    F_IND = 100 * (1/N) * sum_i | y_hat_i  -  mean_{j in kNN(i)} y_hat_j |

where ``y_hat`` is the model's P(fake) and ``kNN(i)`` are the k nearest
neighbours of sample i in a similarity feature space.  Lower F_IND = the model
treats similar faces more consistently = fairer.

!!  IMPORTANT CAVEAT  !!
The paper's appendix does not ship the exact similarity space it used, so the
*absolute* F_IND numbers produced here will not match the paper to the decimal.
What is reproducible and meaningful is the **ranking across detectors** — as
long as you use the SAME embedding for every detector.  Two ways to fix the
embedding:

  1. ``--embeddings emb.npy``  a precomputed (N, d) array aligned row-for-row
     with ``--test-csv`` (recommended: a frozen face-recognition embedding such
     as ArcFace, identical for all detectors).
  2. default: a lightweight 32x32 grayscale-pixel embedding computed on the fly
     (dependency-free, good enough for relative comparison / smoke tests).

Usage
-----
    # (a) from a checkpoint
    python individual_fairness.py --model xception \
        --checkpoint ../training/checkpoints/xception/xception9.pth \
        --test-csv ../dataset/test.csv --k 5 --max-samples 20000

    # (b) from precomputed arrays (no model load)
    python individual_fairness.py --predictions preds.npy --embeddings emb.npy
"""

import argparse
import os

import numpy as np


def consistency_find(preds: np.ndarray, embeddings: np.ndarray, k: int = 5) -> float:
    """Return F_IND = 100 * mean_i |pred_i - mean kNN pred|."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    preds = np.asarray(preds, dtype=np.float64).ravel()
    X = StandardScaler().fit_transform(np.asarray(embeddings, dtype=np.float64))
    n = len(preds)
    k_eff = min(k + 1, n)  # +1 because the first neighbour is the point itself
    nn = NearestNeighbors(n_neighbors=k_eff).fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]                       # drop self
    neigh_mean = preds[idx].mean(axis=1)
    return float(100.0 * np.mean(np.abs(preds - neigh_mean)))


# --------------------------------------------------------------------------- #
# Optional model path: run a detector to obtain predictions + pixel embedding.
# --------------------------------------------------------------------------- #
def _predict_and_embed(model, checkpoint, test_csv, path_col, target_col,
                       batch_size, num_workers, max_samples):
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    import pandas as pd

    from detector_io import build_detector
    from gender_fairness_metrics import to_prob

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    det = build_detector(model, checkpoint, device)

    df = pd.read_csv(test_csv)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=5).reset_index(drop=True)
    paths = df[path_col].astype(str).tolist()
    labels = df[target_col].astype(int).tolist()

    class _DS(Dataset):
        def __init__(self, paths, tf):
            self.paths, self.tf = paths, tf

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            im = Image.open(self.paths[i]).convert("RGB")
            emb = np.asarray(im.resize((32, 32)).convert("L"),
                             dtype=np.float32).ravel() / 255.0
            x = self.tf(image=np.array(im))["image"]
            return {"image": x, "emb": emb, "idx": i}

    tf = det.make_transform([""])
    dl = DataLoader(_DS(paths, tf), batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    logits, embs = [], []
    for batch in tqdm(dl, desc="individual-fairness inference"):
        out = det.logits(batch["image"].to(device))
        logits.append(np.atleast_1d(out) if out.ndim == 1 else out)
        embs.append(batch["emb"].numpy())
    logits = np.concatenate(logits, axis=0)
    probs = to_prob(logits, det.activation)
    embs = np.concatenate(embs, axis=0)
    return probs, embs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5, help="number of neighbours")
    # mode (b): precomputed arrays
    ap.add_argument("--predictions", help=".npy of P(fake) aligned with embeddings")
    ap.add_argument("--embeddings", help=".npy (N,d) similarity features")
    # mode (a): run a model
    ap.add_argument("--model")
    ap.add_argument("--checkpoint")
    ap.add_argument("--test-csv", default="../dataset/test.csv")
    ap.add_argument("--path-col", default="Image Path")
    ap.add_argument("--target-col", default="Target")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=20000,
                    help="subsample for the kNN step (0 = use all)")
    ap.add_argument("--out", default="../results/individual_fairness.csv")
    args = ap.parse_args()

    if args.predictions:
        preds = np.load(args.predictions)
        if args.embeddings:
            embs = np.load(args.embeddings)
        else:
            raise SystemExit("--embeddings is required alongside --predictions")
        tag = os.path.basename(args.predictions)
    elif args.model and args.checkpoint:
        preds, embs = _predict_and_embed(
            args.model, args.checkpoint, args.test_csv, args.path_col,
            args.target_col, args.batch_size, args.num_workers, args.max_samples)
        if args.embeddings:                      # override pixel embedding
            embs = np.load(args.embeddings)
        tag = args.model
    else:
        raise SystemExit("provide either --predictions/--embeddings or "
                         "--model/--checkpoint")

    f_ind = consistency_find(preds, embs, k=args.k)
    print(f"[individual_fairness] {tag}: F_IND = {f_ind:.5f}  "
          f"(k={args.k}, N={len(preds)})")

    import pandas as pd
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    row = pd.DataFrame([{"detector": tag, "F_IND": f_ind, "k": args.k, "N": len(preds)}])
    if os.path.exists(out):
        row = pd.concat([pd.read_csv(out), row], ignore_index=True)
    row.to_csv(out, index=False)
    print(f"[individual_fairness] appended to {out}")


if __name__ == "__main__":
    main()
