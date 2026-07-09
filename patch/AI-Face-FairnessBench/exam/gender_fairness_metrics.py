#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gender_fairness_metrics.py
==========================
Single-attribute (gender) fairness + utility metrics used to reproduce
**Table 5 - Fairness generalization results based on the gender attribute**.

The formulas are copied verbatim from the repository's own
``training/fairness_metrics.py`` so that the numbers are directly comparable to
the AI-Face benchmark; here they are specialised to a *binary* protected
attribute (Female / Male) evaluated on an external test set.

Definitions (paper Sec. 4, "Evaluation Metrics")
-------------------------------------------------
Let the two gender groups be g in {female, male} and let "all" be the union.
Per group we compute the confusion-matrix rates FPR_g, TPR_g and accuracy ACC_g.

    F_OAE  = 100 * ( max_g ACC_g - min_g ACC_g )                 # Overall Accuracy Equality
    F_EO   = 100 * sum_g ( |FPR_g - FPR_all| + |TPR_g - TPR_all| ) # Equal Odds

Utility:  ACC and AUC on the whole external set (fraction*100).

Special case - CCv2 (all real images): there are no positive (fake) samples, so
TPR / AUC are undefined.  Only ``F_OAE`` and ``ACC`` are reported, matching the
paper's footnote for Table 5.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax2(logits: np.ndarray) -> np.ndarray:
    """Return P(fake) for a 2-logit softmax model; logits shape (N, 2)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def to_prob(logits: np.ndarray, activation: str) -> np.ndarray:
    """Map raw model outputs to P(fake) in [0, 1].

    activation: 'sigmoid' for single-logit BCE detectors (xception, ucf, ...),
                'softmax' for 2-logit models (srm, core).
    """
    logits = np.asarray(logits, dtype=np.float64)
    if activation == "softmax":
        if logits.ndim == 2 and logits.shape[1] == 2:
            return softmax2(logits)
        return sigmoid(logits.ravel())            # graceful fallback (never 0-d)
    return sigmoid(logits.ravel())                # ravel, not squeeze: (1,) stays 1-d


def _rates(label: np.ndarray, pred_bin: np.ndarray):
    """Return (FPR, TPR, ACC) handling single-class groups gracefully."""
    label = np.asarray(label).astype(int)
    pred_bin = np.asarray(pred_bin).astype(int)
    acc = accuracy_score(label, pred_bin)
    n_pos = int((label == 1).sum())
    n_neg = int((label == 0).sum())
    fp = int(((pred_bin == 1) & (label == 0)).sum())
    tp = int(((pred_bin == 1) & (label == 1)).sum())
    fpr = fp / n_neg if n_neg > 0 else np.nan
    tpr = tp / n_pos if n_pos > 0 else np.nan
    return fpr, tpr, acc


def compute_gender_metrics(labels, probs, genders, threshold: float = 0.5) -> dict:
    """Compute Table-5 metrics for one external dataset.

    Parameters
    ----------
    labels  : (N,) int   - 0 real / 1 fake ground truth
    probs   : (N,) float - P(fake) in [0, 1]
    genders : (N,) int   - 0 female / 1 male   (paper's convention)
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    genders = np.asarray(genders).astype(int)
    pred_bin = (probs >= threshold).astype(int)

    all_only_real = (labels == 0).all()

    # ---- overall utility --------------------------------------------------
    overall_acc = accuracy_score(labels, pred_bin)
    if all_only_real or len(np.unique(labels)) < 2:
        overall_auc = np.nan
    else:
        overall_auc = roc_auc_score(labels, probs)

    fpr_all, tpr_all, _ = _rates(labels, pred_bin)

    # ---- per-group --------------------------------------------------------
    per_group = {}
    for g, name in ((0, "female"), (1, "male")):
        m = genders == g
        if m.sum() == 0:
            continue
        fpr_g, tpr_g, acc_g = _rates(labels[m], pred_bin[m])
        per_group[name] = {"n": int(m.sum()), "FPR": fpr_g,
                           "TPR": tpr_g, "ACC": acc_g}

    accs = [v["ACC"] for v in per_group.values()]
    f_oae = 100.0 * (max(accs) - min(accs)) if len(accs) >= 2 else np.nan

    if all_only_real:
        f_eo = np.nan  # TPR undefined -> Equal Odds not reported (CCv2 case)
    else:
        f_eo = 0.0
        for v in per_group.values():
            term = 0.0
            if not np.isnan(v["FPR"]) and not np.isnan(fpr_all):
                term += abs(v["FPR"] - fpr_all)
            if not np.isnan(v["TPR"]) and not np.isnan(tpr_all):
                term += abs(v["TPR"] - tpr_all)
            f_eo += term
        f_eo *= 100.0

    return {
        "N": int(len(labels)),
        "all_real": bool(all_only_real),
        "F_OAE": f_oae,
        "F_EO": f_eo,
        "ACC": 100.0 * overall_acc,
        "AUC": np.nan if np.isnan(overall_auc) else 100.0 * overall_auc,
        "per_group": per_group,
    }


def format_report(name: str, det: str, m: dict) -> str:
    lines = [f"[{det} @ {name}]  N={m['N']}  (all_real={m['all_real']})"]
    def s(x):
        return "  n/a " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:6.3f}"
    lines.append(f"    F_OAE(%)↓ = {s(m['F_OAE'])}   F_EO(%)↓ = {s(m['F_EO'])}")
    lines.append(f"    ACC(%)↑   = {s(m['ACC'])}   AUC(%)↑  = {s(m['AUC'])}")
    for gname, gv in m["per_group"].items():
        lines.append(f"      {gname:6s}: n={gv['n']:6d}  ACC={s(100*gv['ACC'])}  "
                     f"FPR={s(gv['FPR'])}  TPR={s(gv['TPR'])}")
    return "\n".join(lines)
