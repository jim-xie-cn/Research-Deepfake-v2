#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_only.py — 只跑「测试/评测」，不训练（inference-only evaluation）
====================================================================
`train_test.py` / `train_test_vit.py` / `train_test_clip.py` 把训练和测试写死在
同一个 epoch 循环里，仓库**没有**单独的 test 入口。本脚本抽出其中的**评测部分**：
加载你已经训练好的 checkpoint → 在 test.csv 上按 11 个子群推理 → 调用仓库自带的
`acc_fairness`（sigmoid）或 `acc_fairness_softmax`（srm/core）打印全部公平性/效用指标。

产出的打印格式与 `train_test.py` 完全一致，因此可直接被
`exam/parse_results_table4.py` 解析成 Table 4。

它相对仓库原代码修正了一个隐藏 bug：原 `train_test.py` 对 srm/core 只 import 了
`acc_fairness_softmax`，却仍调用 `acc_fairness(...)`（会 NameError）。本脚本按模型
自动选择正确的度量函数。

════════════════════════════ 使用说明 (USAGE) ════════════════════════════
前置条件
  1) 已下载/训练好 checkpoint，例如 training/checkpoints/xception/xception9.pth
  2) training/pretrained/xception-b5690688.pth 存在（构造 backbone 时要加载）
  3) dataset/test.csv 含列: Image Path, Target, Intersection(0-5), Predicted Age(0-4)
     （年龄子群才需要 Predicted Age；肤色/性别子群不需要）

在 training/ 目录下运行（脚本会自动 chdir 到 training/，路径用相对 training/ 或绝对路径）：

  # 单个检测器
  cd training
  python ../exam/test_only.py \
      --model xception \
      --checkpoint ./checkpoints/xception/xception9.pth \
      --test-csv ../dataset/test.csv \
      --savepath ../results

  # srm / core（自动走 softmax 度量），vit / UnivFD 同样支持
  python ../exam/test_only.py --model srm    --checkpoint ./checkpoints/srm/srm9.pth
  python ../exam/test_only.py --model vit    --checkpoint ./checkpoints/vit/vit9.pth
  python ../exam/test_only.py --model UnivFD --checkpoint ./checkpoints/UnivFD/UnivFD9.pth

  # 评测日志会同时写到 --log-file（默认 ./checkpoints/<model>/log_test.txt），
  # 之后用它拼 Table 4：
  python ../exam/parse_results_table4.py \
      --log xception=./checkpoints/xception/log_test.txt \
      --out-md ../results/table4.md

参数
  --model         12 个之一: xception efficientnet vit f3net spsl srm ucf UnivFD core
                  daw_fdd dag_fdd fair_df_detector
  --checkpoint    训练好的 .pth（必填）
  --test-csv      默认 ../dataset/test.csv
  --savepath      npy 中间文件与最终指标所在目录，默认 ../results（跑完自动清理 npy）
  --test-batchsize 默认 32
  --num-workers   默认 16
  --inter-attribute 11 个测试子群，默认与 train_test.py 一致，一般无需改
  --log-file      评测输出同时写到此文件；默认 ./checkpoints/<model>/log_test.txt
  --keep-npy      保留中间 npy（默认跑完删除）
══════════════════════════════════════════════════════════════════════════
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# NOTE: importing detector_io first is REQUIRED — it inserts training/ onto
# sys.path and chdir()s into it so the detectors find ./pretrained/... and so
# the `dataset` / `fairness_metrics` imports below resolve.
from detector_io import build_detector, ALL_MODELS, SOFTMAX_MODELS, TRAINING_DIR  # noqa: E402
from dataset.datasets_train import ImageDataset_Test  # noqa: E402

# The 11 test subgroups, identical to train_test.py's default --inter_attribute.
DEFAULT_INTER = ("nomale,skintone1-nomale,skintone2-nomale,skintone3-"
                 "male,skintone1-male,skintone2-male,skintone3-"
                 "child-young-adult-middle-senior")
# group_1 = gender, group_2 = skin tone, group_3 = age (matches acc_fairness).
ATTR_GROUPS = [["nomale", "male"],
               ["skintone1", "skintone2", "skintone3"],
               ["child", "young", "adult", "middle", "senior"]]


class _Tee:
    """Mirror everything printed to stdout into a log file as well."""

    def __init__(self, path):
        self.terminal = sys.__stdout__
        self.log = open(path, "a", buffering=1)

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def _cleanup_npy(directory):
    for item in os.listdir(directory):
        if item.endswith(".npy"):
            os.remove(os.path.join(directory, item))


@torch.no_grad()
def run_subgroup(det, test_csv, eachatt, transform, batch_size, num_workers, device):
    """Inference over one subgroup; return (labels, predictions) numpy arrays.

    predictions shape: (N,) for sigmoid models, (N, 2) for srm/core (softmax).
    This mirrors train_test.py's per-subgroup save exactly.
    """
    ds = ImageDataset_Test(test_csv, eachatt, transform)
    if len(ds) == 0:
        return np.array([]), np.array([])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    preds, labels = [], []
    for data in tqdm(dl, desc=f"test {eachatt}", leave=False):
        out = det.logits(data["image"].to(device))          # (B,) or (B,2)
        preds.append(np.atleast_1d(out) if out.ndim == 1 else out)
        labels += data["label"].numpy().tolist() if torch.is_tensor(data["label"]) \
            else list(np.asarray(data["label"]))
    preds = np.concatenate(preds, axis=0) if preds else np.array([])
    return np.array(labels), preds


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--checkpoint", required=True, help="trained .pth for --model")
    ap.add_argument("--test-csv", default="../dataset/test.csv")
    ap.add_argument("--savepath", default="../results",
                    help="dir for temp .npy + where acc_fairness reads them from")
    ap.add_argument("--test-batchsize", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--inter-attribute", default=DEFAULT_INTER)
    ap.add_argument("--log-file", default=None,
                    help="tee metrics output here (default ./checkpoints/<model>/log_test.txt)")
    ap.add_argument("--keep-npy", action="store_true",
                    help="do not delete the intermediate .npy files")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[test_only] model={args.model} checkpoint={args.checkpoint} device={device}")

    # ---- tee stdout to a log so parse_results_table4.py can consume it ----
    log_file = args.log_file or os.path.join("./checkpoints", args.model, "log_test.txt")
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    sys.stdout = _Tee(os.path.abspath(log_file))
    print(f"[test_only] logging metrics to {os.path.abspath(log_file)}")

    # ---- pick the correct fairness routine (fixes the srm/core repo bug) ---
    if args.model in SOFTMAX_MODELS:                    # srm, core -> 2-logit softmax
        from fairness_metrics_srm import acc_fairness_softmax as fairness_fn
    else:
        from fairness_metrics import acc_fairness as fairness_fn

    # ---- build model + load checkpoint ------------------------------------
    det = build_detector(args.model, args.checkpoint, device)
    transform = det.make_transform([""])               # test-time (no post-proc)

    savedir = os.path.abspath(args.savepath)
    os.makedirs(savedir, exist_ok=True)

    # ---- per-subgroup inference (mirrors train_test.py test loop) ---------
    for eachatt in args.inter_attribute.split("-"):
        labels, preds = run_subgroup(det, args.test_csv, eachatt, transform,
                                     args.test_batchsize, args.num_workers, device)
        base = os.path.join(savedir, eachatt)
        np.save(base + "labels.npy", labels)
        np.save(base + "predictions.npy", preds)
        print(f"[test_only] {eachatt}: N={len(labels)}")

    # ---- compute + print all Table-4 metrics ------------------------------
    print("=" * 70)
    fairness_fn(savedir + os.sep, ATTR_GROUPS)
    print("=" * 70)

    if not args.keep_npy:
        _cleanup_npy(savedir)
    print(f"[test_only] done. metrics logged to {os.path.abspath(log_file)}")


if __name__ == "__main__":
    main()
