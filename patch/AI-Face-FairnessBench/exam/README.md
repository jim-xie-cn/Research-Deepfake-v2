# `exam/` · 复现 Table 4 / Table 5 / Figure 6 的脚本

配套说明文档在 [`../doc/`](../doc/)。所有脚本在仓库自带代码之外**新增**，用来补齐论文结果里
仓库没有直接提供的部分。除画图外，脚本默认在有 GPU 时用 `cuda:0`，否则回退 CPU。

> 运行前提：能 `import` 到 `training/` 下的检测器与 transform。评测脚本
> （`eval_*`、`individual_fairness.py`）通过 `detector_io.py` **自动把工作目录切到 `training/`**，
> 以便检测器构造时找到硬编码的 `./pretrained/xception-b5690688.pth`。因此传给脚本的路径请用
> **绝对路径**或**相对 `training/` 的路径**。建议统一在 `training/` 目录下调用（`python ../exam/xxx.py`）。

## 脚本清单

| 脚本 | 作用 | 对应文档 |
|---|---|---|
| [`make_pair_dataset.py`](make_pair_dataset.py) | 由 `train.csv` 生成 `train_fake_spe.csv` + `train_real.csv`（含 `Specific` 列） | [doc/02](../doc/02_pair_dataset_generation.md) |
| ⭐ [`test_only.py`](test_only.py) | **只测试不训练**：加载已训练 checkpoint → 跑评测 → 打印全部指标（仓库无此入口） | [doc/03 §6](../doc/03_table4.md) |
| ⭐ [`run_test_all.sh`](run_test_all.sh) | 对 12 个已训练 checkpoint 批量 `test_only.py` → 直接拼出 Table 4 | [doc/03 §6](../doc/03_table4.md) |
| [`run_table4.sh`](run_table4.sh) | 一键**训练**+测试全部 12 个检测器（从零训练时用） | [doc/03](../doc/03_table4.md) |
| [`parse_results_table4.py`](parse_results_table4.py) | 把 `log_training.txt` 解析成 Table 4（Markdown + 长表 CSV） | [doc/03](../doc/03_table4.md) |
| [`individual_fairness.py`](individual_fairness.py) | 计算 Table 4 缺失的 `F_IND`（一致性/consistency 度量） | [doc/03 §4](../doc/03_table4.md) |
| [`eval_table5_generalization.py`](eval_table5_generalization.py) | 跨数据集性别公平性泛化（Table 5） | [doc/04](../doc/04_table5.md) |
| [`gender_fairness_metrics.py`](gender_fairness_metrics.py) | 单属性（性别）的 `F_OAE/F_EO/ACC/AUC`（被 Table 5 复用） | [doc/04](../doc/04_table5.md) |
| [`eval_figure6_postprocessing.py`](eval_figure6_postprocessing.py) | 6 种后处理前后 `F_EO(inter)/AUC`（Figure 6 数据） | [doc/05](../doc/05_figure6.md) |
| [`plot_figure6.py`](plot_figure6.py) | 画 Figure 6 的 4 联比值图 | [doc/05](../doc/05_figure6.md) |
| [`detector_io.py`](detector_io.py) | 统一构建/加载 12 个检测器 + 统一推理/transform（被多个 eval 复用） | — |

## 最短复现命令

```bash
cd training     # 关键：在 training/ 下跑

# ---- pair 数据集（仅 ucf / fair_df_detector 需要）----
python ../exam/make_pair_dataset.py --train-csv ../dataset/train.csv --out-dir ../dataset

# ---- Table 4 ----
bash ../exam/run_table4.sh                                    # 训练（或改用官方 checkpoints）
python ../exam/individual_fairness.py --model xception \
       --checkpoint ./checkpoints/xception/xception9.pth --test-csv ../dataset/test.csv
python ../exam/parse_results_table4.py \
       --log xception=./checkpoints/xception/log_training.txt \
       --find-csv ../results/individual_fairness.csv --out-md ../results/table4.md

# ---- Table 5 ----
python ../exam/eval_table5_generalization.py --model xception \
       --checkpoint ./checkpoints/xception/xception9.pth \
       --dataset CCv2=/data/ccv2_gender.csv \
       --dataset DF-Platter=/data/dfplatter_gender.csv \
       --dataset GenData=/data/gendata_gender.csv \
       --out ../results/table5_results.csv

# ---- Figure 6 ----
python ../exam/eval_figure6_postprocessing.py --model xception \
       --checkpoint ./checkpoints/xception/xception9.pth \
       --test-csv ../dataset/test.csv --out ../results/figure6_metrics.csv
python ../exam/plot_figure6.py --in ../results/figure6_metrics.csv --out ../results/figure6.png
```

（把 `xception` 换成 12 个检测器名循环即可；见各文档的批量脚本。）

## 依赖

沿用仓库 `requirements.txt`（`torch, albumentations==1.0.3, numpy, pandas, scikit-learn, matplotlib, tqdm, ftfy`）。
`individual_fairness.py` 额外用到 `sklearn.neighbors`（已在 scikit-learn 内）。

## 已验证

- `parse_results_table4.py` 在合成日志上**精确复现** Xception 的 Table 4 数值（AUC 98.583、
  F_EO 12.132/4.143/42.216/24.315 等），确认了指标映射与缩放正确。
- `gender_fairness_metrics.py` 通过数值自测：混合真假集给出正常 `F_OAE/F_EO/ACC/AUC`；
  全真（CCv2 情形）时 `F_EO/AUC=NaN`、`F_OAE/ACC` 有效；softmax 路径正常。
- `make_pair_dataset.py` / `plot_figure6.py` / `individual_fairness.py` 均在合成数据上跑通。

> 注：以上为逻辑/接口验证。**端到端的真实数值**需要真实的 AI-Face 数据、GPU 与训练好的
> checkpoint，本机未执行。
