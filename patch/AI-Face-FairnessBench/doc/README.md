# AI-Face-FairnessBench · 复现说明文档索引

本目录整理了如何复现论文
*AI-Face: A Million-Scale Demographically Annotated AI-Generated Face Dataset and Fairness Benchmark* (CVPR 2025)
的三个关键结果 **Table 4 / Table 5 / Figure 6**，以及配套的代码使用与数据准备说明。

配套代码在 [`../exam/`](../exam/)（见其 [`README.md`](../exam/README.md)）。

## 文档目录

| # | 文档 | 内容 |
|---|---|---|
| 01 | [`01_code_usage.md`](01_code_usage.md) | 环境、数据准备（**代码真正读的列名**）、预训练权重、训练/测试 12 个检测器、指标含义 |
| 02 | [`02_pair_dataset_generation.md`](02_pair_dataset_generation.md) | UCF / PG-FDD 的 **pair 数据集**生成方法 + `Specific` 列约定 |
| 03 | [`03_table4.md`](03_table4.md) | **Table 4** 复现：`acc_fairness` 打印行 → 表格单元格的精确映射、日志解析、缺失的 `F_IND` |
| 04 | [`04_table5.md`](04_table5.md) | **Table 5** 复现：跨数据集（CCv2/DF-Platter/GenData）性别公平性泛化评测（新代码） |
| 05 | [`05_figure6.md`](05_figure6.md) | **Figure 6** 复现：6 种后处理前后 `F_EO`/`AUC` 比值评测 + 画图（新代码） |
| 06 | [`06_exam_manifest.md`](06_exam_manifest.md) | **`exam/` 所有代码/脚本功能清单**：逐文件功能、输入输出、依赖数据流、验证状态 |

## 复现全流程（Checklist）

```
① 装环境 (01)                : conda env + pip install -r requirements.txt (albumentations==1.0.3)
② 放数据 (01 §2)             : dataset/train.csv, dataset/test.csv  (列名以代码为准!)
③ 放权重 (01 §3)             : training/pretrained/xception-b5690688.pth
④ pair CSV (02)             : python exam/make_pair_dataset.py       # 仅 ucf / fair_df_detector 需要
⑤ 训练 (03)                 : bash exam/run_table4.sh                # 或下载官方 checkpoints
── Table 4 ──
⑥ F_IND (03 §4)            : python exam/individual_fairness.py --model ...
⑦ 解析 (03 §3)             : python exam/parse_results_table4.py --log ... -> results/table4.md
── Table 5 ──
⑧ 外部 CSV (04 §2)         : 准备 CCv2 / DF-Platter / GenData 的 (Image Path, Target, Gender)
⑨ 评测 (04 §3)             : python exam/eval_table5_generalization.py --model ... -> results/table5_results.csv
── Figure 6 ──
⑩ 评测 (05 §2)             : python exam/eval_figure6_postprocessing.py --model ... -> results/figure6_metrics.csv
⑪ 画图 (05 §2)             : python exam/plot_figure6.py -> results/figure6.png
```

## 三个结果分别需要什么

| 结果 | 仓库现成代码？ | 需要补的代码（在 `exam/`） | 需要的额外数据 |
|---|---|---|---|
| **Table 4** | ✅ 训练+`acc_fairness` 已产出 4×4 公平性 + 效用 | 解析器 `parse_results_table4.py`；`F_IND` 计算 `individual_fairness.py` | 无（AI-Face 自带 test.csv） |
| **Table 5** | ❌ 无 | `eval_table5_generalization.py` + `gender_fairness_metrics.py` | CCv2 / DF-Platter / GenData（外部，需自备性别标注 CSV） |
| **Figure 6** | ❌ 无（但后处理算子在 `transform.py` 里已有） | `eval_figure6_postprocessing.py` + `plot_figure6.py` | 无 |

## 重要提醒（Assumptions & Caveats）

1. **列名以源码为准**：`ImageDataset_Test` 读 `Predicted Age` 与 `Intersection`（6 类，性别×肤色），
   与 README「Update Notes」里那张 Race/8 类表不同。详见 [01 §2.2](01_code_usage.md)。
2. **`Specific` 列约定** `{0:Real,1:Deepfakes,2:GANs,3:DMs}` 是基于源码
   （`specific_task_number=4` + 真样本 spe=0）的合理推断，仓库无官方生成脚本。见 [02 §4](02_pair_dataset_generation.md)。
3. **`F_EO` 缩放**：源码打印的 `F_EO`(=b) 未 ×100，其余公平性指标已 ×100。见 [03 §2](03_table4.md)。
4. **`F_IND` 绝对值不可逐位复现**（论文附录未公开相似度空间）；用同一 embedding 时**相对排序**可复现。
5. **Table 5 / Figure 6 的绝对数值**受外部数据版本、人脸裁剪、阈值、后处理随机性影响，
   可复现的是**趋势与定性结论**。
