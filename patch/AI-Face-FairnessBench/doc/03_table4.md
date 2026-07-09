# 03 · 复现 Table 4（Overall Performance Comparison）

> **Table 4. Overall performance comparison of different methods on the AI-Face dataset.**
> 12 个检测器 × (4 属性 × 4 公平性指标 + Individual 的 F_IND) + 5 个效用指标。
> 结论（论文）：PG-FDD 综合最好；UnivFD(CLIP) 次之；10/12 检测器 AUC > 98%。

配套脚本（[`exam/`](../exam)）：
- [`run_table4.sh`](../exam/run_table4.sh) — 一键训练+测试 12 个检测器
- [`parse_results_table4.py`](../exam/parse_results_table4.py) — 把日志解析成 Table 4 网格
- [`individual_fairness.py`](../exam/individual_fairness.py) — 计算 Table 4 里唯一缺失的 **F_IND** 列

---

## 1. 实验方法（论文口径）

1. 数据划分：AI-Face **80% train / 20% test**（用给定的 `train.csv` / `test.csv`）。
2. 对每个检测器：用 §[01](01_code_usage.md) 的训练命令训 **10 epoch**（SGD, lr=5e-4, bs=128）。
3. 训练脚本每个 epoch 后自动在 `test.csv` 上按 **11 个子群**推理并调用 `acc_fairness()`：
   - 性别：`nomale`(女) / `male`(男)
   - 肤色：`skintone1`(Light 1–3) / `skintone2`(Medium 4–6) / `skintone3`(Dark 7–10)
   - 性别×肤色：6 组交集（Intersection 0–5）
   - 年龄：`child/young/adult/middle/senior`（Predicted Age 0–4）
4. 报告 **最后一个 epoch** 的指标。

`acc_fairness` 内部把子群聚成 3 个「大组」+ 交集组：
- `group_1` = 性别（也等于**全体样本**，因为每个样本都有性别）→ 用于**整体效用**
- `group_2` = 肤色，`group_3` = 年龄，`inter` = 性别×肤色交集

---

## 2. ⭐ 指标映射：`acc_fairness` 打印行 → Table 4 单元格

这是复现 Table 4 的核心。`fairness_metrics.py` 打印以下几行（每 epoch 一次，取最后一次）：

```
 number of N with group_1 : AUC: .. | ErrorRate: .. | fpr: .. | tpr:.. | acc: .. | precision: .. | EER: .. | F_G: ..
 F_G_inter, F_EFPR_inter, F_EO_inter, F_A_inter: a_inter | efpr_inter | b_inter | ..
 F_G, F_EFPR, F_EO, F_A, F_COV of group_1: a | efpr | b | ..      # group_1 = Gender
 F_G, F_EFPR, F_EO, F_A, F_COV of group_2: ...                     # group_2 = Skin Tone
 F_G, F_EFPR, F_EO, F_A, F_COV of group_3: ...                     # group_3 = Age
 inter_foae, g1_foae, g2_foae, g3_foae: .. | .. | .. | ..
 inter_F_DP,g1_F_DP,g2_F_DP,g3_F_DP: .. | .. | .. | ..
 inter_F_meo,g1_F_meo,g2_F_meo, g3_F_meo: .. | .. | .. | ..
```

**属性列顺序**：打印里都是 `inter | g1(gender) | g2(skintone) | g3(age)`；
Table 4 的行顺序是 `Skin Tone, Gender, Age, Intersection`。映射如下：

| Table 4 行/列 | 取值来源（打印行） | 缩放 |
|---|---|---|
| **Skin Tone** · F_MEO | `g2_F_meo` | 已 ×100 |
| Skin Tone · F_DP | `g2_F_DP` | 已 ×100 |
| Skin Tone · F_OAE | `g2_foae` | 已 ×100 |
| Skin Tone · F_EO | `group_2` 那行的 **b（第3个数）** | **需 ×100** |
| **Gender** · F_MEO/F_DP/F_OAE | `g1_F_meo` / `g1_F_DP` / `g1_foae` | 已 ×100 |
| Gender · F_EO | `group_1` 行的 b | ×100 |
| **Age** · F_MEO/F_DP/F_OAE | `g3_*` | 已 ×100 |
| Age · F_EO | `group_3` 行的 b | ×100 |
| **Intersection** · F_MEO/F_DP/F_OAE | `inter_*` | 已 ×100 |
| Intersection · F_EO | `F_EO_inter`（`b_inter`） | ×100 |
| **Individual** · F_IND | 见 §4（仓库未实现） | — |
| **Utility** AUC/ACC/AP/EER/FPR | `with group_1` 行的 `AUC/acc/precision/EER/fpr` | 全部 ×100 |

> **缩放陷阱**：`F_MEO / F_DP / F_OAE` 在源码里已经 `×100`，但 `F_EO`（变量 `b`）打印时是
> 0–1 原始分数，**需再 ×100**。效用（AUC/ACC/AP/EER/FPR）源码是 0–1，报表 ×100。

### 验证
用 Xception 的 Table 4 数值反推，映射完全对齐（本文脚本已验证）：
`Skin Tone` F_MEO=8.836 / F_DP=9.751 / F_OAE=1.271 / F_EO=12.132；
`Gender` 4.143(EO)；`Age` 42.216(EO)；`Intersection` 24.315(EO)；
`Utility` AUC=98.583, ACC=96.308, AP=99.350, EER=5.149, FPR=12.961 —— 与论文一致。

---

## 3. 一键运行 + 解析

### 3.1 训练全部 12 个检测器

```bash
cd training
bash ../exam/run_table4.sh          # 内部逐个调用 train_test / _vit / _clip
# 可用环境变量覆盖: DATAPATH= TESTCSV= SAVEPATH= TRAIN_BS= TEST_BS= LR=
```

> ⏱️ 论文里单 epoch 耗时从 SPSL 的 ~5min55s 到 PG-FDD 的 ~7h20min。12 个模型 ×10 epoch
> 是一次很大的计算量，建议用下载好的 checkpoint（§7 of [01](01_code_usage.md)）或多卡并行。

### 3.2 解析日志成 Table 4

```bash
python ../exam/parse_results_table4.py \
    --log xception=./checkpoints/xception/log_training.txt \
    --log efficientnet=./checkpoints/efficientnet/log_training.txt \
    --log vit=./checkpoints/vit/log_training.txt \
    --log f3net=./checkpoints/f3net/log_training.txt \
    --log spsl=./checkpoints/spsl/log_training.txt \
    --log srm=./checkpoints/srm/log_training.txt \
    --log ucf=./checkpoints/ucf/log_training.txt \
    --log UnivFD=./checkpoints/UnivFD/log_training.txt \
    --log core=./checkpoints/core/log_training.txt \
    --log daw_fdd=./checkpoints/daw_fdd/log_training.txt \
    --log dag_fdd=./checkpoints/dag_fdd/log_training.txt \
    --log fair_df_detector=./checkpoints/fair_df_detector/log_training.txt \
    --find-csv ../results/individual_fairness.csv \
    --out-md ../results/table4.md --out-csv ../results/table4_long.csv
```

输出：
- `results/table4.md` — 论文版式的 Markdown 表（Measure/Attribute/Metric 为行，检测器为列）
- `results/table4_long.csv` — 长表，方便再做透视/画图

解析器自动：取每个日志的**最后一次**匹配（=最后 epoch）、按 §2 完成属性重排与 ×100 缩放、
把 `--find-csv` 里的 F_IND 填入 Individual 行。

---

## 4. F_IND（Individual Fairness）——仓库缺失，需单独算

`acc_fairness` **不产出** F_IND。用 [`individual_fairness.py`](../exam/individual_fairness.py) 补齐：

```bash
# 对每个检测器，用其 checkpoint 在 test.csv 上算 F_IND（追加写入同一 CSV）
for m in xception efficientnet vit f3net spsl srm ucf UnivFD core daw_fdd dag_fdd fair_df_detector; do
  python ../exam/individual_fairness.py --model $m \
      --checkpoint ./checkpoints/$m/${m}9.pth \
      --test-csv ../dataset/test.csv --k 5 --max-samples 20000
done
# -> ../results/individual_fairness.csv (detector, F_IND, k, N)
```

**定义（consistency / 一致性）**：
`F_IND = 100 · (1/N) Σ_i | ŷ_i − mean_{j∈kNN(i)} ŷ_j |`，
其中 `ŷ = P(fake)`，`kNN` 在相似度特征空间里找。越低=对「相似个体」预测越一致=越公平。

> ⚠️ **重要说明**：论文附录未公开它用的相似度空间，故这里算出的 **F_IND 绝对值不会和论文逐位对齐**。
> 可复现且有意义的是**检测器之间的相对排序**——前提是所有检测器用**同一套 embedding**。
> 两种固定 embedding 的方式：
> 1. `--embeddings emb.npy`：预先算好、与 `test.csv` 行对齐的 `(N,d)` 特征（推荐用固定的人脸识别
>    embedding，如 ArcFace，对所有检测器都一样）；
> 2. 默认：脚本用 `32×32` 灰度像素作为轻量 embedding（零额外依赖，适合相对比较/冒烟测试）。
>
> 也支持完全脱离模型、直接用两个 npy 计算：
> `python individual_fairness.py --predictions preds.npy --embeddings emb.npy`

---

## 5. 产物一览

| 文件 | 内容 |
|---|---|
| `training/checkpoints/<model>/log_training.txt` | 原始 `acc_fairness` 打印（12 份） |
| `results/individual_fairness.csv` | 各检测器 F_IND |
| `results/table4.md` | 复现的 Table 4（Markdown） |
| `results/table4_long.csv` | 长表版本 |

---

## 6. ⚡ 已训练好模型？只测试不训练（`test_only.py`）

仓库**没有**单独的 test 入口——`train_test*.py` 把评测嵌在训练 epoch 循环里，跑一次要重训 10 epoch。
若你已有 checkpoint，用 [`exam/test_only.py`](../exam/test_only.py)**只跑评测**，输出与 `train_test.py`
完全一致的 `acc_fairness` 指标（因此同样能被 §3.2 的解析器消费）。

它相较原仓库还**修了一个 bug**：`train_test.py` 对 `srm`/`core` 只 import 了
`acc_fairness_softmax` 却调用 `acc_fairness(...)`（会 `NameError`）；`test_only.py` 按模型自动选对函数。

### 单个检测器
```bash
cd training
python ../exam/test_only.py \
    --model xception \
    --checkpoint ./checkpoints/xception/xception9.pth \
    --test-csv ../dataset/test.csv --savepath ../results
# 指标同时打印到屏幕 + ./checkpoints/xception/log_test.txt
```
`srm/core`（自动 softmax）、`vit`、`UnivFD` 同样支持，只需换 `--model` 与 `--checkpoint`。

### 一键跑完 12 个 checkpoint 并拼出 Table 4
```bash
cd training
bash ../exam/run_test_all.sh          # 默认读 ./checkpoints/<model>/<model>9.pth
# 可覆盖: MODELS="xception srm" EPOCH=9 TESTCSV=../dataset/test.csv
# 缺失的 checkpoint 会跳过；跑完自动调用 parse_results_table4.py -> results/table4.md
```

> `run_test_all.sh` 用日志文件 `./checkpoints/<model>/log_test.txt` 汇总，末尾自动调用
> `parse_results_table4.py`；`F_IND` 仍需按 §4 单独跑 `individual_fairness.py`（脚本里已给出命令模板）。

### 与从零训练（`run_table4.sh`）的区别
| | `run_table4.sh` | `run_test_all.sh` |
|---|---|---|
| 是否训练 | ✅ 训练 10 epoch 再测 | ❌ 只加载 checkpoint 测试 |
| 前置 | 数据 + 预训练 backbone | **已训练好的 checkpoint** + 预训练 backbone（构造 backbone 时仍要加载） |
| 耗时 | 很大（单模型数小时） | 一次前向推理（分钟级） |
| 产物 | `log_training.txt` | `log_test.txt` |
