# 06 · `exam/` 代码与脚本功能清单（Manifest）

本文件是 [`../exam/`](../exam/) 目录下**所有交付代码/脚本**的功能清单：每个文件干什么、输入输出、
关键函数/参数、彼此依赖、对应哪个复现目标、以及验证状态。

- 全部为**本项目新增**（仓库原本没有），用来补齐论文 Table 4 / Table 5 / Figure 6 的复现。
- 命名约定：`.py` 为 Python，`.sh` 为 Bash。带 ⚙️ 的是**被 import 的库**（不单独运行），
  其余为**可直接运行的 CLI/脚本**。
- 详细用法见各自对应文档：[01](01_code_usage.md) / [02](02_pair_dataset_generation.md) /
  [03](03_table4.md) / [04](04_table5.md) / [05](05_figure6.md)。

---

## 1. 一览表

| 文件 | 类型 | 一句话功能 | 复现目标 | 详见 |
|---|---|---|---|---|
| [`detector_io.py`](../exam/detector_io.py) | ⚙️Py 库 | 统一构建/加载 12 个检测器，提供统一的推理与 transform 接口 | 通用（Table5/Fig6/F_IND 共用） | 03/04/05 |
| [`gender_fairness_metrics.py`](../exam/gender_fairness_metrics.py) | ⚙️Py 库 | 单属性（性别）的 `F_OAE/F_EO/ACC/AUC` + `to_prob`（sigmoid/softmax） | Table 5 | 04 |
| [`make_pair_dataset.py`](../exam/make_pair_dataset.py) | Py CLI | 由 `train.csv` 生成 `train_fake_spe.csv` + `train_real.csv`（含 `Specific` 列） | pair 训练前置（UCF/PG-FDD） | 02 |
| [`test_only.py`](../exam/test_only.py) | Py CLI | **只加载 checkpoint 跑评测**（不训练），输出与 `train_test.py` 一致的指标 | Table 4（快速路径） | 01/03 |
| [`run_test_all.sh`](../exam/run_test_all.sh) | Bash | 批量对 12 个 checkpoint 跑 `test_only.py`，末尾自动拼 `table4.md` | Table 4（快速路径） | 03 §6 |
| [`run_table4.sh`](../exam/run_table4.sh) | Bash | 从零**训练**+测试 12 个检测器（含自动生成 pair CSV） | Table 4（完整训练） | 03 |
| [`parse_results_table4.py`](../exam/parse_results_table4.py) | Py CLI | 解析 `acc_fairness` 日志 → Table 4（Markdown + 长表 CSV） | Table 4 | 03 |
| [`individual_fairness.py`](../exam/individual_fairness.py) | Py CLI | 计算 `F_IND`（consistency），补齐 Table 4 唯一缺失列 | Table 4（Individual 行） | 03 §4 |
| [`eval_table5_generalization.py`](../exam/eval_table5_generalization.py) | Py CLI | 跨数据集（CCv2/DF-Platter/GenData）性别公平性泛化评测 | Table 5 | 04 |
| [`eval_figure6_postprocessing.py`](../exam/eval_figure6_postprocessing.py) | Py CLI | 6 种后处理前后的交集 `F_EO`/`AUC`，计算 after/before 比值 | Figure 6 | 05 |
| [`plot_figure6.py`](../exam/plot_figure6.py) | Py CLI | 由 `figure6_metrics.csv` 画 4 联比值图 | Figure 6 | 05 |
| [`README.md`](../exam/README.md) | 文档 | `exam/` 目录索引 + 最短复现命令 | — | — |

---

## 2. 依赖与数据流（谁调用谁 / 谁产出谁消费）

**import 依赖（代码级）**
```
detector_io.py ────────────┐  (被 import)
  └─ 被:  test_only.py, eval_table5_generalization.py,
          eval_figure6_postprocessing.py, individual_fairness.py

gender_fairness_metrics.py ─┐  (被 import)
  └─ 被:  eval_table5_generalization.py (compute_gender_metrics/to_prob),
          eval_figure6_postprocessing.py (to_prob),
          individual_fairness.py (to_prob)
```

**脚本调用 + 文件产出/消费（运行级）**
```
[Table 4 · 完整训练]
  run_table4.sh
      ├─ make_pair_dataset.py      → train_fake_spe.csv / train_real.csv
      ├─ train_test*.py (仓库自带)  → checkpoints/<m>/log_training.txt
      └─ (提示) parse_results_table4.py → results/table4.md

[Table 4 · 已有 checkpoint 的快速路径]
  run_test_all.sh
      ├─ test_only.py  (每个模型)  → checkpoints/<m>/log_test.txt
      └─ parse_results_table4.py   → results/table4.md / table4_long.csv
  individual_fairness.py           → results/individual_fairness.csv
      └─ 被 parse_results_table4.py 的 --find-csv 消费（填 F_IND 行）

[Table 5]
  eval_table5_generalization.py    → results/table5_results.csv
      (外部 CSV: CCv2/DF-Platter/GenData, 各含 Image Path,Target,Gender)

[Figure 6]
  eval_figure6_postprocessing.py   → results/figure6_metrics.csv
      └─ plot_figure6.py           → results/figure6.png
```

---

## 3. 逐文件详解

### 3.1 ⚙️ `detector_io.py`（共享库）
- **作用**：屏蔽仓库里 3 种不同的建模方式，对外暴露统一对象。
- **关键内容**：
  - 常量 `REGISTRY_MODELS`(10 个)、`SPECIAL_MODELS`(`vit`,`UnivFD`)、`SOFTMAX_MODELS`(`srm`,`core`)、`ALL_MODELS`(12)。
  - `build_detector(model, checkpoint, device)` → 构建模型并加载 checkpoint，返回 `Detector`。
  - `Detector.logits(imgs)` → 统一前向：registry 走 `model({'image':x}, inference=True)['cls']`，vit/UnivFD 走 `model(x)`；`(N,1)` 压成 `(N,)`，`srm/core` 保留 `(N,2)`。
  - `Detector.make_transform(methods)` → 按模型选 albumentations 测试变换（xception 家族 / vit / clip）。
  - `_load_state()` → 兼容原始 state_dict、`module.` 前缀、`{'state_dict':...}` 包装；`strict=False` 并打印 missing/unexpected。
- **副作用**：import 时 `os.chdir(training/)` 且把 `training/` 加入 `sys.path`（让 detector 找到硬编码的 `./pretrained/xception-b5690688.pth`）。**所以其它脚本务必在 `training/` 下运行、用相对 `training/` 或绝对路径。**
- **依赖**：`torch`（+ 各 detector 建模时的 torchvision/clip/efficientnet_pytorch）。

### 3.2 ⚙️ `gender_fairness_metrics.py`（共享库）
- **作用**：Table 5 的单属性（性别，2 组）公平性/效用指标，公式与仓库 `fairness_metrics.py` 一致。
- **关键函数**：`compute_gender_metrics(labels, probs, genders, threshold)` → `{F_OAE, F_EO, ACC, AUC, per_group}`；`to_prob(logits, activation)`（sigmoid / softmax）；`softmax2`；`_rates`；`format_report`。
- **特例**：全真数据（CCv2）自动令 `F_EO/AUC=NaN`、只给 `F_OAE/ACC`（符合论文表注）。

### 3.3 `make_pair_dataset.py`
- **输入**：`train.csv`（`Image Path,Target,...`）。**输出**：`train_fake_spe.csv`（+`Specific`）、`train_real.csv`。
- **是采样吗**：**不是**，无损全量拆分；采样发生在训练时的 `pair_dataset.py`。详见 [02 §2](02_pair_dataset_generation.md)。
- **关键**：`infer_specific()`（按路径正则归为 1=Deepfakes/2=GANs/3=DMs）、`FAMILY_RULES`。
- **参数**：`--train-csv --out-dir --path-column --target-column --specific-column --default-specific`。

### 3.4 `test_only.py`
- **作用**：抽出 `train_test*.py` 的评测部分，**不训练**；加载 checkpoint → 11 子群推理 → `acc_fairness`。
- **输入**：`--checkpoint` + `--test-csv`。**输出**：屏幕 + `checkpoints/<model>/log_test.txt`（中间 `.npy` 默认清理）。
- **修的 bug**：仓库对 `srm/core` 只 import `acc_fairness_softmax` 却调用 `acc_fairness`（NameError）；本脚本按模型自动选对函数。
- **参数**：`--model --checkpoint --test-csv --savepath --test-batchsize --num-workers --inter-attribute --log-file --keep-npy`。
- **依赖**：`detector_io` + 仓库 `dataset.datasets_train.ImageDataset_Test` + `fairness_metrics(_srm)`。

### 3.5 `run_test_all.sh`
- **作用**：`MODELS` 里每个模型跑一次 `test_only.py`（默认读 `checkpoints/<m>/<m>9.pth`），末尾调 `parse_results_table4.py` 出 `table4.md`；缺失 checkpoint 自动跳过。
- **环境变量**：`TESTCSV SAVEPATH TEST_BS EPOCH MODELS CKPT_<model>`（后者可覆盖单个模型的权重路径）。

### 3.6 `run_table4.sh`
- **作用**：从零训练路径——8 个 no_pair 检测器 + 自动生成 pair CSV 后训 `ucf`/`fair_df_detector` + `vit`/`UnivFD`；结尾打印后续的 `individual_fairness.py` 与 `parse_results_table4.py` 命令模板。
- **环境变量**：`DATAPATH TESTCSV SAVEPATH TRAIN_BS TEST_BS LR`。

### 3.7 `parse_results_table4.py`
- **作用**：把 `acc_fairness` 打印行映射成 Table 4 网格，处理属性重排与 **`F_EO` 需 ×100** 的缩放差异。
- **输入**：`--log MODEL=PATH`（可重复）+ 可选 `--find-csv`（F_IND）。**输出**：`table4.md`（论文版式）+ `table4_long.csv`（长表）。
- **关键**：`parse_log()`（取每个日志**最后一次**匹配=末 epoch）、`build_markdown()`、`ATTRS/FAIR/UTIL` 映射表。
- **已验证**：合成日志下**精确复现** Xception 的 Table 4 全部数值。

### 3.8 `individual_fairness.py`
- **作用**：计算 Table 4 唯一没有现成实现的 `F_IND`（consistency：`100·mean|ŷ_i − 近邻均值|`）。
- **两种模式**：`--model/--checkpoint`（跑模型取 P(fake)+像素/自定义 embedding）或 `--predictions/--embeddings`（直接用 npy 算）。
- **关键**：`consistency_find(preds, embeddings, k)`。
- **注意**：绝对值不逐位复现论文（附录未公开相似度空间）；用**同一 embedding**时**相对排序**可复现。

### 3.9 `eval_table5_generalization.py`
- **作用**：把 AI-Face 上训好的检测器零样本迁移到外部数据集，只评**性别**公平性泛化。
- **输入**：`--checkpoint` + `--dataset NAME=CSV`（可重复；CSV 含 `Image Path,Target,Gender`）。**输出**：`table5_results.csv`（追加）。
- **关键**：`ExternalCSV`、`run_one`、`_coerce_gender`（`male/m/1→1`，`female/f/0→0`）。
- **参数**：`--model --checkpoint --dataset --path-col --target-col --gender-col --batch-size --num-workers --threshold --out --overwrite`。

### 3.10 `eval_figure6_postprocessing.py`
- **作用**：对每个检测器在 6 种后处理（JC/GB/HSV/BC/RT/RC）+ 基线下，测交集 `F_EO` 与 `AUC`，算 after/before 比值。
- **输入**：`--checkpoint` + `--test-csv`（`Image Path,Target,Intersection`）。**输出**：`figure6_metrics.csv`（含 `ratio_F_EO/ratio_AUC`）。
- **关键**：`POSTPROCESS`（7 项含 baseline）、`TestCSV`、`intersectional_feo_auc()`、`_safe_ratio()`。
- **已验证**：`intersectional_feo_auc` 与仓库原生 `acc_fairness` 的 `b_inter`/`group_1 auc` **逐位相等（1e-9）**。

### 3.11 `plot_figure6.py`
- **作用**：把 `figure6_metrics.csv` 画成 4 联子图（Naive/Frequency/Spatial/Fairness-enhanced），实线=`F_EO` 比值、虚线=`AUC` 比值、灰线=1.0 基准。
- **参数**：`--in --out --ymax`。
- **分组**：`PANELS` 覆盖全部 12 个检测器；`X_ORDER=[JC,GB,HSV,BC,RT,RC]`。

---

## 4. 按复现目标分组

| 目标 | 用到的 exam/ 文件（按执行顺序） |
|---|---|
| **pair 数据** | `make_pair_dataset.py` |
| **Table 4（已训练）** | `test_only.py` → `individual_fairness.py` → `parse_results_table4.py`（或一键 `run_test_all.sh`）；库：`detector_io.py`, `gender_fairness_metrics.py` |
| **Table 4（从零训练）** | `run_table4.sh`（内部用 `make_pair_dataset.py` + 仓库 `train_test*.py`）→ `parse_results_table4.py` |
| **Table 5** | `eval_table5_generalization.py`；库：`detector_io.py`, `gender_fairness_metrics.py` |
| **Figure 6** | `eval_figure6_postprocessing.py` → `plot_figure6.py`；库：`detector_io.py`, `gender_fairness_metrics.py` |

---

## 5. 验证状态

所有文件已通过**执行验证 + 与仓库自带函数交叉核对**（详见对话记录），要点：
- `parse_results_table4.py`：合成日志下精确复现 Xception 的 Table 4。
- `eval_figure6_postprocessing.py`：`F_EO`/`AUC` 与仓库 `acc_fairness` 逐位一致（1e-9）。
- `gender_fairness_metrics.py` / `individual_fairness.py`：手算样例与边界（CCv2 全真、单组、k≥N）通过。
- `detector_io.py`：`logits` 形状、`_load_state`（原始/`module.`/wrapper）通过；建模与 `train_test*.py` 逐行比对一致。
- `make_pair_dataset.py`：family 推断 + 4 个 CLI 边界通过。
- `run_*.sh`：`bash -n` + 间接展开/数组在 macOS bash 3.2 实测通过。

> 边界：需要 `torchvision`/`albumentations`/预训练权重的**真实前向路径**（实际建模、增广、`ImageDataset_Test` 读图）
> 在验证机上缺依赖未端到端跑，是通过与 `train_test*.py` 源码逐行比对确认的；在 `FairnessBench` 环境可完整运行。
