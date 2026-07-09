# 01 · 代码使用文档（Code Usage）

> 目标：把 `AI-Face-FairnessBench` 仓库从环境搭建到「训练 → 测试 → 得到公平性指标」跑通。
> 本文只描述**仓库自带**的流程；复现 Table 4 / Table 5 / Figure 6 的额外脚本见
> [`03_table4.md`](03_table4.md)、[`04_table5.md`](04_table5.md)、[`05_figure6.md`](05_figure6.md)。

---

## 1. 环境安装

```bash
cd AI-Face-FairnessBench
conda create -n FairnessBench python=3.9.0
conda activate FairnessBench
pip install -r requirements.txt
```

关键依赖（`requirements.txt`）：`torch 2.4.1`、`albumentations==1.0.3`、`numpy==1.26.4`、
`scikit-learn==1.4.2`、`pandas==2.2.2`、`efficientnet-pytorch==0.7.1`、`ftfy`（CLIP 用）。

> ⚠️ `albumentations==1.0.3` 里用的是旧 API（`A.JpegCompression`、`quality_lower/upper`）。
> 如果你装了新版 albumentations，`transform.py` 里的后处理算子名会报 deprecation/错误——
> 请锁定 `1.0.3`，否则 Figure 6 的后处理会失败。

---

## 2. 数据准备

### 2.1 目录结构
下载 AI-Face 图像 tar 包后解压，按下面结构组织（`Real` 为真实人脸）：

```
AI-Face Dataset/
├── deepfakes/ (dfd, dfdc, ff++, celeb-df ...)
├── GANs/      (AttGAN, STGAN, StarGAN, StyleGANs, MSGGAN, ProGAN, VQGAN ...)
├── DMs/       (Palette, StableDiffusion1.5, DALLE2, Midjourney, DCFace, LatentDiffusion ...)
└── Real/      (FFHQ, imdb_wiki, FF++/DFDC/DFD/Celeb-DF 的真实帧)
```

### 2.2 `train.csv` / `test.csv` 放到 [`dataset/`](../dataset)

⭐ **最容易踩坑的地方：代码实际读取的列名 ≠ README 顶部的说明表。**
下面是**代码真正 index 的列名**（以源码为准），务必让 CSV 里存在这些列：

| 用途 | 读取者（源码） | 需要的列名（精确大小写） | 取值 |
|---|---|---|---|
| 训练（非 pair） | `ImageDataset_Train` (`dataset/datasets_train.py:41`) | `Image Path`(第0列), `Target`, `Intersection` | Target: 0 真 / 1 假；Intersection: 0–5 |
| 测试（分组） | `ImageDataset_Test` (`dataset/datasets_train.py:99-106`) | `Image Path`, `Target`, `Intersection`, **`Predicted Age`** | Intersection 0–5；Predicted Age 0–4 |
| pair 训练（假） | `pairDataset` (`dataset/pair_dataset.py:32-43`) | `Image Path`, `Target`, **`Specific`**, `Intersection` | Specific 1–3（见 [02](02_pair_dataset_generation.md)） |
| pair 训练（真） | `pairDataset` | `Image Path`, `Target`, `Intersection` | — |

**`Intersection`（性别 × 肤色，6 组）编码**（`datasets_train.py:81-86`，也是当前 v2.0 release 用的定义）：

| 值 | 含义 | 测试用 attribute 名 |
|---|---|---|
| 0 | Female, Light (Tone 1–3) | `nomale,skintone1` |
| 1 | Female, Medium (Tone 4–6) | `nomale,skintone2` |
| 2 | Female, Dark (Tone 7–10) | `nomale,skintone3` |
| 3 | Male, Light | `male,skintone1` |
| 4 | Male, Medium | `male,skintone2` |
| 5 | Male, Dark | `male,skintone3` |

**`Predicted Age`（年龄，5 组）**：0=child, 1=young, 2=adult, 3=middle, 4=senior。

> 📌 注意：README「Update Notes」里 v2 那张表写的是 *Race*、Intersection 为 8 类 (0–7)。
> 那是**第一版 v2 论文**的标注；**当前 release 代码**用的是上表的 **Skin Tone + 6 类 Intersection + Predicted Age**。
> 以代码为准。如果你下载的标注列名不同（如 `Age`、`Skin Tone`），需要自己 rename 成 `Predicted Age` / 生成 `Intersection`。

---

## 3. 预训练权重

除 ViT/CLIP 外的检测器 backbone 都是 **Xception(ImageNet)**，各 detector 在
`build_backbone()` 里**硬编码**加载：

```python
torch.load('./pretrained/xception-b5690688.pth')   # 相对 training/ 目录
```

- 从 README「Load Pretrained Weights」的链接下载 `xception-b5690688.pth`，放到 [`training/pretrained/`](../training/pretrained)。
- 因为是相对路径 `./pretrained/...`，**所有训练/测试脚本必须在 `training/` 目录下运行**。
- ViT 用 `torchvision.models.vit_b_16(pretrained=True)`；UnivFD 用 CLIP `ViT-L/14`（首次运行自动下载）。

---

## 4. 训练 + 测试（一体化）

仓库的 `train_test*.py` 把训练和「每个 epoch 后的公平性测试」写在一起：
每个 epoch 结束→存 checkpoint→在 `test.csv` 上按 11 个子群跑推理→调用
`acc_fairness()` 打印所有公平性/效用指标。

> ⚡ **已经训练好、只想跑测试？** 仓库没有单独的 test 入口，重跑 `train_test.py` 会白训 10 epoch。
> 用 [`exam/test_only.py`](../exam/test_only.py) **只加载 checkpoint 跑评测**（输出与本节完全一致），
> 详见 [03 §6](03_table4.md#6--已训练好模型只测试不训练test_onlypy)。

### 4.1 主入口 `train_test.py`（10 个检测器）

```bash
cd training
python train_test.py --model xception --dataset_type no_pair
```

常用参数（`train_test.py:25-54`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | `xception` | `xception, efficientnet, f3net, spsl, srm, core, ucf, daw_fdd, dag_fdd, fair_df_detector` |
| `--dataset_type` | `no_pair` | `ucf` 和 `fair_df_detector` 必须用 `pair`；其余 `no_pair` |
| `--lr` | `0.0005` | SGD, momentum 0.9, weight_decay 5e-3；StepLR(step=60, γ=0.9) |
| `--train_batchsize` | `128` | |
| `--test_batchsize` | `32` | |
| `--datapath` | `../dataset/` | 里面需有 `train.csv`（pair 模式还需 `train_fake_spe.csv`/`train_real.csv`） |
| `--test_datapath` | `../dataset/test.csv` | |
| `--savepath` | `../results` | 测试时中间 `.npy`（label/prediction）暂存处 |
| `--inter_attribute` | 见源码 | 11 个测试子群，用 `-` 分隔 |
| `--seed` | `5` | |

训练固定 **10 epoch**（`num_epochs=10`，`train_test.py:253`），checkpoint 存到
`training/checkpoints/<model>/<model><epoch>.pth`，日志（含 `acc_fairness` 输出）存到
`training/checkpoints/<model>/log_training.txt`。

### 4.2 ViT-B/16 与 UnivFD（单独脚本）

```bash
cd training
python train_test_vit.py  --model vit       # ViT-B/16, 224 输入, ImageNet mean/std
python train_test_clip.py --model UnivFD     # CLIP ViT-L/14 backbone 冻结, 只训 fc
```

### 4.3 pair 检测器（UCF / PG-FDD）

需要先生成配对 CSV（详见 [02](02_pair_dataset_generation.md)）：

```bash
python ../exam/make_pair_dataset.py --train-csv ../dataset/train.csv --out-dir ../dataset
python train_test.py --model ucf              --dataset_type pair
python train_test.py --model fair_df_detector --dataset_type pair   # PG-FDD
```

---

## 5. 12 个检测器与运行方式速查

| 论文名 | `--model` | 脚本 | dataset_type | 类别 | 输出/激活 |
|---|---|---|---|---|---|
| Xception | `xception` | train_test.py | no_pair | Naive | 1-logit / sigmoid |
| EfficientNet-B4 | `efficientnet` | train_test.py | no_pair | Naive | 1-logit / sigmoid |
| ViT-B/16 | `vit` | train_test_vit.py | no_pair | Naive | 1-logit / sigmoid |
| F3Net | `f3net` | train_test.py | no_pair | Frequency | 1-logit / sigmoid |
| SPSL | `spsl` | train_test.py | no_pair | Frequency | 1-logit / sigmoid |
| SRM | `srm` | train_test.py | no_pair | Frequency | **2-logit / softmax** |
| UCF | `ucf` | train_test.py | **pair** | Spatial | 1-logit / sigmoid |
| UnivFD | `UnivFD` | train_test_clip.py | no_pair | Spatial | 1-logit / sigmoid |
| CORE | `core` | train_test.py | no_pair | Spatial | **2-logit / softmax** |
| DAW-FDD | `daw_fdd` | train_test.py | no_pair | Fairness-enh. | 1-logit / sigmoid |
| DAG-FDD | `dag_fdd` | train_test.py | no_pair | Fairness-enh. | 1-logit / sigmoid |
| PG-FDD | `fair_df_detector` | train_test.py | **pair** | Fairness-enh. | 1-logit / sigmoid |

> `srm`/`core` 输出 2 维 logits，用 `fairness_metrics_srm.acc_fairness_softmax`（softmax）；
> 其余用 `fairness_metrics.acc_fairness`（sigmoid）。`train_test.py:57-60` 会自动切换。

---

## 6. 公平性/效用指标：`acc_fairness()` 输出怎么读

测试时对 11 个子群各存一份 `labels.npy`/`predictions.npy`，然后：

```python
acc_fairness('../results/',
             [['nomale','male'],                                  # group_1 = 性别
              ['skintone1','skintone2','skintone3'],              # group_2 = 肤色
              ['child','young','adult','middle','senior']])       # group_3 = 年龄
# group_1 × group_2 的 6 个组合 = Intersection
```

论文 Sec.4「Evaluation Metrics」的 5 个公平性指标（均越低越好）：

| 指标 | 含义 | 在 `acc_fairness` 里的计算 |
|---|---|---|
| `F_DP` | Demographic Parity | 各子群 预测为正率(PR)/负率(NR) 的极差 ×100 |
| `F_MEO` | Max Equalized Odds | 各子群 {FPR,FNR,TNR,TPR} 极差取 max ×100 |
| `F_EO` | Equal Odds | Σ子群(\|FPR−FPR_all\|+\|TPR−TPR_all\|)（源码**未 ×100**） |
| `F_OAE` | Overall Accuracy Equality | 各子群 accuracy 的极差 ×100（源码变量名 `foae`） |
| `F_IND` | Individual Fairness | **仓库未实现**，见 [03](03_table4.md) 的 `individual_fairness.py` |

5 个效用指标：`AUC↑, ACC↑, AP↑(precision), EER↓, FPR↓`（源码里是 0–1 分数，报表 ×100）。

指标打印行 → Table 4 表格的**精确映射**见 [`03_table4.md`](03_table4.md)。

---

## 7. Checkpoints（可跳过训练）

论文提供在 AI-Face 上训好的 checkpoint（README「Checkpoints」链接）。
下载后即可直接用于 [04](04_table5.md)（Table 5）和 [05](05_figure6.md)（Figure 6）的评测脚本，
无需自己训练 10 epoch。约定放到 `training/checkpoints/<model>/<model>9.pth`（第 10 个 epoch）。

---

## 8. 常见问题（Troubleshooting）

- **`FileNotFoundError: ./pretrained/xception-b5690688.pth`** → 没在 `training/` 下运行，或没下载权重。
- **`KeyError: 'Predicted Age'` / `'Intersection'`** → `test.csv` 缺列，见 §2.2。
- **`KeyError: 'Specific'`** → 用 pair 模式但没生成 `train_fake_spe.csv`，见 [02](02_pair_dataset_generation.md)。
- **albumentations 报 `JpegCompression`/`quality_lower` 错** → 版本不是 1.0.3。
- **只有 CPU** → 源码写死 `torch.device('cuda:0')`，无 GPU 需手动改为 `cpu`（`exam/` 里的评测脚本已自动回退到 CPU）。
