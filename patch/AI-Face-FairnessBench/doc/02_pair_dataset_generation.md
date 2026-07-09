# 02 · Pair 数据集生成方法（Pair Dataset Generation）

> 适用对象：`ucf`（UCF）和 `fair_df_detector`（PG-FDD）这两个**解耦式（disentanglement）**检测器。
> 其余 10 个检测器用普通 `train.csv`，**不需要**本文。
> 配套脚本：[`exam/make_pair_dataset.py`](../exam/make_pair_dataset.py)

---

## 1. 为什么需要 pair 数据集

`ucf_detector.py` / `fair_df_detector.py` 里有**重建分支 + 对比学习分支**，要求每个 batch
里真/假样本各占一半（源码里 `cat_data.chunk(2, dim=0)` 把 batch 前一半当 real、后一半当 fake）。

普通 `ImageDataset_Train` 做不到这个保证，于是 `train_test.py` 在 `--dataset_type pair` 时改用
`pairDataset`（`dataset/pair_dataset.py`）：

```python
train_dataset = pairDataset(datapath+'train_fake_spe.csv',   # 只含假样本
                            datapath+'train_real.csv',        # 只含真样本
                            transform)
```

`pairDataset.__getitem__(idx)`：取第 `idx` 个**假**样本，再**随机**取一个**真**样本，组成一对；
`collate_fn` 把一个 batch 里的 real 堆前面、fake 堆后面 → `image = cat([real, fake])`。

所以生成 pair 数据集 = **把 `train.csv` 拆成两份**：`train_fake_spe.csv` + `train_real.csv`。

---

## 2. 采样机制与数量够不够（重要澄清）

**生成阶段不采样。** [`make_pair_dataset.py`](../exam/make_pair_dataset.py) 是**无损全量拆分**：
按 `Target` 把 `train.csv` 拆成「全部真」和「全部假」两份，没有 `.sample()` / `.head()` / `dropna`，
一行都不丢（`make_pair_dataset.py:127-150`）。**故意保留全量**，是为了给训练时的随机配对留一个完整的
样本池——若在生成阶段就采样，反而会削弱覆盖。

**采样发生在训练时**，在 `dataset/pair_dataset.py`：

```python
def __getitem__(self, idx):
    fake = fakes.loc[idx]                        # 第 idx 个假样本（确定性，每 epoch 过一遍）
    real_idx = random.randint(0, len(reals)-1)   # 随机抽 1 个真样本（有放回）
    real = reals.loc[real_idx]
def __len__(self):
    return len(fakes)                            # 一个 epoch 的长度 = 假样本数
```

即：一个 epoch = **遍历全部假样本各一次**，每个假样本随机配一个真样本；`collate_fn` 再把每个 batch
拼成 50% 真 / 50% 假。

**够不够？够——因为 AI-Face 里「假 ≫ 真」。** 按 README v2 规模（`README.md:164-166`），80% 训练集大致：

| | 全量 | 训练集(80%) | 每 epoch 曝光 | 10 epoch 累计 |
|---|---|---|---|---|
| 假 (fake) | 1,245,660 | ≈ 996k | 每个**恰好 1 次**（全覆盖） | 每个 10 次 |
| 真 (real) | 400,885 | ≈ 321k | 有放回抽 ~996k 次 → 每个**期望 ~3.1 次** | 每个 ~31 次 |

- **假样本**：一个 epoch 全覆盖，无遗漏。
- **真样本**：`len = 假样本数(~1M) ≫ 真样本数(~321k)`，每个真样本每 epoch 期望被抽 ~3 次；
  单 epoch 漏掉某个真样本的概率 ≈ e⁻³·¹ ≈ 4.5%，但跨 10 epoch 几乎必然被覆盖（≈ e⁻³¹ ≈ 0）。
- 每 epoch 重新随机配对，对 UCF/PG-FDD 的**重建 + 对比**分支反而增加多样性。

> ⚠️ **前提假设**：这套机制假设 **假样本数 ≥ 真样本数**（epoch 长度由假样本数驱动）。AI-Face 满足
> （假约为真的 3 倍）。反之若某数据集「真 ≫ 假」，一个 epoch 内部分真样本可能抽不到——但那不是
> AI-Face 的情况。`make_pair_dataset.py` 运行结束会打印真/假行数与 `Specific` 三族分布，可当场核对数量。

---

## 3. 两个 CSV 的列要求（源码依据）

`pair_dataset.py` 实际读取的列：

```python
# 假 CSV（train_fake_spe.csv）
fake_img_path      = fake.loc[idx, 'Image Path']
fake_label         = fake.loc[idx, 'Target']          # =1
fake_spe_label     = fake.loc[idx, 'Specific']        # ⭐ 关键新增列
fake_intersec_label= fake.loc[idx, 'Intersection']    # 0–5

# 真 CSV（train_real.csv）
real_img_path      = real.loc[real_idx, 'Image Path']
real_label         = real.loc[real_idx, 'Target']     # =0
real_spe_label     = real.loc[real_idx, 'Target']     # ⭐ 真样本 spe 直接用 Target → 0
real_intersec_label= real.loc[real_idx, 'Intersection']
```

即：
- **真 CSV** 只需 `Image Path, Target(=0), Intersection`（脚本会保留 train.csv 全部列，多余无害）。
- **假 CSV** 需要额外的 **`Specific`** 列。

---

## 4. `Specific` 列的取值约定（重点）

`UCFDetector` 与 `FairDetector` 的 specific-forgery 头都是 `specific_task_number = 4`
（`ucf_detector.py:55`、`fair_df_detector.py:52`），用 `CrossEntropyLoss` 训练，标签范围必须是
`{0,1,2,3}`。

关键推理链：
1. `pairDataset` 把**真样本**的 `spe_label = Target = 0` → 真 = 类别 **0**。
2. 为避免碰撞，**假样本**只能用 `{1,2,3}`。
3. AI-Face 的 37 种生成方法正好归成 3 大族（Deepfake 视频 / GAN / DM）。

因此采用 **DeepfakeBench 沿用的约定**（本仓库改编自 DeepfakeBench）：

| `Specific` | 含义 | 归属目录/关键词 |
|---|---|---|
| `0` | **Real**（由 `pairDataset` 自动赋给真 CSV，不用你写） | `Real/` |
| `1` | **Deepfakes（视频换脸）** | `deepfakes/`, ff++, dfdc, dfd, celeb-df, faceswap … |
| `2` | **GANs** | `GANs/`, AttGAN, STGAN, StarGAN, StyleGAN, MSGGAN, ProGAN, VQGAN … |
| `3` | **DMs（扩散模型）** | `DMs/`, StableDiffusion, Palette, DALLE2, Midjourney, DCFace, LatentDiffusion, IF … |

> ⚠️ **这是基于源码 (`specific_task_number=4` + 真样本 spe=0) 的合理推断**，仓库没有附带官方的
> `Specific` 生成脚本或数据字典。训练**不依赖**这个映射的语义正确性（只要真=0、假∈{1,2,3}、每族内部一致即可）；
> 它只影响 specific 头学到的「forgery 家族」语义。如果你的标注里已有更细的方法列，可直接用
> `--specific-column` 指定，脚本会跳过路径猜测。

---

## 5. 使用 `make_pair_dataset.py`

### 5.1 基本用法（按图像路径自动推断 family）

```bash
cd training     # 或任意目录，注意 --train-csv/--out-dir 的相对路径
python ../exam/make_pair_dataset.py \
    --train-csv ../dataset/train.csv \
    --out-dir   ../dataset
# 生成: ../dataset/train_fake_spe.csv  和  ../dataset/train_real.csv
```

脚本逻辑（[`make_pair_dataset.py`](../exam/make_pair_dataset.py)）：
1. 按 `Target` 拆成 real / fake 两份。
2. 对 fake：若给了 `--specific-column` 且该列存在 → 直接用；否则用 `FAMILY_RULES` 正则匹配
   `Image Path`（大小写不敏感，命中第一条规则的 family 胜出），匹配不到则用 `--default-specific`（默认 1）并告警。
3. 保留 `train.csv` 的所有原始列（`Intersection` 等），只给 fake 追加 `Specific`。
4. 打印 fake 的 `Specific` 分布做 sanity check。

### 5.2 已有方法/家族列时

```bash
python ../exam/make_pair_dataset.py \
    --train-csv ../dataset/train.csv --out-dir ../dataset \
    --specific-column family_id        # family_id 里已是 1/2/3
```

### 5.3 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--train-csv` | `../dataset/train.csv` | 输入扁平 CSV |
| `--out-dir` | `../dataset` | 输出目录 |
| `--path-column` | `Image Path` | 路径列名 |
| `--target-column` | `Target` | 真(0)/假(1)列名 |
| `--specific-column` | `None` | 若给定则跳过路径启发式 |
| `--default-specific` | `1` | 路径无法归类时的兜底 family |

---

## 6. 生成后如何训练

```bash
cd training
python train_test.py --model ucf              --dataset_type pair
python train_test.py --model fair_df_detector --dataset_type pair
```

`train_test_vit.py` / `train_test_clip.py` 也支持 `--dataset_type pair`（读同样两个 CSV），
但论文里 ViT / UnivFD 用的是普通训练（no_pair）。

---

## 7. 校验清单（Checklist）

- [ ] `train_fake_spe.csv` 里所有行 `Target==1`，且有 `Specific ∈ {1,2,3}`。
- [ ] `train_real.csv` 里所有行 `Target==0`。
- [ ] 两个 CSV 都含 `Image Path` 与 `Intersection`（0–5）。
- [ ] `Specific` 分布合理（三族数量与数据集统计大致吻合，DM 占比最大）。
- [ ] 训练启动日志出现 `Load pretrained model successfully!`（说明 backbone + 数据管线 OK）。
