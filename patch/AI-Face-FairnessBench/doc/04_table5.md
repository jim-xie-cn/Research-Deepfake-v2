# 04 · 复现 Table 5（Fairness Generalization, Gender）

> **Table 5. Fairness generalization results based on the gender attribute.**
> 把在 AI-Face 上训好的检测器，**零样本（zero-shot）**迁移到 3 个外部数据集上，
> 只看**性别**属性的公平性泛化：CCv2、DF-Platter、GenData。
> 结论（论文）：即使 UCF/PG-FDD 这类专门做泛化/公平的检测器，跨域时公平性仍不稳定；
> DAW-FDD / PG-FDD 因标注质量高，在 CCv2 上公平性没有崩。

> ⚠️ 仓库**没有**这部分代码。配套脚本已生成：
> [`exam/eval_table5_generalization.py`](../exam/eval_table5_generalization.py)（+ [`gender_fairness_metrics.py`](../exam/gender_fairness_metrics.py)）。

---

## 1. 实验方法（论文口径）

- 训练集：AI-Face（同 Table 4 的 checkpoint，**不再训练**）。
- 测试集（外部、含性别标注）：
  | 数据集 | 内容 | 报告指标 |
  |---|---|---|
  | **CCv2**（Casual Conversations v2） | **只有真实人脸**（参与者自报性别） | `F_OAE`↓, `ACC`↑ |
  | **DF-Platter** | 真 + 假 | `F_OAE`↓, `F_EO`↓, `AUC`↑ |
  | **GenData** | 真 + 假 | `F_OAE`↓, `F_EO`↓, `AUC`↑ |
- CCv2 因为全是真图（没有正类=fake），TPR/AUC 无定义，**只报 `F_OAE` 和 `ACC`**（论文表注明确说明）。
- 性别二分组：Female / Male。

### 性别属性下的指标定义（复用仓库公式，特化为二组）
设两组 `g∈{female, male}`，`all` 为全体：
```
F_OAE = 100 · ( max_g ACC_g − min_g ACC_g )                       # 两性别 accuracy 极差
F_EO  = 100 · Σ_g ( |FPR_g − FPR_all| + |TPR_g − TPR_all| )        # Equal Odds
ACC   = 100 · overall_accuracy
AUC   = 100 · overall_AUC
```
这与 `training/fairness_metrics.py` 里对单个属性组的 `F_EO(=b)` 与 `F_OAE(=foae)` 完全一致
（只是这里手动 ×100，见 [03](03_table4.md) 的缩放说明）。

---

## 2. 准备外部数据集 CSV

每个外部数据集准备一个 CSV，**至少 3 列**：

```csv
Image Path,Target,Gender
/abs/path/img1.png,0,0        # Target: 0=real 1=fake ; Gender: 0=female 1=male
/abs/path/img2.png,1,1
```

- 列名可用 `--path-col / --target-col / --gender-col` 改。
- `Gender` 支持字符串（`male/female/m/f`）或数字，脚本会自动映射（`male/m/1→1`，`female/f/0→0`）。
- **CCv2**：所有 `Target=0`。
- 数据集本身的下载与人脸对齐/裁剪请遵循各数据集官方协议；本脚本只消费「路径+标签+性别」的 CSV。

> CCv2 / DF-Platter / GenData 均非 AI-Face 的一部分，需自行获取并生成上述 CSV。
> AI-Face 训练用的是 256×256 对齐人脸；外部图片建议同样做人脸裁剪，以贴近论文设置。

---

## 3. 运行

```bash
cd training     # 保证 ./pretrained 可被检测器构造时找到
python ../exam/eval_table5_generalization.py \
    --model xception \
    --checkpoint ./checkpoints/xception/xception9.pth \
    --dataset CCv2=/data/ccv2_gender.csv \
    --dataset DF-Platter=/data/dfplatter_gender.csv \
    --dataset GenData=/data/gendata_gender.csv \
    --out ../results/table5_results.csv
```

- 每个检测器跑一次；结果**追加**到同一 `--out` CSV（用 `--overwrite` 重置）。
- `srm`/`core` 自动走 softmax（脚本按模型名判定激活函数），其余 sigmoid。
- ViT 用 `--model vit`，UnivFD 用 `--model UnivFD`（会加载对应 checkpoint）。

批量（12 个检测器）：
```bash
for m in xception efficientnet vit f3net spsl srm ucf UnivFD core daw_fdd dag_fdd fair_df_detector; do
  python ../exam/eval_table5_generalization.py --model $m \
    --checkpoint ./checkpoints/$m/${m}9.pth \
    --dataset CCv2=/data/ccv2_gender.csv \
    --dataset DF-Platter=/data/dfplatter_gender.csv \
    --dataset GenData=/data/gendata_gender.csv \
    --out ../results/table5_results.csv
done
```

---

## 4. 输出

`results/table5_results.csv`（长表）：

| detector | dataset | N | F_OAE | F_EO | ACC | AUC |
|---|---|---|---|---|---|---|
| xception | CCv2 | … | 1.006 | NaN | 86.465 | NaN |
| xception | DF-Platter | … | 6.836 | 9.789 | NaN | 81.273 |
| … | | | | | | |

- CCv2 行的 `F_EO`/`AUC` 为 `NaN`（符合论文只报 `F_OAE`/`ACC`）。
- 想排成论文那种「Model Type 分组、每数据集两列」的版式，直接对该 CSV 做透视
  （`pandas.pivot_table(index='detector', columns='dataset')`）即可。

---

## 5. 与论文对齐的注意点

- 论文里 CCv2 的 `F_OAE` / DF-Platter·GenData 的 `F_EO` 括号内是「相对 AI-Face 内测的变化量」，
  本脚本只给**绝对值**；变化量 = 外部值 − Table 4 对应（gender）值，可自行相减。
- 绝对数值会受你所用外部数据集的**具体版本、裁剪方式、性别标注**影响，难以逐位复现；
  可复现的是**趋势与相对排序**（例如 UnivFD 跨域更稳、fairness-enhanced 未必稳）。
- 阈值默认 0.5（`--threshold` 可调）；这会影响 ACC/FPR/TPR 类指标，但不影响 AUC。
