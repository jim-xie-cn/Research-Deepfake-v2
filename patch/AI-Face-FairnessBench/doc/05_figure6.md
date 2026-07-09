# 05 · 复现 Figure 6（Post-Processing Robustness）

> **Figure 6. Performance ratio after vs. before post-processing.**
> 对每个检测器，在 AI-Face 测试集上施加 6 种图像后处理，比较后处理**前后**的
> 交集公平性 `F_EO(inter)` 与效用 `AUC` 的**比值**（after / before）。
> 比值越接近 **1.0** = 越鲁棒。4 个子图按模型类型分：Naive / Frequency / Spatial / Fairness-enhanced。
> 结论（论文）：后处理会抹掉取证痕迹→普遍性能下降；后处理不一定加剧偏见（UCF/UnivFD/CORE/DAW-FDD
> 旋转后甚至更公平）；空间型检测器公平性鲁棒性更好。

> ⚠️ 仓库**没有**这部分代码。配套脚本已生成：
> [`exam/eval_figure6_postprocessing.py`](../exam/eval_figure6_postprocessing.py) +
> [`exam/plot_figure6.py`](../exam/plot_figure6.py)。

---

## 1. 实验方法

- 6 种后处理（论文缩写 → `transform.py` 里的算子名）：
  | 缩写 | 后处理 | `transform.py` 实现（xception 家族） |
  |---|---|---|
  | **JC** | JPEG Compression | `A.JpegCompression(quality=60)` |
  | **GB** | Gaussian Blur | `A.GaussianBlur(blur_limit=(5,5))` |
  | **HSV** | Hue-Saturation-Value | `A.HueSaturationValue(±50,±50,±50)` |
  | **BC** | Brightness/Contrast | `A.RandomBrightnessContrast(0.8, 0.8)` |
  | **RT** | Rotation | `A.Rotate(limit=45)` |
  | **RC** | Random Crop | `A.RandomCrop(224,224)` |

  > ViT/CLIP 走 `get_albumentations_transforms_vit_clip`，参数略不同（JPEG q=80、blur k=3、BC 0.4、Rot 30、先 Resize 224）。脚本按模型自动选择。

- 基线（before）= 只做 `Resize + Normalize`（`methods=['']`）。
- 每种后处理各在**整个 `test.csv`** 上推理一次，计算：
  - `AUC`：整体 ROC-AUC；
  - `F_EO(inter)`：性别×肤色 6 个交集子群相对整体的 Equal-Odds 差
    `Σ_{g=0..5}(|FPR_g−FPR_all| + |TPR_g−TPR_all|)`——即 `acc_fairness` 里的 `b_inter`。
- **比值**：`ratio_F_EO = F_EO(method) / F_EO(baseline)`，`ratio_AUC = AUC(method)/AUC(baseline)`。
  因为是比值，`F_EO` 的 ×100 缩放会约掉，脚本用原始分数即可。

测试 CSV 只需 `Image Path, Target, Intersection`（0–5）三列，见 [01 §2.2](01_code_usage.md)。

---

## 2. 运行

### 2.1 逐检测器算指标（追加进同一 CSV）

```bash
cd training
python ../exam/eval_figure6_postprocessing.py \
    --model xception \
    --checkpoint ./checkpoints/xception/xception9.pth \
    --test-csv ../dataset/test.csv \
    --out ../results/figure6_metrics.csv
# 输出每种后处理的 F_EO(inter)/AUC，以及相对 baseline 的 ratio
```

批量（12 个检测器）：
```bash
for m in xception efficientnet vit f3net spsl srm ucf UnivFD core daw_fdd dag_fdd fair_df_detector; do
  python ../exam/eval_figure6_postprocessing.py --model $m \
    --checkpoint ./checkpoints/$m/${m}9.pth \
    --test-csv ../dataset/test.csv \
    --out ../results/figure6_metrics.csv
done
```

- `--methods JC,GB` 可只跑部分后处理（baseline 会自动补上）。
- 每个 detector 会跑 **7 次**全测试集推理（baseline + 6），比较耗时；可用 `--methods` 分批。

### 2.2 画图

```bash
python ../exam/plot_figure6.py \
    --in  ../results/figure6_metrics.csv \
    --out ../results/figure6.png
```

生成 4 联子图（Naive / Frequency / Spatial / Fairness-enhanced）：
- 实线+圆点 = `F_EO` 比值；虚线+三角 = `AUC` 比值；
- 灰色虚线 = `y=1.0`（无变化基准）；
- x 轴顺序 `JC, GB, HSV, BC, RT, RC`。

模型→子图归类（论文 Sec.4）：
`Naive={xception, efficientnet, vit}`、`Frequency={f3net, spsl, srm}`、
`Spatial={ucf, UnivFD, core}`、`Fairness-enhanced={daw_fdd, dag_fdd, fair_df_detector}`。

---

## 3. 输出

| 文件 | 内容 |
|---|---|
| `results/figure6_metrics.csv` | detector, method, F_EO, AUC, *_baseline, ratio_F_EO, ratio_AUC |
| `results/figure6.png` | 4 联比值图 |

---

## 4. 与论文对齐的注意点

- **随机性**：RC/RT/HSV/BC 含随机参数（albumentations `p=1.0` 但幅度/裁剪位置随机）。
  想更稳可在 `transform.py` 里设固定 seed，或多跑几次取均值；论文未说明是否取均值。
- 论文 Figure 6 的具体标注是 `F_EO` 与 `AUC` 比值（图例标 `Fᴇᴏ` 与 `AUC`），本脚本产物与之一致。
- ratio 对 baseline 数值很敏感：若某检测器 baseline `F_EO≈0`，比值会被放大/不稳定——
  脚本对 `baseline==0` 会输出 `NaN` 以避免误导，绘图时该点会缺失。
- 数值不会逐位复现（同 Table 5 的原因），但**「哪种后处理伤害最大」「哪类模型更鲁棒」的定性结论**可复现。
