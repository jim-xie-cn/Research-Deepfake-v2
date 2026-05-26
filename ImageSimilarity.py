import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import skew, kurtosis, spearmanr, kendalltau, wasserstein_distance
from math import log2
import seaborn as sns
from scipy.signal import find_peaks
from ImageConstruct import CImageUtils

class CImageSimilarity:

    def __init__(self, bins=256, eps=1e-12, weights=(1/3, 1/3, 1/3), use_range_cache=True):
        self.bins = bins
        self.eps = eps
        self.weights = list(weights)  # [hist, stat, info]
        self.use_range_cache = use_range_cache
        self._range_cache = {}
        self._auto_tuned = False

    @staticmethod
    def find_peaks(ds, prominence=0.00001):
        idx, props = find_peaks(ds, prominence=prominence)
        return idx

    @staticmethod
    def display_plots(df, x, ys, kind="lineplot", cols=None, cell_size=3.0, **kwargs):
        """
        layout 固定为 row：优先一行排满，最多 8 列
        """
        plot_func = getattr(sns, kind)
        n = len(ys)

        if cols is None:
            cols = min(8, n)  # 一行排满
            cols = max(cols, 1)

        rows = int(np.ceil(n / cols))
        figsize = (cols * cell_size, rows * cell_size)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)

        for i, y in enumerate(ys):
            ax = axes[i]
            plot_func(data=df, x=x, y=y, ax=ax, **kwargs)
            ax.set_title(y, fontsize=9)
            ax.grid(True, alpha=0.3)

        for j in range(n, rows * cols):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    # ---------- 缓存与范围 ----------
    def _range_cache_key(self, img):
        ai = img.__array_interface__
        return (ai['data'][0], img.shape, str(img.dtype), img.size)

    def clear_cache(self):
        self._range_cache.clear()

    def data_range(self, img):
        # 对浮点图像：若已是[0,1]，固定1.0，确保PSNR/SSIM跨样本可比
        if np.issubdtype(img.dtype, np.floating):
            mn = float(np.min(img))
            mx = float(np.max(img))
            if mn >= 0.0 - self.eps and mx <= 1.0 + self.eps:
                return 1.0

        if self.use_range_cache:
            key = self._range_cache_key(img)
            if key in self._range_cache:
                return self._range_cache[key]

        if np.issubdtype(img.dtype, np.integer):
            r = float(np.iinfo(img.dtype).max)
        else:
            mn = float(np.min(img))
            mx = float(np.max(img))
            r = mx - mn if mx > mn else 1.0

        if self.use_range_cache:
            self._range_cache[key] = r
        return r

    # ---------- 基础工具 ----------
    def to_gray(self, img):
        img = np.asarray(img)
        if img.ndim == 2:
            return img
        if img.ndim == 3 and img.shape[-1] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        raise ValueError(f"Unsupported image shape for to_gray: {img.shape}")

    def compute_hist(self, img, bins=None, normalize=True):
        bins = bins or self.bins
        img = np.asarray(img)

        # 修正：统一范围（若归一化图像则[0,1]；否则使用数据实际范围）
        mn = float(np.min(img))
        mx = float(np.max(img))
        if np.issubdtype(img.dtype, np.floating) and mn >= 0.0 - self.eps and mx <= 1.0 + self.eps:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = mn, mx
            if hi <= lo:
                hi = lo + 1.0

        hist = cv2.calcHist([img], [0], None, [bins], [float(lo), float(hi)]).flatten().astype(np.float64)
        if normalize:
            hist = hist / (hist.sum() + self.eps)
        return hist

    def channel_histograms(self, img, bins=None):
        bins = bins or self.bins
        img = np.asarray(img)
        if img.ndim == 2:
            return [self.compute_hist(img, bins=bins)]
        if img.ndim == 3 and img.shape[-1] == 3:
            chans = cv2.split(img)
            return [self.compute_hist(c, bins=bins) for c in chans]
        raise ValueError(f"Unsupported image shape for channel_histograms: {img.shape}")

    # ---------- 直方图/分布类 ----------
    def hist_correlation(self, h1, h2):
        h1m = h1 - h1.mean()
        h2m = h2 - h2.mean()
        denom = np.sqrt((h1m**2).sum() * (h2m**2).sum()) + self.eps
        return float((h1m * h2m).sum() / denom)

    def chi_square(self, h1, h2):
        return float(0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + self.eps)))

    def bhattacharyya(self, h1, h2):
        bc = np.sum(np.sqrt(h1 * h2))
        return float(np.sqrt(max(0.0, 1.0 - bc)))

    def kl_divergence(self, p, q):
        return float(np.sum(p * np.log((p + self.eps) / (q + self.eps))))

    def js_divergence(self, p, q):
        m = 0.5 * (p + q)
        return float(0.5 * self.kl_divergence(p, m) + 0.5 * self.kl_divergence(q, m))

    def emd_wasserstein(self, h1, h2):
        bins = np.arange(len(h1))
        return float(wasserstein_distance(bins, bins, h1, h2))

    def hist_metrics(self, img1, img2):
        hists1 = self.channel_histograms(img1)
        hists2 = self.channel_histograms(img2)
        return {
            'Hist_Correlation': np.mean([self.hist_correlation(h1, h2) for h1, h2 in zip(hists1, hists2)]),
            'Chi_Square': np.mean([self.chi_square(h1, h2) for h1, h2 in zip(hists1, hists2)]),
            'Bhattacharyya': np.mean([self.bhattacharyya(h1, h2) for h1, h2 in zip(hists1, hists2)]),
            'KL': np.mean([self.kl_divergence(h1, h2) for h1, h2 in zip(hists1, hists2)]),
            'JS': np.mean([self.js_divergence(h1, h2) for h1, h2 in zip(hists1, hists2)]),
            'EMD': np.mean([self.emd_wasserstein(h1, h2) for h1, h2 in zip(hists1, hists2)]),
        }

    # ---------- 统计相关类 ----------
    def ncc(self, img1, img2):
        a = img1.astype(np.float64).flatten()
        b = img2.astype(np.float64).flatten()
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + self.eps)
        return float(np.dot(a, b) / denom)

    def zncc(self, img1, img2):
        a = img1.astype(np.float64).flatten()
        b = img2.astype(np.float64).flatten()
        a -= a.mean()
        b -= b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + self.eps)
        return float(np.dot(a, b) / denom)

    def covariance(self, img1, img2):
        a = img1.astype(np.float64).flatten()
        b = img2.astype(np.float64).flatten()
        return float(np.mean((a - a.mean()) * (b - b.mean())))

    def correlation_coeff(self, img1, img2):
        a = img1.astype(np.float64).flatten()
        b = img2.astype(np.float64).flatten()
        c = np.corrcoef(a, b)[0, 1]
        if np.isnan(c) or np.isinf(c):
            c = 0.0
        return float(c)

    def correlation_metrics(self, img1, img2):
        g1 = self.to_gray(img1)
        g2 = self.to_gray(img2)
        return {
            'NCC': self.ncc(g1, g2),
            'ZNCC': self.zncc(g1, g2),
            'Covariance': self.covariance(g1, g2),
            'Correlation_Coeff': self.correlation_coeff(g1, g2),
        }

    # ---------- 误差/结构/矩/秩/频域/SVD ----------
    def mse(self, img1, img2):
        a = img1.astype(np.float64)
        b = img2.astype(np.float64)
        return float(np.mean((a - b) ** 2))

    def mae(self, img1, img2):
        a = img1.astype(np.float64)
        b = img2.astype(np.float64)
        return float(np.mean(np.abs(a - b)))

    def psnr(self, img1, img2):
        r = self.data_range(img1)
        m = self.mse(img1, img2)
        if m < self.eps:
            return float('inf')
        return float(20 * np.log10(r + self.eps) - 10 * np.log10(m + self.eps))

    def ssim_score(self, img1, img2):
        g1 = self.to_gray(img1)
        g2 = self.to_gray(img2)
        r = self.data_range(g1)
        return float(ssim(g1, g2, data_range=r))

    def moment_metrics(self, img1, img2):
        g1 = self.to_gray(img1).astype(np.float64).flatten()
        g2 = self.to_gray(img2).astype(np.float64).flatten()
        h1 = self.compute_hist(self.to_gray(img1), bins=self.bins, normalize=True)
        h2 = self.compute_hist(self.to_gray(img2), bins=self.bins, normalize=True)
        H1 = -np.sum(h1 * np.log2(h1 + self.eps))
        H2 = -np.sum(h2 * np.log2(h2 + self.eps))
        return {
            'Mean_Diff': abs(g1.mean() - g2.mean()),
            'Var_Diff': abs(g1.var() - g2.var()),
            'Skew_Diff': abs(skew(g1) - skew(g2)),
            'Kurt_Diff': abs(kurtosis(g1) - kurtosis(g2)),
            'Entropy_Diff': abs(H1 - H2),
        }

    def rank_correlation_metrics(self, img1, img2):
        g1 = self.to_gray(img1).astype(np.float64).flatten()
        g2 = self.to_gray(img2).astype(np.float64).flatten()
        sp = spearmanr(g1, g2).correlation
        kd = kendalltau(g1, g2).correlation
        sp = 0.0 if np.isnan(sp) else float(sp)
        kd = 0.0 if np.isnan(kd) else float(kd)
        return {'Spearman': sp, 'Kendall': kd}

    def svd_spectrum_distance(self, img1, img2, k=50):
        g1 = self.to_gray(img1).astype(np.float64)
        g2 = self.to_gray(img2).astype(np.float64)
        s1 = np.linalg.svd(g1, compute_uv=False)[:k]
        s2 = np.linalg.svd(g2, compute_uv=False)[:k]
        s1 = s1 / (s1.sum() + self.eps)
        s2 = s2 / (s2.sum() + self.eps)
        return float(np.sum(np.abs(s1 - s2)))

    def fft_spectrum_distance(self, img1, img2):
        g1 = self.to_gray(img1).astype(np.float64)
        g2 = self.to_gray(img2).astype(np.float64)
        f1 = np.abs(np.fft.fftshift(np.fft.fft2(g1)))
        f2 = np.abs(np.fft.fftshift(np.fft.fft2(g2)))
        f1 = f1 / (f1.sum() + self.eps)
        f2 = f2 / (f2.sum() + self.eps)
        return float(np.mean(np.abs(f1 - f2)))

    # ---------- 信息论类 ----------
    def mutual_information(self, img1, img2):
        g1 = self.to_gray(img1).astype(np.float64)
        g2 = self.to_gray(img2).astype(np.float64)

        # 与 compute_hist 保持同类范围逻辑
        mn1, mx1 = float(np.min(g1)), float(np.max(g1))
        mn2, mx2 = float(np.min(g2)), float(np.max(g2))

        if mn1 >= -self.eps and mx1 <= 1.0 + self.eps and mn2 >= -self.eps and mx2 <= 1.0 + self.eps:
            r1 = [0.0, 1.0]
            r2 = [0.0, 1.0]
        else:
            if mx1 <= mn1:
                mx1 = mn1 + 1.0
            if mx2 <= mn2:
                mx2 = mn2 + 1.0
            r1 = [mn1, mx1]
            r2 = [mn2, mx2]

        joint_hist, _, _ = np.histogram2d(
            g1.ravel(), g2.ravel(),
            bins=self.bins,
            range=[r1, r2]
        )
        joint_prob = joint_hist / (joint_hist.sum() + self.eps)
        p1 = joint_prob.sum(axis=1, keepdims=True)
        p2 = joint_prob.sum(axis=0, keepdims=True)

        denom = p1 @ p2
        mask = joint_prob > 0
        mi = np.sum(joint_prob[mask] * np.log2((joint_prob[mask] + self.eps) / (denom[mask] + self.eps)))
        return float(max(0.0, mi))

    def normalized_mutual_information(self, img1, img2):
        mi = self.mutual_information(img1, img2)
        g1 = self.to_gray(img1)
        g2 = self.to_gray(img2)
        h1 = self.compute_hist(g1, bins=self.bins, normalize=True)
        h2 = self.compute_hist(g2, bins=self.bins, normalize=True)
        H1 = -np.sum(h1 * np.log2(h1 + self.eps))
        H2 = -np.sum(h2 * np.log2(h2 + self.eps))
        return float(mi / (H1 + H2 + self.eps))

    def info_metrics(self, img1, img2):
        return {
            'MI': self.mutual_information(img1, img2),
            'NMI': self.normalized_mutual_information(img1, img2),
        }

    # ---------- 相似度评分 ----------
    def _sim_from_corr(self, x):
        if np.isnan(x) or np.isinf(x):
            x = 0.0
        x = np.clip(x, -1.0, 1.0)
        return float((x + 1.0) / 2.0)

    def _sim_from_dist(self, d):
        return float(1.0 / (1.0 + d))

    def _sim_from_cov(self, c):
        return float(1.0 - np.exp(-abs(c)))

    def _sim_from_mi(self, mi):
        return float(1.0 - np.exp(-mi))

    def _sim_from_error(self, d):
        return float(1.0 / (1.0 + d))

    def _sim_from_psnr(self, p):
        if np.isinf(p):
            return 1.0
        return float(1.0 - np.exp(-p / 20.0))

    def histogram_score(self, img1, img2):
        m = self.hist_metrics(img1, img2)
        sims = [
            self._sim_from_corr(m['Hist_Correlation']),
            self._sim_from_dist(m['Chi_Square']),
            self._sim_from_dist(m['Bhattacharyya']),
            self._sim_from_dist(m['KL']),
            self._sim_from_dist(m['JS']),
            self._sim_from_dist(m['EMD']),
        ]
        return float(np.mean(sims))

    def statistical_score(self, img1, img2):
        g1 = self.to_gray(img1)
        g2 = self.to_gray(img2)

        corr = self.correlation_metrics(img1, img2)
        mse_val = self.mse(g1, g2)
        mae_val = self.mae(g1, g2)
        psnr_val = self.psnr(g1, g2)
        ssim_val = self.ssim_score(g1, g2)
        moments = self.moment_metrics(img1, img2)
        ranks = self.rank_correlation_metrics(img1, img2)
        svd_dist = self.svd_spectrum_distance(img1, img2)
        fft_dist = self.fft_spectrum_distance(img1, img2)

        sims = [
            self._sim_from_corr(corr['NCC']),
            self._sim_from_corr(corr['ZNCC']),
            self._sim_from_corr(corr['Correlation_Coeff']),
            self._sim_from_error(abs(corr['Covariance'])),
            self._sim_from_error(mse_val),
            self._sim_from_error(mae_val),
            self._sim_from_psnr(psnr_val),
            self._sim_from_corr(ssim_val),
            self._sim_from_error(moments['Mean_Diff']),
            self._sim_from_error(moments['Var_Diff']),
            self._sim_from_error(moments['Skew_Diff']),
            self._sim_from_error(moments['Kurt_Diff']),
            self._sim_from_error(moments['Entropy_Diff']),
            self._sim_from_corr(ranks['Spearman']),
            self._sim_from_corr(ranks['Kendall']),
            self._sim_from_error(svd_dist),
            self._sim_from_error(fft_dist),
        ]
        return float(np.mean(sims))

    def information_score(self, img1, img2):
        m = self.info_metrics(img1, img2)
        nmi = float(np.clip(m['NMI'], 0.0, 1.0))
        mi_sim = self._sim_from_mi(m['MI'])
        return float(0.8 * nmi + 0.2 * mi_sim)

    def overall_score(self, img1, img2):
        hs = self.histogram_score(img1, img2)
        ss = self.statistical_score(img1, img2)
        iscore = self.information_score(img1, img2)
        w_hist, w_stat, w_info = self.weights
        return float(w_hist * hs + w_stat * ss + w_info * iscore)

    # ---------- 权重自动调参 ----------
    def auto_tune_weights(self, pairs, targets=None):
        X = []
        for (a, b) in pairs:
            X.append([
                self.histogram_score(a, b),
                self.statistical_score(a, b),
                self.information_score(a, b),
            ])
        X = np.array(X, dtype=np.float64)

        if targets is not None:
            y = np.array(targets, dtype=np.float64)
            w, *_ = np.linalg.lstsq(X, y, rcond=None)
            w = np.clip(w, 0.0, None)
            if w.sum() < self.eps:
                w = np.array([1/3, 1/3, 1/3], dtype=np.float64)
            else:
                w = w / (w.sum() + self.eps)
        else:
            var = X.var(axis=0)
            w = 1.0 / (var + self.eps)
            w = w / (w.sum() + self.eps)

        self.weights = list(map(float, w))
        return self.weights

    # ---------- 结果解读（保留原逻辑） ----------
    def interpret(self, results):
        overall = results['score']['Overall']
        hist = results['score']['Histogram']
        stat = results['score']['Statistical']
        info = results['score']['Information']

        psnr = results['stat']['PSNR']
        ssim_val = results['stat']['SSIM']
        nmi = results['info']['NMI']

        if overall >= 0.85:
            level = "高度相似"
        elif overall >= 0.65:
            level = "中度相似"
        else:
            level = "相似度较低"

        if ssim_val >= 0.9:
            structure = {
                "conclusion": "像素级结构相似",
                "because": "SSIM 很高（≥0.90）",
                "explain": "说明纹理/边缘结构基本一致"
            }
        elif ssim_val >= 0.75:
            structure = {
                "conclusion": "结构相似度较好",
                "because": "SSIM 处于中高水平（0.75~0.90）",
                "explain": "说明局部纹理可能存在差异"
            }
        else:
            structure = {
                "conclusion": "结构差异明显",
                "because": "SSIM 较低（<0.75）",
                "explain": "说明纹理/局部结构差异明显或整体对比度/亮度分布不一致"
            }

        if np.isinf(psnr):
            pixel = {
                "conclusion": "像素误差极小",
                "because": "PSNR 为无穷大（误差≈0）",
                "explain": "说明两图像几乎完全一致"
            }
        elif np.isfinite(psnr) and psnr >= 30:
            pixel = {
                "conclusion": "像素误差小",
                "because": "PSNR 较高（≥30dB）",
                "explain": "说明整体重建质量较好"
            }
        elif np.isfinite(psnr) and psnr < 20:
            pixel = {
                "conclusion": "像素误差较大",
                "because": "PSNR 低（<20dB）",
                "explain": "说明可能存在明显噪声或错位"
            }
        else:
            pixel = {
                "conclusion": "像素误差中等",
                "because": "PSNR 处于中间水平（20~30dB）",
                "explain": "说明存在一定噪声或轻微错位"
            }

        if nmi >= 0.8:
            info_field = {
                "conclusion": "信息分布高度一致",
                "because": "NMI 很高（≥0.80）",
                "explain": "说明联合分布匹配良好"
            }
        elif nmi >= 0.5:
            info_field = {
                "conclusion": "信息分布一致性中等",
                "because": "NMI 处于中等水平（0.50~0.80）",
                "explain": "说明分布形状仍存在一定偏移"
            }
        else:
            info_field = {
                "conclusion": "信息分布差异明显",
                "because": "NMI 较低（<0.50）",
                "explain": "说明可能是纹理/局部结构差异明显或整体对比度/亮度分布不一致"
            }

        if hist >= 0.7:
            hist_field = {
                "conclusion": "直方图分布整体接近",
                "because": "直方图相似度较高（≥0.70）",
                "explain": "说明亮度/颜色分布较一致"
            }
        elif hist >= 0.5:
            hist_field = {
                "conclusion": "直方图分布相似但存在偏移",
                "because": "直方图相似度中等（0.50~0.70）",
                "explain": "说明分布形状仍存在一定偏移"
            }
        else:
            hist_field = {
                "conclusion": "直方图分布差异较大",
                "because": "直方图相似度较低（<0.50）",
                "explain": "说明整体对比度/亮度分布不一致"
            }

        hints = [
            f"{structure['conclusion']}，因为 {structure['because']}，{structure['explain']}",
            f"{pixel['conclusion']}，因为 {pixel['because']}，{pixel['explain']}",
            f"{info_field['conclusion']}，因为 {info_field['because']}，{info_field['explain']}",
            f"{hist_field['conclusion']}，因为 {hist_field['because']}，{hist_field['explain']}",
        ]

        return {
            "level": level,
            "summary": f"总体相似度={overall:.3f}，直方图={hist:.3f}，统计={stat:.3f}，信息={info:.3f}",
            "structure": structure,
            "pixel": pixel,
            "info": info_field,
            "hist": hist_field,
            "hints": hints
        }

    # ---------- 汇总接口 ----------
    def compare(self, img1, img2, tune_pairs=None, tune_targets=None, auto_tune=True, tune_once=True, auto_normalize=True):
        if auto_normalize:
            imgs = CImageUtils.normalize_by_channel([img1,img2])
            img1 = imgs[0]
            img2 = imgs[1]

        if img1.shape != img2.shape:
            raise ValueError(f"img1/img2 shape mismatch: {img1.shape} vs {img2.shape}")

        if auto_tune:
            need_tune = True
            if tune_once and self._auto_tuned:
                need_tune = False
            if need_tune and tune_pairs:
                self.auto_tune_weights(tune_pairs, tune_targets)
                self._auto_tuned = True

        # 只计算一次各大模块，避免重复重算
        hist_m = self.hist_metrics(img1, img2)

        stat_m = self.correlation_metrics(img1, img2)
        g1 = self.to_gray(img1)
        g2 = self.to_gray(img2)
        stat_m['MSE'] = self.mse(g1, g2)
        stat_m['MAE'] = self.mae(g1, g2)
        stat_m['PSNR'] = self.psnr(g1, g2)
        stat_m['SSIM'] = self.ssim_score(g1, g2)
        stat_m.update(self.moment_metrics(img1, img2))
        stat_m.update(self.rank_correlation_metrics(img1, img2))
        stat_m['SVD_Dist'] = self.svd_spectrum_distance(img1, img2)
        stat_m['FFT_Dist'] = self.fft_spectrum_distance(img1, img2)

        info_m = self.info_metrics(img1, img2)

        hs = float(np.mean([
            self._sim_from_corr(hist_m['Hist_Correlation']),
            self._sim_from_dist(hist_m['Chi_Square']),
            self._sim_from_dist(hist_m['Bhattacharyya']),
            self._sim_from_dist(hist_m['KL']),
            self._sim_from_dist(hist_m['JS']),
            self._sim_from_dist(hist_m['EMD']),
        ]))

        ss = float(np.mean([
            self._sim_from_corr(stat_m['NCC']),
            self._sim_from_corr(stat_m['ZNCC']),
            self._sim_from_corr(stat_m['Correlation_Coeff']),
            self._sim_from_error(abs(stat_m['Covariance'])),
            self._sim_from_error(stat_m['MSE']),
            self._sim_from_error(stat_m['MAE']),
            self._sim_from_psnr(stat_m['PSNR']),
            self._sim_from_corr(stat_m['SSIM']),
            self._sim_from_error(stat_m['Mean_Diff']),
            self._sim_from_error(stat_m['Var_Diff']),
            self._sim_from_error(stat_m['Skew_Diff']),
            self._sim_from_error(stat_m['Kurt_Diff']),
            self._sim_from_error(stat_m['Entropy_Diff']),
            self._sim_from_corr(stat_m['Spearman']),
            self._sim_from_corr(stat_m['Kendall']),
            self._sim_from_error(stat_m['SVD_Dist']),
            self._sim_from_error(stat_m['FFT_Dist']),
        ]))

        nmi = float(np.clip(info_m['NMI'], 0.0, 1.0))
        mi_sim = self._sim_from_mi(info_m['MI'])
        iscore = float(0.8 * nmi + 0.2 * mi_sim)

        w_hist, w_stat, w_info = self.weights
        overall = float(w_hist * hs + w_stat * ss + w_info * iscore)

        results = {
            'score': {
                'Overall': overall,
                'Histogram': hs,
                'Statistical': ss,
                'Information': iscore
            },
            'hist': hist_m,
            'stat': stat_m,
            'info': info_m
        }

        results['interpret'] = self.interpret(results)
        return results
