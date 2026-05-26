import numpy as np

'''
根据SVD对图像进行分解：
get_svd_fadeout：按 k 截断重建，做离散 rank 递减淡出；
get_svd_single：每帧一个 rank-1 分量，每个奇异分量单独长啥样；
get_svd_auto：连续权重衰减（不是整数 k 截断），并可自动估计 r/gamma，“更平滑、更可调、可自动参数”。
'''
class CImageDecompose:
    def __init__(self, img):
        self.m_img = np.asarray(img, dtype=np.float32)

    @staticmethod
    def _check_count(count):
        if not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("count must be an integer >= 1")

    def _svd_one_channel_fadeout(self, ch, count=64, fade_curve='energy'):
        self._check_count(count)

        U, S, Vt = np.linalg.svd(ch, full_matrices=False)
        r = len(S)

        if fade_curve == 'linear':
            ks = np.rint(np.linspace(r, 0, count)).astype(int)

        elif fade_curve == 'energy':
            e = S ** 2
            cume = np.concatenate(([0.0], np.cumsum(e)))  # cume[k] = 前k项能量
            total = cume[-1] + 1e-12
            targets = np.linspace(1.0, 0.0, count)

            ks = np.empty(count, dtype=int)
            for i, t in enumerate(targets):
                target_e = t * total
                k = np.searchsorted(cume, target_e, side='left')
                ks[i] = int(np.clip(k, 0, r))

            # 保证单调不增（视觉更稳）
            for i in range(1, count):
                if ks[i] > ks[i - 1]:
                    ks[i] = ks[i - 1]
        else:
            raise ValueError("fade_curve must be 'linear' or 'energy'")

        out = []
        for k in ks:
            if k <= 0:
                rec = np.zeros_like(ch, dtype=np.float32)
            elif k >= r:
                rec = ch.copy()
            else:
                rec = (U[:, :k] * S[:k]) @ Vt[:k, :]
                rec = rec.astype(np.float32, copy=False)
            out.append(rec)
        return out

    def _svd_one_channel_single(self, ch, count=256):
        self._check_count(count)

        U, S, Vt = np.linalg.svd(ch, full_matrices=False)
        r = len(S)

        n = min(count, r)
        out = [
            (S[i] * np.outer(U[:, i], Vt[i, :])).astype(np.float32, copy=False)
            for i in range(n)
        ]
        if count > n:
            z = np.zeros_like(ch, dtype=np.float32)
            out.extend([z.copy() for _ in range(count - n)])
        return out

    def get_svd_fadeout(self, count=64, fade_curve='energy'):
        """
        渐隐 SVD 序列
        fade_curve:
            - 'linear': k 线性下降
            - 'energy': 保留能量线性下降（更平滑）
        """
        self._check_count(count)
        x = self.m_img

        if x.ndim == 2:
            return self._svd_one_channel_fadeout(x, count=count, fade_curve=fade_curve)
        elif x.ndim == 3:
            C = x.shape[2]
            per_ch = [
                self._svd_one_channel_fadeout(x[..., c], count=count, fade_curve=fade_curve)
                for c in range(C)
            ]
            return [
                np.stack([per_ch[c][i] for c in range(C)], axis=2).astype(np.float32, copy=False)
                for i in range(count)
            ]
        else:
            raise ValueError("img must be 2D or 3D")

    def get_svd_single(self, count=256):
        """
        每帧仅一个 rank-1 分量的 SVD 序列
        """
        self._check_count(count)
        x = self.m_img

        if x.ndim == 2:
            return self._svd_one_channel_single(x, count=count)
        elif x.ndim == 3:
            C = x.shape[2]
            per_ch = [self._svd_one_channel_single(x[..., c], count=count) for c in range(C)]
            return [
                np.stack([per_ch[c][i] for c in range(C)], axis=2).astype(np.float32, copy=False)
                for i in range(count)
            ]
        else:
            raise ValueError("img must be 2D or 3D")

    # 逐渐衰减主成分分量（自动估计衰减参数）
    def get_svd_auto(self,
        count=64,
        mode='fade_energy',          # 'fade_principal' | 'fade_energy'
        r=None,                      # None => 自动估计
        gamma=None,                  # None => 自动估计
        curve='smooth',              # 'linear' | 'smooth'
        auto_energy_ratio=0.9,       # 自动r的能量阈值
        auto_r_max=48,               # 自动r上限
        auto_gamma_range=(0.8, 2.2),   # 自动gamma范围（统一默认）
        color_param_source='luma',   # 'luma' | 'per_channel'
        return_params=False          # True时额外返回估计参数
    ):
        """
        连续SVD衰减（非整数k截断）+ 自适应参数
        - fade_principal: 仅衰减前r个奇异值（连续）
        - fade_energy: 全谱连续衰减（大奇异值衰减更多）
        支持灰度2D / 彩色3D(HWC)
        """
        self._check_count(count)

        def smoothstep(x: float) -> float:
            return x * x * (3.0 - 2.0 * x)

        def auto_r_gamma_from_s(S, e_ratio=0.90, r_max=20, gmin=0.5, gmax=3.0):
            e = S.astype(np.float64) ** 2
            total = e.sum() + 1e-12
            ce = np.cumsum(e)

            r_est = int(np.searchsorted(ce / total, e_ratio) + 1)
            r_est = int(np.clip(r_est, 1, min(len(S), r_max)))

            c1 = float(e[0] / total)  # 第一奇异值能量占比
            t = np.clip((c1 - 0.2) / (0.8 - 0.2), 0.0, 1.0)
            gamma_est = float(gmin + (gmax - gmin) * t)
            return r_est, gamma_est

        def _build_frames_from_svd(U, S, Vt, r_use, gamma_use):
            n = len(S)
            rr = int(np.clip(r_use, 1, n))
            ts = np.linspace(0.0, 1.0, count, dtype=np.float32)
        
            idx = np.arange(rr, dtype=np.float32)
            pr = ((rr - idx) / rr) ** float(gamma_use)  # 前大后小
        
            out = []
            for t in ts:
                a = smoothstep(float(t)) if curve == 'smooth' else float(t)
        
                if mode == 'fade_principal':
                    base = np.ones_like(S, dtype=np.float32)
                    base[:rr] = np.clip(1.0 - a * pr, 0.0, 1.0)
                    w = (1.0 - a) * base
                else:  # fade_energy
                    p = S / (S[0] + 1e-12)
                    base = np.clip(1.0 - a * p, 0.0, 1.0)
                    w = (1.0 - a) * base
        
                w = np.clip(w, 0.0, 1.0)
                rec = (U * (S * w)) @ Vt
                out.append(rec.astype(np.float32, copy=False))
            return out


        x = self.m_img

        if curve not in ('linear', 'smooth'):
            raise ValueError("curve must be 'linear' or 'smooth'")
        if mode not in ('fade_principal', 'fade_energy'):
            raise ValueError("mode must be 'fade_principal' or 'fade_energy'")
        if color_param_source not in ('luma', 'per_channel'):
            raise ValueError("color_param_source must be 'luma' or 'per_channel'")
        if not (0 < auto_energy_ratio <= 1):
            raise ValueError("auto_energy_ratio must be in (0, 1]")
        if auto_r_max < 1:
            raise ValueError("auto_r_max must be >= 1")
        if (not isinstance(auto_gamma_range, (tuple, list)) or len(auto_gamma_range) != 2
                or auto_gamma_range[0] > auto_gamma_range[1]):
            raise ValueError("auto_gamma_range must be (gmin, gmax) with gmin <= gmax")

        gmin, gmax = float(auto_gamma_range[0]), float(auto_gamma_range[1])

        if x.ndim == 2:
            U, S, Vt = np.linalg.svd(x, full_matrices=False)

            if r is None or gamma is None:
                r_auto, g_auto = auto_r_gamma_from_s(
                    S, e_ratio=auto_energy_ratio, r_max=auto_r_max, gmin=gmin, gmax=gmax
                )
            else:
                r_auto, g_auto = None, None

            r_use = int(r_auto if r is None else r)
            g_use = float(g_auto if gamma is None else gamma)

            frames = _build_frames_from_svd(U, S.astype(np.float32), Vt, r_use, g_use)
            params = {
                'r': r_use, 'gamma': g_use, 'mode': mode, 'curve': curve,
                'auto_used': {'r': r is None, 'gamma': gamma is None}
            }
            return (frames, params) if return_params else frames

        elif x.ndim == 3:
            H, W, C = x.shape
            if C < 1:
                raise ValueError("invalid channel count")

            if (r is None or gamma is None) and color_param_source == 'luma':
                if C >= 3:
                    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
                else:
                    y = x[..., 0]

                _, S_l, _ = np.linalg.svd(y.astype(np.float32), full_matrices=False)
                r_auto, g_auto = auto_r_gamma_from_s(
                    S_l, e_ratio=auto_energy_ratio, r_max=auto_r_max, gmin=gmin, gmax=gmax
                )
                r_use_global = int(r_auto if r is None else r)
                g_use_global = float(g_auto if gamma is None else gamma)

                per_ch_frames = []
                for c in range(C):
                    U, S, Vt = np.linalg.svd(x[..., c], full_matrices=False)
                    per_ch_frames.append(
                        _build_frames_from_svd(U, S.astype(np.float32), Vt, r_use_global, g_use_global)
                    )

                frames = [
                    np.stack([per_ch_frames[c][i] for c in range(C)], axis=2).astype(np.float32, copy=False)
                    for i in range(count)
                ]
                params = {
                    'r': r_use_global, 'gamma': g_use_global, 'mode': mode, 'curve': curve,
                    'color_param_source': 'luma',
                    'auto_used': {'r': r is None, 'gamma': gamma is None}
                }
                return (frames, params) if return_params else frames

            else:
                per_ch = []
                r_list, g_list = [], []

                for c in range(C):
                    U, S, Vt = np.linalg.svd(x[..., c], full_matrices=False)

                    if r is None or gamma is None:
                        r_auto, g_auto = auto_r_gamma_from_s(
                            S, e_ratio=auto_energy_ratio, r_max=auto_r_max, gmin=gmin, gmax=gmax
                        )
                    else:
                        r_auto, g_auto = None, None

                    r_use = int(r_auto if r is None else r)
                    g_use = float(g_auto if gamma is None else gamma)

                    r_list.append(r_use)
                    g_list.append(g_use)
                    per_ch.append(_build_frames_from_svd(U, S.astype(np.float32), Vt, r_use, g_use))

                frames = [
                    np.stack([per_ch[c][i] for c in range(C)], axis=2).astype(np.float32, copy=False)
                    for i in range(count)
                ]
                params = {
                    'r_per_channel': r_list, 'gamma_per_channel': g_list,
                    'mode': mode, 'curve': curve, 'color_param_source': 'per_channel',
                    'auto_used': {'r': r is None, 'gamma': gamma is None}
                }
                return (frames, params) if return_params else frames

        else:
            raise ValueError("img must be 2D or 3D (HWC)")
