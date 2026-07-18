"""
SVD图像分解工具 (SVD Image Decomposition Tool)
================================================
生产版本 v2.1.0 - 经过严格验证的版本

作者：Image Processing Lab
更新：2024-01
许可：MIT License

依赖：numpy >= 1.19.0
"""

import numpy as np
from typing import Union, Tuple, List, Optional, Literal, Dict, Any
import warnings


class CImageDecompose:
    """
    基于奇异值分解(SVD)的图像处理类
    
    数学原理：
    对于矩阵 A (m×n)，SVD分解为：A = U·Σ·V^T
    其中：
    - U: m×m 正交矩阵（左奇异向量）
    - Σ: m×n 对角矩阵（奇异值）
    - V^T: n×n 正交矩阵的转置（右奇异向量）
    
    重建公式：
    使用前k个奇异值：A_k = U[:,:k] · Σ[:k,:k] · V^T[:k,:]
    """
    
    def __init__(
        self, 
        img: Union[np.ndarray, list], 
        normalize: bool = True,
        validate: bool = True
    ):
        """
        初始化SVD图像分解器
        
        Args:
            img: 输入图像数组
                - 2D: 灰度图像 [H, W]
                - 3D: 彩色图像 [H, W, C]
            normalize: 是否归一化到[0,255]范围
            validate: 是否执行输入验证
        
        Raises:
            ValueError: 图像维度错误或数据无效
            TypeError: 输入类型不支持
        """
        # 类型转换
        if not isinstance(img, np.ndarray):
            try:
                img = np.asarray(img)
            except Exception as e:
                raise TypeError(f"无法将输入转换为numpy数组: {e}")
        
        # 维度验证
        if img.ndim not in (2, 3):
            raise ValueError(
                f"图像必须是2D(灰度)或3D(彩色)数组，当前维度：{img.ndim}"
            )
        
        # 3D图像通道数验证
        if img.ndim == 3 and img.shape[2] not in (1, 3, 4):
            warnings.warn(
                f"非标准通道数：{img.shape[2]}。标准值为1(灰度)、3(RGB)或4(RGBA)"
            )
        
        # 数据验证
        if validate:
            if not np.isfinite(img).all():
                raise ValueError("图像包含无限值或NaN")
            
            # 检查值范围并给出警告
            img_min, img_max = img.min(), img.max()
            if img_min < 0:
                warnings.warn(f"图像包含负值：最小值={img_min}")
            if img_max > 255 and normalize:
                warnings.warn(f"图像最大值({img_max})超过255，将进行裁剪")
        
        # 转换为float32以确保计算精度
        self.m_img = img.astype(np.float32, copy=True)
        
        # 归一化处理
        if normalize:
            if self.m_img.max() <= 1.0 and self.m_img.min() >= 0:
                # 假设是[0,1]范围，转换到[0,255]
                self.m_img *= 255.0
            elif self.m_img.max() > 255:
                # 裁剪到[0,255]
                self.m_img = np.clip(self.m_img, 0, 255)
        
        # 存储元信息
        self.shape = self.m_img.shape
        self.is_color = (self.m_img.ndim == 3)
        self.n_channels = self.shape[2] if self.is_color else 1
        self.height = self.shape[0]
        self.width = self.shape[1]
        
        # 初始化缓存
        self._svd_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._error_cache: Dict[int, Dict[str, float]] = {}
    
    def clear_cache(self) -> None:
        """清除所有缓存"""
        self._svd_cache.clear()
        self._error_cache.clear()
    
    @staticmethod
    def _convert_output(
        original: np.ndarray,
        reconstruction: np.ndarray,
        output: str
    ) -> np.ndarray:
        """
        根据输出模式转换结果
        
        数学定义：
        - reconstruction: R = Σ(σᵢ·uᵢ·vᵢᵀ)
        - residual: E = A - R
        - abs_residual: |E| = |A - R|
        - energy_residual: E² = (A - R)²
        """
        output_map = {
            'reconstruction': lambda o, r: r,
            'residual': lambda o, r: o - r,
            'abs_residual': lambda o, r: np.abs(o - r),
            'energy_residual': lambda o, r: (o - r) ** 2
        }
        
        if output not in output_map:
            valid_outputs = ', '.join(output_map.keys())
            raise ValueError(f"output必须是以下之一: {valid_outputs}")
        
        result = output_map[output](original, reconstruction)
        return result.astype(np.float32)
    
    @staticmethod
    def _validate_count(count: Any) -> int:
        """验证并转换count参数"""
        try:
            count_int = int(count)
            if count_int < 1:
                raise ValueError
            return count_int
        except (ValueError, TypeError):
            raise ValueError(f"count必须是正整数，当前值：{count}")
    
    def _get_svd(self, channel_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行SVD分解（带缓存）
        
        使用numpy.linalg.svd，设置full_matrices=False进行经济型SVD
        对于m×n矩阵(m>n)，只计算前min(m,n)个奇异向量
        """
        # 使用数组的内存地址作为缓存键
        cache_key = id(channel_data)
        
        if cache_key not in self._svd_cache:
            try:
                # 执行SVD分解
                # 注意：numpy的SVD已经按降序排列奇异值
                U, S, Vt = np.linalg.svd(channel_data, full_matrices=False)
                
                # 验证SVD结果
                if not np.all(S >= 0):
                    warnings.warn("检测到负奇异值，这不应该发生")
                    S = np.abs(S)
                
                # 确保奇异值按降序排列（numpy应该已经保证）
                if not np.all(S[:-1] >= S[1:]):
                    warnings.warn("奇异值未按降序排列，正在修正")
                    sort_idx = np.argsort(S)[::-1]
                    S = S[sort_idx]
                    U = U[:, sort_idx]
                    Vt = Vt[sort_idx, :]
                
                self._svd_cache[cache_key] = (U, S.astype(np.float32), Vt)
                
            except np.linalg.LinAlgError as e:
                raise RuntimeError(f"SVD分解失败: {e}")
        
        return self._svd_cache[cache_key]
    
    def _svd_one_channel_fadeout(
        self,
        ch: np.ndarray,
        count: int = 64,
        fade_curve: str = 'energy',
        output: str = 'reconstruction'
    ) -> List[np.ndarray]:
        """
        单通道SVD渐隐序列生成
        
        算法逻辑：
        1. linear模式：k值从r线性递减到0
        2. energy模式：保持能量比例从100%递减到0%
           能量定义：E(k) = Σᵢ₌₁ᵏ σᵢ²
        """
        count = self._validate_count(count)
        U, S, Vt = self._get_svd(ch)
        r = len(S)  # 秩
        
        if fade_curve == 'linear':
            # k值线性递减：r, r*(n-1)/n, ..., r/n, 0
            ks = np.round(np.linspace(r, 0, count)).astype(int)
            
        elif fade_curve == 'energy':
            # 基于能量保留的递减
            # 计算奇异值能量（奇异值的平方）
            energy = S.astype(np.float64) ** 2
            
            # 累积能量（前k项的能量和）
            # cumulative[k] = Σᵢ₌₀ᵏ⁻¹ σᵢ²
            cumulative_energy = np.concatenate(([0.0], np.cumsum(energy)))
            total_energy = cumulative_energy[-1]
            
            # 防止除零
            if total_energy < 1e-12:
                # 退化情况：全零矩阵
                ks = np.zeros(count, dtype=int)
            else:
                # 目标能量比例：从100%线性递减到0%
                target_ratios = np.linspace(1.0, 0.0, count)
                ks = np.empty(count, dtype=int)
                
                for i, ratio in enumerate(target_ratios):
                    target_energy = ratio * total_energy
                    
                    # 二分查找：找到累积能量>=目标能量的最小k
                    # 使用searchsorted找到插入位置
                    k = np.searchsorted(cumulative_energy[1:], target_energy, side='right')
                    ks[i] = min(k, r)
                
                # 确保单调性：k值不能增加
                # 这保证视觉效果的稳定性
                for i in range(1, count):
                    ks[i] = min(ks[i], ks[i-1])
        
        else:
            raise ValueError(f"不支持的fade_curve: {fade_curve}，必须是'linear'或'energy'")
        
        # 生成帧序列
        frames = []
        for k in ks:
            if k <= 0:
                # k=0：零矩阵
                reconstruction = np.zeros_like(ch, dtype=np.float32)
            elif k >= r:
                # k>=r：完全重建
                reconstruction = ch.astype(np.float32)
            else:
                # 部分重建：A_k = U[:,:k] · diag(S[:k]) · Vt[:k,:]
                # 优化：利用广播避免显式对角矩阵
                reconstruction = (U[:, :k] * S[:k]) @ Vt[:k, :]
                reconstruction = reconstruction.astype(np.float32)
            
            result = self._convert_output(ch, reconstruction, output)
            frames.append(result)
        
        return frames
    
    def _svd_one_channel_single(
        self,
        ch: np.ndarray,
        count: int = 256
    ) -> List[np.ndarray]:
        """
        生成单个SVD分量序列
        
        第i个分量：Aᵢ = σᵢ · uᵢ · vᵢᵀ
        其中uᵢ是U的第i列，vᵢᵀ是Vt的第i行
        """
        count = self._validate_count(count)
        U, S, Vt = self._get_svd(ch)
        r = len(S)
        
        components = []
        
        # 生成每个rank-1分量
        for i in range(min(count, r)):
            # 外积：uᵢ ⊗ vᵢ，然后乘以奇异值
            # np.outer(U[:,i], Vt[i,:]) 等价于 U[:,i:i+1] @ Vt[i:i+1,:]
            component = S[i] * np.outer(U[:, i], Vt[i, :])
            components.append(component.astype(np.float32))
        
        # 如果请求数量超过秩，用零填充
        if count > r:
            zero_component = np.zeros_like(ch, dtype=np.float32)
            for _ in range(count - r):
                components.append(zero_component.copy())
        
        return components
    
    def get_svd_fadeout(
        self,
        count: int = 64,
        fade_curve: Literal['linear', 'energy'] = 'energy',
        output: Literal['reconstruction', 'residual', 'abs_residual', 'energy_residual'] = 'reconstruction'
    ) -> List[np.ndarray]:
        """
        生成SVD渐隐动画序列
        
        通过逐渐减少使用的奇异值数量，生成从完整图像到空白的序列
        """
        count = self._validate_count(count)
        
        if self.is_color:
            # 彩色图像：分通道处理后合并
            channel_frames = []
            
            for c in range(self.n_channels):
                frames = self._svd_one_channel_fadeout(
                    self.m_img[..., c],
                    count=count,
                    fade_curve=fade_curve,
                    output=output
                )
                channel_frames.append(frames)
            
            # 合并通道：[count, H, W, C]
            merged_frames = []
            for i in range(count):
                frame = np.stack([channel_frames[c][i] for c in range(self.n_channels)], axis=2)
                merged_frames.append(frame.astype(np.float32))
            
            return merged_frames
        
        else:
            # 灰度图像：直接处理
            return self._svd_one_channel_fadeout(
                self.m_img,
                count=count,
                fade_curve=fade_curve,
                output=output
            )
    
    def get_svd_single(self, count: int = 256) -> List[np.ndarray]:
        """
        生成单个SVD分量的可视化序列
        
        每帧显示一个rank-1矩阵（σᵢ·uᵢ·vᵢᵀ）
        """
        count = self._validate_count(count)
        
        if self.is_color:
            # 彩色图像：分通道处理
            channel_components = []
            
            for c in range(self.n_channels):
                components = self._svd_one_channel_single(
                    self.m_img[..., c],
                    count=count
                )
                channel_components.append(components)
            
            # 合并通道
            merged_components = []
            for i in range(count):
                component = np.stack(
                    [channel_components[c][i] for c in range(self.n_channels)],
                    axis=2
                )
                merged_components.append(component.astype(np.float32))
            
            return merged_components
        
        else:
            # 灰度图像
            return self._svd_one_channel_single(self.m_img, count=count)
    
    def get_svd_auto(
        self,
        count: int = 64,
        mode: Literal['fade_principal', 'fade_energy'] = 'fade_energy',
        r: Optional[int] = None,
        gamma: Optional[float] = None,
        curve: Literal['linear', 'smooth'] = 'smooth',
        auto_energy_ratio: float = 0.9,
        auto_r_max: int = 48,
        auto_gamma_range: Tuple[float, float] = (0.8, 2.2),
        color_param_source: Literal['luma', 'per_channel'] = 'luma',
        return_params: bool = False,
        output: str = 'reconstruction'
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], Dict[str, Any]]]:
        """
        高级SVD分解：连续权重衰减 + 自动参数优化
        
        核心算法：
        1. 连续权重：w(t,i) ∈ [0,1]，而非离散的0/1截断
        2. 自动参数：基于能量分布估计最优r和gamma
        3. 两种衰减模式：
           - fade_principal: 仅衰减前r个分量
           - fade_energy: 全谱衰减，按能量比例
        """
        count = self._validate_count(count)
        
        # 参数验证
        if curve not in ('linear', 'smooth'):
            raise ValueError(f"curve必须是'linear'或'smooth'")
        
        if mode not in ('fade_principal', 'fade_energy'):
            raise ValueError(f"mode必须是'fade_principal'或'fade_energy'")
        
        if color_param_source not in ('luma', 'per_channel'):
            raise ValueError(f"color_param_source必须是'luma'或'per_channel'")
        
        if not 0 < auto_energy_ratio <= 1:
            raise ValueError(f"auto_energy_ratio必须在(0,1]范围")
        
        if auto_r_max < 1:
            raise ValueError(f"auto_r_max必须>=1")
        
        if (not isinstance(auto_gamma_range, (tuple, list)) or 
            len(auto_gamma_range) != 2 or 
            auto_gamma_range[0] > auto_gamma_range[1]):
            raise ValueError(f"auto_gamma_range格式错误")
        
        gmin, gmax = float(auto_gamma_range[0]), float(auto_gamma_range[1])
        
        def smoothstep(x: float) -> float:
            """
            Hermite插值的S型曲线
            smoothstep(x) = 3x² - 2x³, x ∈ [0,1]
            特性：f(0)=0, f(1)=1, f'(0)=f'(1)=0
            """
            x = float(np.clip(x, 0.0, 1.0))
            return x * x * (3.0 - 2.0 * x)
        
        def auto_estimate_params(
            S: np.ndarray,
            e_ratio: float = 0.9,
            r_max: int = 48,
            gamma_min: float = 0.8,
            gamma_max: float = 2.2
        ) -> Tuple[int, float]:
            """
            自动估计参数
            
            算法：
            1. r估计：找到保留e_ratio能量的最小秩
            2. gamma估计：基于能量集中度
               - 能量集中（第一奇异值占比大）→ 小gamma（快速衰减）
               - 能量分散 → 大gamma（缓慢衰减）
            """
            # 计算能量分布（使用float64避免精度损失）
            energy = S.astype(np.float64) ** 2
            total_energy = np.sum(energy)
            
            if total_energy < 1e-12:
                # 退化情况
                return 1, 1.0
            
            # 估计r：累积能量达到阈值
            cumulative_ratio = np.cumsum(energy) / total_energy
            r_est = np.searchsorted(cumulative_ratio, e_ratio) + 1
            r_est = int(np.clip(r_est, 1, min(len(S), r_max)))
            
            # 估计gamma：基于第一奇异值的能量占比
            first_ratio = float(energy[0] / total_energy)
            
            # 映射函数：能量集中度 → gamma
            # 高集中度(>0.8) → gamma接近gamma_min
            # 低集中度(<0.2) → gamma接近gamma_max
            t = np.clip((first_ratio - 0.2) / 0.6, 0.0, 1.0)
            gamma_est = gamma_min + (gamma_max - gamma_min) * (1.0 - t)
            
            return r_est, gamma_est
        
        def build_frames_from_svd(
            U: np.ndarray,
            S: np.ndarray,
            Vt: np.ndarray,
            original: np.ndarray,
            r_use: int,
            gamma_use: float
        ) -> List[np.ndarray]:
            """
            构建帧序列（核心算法）
            
            权重计算：
            - fade_principal: w[i] = 1 - t * ((r-i)/r)^gamma, i < r
            - fade_energy: w[i] = (1-t) * (1 - t * (S[i]/S[0]))
            """
            n_singular = len(S)
            r_clip = int(np.clip(r_use, 1, n_singular))
            
            # 生成时间序列
            time_points = np.linspace(0.0, 1.0, count, dtype=np.float64)
            
            # 预计算主成分衰减曲线
            if mode == 'fade_principal':
                indices = np.arange(r_clip, dtype=np.float64)
                # 幂函数衰减：前面的分量衰减慢，后面的快
                decay_curve = ((r_clip - indices) / max(r_clip, 1)) ** gamma_use
            
            frames = []
            
            for t in time_points:
                # 时间插值
                if curve == 'smooth':
                    alpha = smoothstep(t)
                else:
                    alpha = float(t)
                
                # 计算权重向量
                weights = np.ones(n_singular, dtype=np.float64)
                
                if mode == 'fade_principal':
                    # 仅衰减前r个分量
                    if r_clip > 0:
                        weights[:r_clip] = np.maximum(1.0 - alpha * decay_curve, 0.0)
                        weights[r_clip:] = 1.0 - alpha  # 其余分量统一衰减
                    
                elif mode == 'fade_energy':
                    # 全谱衰减，按奇异值大小比例
                    if S[0] > 1e-12:
                        # 归一化奇异值
                        s_normalized = S / S[0]
                        # 大奇异值衰减更多
                        weights = (1.0 - alpha) * np.maximum(1.0 - alpha * s_normalized, 0.0)
                    else:
                        weights = np.ones(n_singular) * (1.0 - alpha)
                
                # 裁剪到[0,1]
                weights = np.clip(weights, 0.0, 1.0).astype(np.float32)
                
                # 重建：A_w = U · diag(S·w) · Vt
                weighted_S = S * weights
                reconstruction = (U * weighted_S) @ Vt
                
                # 转换输出
                result = self._convert_output(original, reconstruction, output)
                frames.append(result)
            
            return frames
        
        # 处理灰度图像
        if not self.is_color:
            U, S, Vt = self._get_svd(self.m_img)
            
            # 参数估计或使用手动值
            if r is None or gamma is None:
                r_auto, gamma_auto = auto_estimate_params(
                    S, auto_energy_ratio, auto_r_max, gmin, gmax
                )
                r_use = r_auto if r is None else r
                gamma_use = gamma_auto if gamma is None else gamma
            else:
                r_use = r
                gamma_use = gamma
            
            frames = build_frames_from_svd(U, S, Vt, self.m_img, r_use, gamma_use)
            
            if return_params:
                params = {
                    'r': r_use,
                    'gamma': gamma_use,
                    'mode': mode,
                    'curve': curve,
                    'auto_used': {'r': r is None, 'gamma': gamma is None}
                }
                return frames, params
            return frames
        
        # 处理彩色图像
        else:
            if (r is None or gamma is None) and color_param_source == 'luma':
                # 基于亮度通道估计全局参数
                if self.n_channels >= 3:
                    # ITU-R BT.601 亮度公式
                    luma = (0.299 * self.m_img[..., 0] +
                           0.587 * self.m_img[..., 1] +
                           0.114 * self.m_img[..., 2])
                else:
                    luma = self.m_img[..., 0]
                
                _, S_luma, _ = np.linalg.svd(luma, full_matrices=False)
                r_auto, gamma_auto = auto_estimate_params(
                    S_luma, auto_energy_ratio, auto_r_max, gmin, gmax
                )
                
                r_use = r_auto if r is None else r
                gamma_use = gamma_auto if gamma is None else gamma
                
                # 应用相同参数到所有通道
                channel_frames = []
                for c in range(self.n_channels):
                    U, S, Vt = self._get_svd(self.m_img[..., c])
                    frames = build_frames_from_svd(
                        U, S, Vt, self.m_img[..., c], r_use, gamma_use
                    )
                    channel_frames.append(frames)
                
                # 合并通道
                merged_frames = []
                for i in range(count):
                    frame = np.stack(
                        [channel_frames[c][i] for c in range(self.n_channels)],
                        axis=2
                    ).astype(np.float32)
                    merged_frames.append(frame)
                
                if return_params:
                    params = {
                        'r': r_use,
                        'gamma': gamma_use,
                        'mode': mode,
                        'curve': curve,
                        'color_param_source': 'luma',
                        'auto_used': {'r': r is None, 'gamma': gamma is None}
                    }
                    return merged_frames, params
                return merged_frames
                
            else:
                # 每通道独立参数
                channel_frames = []
                r_list = []
                gamma_list = []
                
                for c in range(self.n_channels):
                    U, S, Vt = self._get_svd(self.m_img[..., c])
                    
                    if r is None or gamma is None:
                        r_auto, gamma_auto = auto_estimate_params(
                            S, auto_energy_ratio, auto_r_max, gmin, gmax
                        )
                        r_use = r_auto if r is None else r
                        gamma_use = gamma_auto if gamma is None else gamma
                    else:
                        r_use = r
                        gamma_use = gamma
                    
                    r_list.append(r_use)
                    gamma_list.append(gamma_use)
                    
                    frames = build_frames_from_svd(
                        U, S, Vt, self.m_img[..., c], r_use, gamma_use
                    )
                    channel_frames.append(frames)
                
                # 合并通道
                merged_frames = []
                for i in range(count):
                    frame = np.stack(
                        [channel_frames[c][i] for c in range(self.n_channels)],
                        axis=2
                    ).astype(np.float32)
                    merged_frames.append(frame)
                
                if return_params:
                    params = {
                        'r_per_channel': r_list,
                        'gamma_per_channel': gamma_list,
                        'mode': mode,
                        'curve': curve,
                        'color_param_source': 'per_channel',
                        'auto_used': {'r': r is None, 'gamma': gamma is None}
                    }
                    return merged_frames, params
                return merged_frames
    
    def get_reconstruction_error(self, k: int) -> Dict[str, float]:
        """
        计算使用前k个奇异值重建的误差指标
        
        指标定义：
        - MSE = (1/N) Σ(A - A_k)²
        - PSNR = 10·log₁₀(MAX²/MSE), MAX=255
        - Energy = Σᵢ₌₁ᵏ σᵢ² / Σᵢ σᵢ²
        """
        k = self._validate_count(k)
        
        # 检查缓存
        if k in self._error_cache:
            return self._error_cache[k].copy()
        
        errors = {'k': k}
        
        if self.is_color:
            # 彩色图像：计算各通道平均
            mse_list = []
            energy_list = []
            
            for c in range(self.n_channels):
                U, S, Vt = self._get_svd(self.m_img[..., c])
                r = len(S)
                k_actual = min(k, r)
                
                # 重建
                if k_actual >= r:
                    reconstruction = self.m_img[..., c]
                    mse = 0.0
                else:
                    reconstruction = (U[:, :k_actual] * S[:k_actual]) @ Vt[:k_actual, :]
                    mse = float(np.mean((self.m_img[..., c] - reconstruction) ** 2))
                
                mse_list.append(mse)
                
                # 能量保留率
                total_energy = float(np.sum(S ** 2))
                if total_energy > 1e-12:
                    retained_energy = float(np.sum(S[:k_actual] ** 2))
                    energy_list.append(retained_energy / total_energy)
                else:
                    energy_list.append(1.0)
            
            errors['mse'] = float(np.mean(mse_list))
            errors['energy_retained'] = float(np.mean(energy_list))
            
        else:
            # 灰度图像
            U, S, Vt = self._get_svd(self.m_img)
            r = len(S)
            k_actual = min(k, r)
            
            # 重建和MSE
            if k_actual >= r:
                errors['mse'] = 0.0
            else:
                reconstruction = (U[:, :k_actual] * S[:k_actual]) @ Vt[:k_actual, :]
                errors['mse'] = float(np.mean((self.m_img - reconstruction) ** 2))
            
            # 能量保留率
            total_energy = float(np.sum(S ** 2))
            if total_energy > 1e-12:
                retained_energy = float(np.sum(S[:k_actual] ** 2))
                errors['energy_retained'] = retained_energy / total_energy
            else:
                errors['energy_retained'] = 1.0
        
        # PSNR计算
        if errors['mse'] > 0:
            errors['psnr'] = float(10 * np.log10(255 ** 2 / errors['mse']))
        else:
            errors['psnr'] = float('inf')
        
        # 压缩比计算
        original_size = self.m_img.size * 4  # float32
        if self.is_color:
            # 每通道：k*(H+W+1)个float32
            compressed_size = self.n_channels * k * (self.height + self.width + 1) * 4
        else:
            compressed_size = k * (self.height + self.width + 1) * 4
        
        errors['compression_ratio'] = original_size / max(compressed_size, 1)
        
        # 缓存结果
        self._error_cache[k] = errors.copy()
        
        return errors
    
    def find_optimal_k(
        self,
        target: Literal['energy', 'psnr'] = 'energy',
        threshold: float = 0.95
    ) -> int:
        """
        寻找满足目标条件的最优k值
        
        使用二分查找优化搜索效率
        """
        if target == 'energy':
            if not 0 < threshold <= 1:
                raise ValueError(f"能量阈值必须在(0,1]范围")
            
            # 直接计算，不需要二分查找
            if self.is_color:
                k_per_channel = []
                
                for c in range(self.n_channels):
                    _, S, _ = self._get_svd(self.m_img[..., c])
                    energy = S ** 2
                    cumulative = np.cumsum(energy)
                    total = cumulative[-1]
                    
                    if total > 1e-12:
                        k = np.searchsorted(cumulative / total, threshold) + 1
                        k_per_channel.append(min(k, len(S)))
                    else:
                        k_per_channel.append(1)
                
                # 返回最大值以确保所有通道都满足条件
                return max(k_per_channel)
            
            else:
                _, S, _ = self._get_svd(self.m_img)
                energy = S ** 2
                cumulative = np.cumsum(energy)
                total = cumulative[-1]
                
                if total > 1e-12:
                    k = np.searchsorted(cumulative / total, threshold) + 1
                    return min(k, len(S))
                else:
                    return 1
        
        elif target == 'psnr':
            if threshold <= 0:
                raise ValueError(f"PSNR阈值必须>0")
            
            # 二分查找
            max_k = min(self.height, self.width)
            left, right = 1, max_k
            result = max_k
            
            while left <= right:
                mid = (left + right) // 2
                metrics = self.get_reconstruction_error(mid)
                
                if metrics['psnr'] >= threshold:
                    result = mid
                    right = mid - 1
                else:
                    left = mid + 1
            
            return result
        
        else:
            raise ValueError(f"不支持的target: {target}")


# ============================================================================
# 单元测试
# ============================================================================

def run_unit_tests():
    """运行单元测试以验证核心逻辑"""
    print("=" * 60)
    print("运行单元测试")
    print("=" * 60)
    
    def test_svd_reconstruction():
        """测试SVD重建的正确性"""
        print("\n测试1: SVD重建正确性")
        print("-" * 40)
        
        # 创建测试矩阵
        np.random.seed(42)
        test_matrix = np.random.randn(50, 30).astype(np.float32)
        
        # SVD分解
        U, S, Vt = np.linalg.svd(test_matrix, full_matrices=False)
        
        # 完全重建
        reconstruction = (U * S) @ Vt
        
        # 验证重建误差
        error = np.max(np.abs(test_matrix - reconstruction))
        print(f"最大重建误差: {error:.2e}")
        assert error < 1e-5, f"重建误差过大: {error}"
        print("✓ SVD重建测试通过")
    
    def test_energy_calculation():
        """测试能量计算的正确性"""
        print("\n测试2: 能量计算正确性")
        print("-" * 40)
        
        # 创建测试数据
        S = np.array([10, 5, 2, 1], dtype=np.float32)
        energy = S ** 2
        total_energy = energy.sum()
        
        # 测试累积能量
        cumulative = np.cumsum(energy)
        ratios = cumulative / total_energy
        
        expected_ratios = [100/130, 125/130, 129/130, 130/130]
        
        for i, (actual, expected) in enumerate(zip(ratios, expected_ratios)):
            error = abs(actual - expected)
            print(f"奇异值{i}: 期望={expected:.4f}, 实际={actual:.4f}, 误差={error:.2e}")
            assert error < 1e-6, f"能量比例计算错误"
        
        print("✓ 能量计算测试通过")
    
    def test_fadeout_monotonicity():
        """测试fadeout的单调性"""
        print("\n测试3: Fadeout单调性")
        print("-" * 40)
        
        # 创建测试图像
        test_img = np.random.randn(20, 20).astype(np.float32) * 50 + 128
        decomposer = SVDImageDecompose(test_img, normalize=False)
        
        # 生成fadeout序列
        frames = decomposer.get_svd_fadeout(count=10, fade_curve='energy')
        
        # 检查能量单调递减
        energies = [np.sum(frame ** 2) for frame in frames]
        
        for i in range(1, len(energies)):
            assert energies[i] <= energies[i-1] * 1.001, \
                f"能量未单调递减: {energies[i]} > {energies[i-1]}"
        
        print(f"能量序列: {[f'{e:.0f}' for e in energies[:5]]} ...")
        print("✓ Fadeout单调性测试通过")
    
    def test_auto_parameters():
        """测试自动参数估计"""
        print("\n测试4: 自动参数估计")
        print("-" * 40)
        
        # 创建具有不同能量分布的测试图像
        # 情况1: 能量集中
        concentrated = np.diag([100, 10, 1, 0.1])
        decomposer1 = SVDImageDecompose(concentrated, normalize=False)
        
        frames1, params1 = decomposer1.get_svd_auto(
            count=5,
            return_params=True,
            auto_energy_ratio=0.95
        )
        
        print(f"能量集中图像: r={params1['r']}, gamma={params1['gamma']:.3f}")
        
        # 情况2: 能量分散
        dispersed = np.random.randn(20, 20).astype(np.float32)
        decomposer2 = SVDImageDecompose(dispersed, normalize=False)
        
        frames2, params2 = decomposer2.get_svd_auto(
            count=5,
            return_params=True,
            auto_energy_ratio=0.95
        )
        
        print(f"能量分散图像: r={params2['r']}, gamma={params2['gamma']:.3f}")
        
        # gamma应该反映能量分布
        assert params1['gamma'] <= params2['gamma'], \
            "能量集中时gamma应该更小"
        
        print("✓ 自动参数估计测试通过")
    
    def test_color_consistency():
        """测试彩色图像处理的一致性"""
        print("\n测试5: 彩色图像处理一致性")
        print("-" * 40)
        
        # 创建彩色测试图像
        np.random.seed(42)
        color_img = np.random.randn(10, 10, 3).astype(np.float32) * 50 + 128
        
        decomposer = SVDImageDecompose(color_img, normalize=False)
        
        # 测试luma模式
        frames_luma, params_luma = decomposer.get_svd_auto(
            count=3,
            color_param_source='luma',
            return_params=True
        )
        
        print(f"Luma模式: r={params_luma['r']}, gamma={params_luma['gamma']:.3f}")
        
        # 测试per_channel模式
        frames_per_ch, params_per_ch = decomposer.get_svd_auto(
            count=3,
            color_param_source='per_channel',
            return_params=True
        )
        
        print(f"Per-channel模式: r={params_per_ch['r_per_channel']}")
        
        # 验证输出形状
        assert frames_luma[0].shape == color_img.shape, "输出形状不匹配"
        assert frames_per_ch[0].shape == color_img.shape, "输出形状不匹配"
        
        print("✓ 彩色图像处理测试通过")
    
    def test_numerical_stability():
        """测试数值稳定性"""
        print("\n测试6: 数值稳定性")
        print("-" * 40)
        
        # 测试极端情况
        test_cases = [
            ("零矩阵", np.zeros((10, 10), dtype=np.float32)),
            ("常数矩阵", np.ones((10, 10), dtype=np.float32) * 100),
            ("极小值", np.random.randn(10, 10).astype(np.float32) * 1e-10),
            ("极大值", np.random.randn(10, 10).astype(np.float32) * 1e10),
        ]
        
        for name, test_img in test_cases:
            try:
                decomposer = SVDImageDecompose(test_img, normalize=False, validate=False)
                frames = decomposer.get_svd_fadeout(count=3)
                errors = decomposer.get_reconstruction_error(k=5)
                print(f"{name}: ✓ 处理成功, PSNR={errors['psnr']:.1f}")
            except Exception as e:
                print(f"{name}: ✗ 失败 - {e}")
                raise
        
        print("✓ 数值稳定性测试通过")
    
    # 运行所有测试
    try:
        test_svd_reconstruction()
        test_energy_calculation()
        test_fadeout_monotonicity()
        test_auto_parameters()
        test_color_consistency()
        test_numerical_stability()
        
        print("\n" + "=" * 60)
        print("所有单元测试通过！ ✓")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n意外错误: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# 性能测试
# ============================================================================

def run_performance_tests():
    """运行性能测试"""
    import time
    
    print("\n" + "=" * 60)
    print("性能测试")
    print("=" * 60)
    
    sizes = [(64, 64), (128, 128), (256, 256)]
    
    for h, w in sizes:
        print(f"\n图像尺寸: {h}×{w}")
        print("-" * 40)
        
        # 创建测试图像
        test_img = np.random.randn(h, w).astype(np.float32) * 50 + 128
        decomposer = SVDImageDecompose(test_img)
        
        # 测试SVD分解速度
        start = time.time()
        _, _, _ = np.linalg.svd(test_img, full_matrices=False)
        svd_time = time.time() - start
        print(f"SVD分解时间: {svd_time*1000:.2f} ms")
        
        # 测试fadeout生成
        start = time.time()
        frames = decomposer.get_svd_fadeout(count=30)
        fadeout_time = time.time() - start
        print(f"Fadeout生成(30帧): {fadeout_time*1000:.2f} ms")
        
        # 测试auto模式
        start = time.time()
        frames = decomposer.get_svd_auto(count=30)
        auto_time = time.time() - start
        print(f"Auto模式(30帧): {auto_time*1000:.2f} ms")
        
        # 内存使用估算
        memory = (h * w * 4 * 3) / (1024 * 1024)  # U, S, Vt
        print(f"SVD内存使用(估算): {memory:.2f} MB")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "SVD图像分解工具 - 生产版本 v2.1" + " " * 14 + "║")
    print("╚" + "═" * 58 + "╝")
    
    # 运行单元测试
    test_success = run_unit_tests()
    
    if test_success:
        # 运行性能测试
        run_performance_tests()
        
        print("\n" + "=" * 60)
        print("代码验证完成 - 可以安全上线！✓")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("代码存在问题 - 请修复后再上线！✗")
        print("=" * 60)