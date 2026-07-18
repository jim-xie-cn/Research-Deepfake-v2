import numpy as np
from typing import Union, Tuple, List, Optional, Literal, Dict, Any
import warnings
from ImageConstruct import CImageUtils, CImageAlpha, CImageMFS, CImageFD
from ImageDecompose import CImageDecompose


class EnhancedSVDImageDecompose(CImageDecompose):
    """增强版SVD分解，专门用于分析细节分量"""
    
    def get_component_range(self, start=2, end=10, mode='individual'):
        """
        获取特定范围的分量
        
        Args:
            start: 起始分量（1-based）
            end: 结束分量（包含）
            mode: 'individual' - 单独的分量
                  'cumulative' - 累积分量
                  'reconstruction' - 使用这些分量的重建
                  'residual' - 原图减去指定范围的重建（剩余分量）
                  'residual_cumulative' - 累积残差序列
                  'residual_individual' - 每个分量对应的残差
        """
        if self.is_color:
            results = []
            for c in range(self.n_channels):
                channel_result = self._get_component_range_single(
                    self.m_img[..., c], start, end, mode
                )
                results.append(channel_result)
            
            # 合并通道
            merged = []
            for i in range(len(results[0])):
                frame = np.stack([results[c][i] for c in range(self.n_channels)], axis=2)
                merged.append(frame.astype(np.float32))
            return merged
        else:
            return self._get_component_range_single(self.m_img, start, end, mode)
    
    def _get_component_range_single(self, channel, start, end, mode):
        """单通道的分量提取"""
        U, S, Vt = self._get_svd(channel)
        
        results = []
        
        if mode == 'individual':
            # 返回单独的每个分量
            for i in range(start-1, min(end, len(S))):
                component = S[i] * np.outer(U[:, i], Vt[i, :])
                results.append(component.astype(np.float32))
        
        elif mode == 'cumulative':
            # 返回累积的分量
            cumsum = np.zeros_like(channel, dtype=np.float32)
            for i in range(start-1, min(end, len(S))):
                component = S[i] * np.outer(U[:, i], Vt[i, :])
                cumsum += component
                results.append(cumsum.copy())
        
        elif mode == 'reconstruction':
            # 只使用指定范围重建
            reconstruction = np.zeros_like(channel, dtype=np.float32)
            for i in range(start-1, min(end, len(S))):
                reconstruction += S[i] * np.outer(U[:, i], Vt[i, :])
            results.append(reconstruction)
        
        elif mode == 'residual':
            # 残差：原图减去指定范围的重建
            # 等价于：使用范围外的所有分量重建
            
            # 方法1：计算指定范围的重建，然后从原图减去
            reconstruction_range = np.zeros_like(channel, dtype=np.float32)
            for i in range(start-1, min(end, len(S))):
                reconstruction_range += S[i] * np.outer(U[:, i], Vt[i, :])
            residual = channel - reconstruction_range
            results.append(residual.astype(np.float32))
            
        elif mode == 'residual_cumulative':
            # 累积残差序列：逐步减去更多分量
            cumulative_reconstruction = np.zeros_like(channel, dtype=np.float32)
            for i in range(start-1, min(end, len(S))):
                component = S[i] * np.outer(U[:, i], Vt[i, :])
                cumulative_reconstruction += component
                residual = channel - cumulative_reconstruction
                results.append(residual.astype(np.float32))
        
        elif mode == 'residual_individual':
            # 每个分量对应的残差：原图减去单个分量
            for i in range(start-1, min(end, len(S))):
                # 重建除了第i个分量外的所有分量
                reconstruction_without_i = np.zeros_like(channel, dtype=np.float32)
                
                # 添加第i个分量之前的所有分量
                for j in range(min(i, len(S))):
                    if j != i:  # 跳过第i个
                        reconstruction_without_i += S[j] * np.outer(U[:, j], Vt[j, :])
                
                # 添加第i个分量之后的所有分量
                for j in range(i+1, len(S)):
                    reconstruction_without_i += S[j] * np.outer(U[:, j], Vt[j, :])
                
                residual = channel - reconstruction_without_i
                results.append(residual.astype(np.float32))
        
        elif mode == 'residual_progressive':
            # 渐进残差：从完整重建逐步移除分量
            # 首先构建完整重建
            full_reconstruction = (U * S) @ Vt
            
            # 逐步移除指定范围的分量
            current_reconstruction = full_reconstruction.copy()
            for i in range(start-1, min(end, len(S))):
                # 移除第i个分量
                component = S[i] * np.outer(U[:, i], Vt[i, :])
                current_reconstruction -= component
                results.append(current_reconstruction.copy().astype(np.float32))
        
        else:
            raise ValueError(f"不支持的mode: {mode}")
        
        return results
    
    def analyze_components(self, component_range=(2, 10), include_residual=True):
        """全面分析指定范围的分量"""
        start, end = component_range
        
        # 获取不同模式的结果
        individual = self.get_component_range(start, end, 'individual')
        cumulative = self.get_component_range(start, end, 'cumulative')
        reconstruction = self.get_component_range(start, end, 'reconstruction')
        
        result = {
            'individual': individual,
            'cumulative': cumulative,
            'reconstruction': reconstruction,
        }
        
        # 添加残差分析
        if include_residual:
            residual = self.get_component_range(start, end, 'residual')
            residual_cumulative = self.get_component_range(start, end, 'residual_cumulative')
            residual_progressive = self.get_component_range(start, end, 'residual_progressive')
            
            result.update({
                'residual': residual,
                'residual_cumulative': residual_cumulative,
                'residual_progressive': residual_progressive
            })
        
        # 计算统计信息
        stats = []
        for i, comp in enumerate(individual):
            stat = {
                'index': start + i,
                'mean': np.mean(comp),
                'std': np.std(comp),
                'min': np.min(comp),
                'max': np.max(comp),
                'energy': np.sum(comp ** 2),
                'norm': np.linalg.norm(comp)
            }
            
            # 如果有残差，添加残差统计
            if include_residual and i < len(result['residual_cumulative']):
                residual_comp = result['residual_cumulative'][i]
                stat['residual_energy'] = np.sum(residual_comp ** 2)
                stat['residual_std'] = np.std(residual_comp)
            
            stats.append(stat)
        
        result['stats'] = stats
        return result
    
    def compare_with_original(self, component_range=(2, 10)):
        """比较指定范围分量与原图的关系"""
        start, end = component_range
        
        # 获取原图
        if self.is_color:
            original = self.m_img
        else:
            original = self.m_img
        
        # 获取各种重建和残差
        reconstruction = self.get_component_range(start, end, 'reconstruction')[0]
        residual = self.get_component_range(start, end, 'residual')[0]
        
        # 计算误差指标
        mse_reconstruction = np.mean((reconstruction) ** 2)
        mse_residual = np.mean((residual) ** 2)
        
        # 能量分析
        total_energy = np.sum(original ** 2)
        reconstruction_energy = np.sum(reconstruction ** 2)
        residual_energy = np.sum(residual ** 2)
        
        metrics = {
            'reconstruction_mse': mse_reconstruction,
            'residual_mse': mse_residual,
            'total_energy': total_energy,
            'reconstruction_energy': reconstruction_energy,
            'residual_energy': residual_energy,
            'reconstruction_energy_ratio': reconstruction_energy / total_energy,
            'residual_energy_ratio': residual_energy / total_energy
        }
        
        return metrics


def test_decompose_enhanced(img_real, img_fake, n_start, n_end, show_residual=True):
    """增强的测试函数，包含残差分析"""
    
    enhanced_decomposer_real = EnhancedSVDImageDecompose(img_real)
    enhanced_decomposer_fake = EnhancedSVDImageDecompose(img_fake)
    
    # 完整分析
    analysis_real = enhanced_decomposer_real.analyze_components(
        (n_start, n_end), include_residual=True
    )
    analysis_fake = enhanced_decomposer_fake.analyze_components(
        (n_start, n_end), include_residual=True
    )
    
    # 可视化结果
    print(f"\n{'='*60}")
    print(f"SVD分量分析: 第{n_start}-{n_end}个分量")
    print('='*60)
    
    # 1. 单独分量
    print(f"\nReal - 第{n_start}-{n_end}个分量（单独）:")
    CImageUtils.display(analysis_real['individual'], auto_normalize=True)
    
    print(f"\nFake - 第{n_start}-{n_end}个分量（单独）:")
    CImageUtils.display(analysis_fake['individual'], auto_normalize=True)
    
    # 2. 累积分量
    print(f"\nReal - 第{n_start}-{n_end}个分量（累积）:")
    CImageUtils.display(analysis_real['cumulative'], auto_normalize=True)
    
    print(f"\nFake - 第{n_start}-{n_end}个分量（累积）:")
    CImageUtils.display(analysis_fake['cumulative'], auto_normalize=True)
    
    # 3. 残差分析
    if show_residual:
        print(f"\n{'='*60}")
        print("残差分析")
        print('='*60)
        
        # 累积残差
        print(f"\nReal - 累积残差（原图-累积重建）:")
        CImageUtils.display(analysis_real['residual_cumulative'], auto_normalize=True)
        
        print(f"\nFake - 累积残差（原图-累积重建）:")
        CImageUtils.display(analysis_fake['residual_cumulative'], auto_normalize=True)
        
        # 渐进残差
        print(f"\nReal - 渐进残差（逐步移除分量）:")
        CImageUtils.display(analysis_real['residual_progressive'], auto_normalize=True)
        
        print(f"\nFake - 渐进残差（逐步移除分量）:")
        CImageUtils.display(analysis_fake['residual_progressive'], auto_normalize=True)
    
    # 4. 统计信息对比
    print(f"\n{'='*60}")
    print("分量统计信息对比")
    print('='*60)
    
    for i in range(len(analysis_real['stats'])):
        stat_r = analysis_real['stats'][i]
        stat_f = analysis_fake['stats'][i]
        print(f"\n分量 {stat_r['index']}:")
        print(f"  Real:")
        print(f"    能量: {stat_r['energy']:.4f}")
        print(f"    标准差: {stat_r['std']:.6f}")
        print(f"    范数: {stat_r['norm']:.4f}")
        if 'residual_energy' in stat_r:
            print(f"    残差能量: {stat_r['residual_energy']:.4f}")
        
        print(f"  Fake:")
        print(f"    能量: {stat_f['energy']:.4f}")
        print(f"    标准差: {stat_f['std']:.6f}")
        print(f"    范数: {stat_f['norm']:.4f}")
        if 'residual_energy' in stat_f:
            print(f"    残差能量: {stat_f['residual_energy']:.4f}")
        
        # 计算差异
        energy_diff = abs(stat_r['energy'] - stat_f['energy'])
        std_diff = abs(stat_r['std'] - stat_f['std'])
        print(f"  差异:")
        print(f"    能量差: {energy_diff:.6f}")
        print(f"    标准差差: {std_diff:.6f}")
    
    # 5. 与原图对比
    print(f"\n{'='*60}")
    print("与原图对比分析")
    print('='*60)
    
    metrics_real = enhanced_decomposer_real.compare_with_original((n_start, n_end))
    metrics_fake = enhanced_decomposer_fake.compare_with_original((n_start, n_end))
    
    print(f"\nReal图像:")
    print(f"  第{n_start}-{n_end}分量能量占比: {metrics_real['reconstruction_energy_ratio']*100:.4f}%")
    print(f"  剩余分量能量占比: {metrics_real['residual_energy_ratio']*100:.4f}%")
    
    print(f"\nFake图像:")
    print(f"  第{n_start}-{n_end}分量能量占比: {metrics_fake['reconstruction_energy_ratio']*100:.4f}%")
    print(f"  剩余分量能量占比: {metrics_fake['residual_energy_ratio']*100:.4f}%")
    
    return analysis_real, analysis_fake


def visualize_residual_difference(img_real, img_fake, component_range=(2, 10)):
    """可视化残差差异"""
    import matplotlib.pyplot as plt
    
    decomposer_real = EnhancedSVDImageDecompose(img_real)
    decomposer_fake = EnhancedSVDImageDecompose(img_fake)
    
    # 获取残差
    residual_cum_real = decomposer_real.get_component_range(
        component_range[0], component_range[1], 'residual_cumulative'
    )
    residual_cum_fake = decomposer_fake.get_component_range(
        component_range[0], component_range[1], 'residual_cumulative'
    )
    
    # 可视化
    n_frames = len(residual_cum_real)
    fig, axes = plt.subplots(3, n_frames, figsize=(n_frames*2, 6))
    
    for i in range(n_frames):
        # Real残差
        res_real = residual_cum_real[i]
        if len(res_real.shape) == 3:
            res_real = np.mean(res_real, axis=2)
        
        vmax = np.max(np.abs(res_real))
        axes[0, i].imshow(res_real, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[0, i].set_title(f'Real (k={component_range[0]+i})', fontsize=10)
        axes[0, i].axis('off')
        
        # Fake残差
        res_fake = residual_cum_fake[i]
        if len(res_fake.shape) == 3:
            res_fake = np.mean(res_fake, axis=2)
        
        vmax = np.max(np.abs(res_fake))
        axes[1, i].imshow(res_fake, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1, i].set_title(f'Fake (k={component_range[0]+i})', fontsize=10)
        axes[1, i].axis('off')
        
        # 差异
        diff = res_real - res_fake
        vmax = np.max(np.abs(diff))
        axes[2, i].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        # 计算差异统计
        mse = np.mean(diff ** 2)
        axes[2, i].set_title(f'Diff (MSE={mse:.2e})', fontsize=10)
        axes[2, i].axis('off')
    
    plt.suptitle(f'残差对比: 移除第{component_range[0]}-{component_range[1]}个分量后', fontsize=14)
    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 假设您有real_mean和fake_mean图像
    #test_decompose_enhanced(real_mean, fake_mean, 2, 10, show_residual=True)
    #visualize_residual_difference(real_mean, fake_mean, (2, 10))
    pass
