import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from config import *
import matplotlib
import warnings

# ============================================================================
# 中文字体配置
# ============================================================================
# 忽略所有 matplotlib 和 tkinter 字体相关的警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='tkinter')
warnings.filterwarnings('ignore', message='.*Glyph.*missing.*')
warnings.filterwarnings('ignore', message='.*does not have a glyph.*')
warnings.filterwarnings('ignore', message='.*Substituting.*')

# 使用 Agg 后端避免 Tkinter 字形警告
matplotlib.use('Agg')

# 禁用 Unicode 减号，使用 ASCII 减号
plt.rcParams['axes.unicode_minus'] = False
# 使用 DejaVu 字体作为数学文本字体
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

# ============================================================================
# 步骤 4：数字波束形成 (Digital Beamforming)
# ============================================================================

def steering_vector(angle_deg, num_antennas=3, wavelength=0.06):
    """
    生成导向矢量 a(θ)
    
    输入：
        angle_deg: 角度（度）
        num_antennas: 天线数（默认3）
        wavelength: 波长，单位 m（默认0.06m，对应5GHz）
    
    输出：
        a: 导向矢量，形状 (num_antennas, 1)
    """
    angle_rad = np.deg2rad(angle_deg)
    d = wavelength / 2
    phase_step = 2 * np.pi * d * np.sin(angle_rad) / wavelength
    
    a = np.array([
        np.exp(1j * 0),
        np.exp(1j * phase_step),
        np.exp(1j * 0)
    ]).reshape(-1, 1)
    
    return a


def compute_mvdr_weights(R_inv, target_angle, wavelength=0.06):
    """
    计算MVDR最优权重
    
    原理：
    W_opt = R^(-1) * a(θ) / (a(θ)^H * R^(-1) * a(θ))
    
    输入：
        R_inv: 协方差矩阵的逆，形状 (3, 3)
        target_angle: 目标方位角（度）
        wavelength: 波长 (m)
    
    输出：
        W_opt: 最优权重，形状 (3, 1)
    """
    
    print("\n" + "=" * 70)
    print("步骤4.1：计算MVDR最优权重")
    print("=" * 70)
    
    a = steering_vector(target_angle, wavelength=wavelength)
    
    numerator = R_inv @ a
    denominator = (a.conj().T @ R_inv @ a)[0, 0]
    W_opt = numerator / denominator
    
    print(f"\n✓ 目标方位角: {target_angle:.2f}°")
    print(f"  权重向量形状: {W_opt.shape}")
    print(f"  权重幅度: {np.abs(W_opt).flatten()}")
    print(f"  权重相位: {np.angle(W_opt).flatten()}")
    
    return W_opt


def apply_beamforming(target_signal, W_opt):
    """
    应用波束形成
    
    输入：
        target_signal: 目标信号，形状 (frames, rx=3)
        W_opt: 最优权重，形状 (3, 1)
    
    输出：
        beamformed_signal: 波束形成后的信号，形状 (frames,)
    """
    
    print("\n" + "=" * 70)
    print("步骤4.2：应用波束形成")
    print("=" * 70)
    
    print(f"\n✓ 输入信号形状: {target_signal.shape}")
    
    # 波束形成：y(t) = W^H * x(t)
    beamformed_signal = (target_signal @ W_opt).flatten()
    
    print(f"  输出信号形状: {beamformed_signal.shape}")
    print(f"  信号功率: {np.mean(np.abs(beamformed_signal)**2):.6f}")
    
    return beamformed_signal


def compare_single_vs_beamformed(target_signal, beamformed_signal, 
                                 save_path="beamforming_comparison.png"):
    """
    对比单通道和波束形成信号
    
    参数：
        target_signal: 原始目标信号，形状 (frames, rx=3)
        beamformed_signal: 波束形成后的信号，形状 (frames,)
        save_path: 保存路径
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('波束形成效果对比', fontsize=14, fontweight='bold')
    
    frames = min(100, target_signal.shape[0])  # 显示前100帧
    
    # 三个通道的原始信号
    for rx in range(3):
        ax = axes[rx, 0]
        ax.plot(target_signal[:frames, rx], linewidth=1, label=f'RX{rx+1}')
        ax.set_title(f'RX{rx+1} 原始信号（前100帧）', fontsize=12, fontweight='bold')
        ax.set_xlabel('帧索引')
        ax.set_ylabel('幅度')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 波束形成后的信号
    for rx in range(3):
        ax = axes[rx, 1]
        if rx == 0:
            ax.plot(beamformed_signal[:frames], linewidth=1.5, color='green', 
                   label='波束形成输出')
        else:
            ax.plot(target_signal[:frames, rx], linewidth=1, alpha=0.5, 
                   label=f'RX{rx+1} (参考)')
        ax.set_title(f'波束形成输出（前100帧）' if rx == 0 else f'通道对比', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('帧索引')
        ax.set_ylabel('幅度')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    print(f"\n✓ 保存波束形成对比图到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存


def analyze_beamformed_signal(beamformed_signal, save_path="beamformed_signal_analysis.png"):
    """
    分析波束形成后的信号
    
    参数：
        beamformed_signal: 波束形成信号，形状 (frames,)
        save_path: 保存路径
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('波束形成信号分析', fontsize=14, fontweight='bold')
    
    # 时域波形
    ax = axes[0, 0]
    ax.plot(beamformed_signal, linewidth=1)
    ax.set_title('时域波形（所有帧）', fontsize=12, fontweight='bold')
    ax.set_xlabel('帧索引')
    ax.set_ylabel('幅度')
    ax.grid(True, alpha=0.3)
    
    # 功率
    ax = axes[0, 1]
    power = np.abs(beamformed_signal)**2
    ax.plot(power, linewidth=1, color='orange')
    ax.set_title('瞬时功率', fontsize=12, fontweight='bold')
    ax.set_xlabel('帧索引')
    ax.set_ylabel('功率')
    ax.grid(True, alpha=0.3)
    
    # 频域谱
    ax = axes[1, 0]
    fft_result = np.fft.fft(beamformed_signal)
    freqs = np.fft.fftfreq(len(beamformed_signal))
    magnitude = np.abs(fft_result)
    ax.semilogy(freqs[:len(freqs)//2], magnitude[:len(freqs)//2], linewidth=1.5)
    ax.set_title('频域谱（FFT）', fontsize=12, fontweight='bold')
    ax.set_xlabel('归一化频率')
    ax.set_ylabel('幅度')
    ax.grid(True, alpha=0.3, which='both')
    
    # 统计信息
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    信号统计信息:
    ━━━━━━━━━━━━━━━━━━━━━━━
    
    • 帧数: {len(beamformed_signal)}
    • 平均功率: {np.mean(power):.6f}
    • 最大功率: {np.max(power):.6f}
    • 最小功率: {np.min(power):.6f}
    • 标准差: {np.std(power):.6f}
    
    • 信号峰值: {np.max(np.abs(beamformed_signal)):.6f}
    • 信号均值: {np.mean(np.abs(beamformed_signal)):.6f}
    • 信噪比提升: 预期 ~3-6 dB
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
           verticalalignment='center', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    print(f"\n✓ 保存波束形成信号分析图到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存


def visualize_signal_improvement(target_signal, beamformed_signal, 
                                save_path="signal_improvement.png"):
    """
    可视化信号改进效果
    
    参数：
        target_signal: 原始目标信号，形状 (frames, rx=3)
        beamformed_signal: 波束形成信号，形状 (frames,)
        save_path: 保存路径
    """
    
    # 计算每个通道的功率
    power_rx = np.mean(np.abs(target_signal)**2, axis=0)
    
    # 计算波束形成后的功率
    power_beamformed = np.mean(np.abs(beamformed_signal)**2)
    
    # 计算平均功率
    power_avg = np.mean(power_rx)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('波束形成信号改进效果', fontsize=14, fontweight='bold')
    
    # 左图：功率对比
    ax = axes[0]
    channels = ['RX1', 'RX2', 'RX3', '平均', '波束形成']
    powers = list(power_rx) + [power_avg, power_beamformed]
    colors = ['blue', 'blue', 'blue', 'orange', 'green']
    
    bars = ax.bar(channels, powers, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('平均功率', fontsize=12, fontweight='bold')
    ax.set_title('各通道功率对比', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, power in zip(bars, powers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{power:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 右图：增益对比
    ax = axes[1]
    gain_rx = 10 * np.log10(power_beamformed / power_rx + 1e-10)
    
    bars = ax.bar(['RX1', 'RX2', 'RX3'], gain_rx, color='red', alpha=0.7, 
                  edgecolor='black', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('增益 (dB)', fontsize=12, fontweight='bold')
    ax.set_title('波束形成增益', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, gain in zip(bars, gain_rx):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{gain:.2f}dB', ha='center', va='bottom' if gain > 0 else 'top', 
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    print(f"\n✓ 保存信号改进对比图到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("开始处理雷达信号 - 步骤4：数字波束形成")
    print("=" * 70 + "\n")
    
    # 加载数据
    print("加载处理结果...")
    target_signal = np.load(TARGET_SIGNAL_FILE)
    R_inv = np.load(COVARIANCE_MATRIX_FILE)  # 需要在step3中保存
    mvdr_spectrum_data = np.load(MVDR_SPECTRUM_FILE)
    
    print(f"✓ 已加载目标信号: {target_signal.shape}")
    print(f"✓ 已加载协方差矩阵逆: {R_inv.shape}\n")
    
    # 从MVDR谱中获取峰值角度 (这里简化处理，实际应从step3传入)
    angles = np.linspace(-60, 60, len(mvdr_spectrum_data))
    peak_idx = np.argmax(mvdr_spectrum_data)
    peak_angle = angles[peak_idx]
    
    # 步骤4.1-4.2: 计算权重和应用波束形成
    W_opt = compute_mvdr_weights(R_inv, peak_angle)
    beamformed_signal = apply_beamforming(target_signal, W_opt)
    
    # 可视化
    print("\n" + "=" * 70)
    print("可视化波束形成效果")
    print("=" * 70)
    
    compare_single_vs_beamformed(target_signal, beamformed_signal,
                                save_path=get_image_path("beamforming_comparison.png"))
    
    analyze_beamformed_signal(beamformed_signal,
                             save_path=get_image_path("beamformed_signal_analysis.png"))
    
    visualize_signal_improvement(target_signal, beamformed_signal,
                                save_path=get_image_path("signal_improvement.png"))
    
    # 保存结果
    print("\n" + "=" * 70)
    print("保存处理结果")
    print("=" * 70)
    np.save(BEAMFORMING_WEIGHTS_FILE, W_opt)
    np.save(BEAMFORMED_SIGNAL_FILE, beamformed_signal)
    
    print(f"✓ 已保存到 {RESULT_DIR}/")
    print(f"  beamforming_weights.npy")
    print(f"  beamformed_signal.npy")
    
    print(f"\n✓ 步骤4处理完成！")