import numpy as np
import matplotlib.pyplot as plt
from config import *

# ============================================================================
# 中文字体配置
# ============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 步骤 2：距离处理 (Range Processing / Range FFT)
# ============================================================================

def range_processing(data_mti_reshaped):
    """
    距离处理：对采样点维度做FFT，找到目标距离
    
    输入：
        data_mti_reshaped: MTI处理后的数据，形状 (frames, samples, rx)
    
    输出：
        range_spectrum: 距离谱，形状 (samples,) - 3个天线合成
        range_idx: 目标所在的距离索引
        range_peak_power: 峰值功率
        range_fft: FFT后的复数数据
    """
    
    print("\n" + "=" * 70)
    print("步骤2：距离处理 (Range FFT)")
    print("=" * 70)
    
    frames, num_samples, rx = data_mti_reshaped.shape
    print(f"  输入数据形状: {data_mti_reshaped.shape}")
    print(f"  采样点数 (Fast Time): {num_samples}")
    print(f"  接收通道数: {rx}")
    
    # ========== 步骤2.1：距离压缩 (Range FFT) ==========
    # 对每一帧的采样点维度做FFT
    # 输入: (frames, samples, rx)
    # 输出: (frames, samples, rx) - 复数域
    
    print(f"\n✓ 对采样点维度(Fast Time)做FFT...")
    range_fft = np.fft.fft(data_mti_reshaped, axis=1)  # 沿axis=1(采样点)做FFT
    
    # 取模得到幅度谱
    range_magnitude = np.abs(range_fft)  # (frames, samples, rx)
    
    print(f"  FFT结果形状: {range_fft.shape}")
    print(f"  幅度谱形状: {range_magnitude.shape}")
    
    # ========== 步骤2.2：确定距离门 (Range Bin Selection) ==========
    # 将3个天线的结果相加，得到综合距离谱
    # 逻辑：多个天线的信号叠加，能增强目标反射，抑制噪声
    
    print(f"\n✓ 合成多天线距离谱...")
    
    # 沿着接收天线维度求和（将3个通道合并）
    # 结果: (frames, samples)
    range_spectrum_all_frames = np.sum(range_magnitude, axis=2)
    
    # 对所有帧取平均，得到总体距离谱（更稳定）
    # 结果: (samples,)
    range_spectrum = np.mean(range_spectrum_all_frames, axis=0)
    
    print(f"  多天线合成距离谱形状: {range_spectrum.shape}")
    
    # ========== 步骤2.3：峰值检测，确定目标距离 ==========
    # 找到距离谱中能量最强的地方
    
    print(f"\n✓ 检测目标距离...")
    
    # 只考虑正频率部分（0 到 N/2）
    range_spectrum_positive = range_spectrum[:num_samples//2]
    
    # 找峰值
    range_idx = np.argmax(range_spectrum_positive)
    range_peak_power = range_spectrum_positive[range_idx]
    
    print(f"  目标距离索引 (Range Bin): {range_idx}")
    print(f"  峰值功率: {range_peak_power:.6f}")
    
    # 根据参数计算物理距离
    # 距离分辨率 = c / (2 * B)
    # 其中 c = 3e8 m/s, B = 5GHz
    # 距离分辨率 ≈ 0.03m
    distance_resolution = 3e8 / (2 * 5e9)  # 约 0.03m
    physical_distance = range_idx * distance_resolution
    
    print(f"  距离分辨率: {distance_resolution*100:.2f} cm")
    print(f"  目标物理距离: {physical_distance:.3f} m")
    
    return range_spectrum, range_idx, range_peak_power, range_fft


def visualize_range_spectrum(range_spectrum, range_idx, save_path="range_spectrum.png"):
    """
    可视化距离谱
    
    参数：
        range_spectrum: 距离谱数据
        range_idx: 目标距离索引
        save_path: 保存图片路径
    """
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 绘制距离谱
    ax.plot(range_spectrum[:len(range_spectrum)//2], 'b-', linewidth=2, label='距离谱')
    
    # 标记峰值
    ax.plot(range_idx, range_spectrum[range_idx], 'ro', markersize=10, label=f'目标位置 (Index={range_idx})')
    ax.axvline(range_idx, color='r', linestyle='--', alpha=0.5)
    
    ax.set_title('多天线合成距离谱', fontsize=14, fontweight='bold')
    ax.set_xlabel('距离索引 (Range Bin)', fontsize=12)
    ax.set_ylabel('幅度', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    print(f"✓ 保存距离谱图到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def extract_target_signal(data_mti_reshaped, range_fft, range_idx):
    """
    在锁定的距离处提取目标信号
    
    输入：
        data_mti_reshaped: 原始MTI处理数据 (frames, samples, rx)
        range_fft: FFT后的复数数据 (frames, samples, rx)
        range_idx: 目标所在的距离索引
    
    输出：
        target_signal: 目标复数信号 (frames, rx)
            - 形状: 每帧在目标距离处的3个天线接收信号
    """
    
    print(f"\n✓ 在距离索引 {range_idx} 处提取目标信号...")
    
    # 在频域内，取出目标距离处的复数数据
    # range_fft[:, range_idx, :] 表示所有帧在该距离的信号
    # 形状: (frames, rx) - 这就是我们关心的目标回波
    
    target_signal = range_fft[:, range_idx, :]  # (frames, rx)
    
    print(f"  提取的目标信号形状: {target_signal.shape}")
    print(f"  = (帧数={target_signal.shape[0]}, 天线数={target_signal.shape[1]})")
    
    # 统计目标信号的信息
    target_power = np.mean(np.abs(target_signal)**2)
    print(f"  目标信号功率: {target_power:.6f}")
    
    return target_signal


def visualize_target_signal(target_signal, save_path="target_signal.png"):
    """
    可视化目标信号的幅度和相位
    
    参数：
        target_signal: 目标复数信号 (frames, rx)
        save_path: 保存图片路径
    """
    
    frames = target_signal.shape[0]
    time_axis = np.arange(frames)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('目标复数信号分析 (3个接收天线)', fontsize=14, fontweight='bold')
    
    for ch in range(3):
        # 提取幅度和相位
        amplitude = np.abs(target_signal[:, ch])
        phase = np.angle(target_signal[:, ch])
        
        # 幅度子图
        ax = axes[ch, 0]
        ax.plot(time_axis, amplitude, 'b-', linewidth=1.5)
        ax.set_title(f'RX{ch+1} - 幅度', fontsize=12)
        ax.set_xlabel('帧索引', fontsize=11)
        ax.set_ylabel('幅度', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 相位子图
        ax = axes[ch, 1]
        ax.plot(time_axis, phase, 'r-', linewidth=1.5)
        ax.set_title(f'RX{ch+1} - 相位', fontsize=12)
        ax.set_xlabel('帧索引', fontsize=11)
        ax.set_ylabel('相位 (rad)', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print(f"✓ 保存目标信号分析图到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("开始处理雷达信号 - 步骤2：距离处理")
    print("=" * 70 + "\n")
    
    # 加载步骤1的处理结果
    print("加载步骤1的处理结果...")
    data_mti_reshaped = np.load(MTI_RESHAPED_FILE)  # 使用配置文件中的路径
    print(f"✓ 已加载 MTI 处理数据: {data_mti_reshaped.shape}\n")
    
    # 步骤2：距离处理
    range_spectrum, range_idx, range_peak_power, range_fft = range_processing(data_mti_reshaped)
    
    # 可视化距离谱
    print("\n" + "=" * 70)
    print("可视化距离谱")
    print("=" * 70)
    visualize_range_spectrum(range_spectrum, range_idx, 
                            save_path=get_image_path("range_spectrum.png"))
    
    # 提取目标信号
    print("\n" + "=" * 70)
    print("提取目标信号")
    print("=" * 70)
    target_signal = extract_target_signal(data_mti_reshaped, range_fft, range_idx)
    
    # 可视化目标信号
    print("\n" + "=" * 70)
    print("可视化目标信号")
    print("=" * 70)
    visualize_target_signal(target_signal, 
                           save_path=get_image_path("target_signal.png"))
    
    # 保存结果
    print("\n" + "=" * 70)
    print("保存处理结果")
    print("=" * 70)
    np.save(TARGET_SIGNAL_FILE, target_signal)
    np.save(RANGE_SPECTRUM_FILE, range_spectrum)
    np.save(RANGE_FFT_FILE, range_fft)
    
    print(f"✓ 已保存到 {RESULT_DIR}/")
    print(f"  target_signal.npy")
    print(f"  range_spectrum.npy")
    print(f"  range_fft.npy")
    
    print("\n✓ 步骤2处理完成！")