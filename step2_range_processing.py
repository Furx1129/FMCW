import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from config import *

# ============================================================================
# 中文字体配置
# ============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 步骤 2：距离处理 (Range Processing / Range FFT) - 多目标检测版
# ============================================================================

def range_processing(data_mti_reshaped, threshold_db=-25, min_distance_m=0.3):
    """
    修改版：支持多目标检测
    
    输入：
        data_mti_reshaped: MTI处理后的数据，形状 (frames, samples, rx)
        threshold_db: 峰值检测阈值 (dB)，相对于最大值的下降量
        min_distance_m: 最小峰值间距 (m)，避免旁瓣被当成第二个目标
    
    输出：
        range_spectrum_half: 距离谱正半轴，形状 (samples//2,)
        peaks: 检测到的目标距离索引列表
        range_fft: FFT后的复数数据 (frames, samples, rx)
    """
    
    print("\n" + "=" * 70)
    print("步骤2：多目标距离检测 (Multi-Target Range Processing)")
    print("=" * 70)
    
    frames, num_samples, rx = data_mti_reshaped.shape
    print(f"  输入数据形状: {data_mti_reshaped.shape}")
    print(f"  采样点数 (Fast Time): {num_samples}")
    print(f"  接收通道数: {rx}")
    
    # ========== 步骤2.1：距离压缩 (Range FFT) ==========
    print(f"\n✓ 对采样点维度(Fast Time)做FFT...")
    range_fft = np.fft.fft(data_mti_reshaped, axis=1)
    
    # 取模得到幅度谱
    range_magnitude = np.abs(range_fft)
    
    print(f"  FFT结果形状: {range_fft.shape}")
    print(f"  幅度谱形状: {range_magnitude.shape}")
    
    # ========== 步骤2.2：确定距离门 ==========
    print(f"\n✓ 合成多天线距离谱...")
    
    # 沿着接收天线维度求和（将3个通道合并）
    range_spectrum_all_frames = np.sum(range_magnitude, axis=2)
    
    # 对所有帧取平均，得到总体距离谱
    range_spectrum = np.mean(range_spectrum_all_frames, axis=0)
    
    # 只取正半轴
    half_samples = num_samples // 2
    range_spectrum_half = range_spectrum[:half_samples]
    
    print(f"  多天线合成距离谱形状: {range_spectrum_half.shape}")
    
    # ========== 步骤2.3：多目标峰值检测 ==========
    print(f"\n✓ 寻找多目标峰值...")
    
    # 计算距离分辨率
    # 距离分辨率 = c / (2 * B)，其中 B = 5GHz
    distance_resolution = 3e8 / (2 * 5e9)  # 约 0.03m
    print(f"  距离分辨率: {distance_resolution*100:.2f} cm")
    
    # 动态阈值：相对于最大值下降 X dB
    max_val = np.max(range_spectrum_half)
    threshold = max_val * (10 ** (threshold_db / 20))
    
    print(f"  阈值 ({threshold_db} dB): {threshold:.6f}")
    
    # 最小峰值间距 (避免同一个人的旁瓣被当成第二个人)
    # 单位从 m 转换为点数
    distance_indices = int(min_distance_m / distance_resolution)
    if distance_indices < 1:
        distance_indices = 1
    
    print(f"  最小峰值间距: {min_distance_m:.2f} m = {distance_indices} 个点")
    
    # 调用 find_peaks 检测峰值
    peaks, properties = find_peaks(range_spectrum_half, height=threshold, distance=distance_indices)
    
    # 如果没找到，退化为找最大值
    if len(peaks) == 0:
        print("  ⚠️ 未检测到显著峰值，回退到最大值模式")
        peaks = np.array([np.argmax(range_spectrum_half)])
    
    print(f"\n✓ 检测到 {len(peaks)} 个目标")
    for i, peak_idx in enumerate(peaks):
        physical_distance = peak_idx * distance_resolution
        print(f"  目标 {i+1}: 索引={peak_idx}, 距离={physical_distance:.3f} m, 功率={range_spectrum_half[peak_idx]:.6f}")
    
    return range_spectrum_half, peaks, range_fft


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


def extract_target_signal(range_fft, range_idx):
    """
    修改版：提取特定距离索引处的信号
    
    输入：
        range_fft: FFT后的复数数据，形状 (frames, samples, rx)
        range_idx: 单个整数，目标的距离索引
    
    输出：
        target_signal: 目标复数信号 (frames, rx)
    """
    
    # 直接切片提取该距离点的数据
    # range_fft 维度: (frames, samples, rx)
    # 结果：取所有帧、只有 range_idx 这个点、所有天线
    target_signal = range_fft[:, range_idx, :]
    
    print(f"\n✓ 在距离索引 {range_idx} 处提取目标信号...")
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
    print("开始处理雷达信号 - 步骤2：多目标距离处理")
    print("=" * 70 + "\n")
    
    # 加载步骤1的处理结果
    print("加载步骤1的处理结果...")
    data_mti_reshaped = np.load(MTI_RESHAPED_FILE)
    print(f"✓ 已加载 MTI 处理数据: {data_mti_reshaped.shape}\n")
    
    # 步骤2：距离处理（多目标检测版）
    range_spectrum, peaks, range_fft = range_processing(data_mti_reshaped, 
                                                        threshold_db=-25, 
                                                        min_distance_m=0.3)
    
    # 可视化距离谱
    print("\n" + "=" * 70)
    print("可视化距离谱")
    print("=" * 70)
    # 这里暂时使用第一个峰值来可视化
    if len(peaks) > 0:
        visualize_range_spectrum(range_spectrum, peaks[0], 
                                save_path=get_image_path("range_spectrum.png"))
    
    # 提取所有目标信号
    print("\n" + "=" * 70)
    print("提取目标信号")
    print("=" * 70)
    
    # 对于当前版本，处理第一个目标（后续可扩展为处理多个）
    if len(peaks) > 0:
        primary_target_idx = peaks[0]
        print(f"\n  处理主要目标 (距离索引={primary_target_idx})")
        target_signal = extract_target_signal(range_fft, primary_target_idx)
        
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
        
        print("\n" + "=" * 70)
        print("✅ 步骤2处理完成！")
        print("=" * 70)
        print(f"\n📊 检测结果:")
        print(f"  检测到 {len(peaks)} 个目标")
        print(f"  处理主要目标 (距离={primary_target_idx * 0.03:.3f}m)")
        print(f"\n✅ 下一步：高阶角度估计（步骤3）")
    else:
        print("\n❌ 未检测到任何目标，请检查数据质量！")
    print(f"  target_signal.npy")
    print(f"  range_spectrum.npy")
    print(f"  range_fft.npy")
    
    print("\n✓ 步骤2处理完成！")