import numpy as np
import warnings
import matplotlib

# 必须在导入 pyplot 之前设置后端和过滤警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*glyph.*')
warnings.filterwarnings('ignore', message='.*font.*')
matplotlib.use('Agg')  # 使用非交互式后端避免 Tkinter 字形问题

import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
from config import *

# ============================================================================
# 中文字体配置
# ============================================================================
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.default'] = 'regular'

# ============================================================================
# 步骤 5：生命体征提取 (Vital Sign Extraction)
# 关键：相位处理的正确顺序
#    1. 提取相位 (extract_phase) → 包裹相位 (-π, π]
#    2. 相位解缠 (unwrap) → 连续相位（移除 2π 跳跃）
#    3. 相位差分 (differentiation) → 相位变化率
# ============================================================================

def extract_phase(beamformed_signal):
    """
    从复信号中提取包裹相位
    
    输入：
        beamformed_signal: 波束形成信号，形状 (frames,) 复数
    
    输出：
        phase_wrapped: 包裹相位，范围 (-π, π]
        
    说明：
        • np.angle() 返回的是包裹在 (-π, π] 范围内的相位
        • 会在相位跨越 ±π 时产生 2π 的跳跃
        • 这种跳跃会被下一步的 unwrap 移除
    """
    
    print("\n" + "=" * 70)
    print("步骤5.1：相位提取")
    print("=" * 70)
    
    phase_wrapped = np.angle(beamformed_signal)
    
    print(f"\n✓ 相位提取完成")
    print(f"  信号长度: {len(beamformed_signal)}")
    print(f"  包裹相位范围: [{np.min(phase_wrapped):.4f}, {np.max(phase_wrapped):.4f}] rad")
    print(f"  包裹相位范围: [{np.min(phase_wrapped)*180/np.pi:.2f}, {np.max(phase_wrapped)*180/np.pi:.2f}]°")
    
    # 统计相位跳跃次数（诊断用）
    phase_jumps = np.abs(np.diff(phase_wrapped))
    jump_threshold = np.pi  # 2π 跳跃会导致差分 > π
    num_jumps = np.sum(phase_jumps > jump_threshold)
    print(f"\n⚠️ 诊断信息:")
    print(f"  相位包裹次数 (2π 跳跃): {num_jumps}")
    print(f"  预期: 正常信号应有若干次跳跃")
    
    return phase_wrapped


def unwrap_phase(phase_wrapped, discontinuity=np.pi):
    """
    相位解缠（移除 2π 的循环跳跃）
    
    输入：
        phase_wrapped: 包裹相位，范围 (-π, π]
        discontinuity: 检测跳跃的阈值 (默认 π，对应 2π 跳跃)
    
    输出：
        phase_unwrapped: 解缠后的连续相位
        
    物理原理：
        信号相位实际上是连续变化的，但 np.angle() 返回的值被限制在 (-π, π]。
        当真实相位跨越 π 时，会产生一个 -2π 的跳跃（从接近 π 跳到接近 -π）。
        unwrap 的作用就是检测这些跳跃并修复，使相位曲线连续。
        
        例：-2.5, -0.5 (跳跃), 2.5 → -2.5, 2.5, 2.5 (修复后)
    """
    
    print("\n" + "=" * 70)
    print("步骤5.2：相位解缠")
    print("=" * 70)
    print(f"  跳跃检测阈值: {discontinuity:.4f} rad")
    
    # np.unwrap 会自动检测并修复 2π 的跳跃
    phase_unwrapped = np.unwrap(phase_wrapped, discont=discontinuity)
    
    print(f"\n✓ 相位解缠完成")
    print(f"  解缠前范围: [{np.min(phase_wrapped):.4f}, {np.max(phase_wrapped):.4f}] rad")
    print(f"  解缠后范围: [{np.min(phase_unwrapped):.4f}, {np.max(phase_unwrapped):.4f}] rad")
    print(f"  总变化: {np.max(phase_unwrapped) - np.min(phase_unwrapped):.4f} rad")
    print(f"            ({(np.max(phase_unwrapped) - np.min(phase_unwrapped))*180/np.pi:.2f}°)")
    
    # 验证：解缠后的相位应该是单调或分段单调的
    phase_diff_verify = np.diff(phase_unwrapped)
    large_jumps = np.sum(np.abs(phase_diff_verify) > np.pi)
    if large_jumps == 0:
        print(f"  ✓ 验证通过: 没有检测到大的相位跳跃（解缠成功）")
    else:
        print(f"  ⚠️ 警告: 仍检测到 {large_jumps} 个大的相位跳跃（可能需要调整阈值）")
    
    return phase_unwrapped


def phase_differentiation(phase_unwrapped):
    """
    对解缠相位求导得到相位变化率
    
    重要：必须在 unwrap 之后调用！
    
    输入：
        phase_unwrapped: 解缠后的连续相位，形状 (frames,)
    
    输出：
        phase_diff: 相位差分，形状 (frames-1,)
        
    说明：
        • 直接对 wrap 的相位求差会产生巨大的尖峰噪声
        • 必须先 unwrap 再 diff，才能得到连续的相位变化率
        • phase_diff 与人体呼吸/心跳产生的相位调制成正比
    """
    
    print("\n" + "=" * 70)
    print("步骤5.3：相位微分")
    print("=" * 70)
    
    print(f"\n✓ 计算相位变化率: phase_diff = diff(phase_unwrapped)")
    phase_diff = np.diff(phase_unwrapped)
    
    print(f"\n✓ 相位微分完成")
    print(f"  差分长度: {len(phase_diff)}")
    print(f"  差分范围: [{np.min(phase_diff):.6f}, {np.max(phase_diff):.6f}] rad/frame")
    print(f"  差分均值: {np.mean(phase_diff):.6f} rad/frame")
    print(f"  差分标准差: {np.std(phase_diff):.6f} rad/frame")
    
    # 检查是否有异常尖峰（如果有，说明 unwrap 可能失败）
    threshold = 3 * np.std(phase_diff)
    num_spikes = np.sum(np.abs(phase_diff) > threshold)
    if num_spikes > 0:
        print(f"\n 警告: 检测到 {num_spikes} 个异常尖峰")
        print(f"           这可能表示 unwrap 不完全或数据质量问题")
    else:
        print(f"\n  ✓ 无异常尖峰，差分质量良好")
    
    return phase_diff


def design_bandpass_filter(lowcut, highcut, fs, order=5):
    """
    设计巴特沃斯带通滤波器
    
    参数：
        lowcut: 低截止频率 (Hz)
        highcut: 高截止频率 (Hz)
        fs: 采样率 (Hz)
        order: 滤波器阶数
    
    输出：
        b, a: 滤波器系数
    """
    
    print(f"\n✓ 设计带通滤波器")
    print(f"  频率范围: [{lowcut:.2f}, {highcut:.2f}] Hz")
    print(f"  采样率: {fs:.2f} Hz")
    print(f"  阶数: {order}")
    
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    
    return b, a


def extract_respiration_and_heartbeat(phase_diff, frame_rate=10.0):
    """
    分离呼吸和心跳信号
    
    呼吸频率: 12-20 次/分钟 = 0.2-0.33 Hz
    心跳频率: 60-100 次/分钟 = 1-1.67 Hz
    
    参数：
        phase_diff: 相位差分，形状 (frames,)
        frame_rate: 帧率 (Hz)
    
    输出：
        respiration: 呼吸信号
        heartbeat: 心跳信号
    """
    
    print("\n" + "=" * 70)
    print("步骤5.3：呼吸和心跳分离")
    print("=" * 70)
    
    fs = frame_rate
    
    # 设计两个带通滤波器
    print(f"\n✓ 呼吸信号提取 (0.2-0.5 Hz)...")
    b_resp, a_resp = design_bandpass_filter(0.2, 0.5, fs, order=4)
    respiration = filtfilt(b_resp, a_resp, phase_diff)
    
    print(f"\n✓ 心跳信号提取 (0.8-2.5 Hz)...")
    b_hr, a_hr = design_bandpass_filter(0.8, 2.5, fs, order=4)
    heartbeat = filtfilt(b_hr, a_hr, phase_diff)
    
    print(f"\n✓ 信号分离完成")
    print(f"  呼吸信号范围: [{np.min(respiration):.6f}, {np.max(respiration):.6f}]")
    print(f"  心跳信号范围: [{np.min(heartbeat):.6f}, {np.max(heartbeat):.6f}]")
    
    return respiration, heartbeat


def extract_vital_signs(respiration, heartbeat, frame_rate=10.0):
    """
    从信号中提取呼吸率和心率
    
    参数：
        respiration: 呼吸信号
        heartbeat: 心跳信号
        frame_rate: 帧率 (Hz)
    
    输出：
        breathing_rate: 呼吸率 (次/分钟)
        heart_rate: 心率 (次/分钟)
    """
    
    print("\n" + "=" * 70)
    print("步骤5.4：生命体征估计")
    print("=" * 70)
    
    # 找到峰值
    peaks_resp, _ = find_peaks(respiration, distance=frame_rate*2)  # 最小间隔2秒
    peaks_hr, _ = find_peaks(heartbeat, distance=frame_rate*0.4)    # 最小间隔0.4秒
    
    print(f"\n✓ 检测到 {len(peaks_resp)} 个呼吸周期")
    print(f"✓ 检测到 {len(peaks_hr)} 个心跳")
    
    # 计算频率
    if len(peaks_resp) > 1:
        respiration_intervals = np.diff(peaks_resp) / frame_rate  # 秒
        breathing_rate = 60.0 / np.mean(respiration_intervals)
    else:
        breathing_rate = 0
    
    if len(peaks_hr) > 1:
        hr_intervals = np.diff(peaks_hr) / frame_rate  # 秒
        heart_rate = 60.0 / np.mean(hr_intervals)
    else:
        heart_rate = 0
    
    print(f"\n✓ 生命体征提取完成:")
    print(f"  呼吸率: {breathing_rate:.1f} 次/分钟")
    print(f"  心率: {heart_rate:.1f} 次/分钟")
    
    return breathing_rate, heart_rate


def visualize_phase_extraction(phase_wrapped, phase_unwrapped, phase_diff, 
                              save_path="phase_extraction.png"):
    """
    可视化相位提取过程
    
    参数：
        phase_wrapped: 包裹相位
        phase_unwrapped: 展开相位
        phase_diff: 相位差分
        save_path: 保存路径
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('相位提取过程', fontsize=14, fontweight='bold')
    
    frames = range(min(500, len(phase_wrapped)))
    
    # 包裹相位
    ax = axes[0]
    ax.plot(frames, phase_wrapped[:len(frames)], 'b-', linewidth=1.5)
    ax.set_ylabel('相位 (rad)', fontsize=11, fontweight='bold')
    ax.set_title('包裹相位 (Wrapped Phase)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-np.pi, np.pi)
    
    # 展开相位
    ax = axes[1]
    ax.plot(frames, phase_unwrapped[:len(frames)], 'g-', linewidth=1.5)
    ax.set_ylabel('相位 (rad)', fontsize=11, fontweight='bold')
    ax.set_title('展开相位 (Unwrapped Phase)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 相位差分
    ax = axes[2]
    ax.plot(frames[:-1], phase_diff[:len(frames)-1], 'r-', linewidth=1.5)
    ax.set_ylabel('相位差分', fontsize=11, fontweight='bold')
    ax.set_xlabel('帧索引', fontsize=11, fontweight='bold')
    ax.set_title('相位差分 (Phase Differentiation)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print(f"\n✓ 保存相位提取过程图到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_vital_signs(respiration, heartbeat, breathing_rate, heart_rate, 
                         frame_rate=10.0, save_path="vital_signs.png"):
    """
    可视化生命体征信号
    
    参数：
        respiration: 呼吸信号
        heartbeat: 心跳信号
        breathing_rate: 呼吸率 (次/分钟)
        heart_rate: 心率 (次/分钟)
        frame_rate: 帧率 (Hz)
        save_path: 保存路径
    """
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('提取的生命体征信号', fontsize=14, fontweight='bold')
    
    time_axis = np.arange(len(respiration)) / frame_rate
    
    # 呼吸信号
    ax = axes[0]
    ax.plot(time_axis, respiration, 'b-', linewidth=1.5, label='呼吸信号')
    peaks_resp, _ = find_peaks(respiration, distance=frame_rate*2)
    ax.plot(time_axis[peaks_resp], respiration[peaks_resp], 'b*', markersize=10, label='呼吸峰值')
    ax.set_ylabel('幅度', fontsize=11, fontweight='bold')
    ax.set_title(f'呼吸信号 (呼吸率: {breathing_rate:.1f} 次/分钟)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # 心跳信号
    ax = axes[1]
    ax.plot(time_axis, heartbeat, 'r-', linewidth=1.5, label='心跳信号')
    peaks_hr, _ = find_peaks(heartbeat, distance=frame_rate*0.4)
    ax.plot(time_axis[peaks_hr], heartbeat[peaks_hr], 'r*', markersize=10, label='心跳峰值')
    ax.set_ylabel('幅度', fontsize=11, fontweight='bold')
    ax.set_xlabel('时间 (秒)', fontsize=11, fontweight='bold')
    ax.set_title(f'心跳信号 (心率: {heart_rate:.1f} 次/分钟)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    print(f"\n✓ 保存生命体征信号图到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_frequency_spectrum(positive_frequencies, respiration_magnitude, 
                                heartbeat_magnitude, breathing_rate, heart_rate,
                                save_path="frequency_spectrum.png"):
    """
    可视化频域谱
    
    参数：
        positive_frequencies: 正频率数组
        respiration_magnitude: 呼吸信号频域幅度
        heartbeat_magnitude: 心跳信号频域幅度
        breathing_rate: 呼吸率
        heart_rate: 心率
        save_path: 保存路径
    """
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('生命体征频域谱', fontsize=14, fontweight='bold')
    
    # 呼吸频域
    ax = axes[0]
    ax.semilogy(positive_frequencies, respiration_magnitude, 'b-', linewidth=2)
    ax.axvline(breathing_rate/60, color='b', linestyle='--', linewidth=2, 
              label=f'呼吸频率: {breathing_rate:.1f} 次/分钟')
    ax.set_xlim(0, 1)
    ax.set_ylabel('幅度', fontsize=11, fontweight='bold')
    ax.set_title('呼吸信号频域谱', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    
    # 心跳频域
    ax = axes[1]
    ax.semilogy(positive_frequencies, heartbeat_magnitude, 'r-', linewidth=2)
    ax.axvline(heart_rate/60, color='r', linestyle='--', linewidth=2, 
              label=f'心率: {heart_rate:.1f} 次/分钟')
    ax.set_xlim(0, 3)
    ax.set_ylabel('幅度', fontsize=11, fontweight='bold')
    ax.set_xlabel('频率 (Hz)', fontsize=11, fontweight='bold')
    ax.set_title('心跳信号频域谱', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    print(f"\n✓ 保存频域谱图到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("开始处理雷达信号 - 步骤5：生命体征提取")
    print("=" * 70 + "\n")
    
    # 加载数据
    print("加载处理结果...")
    beamformed_signal = np.load(BEAMFORMED_SIGNAL_FILE)
    print(f"✓ 已加载波束形成信号: {beamformed_signal.shape}\n")
    
    # 帧率配置
    frame_rate = FRAME_RATE  # 从config.py读取
    print(f"✓ 采样频率: {frame_rate} Hz\n")
    
    # ========================================================================
    # 核心步骤：相位处理 (顺序很重要！)
    # ========================================================================
    print("=" * 70)
    print("相位处理流程（正确顺序）")
    print("=" * 70)
    
    # 步骤1：提取包裹相位 (wrapped phase in -π to π)
    phase_wrapped = extract_phase(beamformed_signal)
    
    # 步骤2：相位解缠 (移除 2π 跳跃，得到连续相位)
    phase_unwrapped = unwrap_phase(phase_wrapped)
    
    # 步骤3：相位差分 (得到相位变化率，与呼吸/心跳成正比)
    phase_diff = phase_differentiation(phase_unwrapped)
    
    # ========================================================================
    # 步骤4：分离呼吸和心跳
    # ========================================================================
    respiration, heartbeat = extract_respiration_and_heartbeat(phase_diff, frame_rate)
    breathing_rate, heart_rate = extract_vital_signs(respiration, heartbeat, frame_rate)
    
    # 可视化
    print("\n" + "=" * 70)
    print("可视化生命体征提取结果")
    print("=" * 70)
    
    visualize_phase_extraction(phase_wrapped, phase_unwrapped, phase_diff,
                              save_path=get_image_path("phase_extraction.png"))
    
    visualize_vital_signs(respiration, heartbeat, breathing_rate, heart_rate,
                         frame_rate=frame_rate,
                         save_path=get_image_path("vital_signs.png"))
    
    # 频域分析
    fft_resp = np.fft.fft(respiration)
    fft_hr = np.fft.fft(heartbeat)
    freqs = np.fft.fftfreq(len(respiration), 1/frame_rate)
    positive_freqs_idx = freqs > 0
    
    visualize_frequency_spectrum(freqs[positive_freqs_idx], 
                                np.abs(fft_resp[positive_freqs_idx]),
                                np.abs(fft_hr[positive_freqs_idx]),
                                breathing_rate, heart_rate,
                                save_path=get_image_path("frequency_spectrum.png"))
    
    # 保存结果
    print("\n" + "=" * 70)
    print("保存处理结果")
    print("=" * 70)
    np.save(PHASE_WRAPPED_FILE, phase_wrapped)
    np.save(PHASE_UNWRAPPED_FILE, phase_unwrapped)
    np.save(PHASE_DIFF_FILE, phase_diff)
    np.save(RESPIRATION_FILE, respiration)
    np.save(HEARTBEAT_FILE, heartbeat)
    
    print(f"✓ 已保存到 {RESULT_DIR}/")
    print(f"  phase_wrapped.npy")
    print(f"  phase_unwrapped.npy")
    print(f"  phase_diff.npy")
    print(f"  respiration.npy")
    print(f"  heartbeat.npy")
    
    # 最终总结
    print("\n" + "=" * 70)
    print("✅ 步骤5处理完成！")
    print("=" * 70)
    print(f"\n📊 最终生命体征提取结果:")
    print(f"   呼吸率 (RR): {breathing_rate:.1f} 次/分钟")
    print(f"   心率 (HR): {heart_rate:.1f} 次/分钟")
    print(f"\n 说明:")
    print(f"   • 正常成人呼吸率: 12-20 次/分钟")
    print(f"   • 正常成人心率: 60-100 次/分钟")
    print(f"   • 结果精度受数据质量影响")
    print(f"\n✓ 所有5个步骤处理完成！")