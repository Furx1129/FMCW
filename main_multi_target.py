import numpy as np
import matplotlib.pyplot as plt
from config import *

# 导入现有的模块
from step1_preprocessing import load_and_preprocess_data
# 导入刚才修改过的 step2 (多目标版)
from step2_range_processing import range_processing, extract_target_signal
# 导入 step3 (已合并完整版，包含 BGT60AntennaArray 等)
from step3_angle_estimation import (
    calculate_covariance_matrix, 
    compute_inverse_covariance, 
    mvdr_spectrum, 
    BGT60AntennaArray
)
# 导入 step4
from step4_digital_beamforming import compute_mvdr_weights, apply_beamforming
# 导入 step5
from step5_vital_sign_extraction import (
    extract_phase, 
    unwrap_phase,
    phase_differentiation, 
    extract_vital_signs
)
from scipy.signal import filtfilt, butter

# ============================================================================
# 辅助函数
# ============================================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Butterworth 带通滤波器（辅助函数）
    
    参数：
        data: 输入信号
        lowcut: 低截止频率 (Hz)
        highcut: 高截止频率 (Hz)
        fs: 采样频率 (Hz)
        order: 滤波器阶数
    
    输出：
        y: 滤波后的信号
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def process_single_target(range_fft, range_idx, antenna_array, target_num=1):
    """
    处理单个目标的完整流水线
    
    参数：
        range_fft: FFT后的复数数据 (frames, samples, rx)
        range_idx: 该目标的距离索引
        antenna_array: 天线阵列配置对象
        target_num: 目标序号（用于打印）
    
    返出：
        dict: 包含处理结果的字典
            - 'phase_wrapped': 原始相位
            - 'phase_unwrapped': 展开后相位
            - 'phase_diff': 相位差分
            - 'respiration': 呼吸波形
            - 'heartbeat': 心跳波形
            - 'angle': 估计的方位角
            - 'distance_m': 物理距离
    """
    
    print(f"\n>>> 🎯 正在处理第 {target_num} 个目标 (距离索引: {range_idx}) <<<")
    
    # ========== Step 2.5: 提取该距离的数据 ==========
    target_signal_raw = extract_target_signal(range_fft, range_idx)
    print(f"    ✓ 提取目标信号: {target_signal_raw.shape}")
    
    # 重要：只使用前两个天线（RX1, RX2）与 Step 3 的简化方案保持一致
    if target_signal_raw.shape[1] > 2:
        print(f"    ⚠️ 原始信号有 {target_signal_raw.shape[1]} 个天线，截取前2个（RX1, RX2）")
        target_signal_raw = target_signal_raw[:, :2]
        print(f"    ✓ 截取后信号形状: {target_signal_raw.shape}")
    
    # ========== Step 3: 角度估计 (MVDR) ==========
    # 计算协方差矩阵（只用前两个天线）
    R = calculate_covariance_matrix(target_signal_raw)
    R_inv = compute_inverse_covariance(R)
    
    # 搜索该目标的方位角
    # 注意：mvdr_spectrum 只需要 R_inv 和可选的 angle_range
    _, _, peak_angle = mvdr_spectrum(R_inv, angle_range=(-60, 60, 0.5))
    print(f"    📐 锁定角度: {peak_angle:.1f}°")
    
    # ========== Step 4: 波束形成 (Beamforming) ==========
    # 生成指向该角度的权重 (使用正确的波长)
    W_opt = compute_mvdr_weights(R_inv, peak_angle, wavelength=WAVELENGTH)
    
    # 融合信号（获得复数信号）
    beamformed_signal = apply_beamforming(target_signal_raw, W_opt)
    print(f"    🔄 波束形成完成: {beamformed_signal.shape}")
    
    # ========== Step 5: 体征提取 ==========
    # 提取相位
    phase_wrapped = extract_phase(beamformed_signal)
    
    # 展开相位
    phase_unwrapped = unwrap_phase(phase_wrapped)
    
    # 相位差分
    phase_diff = phase_differentiation(phase_unwrapped)
    
    # 带通滤波提取呼吸和心跳
    # 呼吸: 0.2-0.5 Hz (12-30 次/分)
    respiration_wave = butter_bandpass_filter(phase_diff, 0.2, 0.5, FRAME_RATE, order=4)
    
    # 心跳: 0.8-2.5 Hz (48-150 次/分)
    heartbeat_wave = butter_bandpass_filter(phase_diff, 0.8, 2.5, FRAME_RATE, order=4)
    
    print(f"    💓 生命体征提取完成")
    
    # 计算物理距离
    distance_resolution = 3e8 / (2 * 5e9)  # 约 0.03m
    physical_distance = range_idx * distance_resolution
    
    return {
        'phase_wrapped': phase_wrapped,
        'phase_unwrapped': phase_unwrapped,
        'phase_diff': phase_diff,
        'respiration': respiration_wave,
        'heartbeat': heartbeat_wave,
        'angle': peak_angle,
        'distance_m': physical_distance,
        'range_idx': range_idx
    }

def plot_final_waveforms_and_spectra(target_results_list):
    """
    为每个目标绘制最终的波形与频谱分析（4子图布局）
    
    参数：
        target_results_list: 目标处理结果的列表
    
    布局：
        左上：呼吸时域波形
        右上：心跳时域波形
        左下：呼吸频谱（RPM）
        右下：心跳频谱（BPM）
    """
    
    frame_rate = FRAME_RATE
    
    for i, result in enumerate(target_results_list):
        # 提取数据
        respiration_wave = result['respiration']
        heartbeat_wave = result['heartbeat']
        
        # 时间轴
        num_frames = len(respiration_wave)
        time_axis = np.arange(num_frames) / frame_rate
        
        # FFT 计算频谱
        fft_resp = np.fft.fft(respiration_wave)
        fft_heart = np.fft.fft(heartbeat_wave)
        
        # 频率轴（单位：Hz）
        freqs = np.fft.fftfreq(num_frames, d=1.0/frame_rate)
        
        # 只取正频率部分
        positive_idx = freqs > 0
        freqs_positive = freqs[positive_idx]
        fft_resp_positive = np.abs(fft_resp[positive_idx])
        fft_heart_positive = np.abs(fft_heart[positive_idx])
        
        # 计算呼吸率（RPM）
        resp_range = (freqs_positive >= 0.2) & (freqs_positive <= 0.5)
        if np.any(resp_range):
            peak_resp_idx = np.argmax(fft_resp_positive[resp_range])
            peak_resp_freq = freqs_positive[resp_range][peak_resp_idx]
            calculated_breath_rate = peak_resp_freq * 60  # Hz -> RPM
        else:
            calculated_breath_rate = 0
        
        # 计算心率（BPM）
        heart_range = (freqs_positive >= 0.8) & (freqs_positive <= 2.5)
        if np.any(heart_range):
            peak_heart_idx = np.argmax(fft_heart_positive[heart_range])
            peak_heart_freq = freqs_positive[heart_range][peak_heart_idx]
            calculated_heart_rate = peak_heart_freq * 60  # Hz -> BPM
        else:
            calculated_heart_rate = 0
        
        # ====================================================================
        # 绘图：4子图布局
        # ====================================================================
        fig = plt.figure(figsize=(12, 9))
        fig.suptitle(f'目标 {i+1} - 最终波形与频谱分析\n'
                    f'距离={result["distance_m"]:.3f}m, 角度={result["angle"]:.1f}°',
                    fontsize=14, fontweight='bold')
        
        # ----- 左上：呼吸时域波形 -----
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(time_axis, respiration_wave, 'b-', linewidth=1.5)
        ax1.set_title('呼吸时域波形 (Respiration Waveform)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('时间 (s)', fontsize=10)
        ax1.set_ylabel('相位变化 (rad)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # ----- 右上：心跳时域波形 -----
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(time_axis, heartbeat_wave, 'r-', linewidth=1.5)
        ax2.set_title('心跳时域波形 (Heartbeat Waveform)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('时间 (s)', fontsize=10)
        ax2.set_ylabel('相位变化 (rad)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # ----- 左下：呼吸频谱 -----
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(freqs_positive * 60, fft_resp_positive, 'b-', linewidth=2)
        ax3.axvline(calculated_breath_rate, color='red', linestyle='--', 
                   linewidth=2, label=f'峰值: {calculated_breath_rate:.1f} RPM')
        ax3.set_title(f'呼吸频谱 (Respiration Spectrum)\n峰值: {calculated_breath_rate:.1f} RPM',
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('频率 (RPM)', fontsize=10)
        ax3.set_ylabel('幅度', fontsize=10)
        ax3.set_xlim(0, 40)  # 呼吸频率范围: 0-40 RPM
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        
        # ----- 右下：心跳频谱 -----
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(freqs_positive * 60, fft_heart_positive, 'r-', linewidth=2)
        ax4.axvline(calculated_heart_rate, color='blue', linestyle='--', 
                   linewidth=2, label=f'峰值: {calculated_heart_rate:.1f} BPM')
        ax4.set_title(f'心跳频谱 (Heartbeat Spectrum)\n峰值: {calculated_heart_rate:.1f} BPM',
                     fontsize=12, fontweight='bold')
        ax4.set_xlabel('频率 (BPM)', fontsize=10)
        ax4.set_ylabel('幅度', fontsize=10)
        ax4.set_xlim(40, 150)  # 心率频率范围: 40-150 BPM
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = get_image_path(f"target_{i+1}_final_waveforms_spectra.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ 保存图片到: {save_path}")
        
        plt.show()
        
        # 打印生理参数
        print(f"\n    📊 目标 {i+1} 生理参数:")
        print(f"       呼吸率: {calculated_breath_rate:.1f} RPM")
        print(f"       心率: {calculated_heart_rate:.1f} BPM")
        
        # 存储计算结果到 result 字典
        result['breathing_rate'] = calculated_breath_rate
        result['heart_rate'] = calculated_heart_rate
def plot_results(target_results_list):
    """
    绘制所有目标的处理结果
    
    参数：
        target_results_list: 目标处理结果的列表
    """
    
    n_targets = len(target_results_list)
    
    # 创建画布
    fig, axes = plt.subplots(n_targets, 1, figsize=(12, 4 * n_targets), sharex=True)
    if n_targets == 1:
        axes = [axes]  # 统一为列表
    
    # 时间轴
    frame_rate = FRAME_RATE
    num_frames = len(target_results_list[0]['respiration'])
    time_axis = np.arange(num_frames) / frame_rate
    
    # 对每个目标绘图
    for i, result in enumerate(target_results_list):
        ax = axes[i]
        
        # 绘制呼吸和心跳波形
        ax.plot(time_axis, result['respiration'], label='呼吸 (0.2-0.5 Hz)', 
               linewidth=2, color='blue', alpha=0.8)
        ax.plot(time_axis, result['heartbeat'], label='心跳 (0.8-2.5 Hz)', 
               linewidth=2, color='red', alpha=0.7)
        
        # 标题包含目标信息
        title = f"目标 {i+1}: 距离={result['distance_m']:.3f}m, 角度={result['angle']:.1f}°"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # 网格和图例
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylabel('相位变化 (rad)', fontsize=10)
    
    # 总的时间轴标签
    axes[-1].set_xlabel('时间 (s)', fontsize=11, fontweight='bold')
    
    plt.suptitle('多目标生命体征监测结果', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    print(f"\n✓ 绘制 {n_targets} 个目标的监测结果")
    plt.savefig(get_image_path("multi_target_vital_signs.png"), dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# 主程序
# ============================================================================

def main():
    """
    多目标生命体征监测流水线
    
    流程：
    1. 预处理（MTI）
    2. 多目标距离检测（Range Processing）
    3. **强制筛选 Top-N 目标**（新增）
    4. 对每个目标循环处理：
       - 角度估计 (MVDR)
       - 波束形成 (Beamforming)
       - 体征提取 (Vital Signs Extraction)
    5. 绘制和保存结果
    6. 绘制最终波形与频谱分析
    """
    
    print("=" * 70)
    print("🚀 启动多目标生命体征监测流水线...")
    print("=" * 70 + "\n")
    
    # ====================================================================
    # Step 1: 预处理 (全局只做一次)
    # ====================================================================
    print("Step 1: 预处理数据 (MTI)")
    print("-" * 70)
    data_mti, _ = load_and_preprocess_data(INPUT_FILE)
    print(f"✓ 预处理完成: {data_mti.shape}\n")
    
    # ====================================================================
    # Step 2: 多目标定位
    # ====================================================================
    print("Step 2: 多目标距离检测")
    print("-" * 70)
    # 1. 提高阈值到 -12 dB
    range_spectrum_half, target_indices, range_fft = range_processing(
        data_mti, 
        threshold_db=-12,  # <--- 从 -20 改为 -12
        min_distance_m=0.3
    )
    
    # 2. 加入强制筛选逻辑 (只取能量最大的 1 个)
    MAX_TARGETS = 1  # <--- 你的场景只有1个人
    
    if len(target_indices) > MAX_TARGETS:
        print(f"⚠️ 检测到 {len(target_indices)} 个目标，仅保留能量最大的 {MAX_TARGETS} 个")
        # 获取这些目标的能量值
        target_powers = range_spectrum_half[target_indices]
        # 排序并取前 N 个
        sorted_indices = np.argsort(target_powers)[::-1][:MAX_TARGETS]
        target_indices = target_indices[sorted_indices]
    
    if len(target_indices) == 0:
        print("\n❌ 未检测到任何目标，程序退出")
        return
        
    print(f"✓ 最终锁定 {len(target_indices)} 个目标: 索引 {target_indices}\n")
    
    # ====================================================================
    # Step 2.5: 备用筛选策略说明 (已整合到上面)
    # ====================================================================
    print("\n" + "-" * 70)
    print("Step 2.5: 强制筛选 Top-N 目标（基于能量排序）")
    print("-" * 70)
    
    MAX_TARGETS = 1  # 根据实际场景调整
                     # 1 = 单人场景
                     # 2-3 = 多人场景
    
    if len(target_indices) > MAX_TARGETS:
        print(f"⚠️ 检测到 {len(target_indices)} 个目标，仅保留能量最大的 {MAX_TARGETS} 个")
        
        # 1. 获取这些索引对应的能量值
        target_powers = range_spectrum_half[target_indices]
        
        # 2. 对能量进行排序 (从大到小)
        # argsort 返回从小到大的索引，[::-1] 反转为从大到小
        sorted_indices_of_indices = np.argsort(target_powers)[::-1]
        
        # 3. 取前 MAX_TARGETS 个
        top_indices_of_indices = sorted_indices_of_indices[:MAX_TARGETS]
        
        # 4. 更新 target_indices（保留原始距离索引的顺序）
        target_indices_filtered = target_indices[top_indices_of_indices]
        
        # 5. 按距离索引重新排序（从近到远）
        target_indices = np.sort(target_indices_filtered)
        
        print(f"✓ 筛选后目标索引: {target_indices}")
        print(f"  对应能量: {range_spectrum_half[target_indices]}")
    else:
        print(f"✓ 检测到 {len(target_indices)} 个目标，无需筛选（≤ {MAX_TARGETS}）")
    
    print(f"\n✓ 最终处理 {len(target_indices)} 个目标\n")
    
    # ====================================================================
    # 初始化天线配置 (Step 3 需要)
    # ====================================================================
    print("初始化天线阵列配置")
    print("-" * 70)
    antenna_array = BGT60AntennaArray(wavelength=WAVELENGTH)
    
    # ====================================================================
    # Step 3-5: 循环处理每个目标
    # ====================================================================
    print("\nStep 3-5: 处理每个目标")
    print("=" * 70)
    
    target_results_list = []
    
    for i, range_idx in enumerate(target_indices):
        try:
            result = process_single_target(range_fft, range_idx, antenna_array, target_num=i+1)
            target_results_list.append(result)
        except Exception as e:
            print(f"    ⚠️ 处理目标 {i+1} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # ====================================================================
    # 绘制结果
    # ====================================================================
    if len(target_results_list) > 0:
        print("\n" + "=" * 70)
        print("绘制处理结果")
        print("=" * 70)
        plot_results(target_results_list)
        
        # ====================================================================
        # 绘制最终波形与频谱分析
        # ====================================================================
        print("\n" + "=" * 70)
        print("绘制最终波形与频谱分析")
        print("=" * 70)
        plot_final_waveforms_and_spectra(target_results_list)
        
        # ====================================================================
        # 最终总结
        # ====================================================================
        print("\n" + "=" * 70)
        print("✅ 处理完成！")
        print("=" * 70)
        print(f"\n📊 处理统计:")
        print(f"  原始检测: {len(target_indices)} 个目标")
        print(f"  成功处理: {len(target_results_list)} 个目标")
        
        print(f"\n📍 目标信息:")
        for i, result in enumerate(target_results_list):
            print(f"  目标 {i+1}:")
            print(f"    - 距离: {result['distance_m']:.3f} m")
            print(f"    - 角度: {result['angle']:.1f}°")
            print(f"    - 距离索引: {result['range_idx']}")
            print(f"    - 呼吸率: {result.get('breathing_rate', 'N/A')} RPM")
            print(f"    - 心率: {result.get('heart_rate', 'N/A')} BPM")
        
        print(f"\n💡 提示：")
        print(f"  - 修改 MAX_TARGETS 可以改变最大处理目标数")
        print(f"  - 修改 threshold_db 可以调整峰值检测灵敏度")
        print(f"  - 修改 config.py 中的 DATA_NAME 可以切换数据集\n")
    else:
        print("\n❌ 所有目标处理失败")


if __name__ == "__main__":
    main()
