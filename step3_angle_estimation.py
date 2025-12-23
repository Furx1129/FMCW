import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ============================================================================
# 中文字体配置
# ============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 步骤 3：高阶角度估计 (MVDR Angle Estimation)
# ============================================================================

def calculate_covariance_matrix(target_signal):
    """
    计算协方差矩阵 R = X^H X
    
    输入：
        target_signal: 目标信号，形状 (frames, rx=3)
                      复数矩阵，每一行是一帧的3个天线接收信号
    
    输出：
        R: 协方差矩阵，形状 (3, 3)
           R = X^H X，其中 X^H 是 X 的共轭转置
    """
    
    print("\n" + "=" * 70)
    print("步骤3.1：计算协方差矩阵")
    print("=" * 70)
    
    print(f"  输入信号形状: {target_signal.shape}")
    print(f"  = (帧数={target_signal.shape[0]}, 天线数={target_signal.shape[1]})")
    
    # X^H 是 X 的共轭转置
    # target_signal.conj().T 即为 X^H，形状为 (3, frames)
    # 相乘得到 (3, frames) @ (frames, 3) = (3, 3)
    
    print(f"\n✓ 计算 R = X^H * X...")
    R = target_signal.conj().T @ target_signal
    
    print(f"  协方差矩阵形状: {R.shape}")
    print(f"  协方差矩阵是 Hermitian 矩阵: {np.allclose(R, R.conj().T)}")
    
    # 打印协方差矩阵的特征值（用于判断矩阵质量）
    eigenvalues = np.linalg.eigvals(R)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    
    print(f"\n✓ 协方差矩阵特征值:")
    for i, eigval in enumerate(eigenvalues_sorted):
        print(f"    λ{i+1} = {eigval:.6f}")
    
    # 条件数（用于判断矩阵是否接近奇异）
    condition_number = np.linalg.cond(R)
    print(f"\n✓ 条件数 (Condition Number): {condition_number:.2f}")
    if condition_number > 1e10:
        print(f"  ⚠️ 警告: 矩阵接近奇异，建议检查数据质量")
    else:
        print(f"  ✓ 矩阵质量良好")
    
    return R


def compute_inverse_covariance(R):
    """
    计算协方差矩阵的逆（使用伪逆防止奇异）
    
    输入：
        R: 协方差矩阵，形状 (3, 3)
    
    输出：
        R_inv: 逆矩阵，形状 (3, 3)
    """
    
    print("\n" + "=" * 70)
    print("步骤3.2：计算协方差矩阵的逆")
    print("=" * 70)
    
    # 方法1：直接求逆（可能不稳定）
    # R_inv = np.linalg.inv(R)
    
    # 方法2：使用伪逆（更稳定）
    print(f"\n✓ 使用伪逆 (pinv) 计算 R^(-1)...")
    R_inv = np.linalg.pinv(R)
    
    print(f"  逆矩阵形状: {R_inv.shape}")
    
    # 验证：R * R_inv 应该接近单位矩阵
    identity_check = R @ R_inv
    error = np.linalg.norm(identity_check - np.eye(3))
    print(f"  验证 R * R^(-1) ≈ I: 误差 = {error:.6f}")
    
    return R_inv


def steering_vector(angle_deg, num_antennas=3, wavelength=0.06):
    """
    生成导向矢量 a(θ)
    
    物理原理：
    三根天线均匀排列，相邻天线间距 d = λ/2 = 0.03m
    当信号从角度 θ 到达时，不同天线接收到的相位差为：
        Δφ = 2π * d * sin(θ) / λ
    
    导向矢量：a(θ) = [1, exp(j*2π*d*sin(θ)/λ), exp(j*4π*d*sin(θ)/λ)]
    
    输入：
        angle_deg: 角度（度），范围 -90° ~ +90°
        num_antennas: 天线数（默认3）
        wavelength: 波长，单位 m（默认0.06m，对应5GHz）
    
    输出：
        a: 导向矢量，形状 (num_antennas, 1)
    """
    
    # 转换为弧度
    angle_rad = np.deg2rad(angle_deg)
    
    # 天线间距
    d = wavelength / 2  # λ/2
    
    # 相位差步长
    phase_step = 2 * np.pi * d * np.sin(angle_rad) / wavelength
    
    # 生成导向矢量
    a = np.array([
        np.exp(1j * 0 * phase_step),           # 天线0：参考点
        np.exp(1j * 1 * phase_step),           # 天线1
        np.exp(1j * 2 * phase_step)            # 天线2
    ]).reshape(-1, 1)  # 形状 (3, 1)
    
    return a


def mvdr_spectrum(R_inv, angle_range=None):
    """
    计算MVDR谱
    
    原理：
    P(θ) = 1 / (a(θ)^H * R^(-1) * a(θ))
    
    其中：
    - a(θ) 是导向矢量
    - R^(-1) 是协方差矩阵的逆
    - ^H 表示共轭转置
    
    输入：
        R_inv: 协方差矩阵的逆，形状 (3, 3)
        angle_range: 角度范围，tuple (start, end, step)
                    默认 (-60, 60, 0.5)
    
    输出：
        spectrum: MVDR谱，shape (num_angles,)
        angles: 对应的角度数组，shape (num_angles,)
        peak_angle: 峰值对应的角度
    """
    
    print("\n" + "=" * 70)
    print("步骤3.3：空间谱扫描与MVDR谱计算")
    print("=" * 70)
    
    if angle_range is None:
        angle_range = (-60, 60, 0.5)  # 默认范围和分辨率
    
    start_angle, end_angle, angle_step = angle_range
    
    # 生成角度扫描范围
    angles = np.arange(start_angle, end_angle + angle_step, angle_step)
    num_angles = len(angles)
    
    print(f"\n✓ 扫描角度范围: [{start_angle}°, {end_angle}°]")
    print(f"  角度分辨率: {angle_step}°")
    print(f"  扫描点数: {num_angles}")
    
    # 初始化MVDR谱
    spectrum = np.zeros(num_angles)
    
    print(f"\n✓ 计算MVDR谱: P(θ) = 1 / (a(θ)^H * R^(-1) * a(θ))...")
    
    # 对每个角度计算MVDR谱值
    for i, angle in enumerate(angles):
        # 生成该角度的导向矢量
        a = steering_vector(angle)  # 形状 (3, 1)
        
        # 计算分母：a^H * R^(-1) * a
        # a.conj().T: (1, 3)
        # R_inv: (3, 3)
        # a: (3, 1)
        # 结果: (1, 1) 的复数
        denominator = (a.conj().T @ R_inv @ a)[0, 0]
        
        # MVDR谱值：1 / denominator
        spectrum[i] = 1.0 / np.abs(denominator)
    
    # 归一化谱
    spectrum = spectrum / np.max(spectrum)
    
    # 找峰值
    peak_idx = np.argmax(spectrum)
    peak_angle = angles[peak_idx]
    peak_power = spectrum[peak_idx]
    
    print(f"\n✓ 峰值检测结果:")
    print(f"  目标角度: {peak_angle:.2f}°")
    print(f"  峰值功率: {peak_power:.6f}")
    
    # 计算3dB带宽（角度分辨率）
    threshold = peak_power / 2
    indices_above_threshold = np.where(spectrum > threshold)[0]
    if len(indices_above_threshold) > 0:
        angle_3db = (angles[indices_above_threshold[-1]] - angles[indices_above_threshold[0]])
        print(f"  3dB带宽: {angle_3db:.2f}°")
    
    return spectrum, angles, peak_angle


def visualize_mvdr_spectrum(spectrum, angles, peak_angle, save_path="mvdr_spectrum.png"):
    """
    可视化MVDR谱
    
    参数：
        spectrum: MVDR谱
        angles: 对应的角度数组
        peak_angle: 峰值角度
        save_path: 保存图片路径
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('MVDR角度估计谱', fontsize=14, fontweight='bold')
    
    # 左图：线性谱
    ax1.plot(angles, spectrum, 'b-', linewidth=2.5, label='MVDR谱')
    ax1.plot(peak_angle, spectrum[np.argmax(spectrum)], 'r*', 
            markersize=20, label=f'检测角度: {peak_angle:.2f}°')
    ax1.axvline(peak_angle, color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_xlabel('角度 (度)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('归一化功率', fontsize=12, fontweight='bold')
    ax1.set_title('MVDR谱 (线性)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.set_xlim(angles[0], angles[-1])
    
    # 右图：dB谱（更易观察细节）
    spectrum_db = 10 * np.log10(spectrum + 1e-10)
    ax2.plot(angles, spectrum_db, 'g-', linewidth=2.5, label='MVDR谱 (dB)')
    peak_idx = np.argmax(spectrum)
    ax2.plot(peak_angle, spectrum_db[peak_idx], 'r*', 
            markersize=20, label=f'检测角度: {peak_angle:.2f}°')
    ax2.axvline(peak_angle, color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax2.set_xlabel('角度 (度)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('功率 (dB)', fontsize=12, fontweight='bold')
    ax2.set_title('MVDR谱 (dB)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.set_xlim(angles[0], angles[-1])
    
    plt.tight_layout()
    
    print(f"\n✓ 保存MVDR谱图到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_beampattern(R_inv, save_path="beampattern.png"):
    """
    可视化波束方向图（Beam Pattern）
    
    参数：
        R_inv: 协方差矩阵的逆
        save_path: 保存图片路径
    """
    
    # 计算MVDR谱（更细的分辨率用于绘图）
    spectrum, angles, _ = mvdr_spectrum(R_inv, angle_range=(-90, 90, 0.1))
    
    fig = plt.figure(figsize=(12, 10))
    
    # 极坐标图
    ax = fig.add_subplot(111, projection='polar')
    
    # 转换为极坐标
    angles_rad = np.deg2rad(angles)
    spectrum_normalized = spectrum / np.max(spectrum)
    
    ax.plot(angles_rad, spectrum_normalized, 'b-', linewidth=2.5)
    ax.fill(angles_rad, spectrum_normalized, alpha=0.25, color='blue')
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_title('MVDR波束方向图\n(极坐标)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print(f"✓ 保存波束方向图到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compare_with_conventional_beamformer(target_signal, save_path="beamformer_comparison.png"):
    """
    对比MVDR与传统波束合成的性能
    
    输入：
        target_signal: 目标信号 (frames, 3)
        save_path: 保存路径
    """
    
    print("\n" + "=" * 70)
    print("步骤3.4：与传统波束合成对比")
    print("=" * 70)
    
    # 计算MVDR
    R = calculate_covariance_matrix(target_signal)
    R_inv = compute_inverse_covariance(R)
    mvdr_spectrum_data, angles, mvdr_peak = mvdr_spectrum(R_inv)
    
    # 计算传统波束合成（ULA, Uniform Linear Array）
    print(f"\n✓ 计算传统波束合成 (Conventional Beamformer)...")
    
    conventional_spectrum = np.zeros(len(angles))
    for i, angle in enumerate(angles):
        a = steering_vector(angle)
        # 传统波束合成：|a^H * X|^2
        beamformer_output = a.conj().T @ target_signal.T
        conventional_spectrum[i] = np.mean(np.abs(beamformer_output)**2)
    
    # 归一化
    conventional_spectrum = conventional_spectrum / np.max(conventional_spectrum)
    
    # 找传统波束合成的峰值
    conventional_peak_idx = np.argmax(conventional_spectrum)
    conventional_peak = angles[conventional_peak_idx]
    
    print(f"  传统方法检测角度: {conventional_peak:.2f}°")
    print(f"  MVDR方法检测角度: {mvdr_peak:.2f}°")
    
    # 绘图对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('MVDR vs 传统波束合成对比', fontsize=14, fontweight='bold')
    
    # 左图：线性对比
    ax = axes[0]
    ax.plot(angles, conventional_spectrum, 'b-', linewidth=2.5, label='传统波束合成')
    ax.plot(angles, mvdr_spectrum_data, 'r-', linewidth=2.5, label='MVDR')
    ax.plot(conventional_peak, conventional_spectrum[conventional_peak_idx], 
           'b*', markersize=15, label=f'传统峰值: {conventional_peak:.2f}°')
    ax.plot(mvdr_peak, mvdr_spectrum_data[np.argmax(mvdr_spectrum_data)], 
           'r*', markersize=15, label=f'MVDR峰值: {mvdr_peak:.2f}°')
    ax.set_xlabel('角度 (度)', fontsize=12, fontweight='bold')
    ax.set_ylabel('归一化功率', fontsize=12, fontweight='bold')
    ax.set_title('线性谱对比', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(angles[0], angles[-1])
    
    # 右图：dB对比
    ax = axes[1]
    conventional_db = 10 * np.log10(conventional_spectrum + 1e-10)
    mvdr_db = 10 * np.log10(mvdr_spectrum_data + 1e-10)
    ax.plot(angles, conventional_db, 'b-', linewidth=2.5, label='传统波束合成')
    ax.plot(angles, mvdr_db, 'r-', linewidth=2.5, label='MVDR')
    ax.plot(conventional_peak, conventional_db[conventional_peak_idx], 
           'b*', markersize=15, label=f'传统峰值: {conventional_peak:.2f}°')
    ax.plot(mvdr_peak, mvdr_db[np.argmax(mvdr_spectrum_data)], 
           'r*', markersize=15, label=f'MVDR峰值: {mvdr_peak:.2f}°')
    ax.set_xlabel('角度 (度)', fontsize=12, fontweight='bold')
    ax.set_ylabel('功率 (dB)', fontsize=12, fontweight='bold')
    ax.set_title('dB谱对比', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(angles[0], angles[-1])
    
    plt.tight_layout()
    
    print(f"\n✓ 保存对比图到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return mvdr_peak, conventional_peak


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("开始处理雷达信号 - 步骤3：高阶角度估计(MVDR)")
    print("=" * 70 + "\n")
    
    # 加载步骤2的处理结果
    print("加载步骤2的处理结果...")
    target_signal = np.load("results/test1_target_signal.npy")
    print(f"✓ 已加载目标信号: {target_signal.shape}\n")
    
    # 步骤3.1：计算协方差矩阵
    R = calculate_covariance_matrix(target_signal)
    
    # 步骤3.2：计算协方差矩阵的逆
    R_inv = compute_inverse_covariance(R)
    
    # 步骤3.3：计算MVDR谱
    print("\n" + "=" * 70)
    print("计算MVDR谱")
    print("=" * 70)
    mvdr_spectrum_data, angles, peak_angle = mvdr_spectrum(R_inv)
    
    # 可视化MVDR谱
    print("\n" + "=" * 70)
    print("可视化MVDR角度估计结果")
    print("=" * 70)
    visualize_mvdr_spectrum(mvdr_spectrum_data, angles, peak_angle, 
                           save_path="results/mvdr_spectrum.png")
    
    # 可视化波束方向图
    print("\n" + "=" * 70)
    print("绘制波束方向图")
    print("=" * 70)
    visualize_beampattern(R_inv, save_path="results/beampattern.png")
    
    # 对比MVDR与传统波束合成
    print("\n" + "=" * 70)
    print("对比MVDR与传统波束合成")
    print("=" * 70)
    mvdr_angle, conventional_angle = compare_with_conventional_beamformer(
        target_signal, 
        save_path="results/beamformer_comparison.png"
    )
    
    # 保存结果
    print("\n" + "=" * 70)
    print("保存处理结果")
    print("=" * 70)
    np.save("results/test1_covariance_matrix.npy", R)
    np.save("results/test1_mvdr_spectrum.npy", mvdr_spectrum_data)
    
    print("✓ test1_covariance_matrix.npy (协方差矩阵)")
    print("✓ test1_mvdr_spectrum.npy (MVDR谱)")
    
    # 最终总结
    print("\n" + "=" * 70)
    print("✅ 步骤3处理完成！")
    print("=" * 70)
    print(f"\n📍 角度估计结果:")
    print(f"   MVDR方法: {mvdr_angle:.2f}°")
    print(f"   传统方法: {conventional_angle:.2f}°")
    print(f"\n📊 生成的可视化图片:")
    print(f"   ✓ results/mvdr_spectrum.png (MVDR谱)")
    print(f"   ✓ results/beampattern.png (波束方向图)")
    print(f"   ✓ results/beamformer_comparison.png (方法对比)")
    print("\n下一步：相位提取与心跳呼吸分离（步骤4）")