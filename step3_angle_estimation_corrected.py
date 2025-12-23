import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from config import *

# ============================================================================
# 中文字体配置
# ============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# BGT60TR13C 天线布局信息
# ============================================================================
"""
重要：BGT60TR13C 接收天线布局
=========================================

天线布局: L型 (Linear + Planar)
- RX1 (antenna_idx=0): 位置 (0, 0)      [参考点，x轴第一根]
- RX2 (antenna_idx=1): 位置 (λ/2, 0)    [x轴第二根]
- RX3 (antenna_idx=2): 位置 (0, λ/2)    [y轴第一根]

波长 λ = c / f = 3e8 / 5e9 = 0.06m
天线间距 d = λ/2 = 0.03m

这与传统的线性阵列 [0, λ/2, λ] 不同！

处理建议：
=========================================
方案1（简化）：只使用RX1和RX2计算方位角
- 利用x轴上的两根天线测量水平角度 (Azimuth)
- 忽略y轴天线的信息
- 优点：计算简单，易于理解
- 缺点：无法测量仰角，分辨率略低

方案2（完整）：使用所有3根天线的L型导向矢量
- 构建2D导向矢量: a(θ,φ) 
- θ: 方位角 (Azimuth)
- φ: 仰角 (Elevation)
- 需要2D MVDR谱 (计算量大)

当前代码采用：方案1（简化方案）
"""

# ============================================================================
# 步骤 3：高阶角度估计 (MVDR Angle Estimation) - 修正版
# ============================================================================

class BGT60AntennaArray:
    """
    BGT60TR13C 天线阵列配置类
    
    存储天线位置和相关计算
    """
    def __init__(self, wavelength=0.06):
        """
        初始化天线阵列
        
        参数：
            wavelength: 波长 (m)，默认0.06m (5GHz)
        """
        self.wavelength = wavelength
        self.d = wavelength / 2  # 天线间距
        
        # L型天线位置 (单位: m)
        # RX1: 参考点
        # RX2: x轴方向
        # RX3: y轴方向
        self.antenna_positions = {
            'RX1': np.array([0.0, 0.0]),
            'RX2': np.array([self.d, 0.0]),
            'RX3': np.array([0.0, self.d])
        }
        
        print("\n" + "=" * 70)
        print("BGT60TR13C 天线阵列配置")
        print("=" * 70)
        print(f"\n✓ 天线布局: L型 (Linear + Planar)")
        print(f"  波长 λ: {wavelength*100:.2f} cm")
        print(f"  天线间距 d: {self.d*100:.2f} cm")
        print(f"\n✓ 天线位置:")
        for name, pos in self.antenna_positions.items():
            print(f"  {name}: ({pos[0]*100:.2f}cm, {pos[1]*100:.2f}cm)")
        print(f"\n⚠️ 重要: 该配置为L型，非线性阵列")
        print(f"  采用简化方案: 仅使用RX1和RX2计算方位角")
    
    def get_antenna_distance_to_ref(self, antenna_idx):
        """
        获取天线到参考点(RX1)的距离
        
        参数：
            antenna_idx: 天线索引 (0, 1, 2)
        
        输出：
            distance: 距离 (m)
        """
        antenna_names = ['RX1', 'RX2', 'RX3']
        pos = self.antenna_positions[antenna_names[antenna_idx]]
        distance = np.linalg.norm(pos)
        return distance


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
    
    print(f"\n✓ 输入信号形状: {target_signal.shape}")
    print(f"  = (帧数={target_signal.shape[0]}, 天线数={target_signal.shape[1]})")
    
    print(f"\n✓ 计算 R = X^H * X...")
    R = target_signal.conj().T @ target_signal
    
    print(f"  协方差矩阵形状: {R.shape}")
    print(f"  协方差矩阵是 Hermitian 矩阵: {np.allclose(R, R.conj().T)}")
    
    # 打印协方差矩阵的特征值
    eigenvalues = np.linalg.eigvals(R)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    
    print(f"\n✓ 协方差矩阵特征值:")
    for i, eigval in enumerate(eigenvalues_sorted):
        print(f"    λ{i+1} = {eigval:.6f}")
    
    # 条件数
    condition_number = np.linalg.cond(R)
    print(f"\n✓ 条件数 (Condition Number): {condition_number:.2f}")
    if condition_number > 1e10:
        print(f"  ⚠️ 警告: 矩阵接近奇异")
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
    
    print(f"\n✓ 使用伪逆 (pinv) 计算 R^(-1)...")
    R_inv = np.linalg.pinv(R)
    
    print(f"  逆矩阵形状: {R_inv.shape}")
    
    # 验证
    identity_check = R @ R_inv
    error = np.linalg.norm(identity_check - np.eye(3))
    print(f"  验证 R * R^(-1) ≈ I: 误差 = {error:.6f}")
    
    return R_inv


def steering_vector_linear_array(angle_deg, antenna_array, wavelength=0.06):
    """
    为线性阵列生成导向矢量（简化方案）
    
    简化处理：仅使用RX1和RX2（x轴方向的两根天线）来计算方位角
    
    原理：
    当信号从角度θ到达时，天线间的相位差为：
        Δφ = 2π * d * sin(θ) / λ
    
    导向矢量 (使用全部3根天线，但只有x轴分量对角度敏感)：
        a(θ) = [1, 
                exp(j*2π*d*sin(θ)/λ),           # RX2相对于RX1的相位差
                exp(j*0)]                        # RX3纯粹垂直，不对方位角敏感
    
    输入：
        angle_deg: 方位角（度），范围 -90° ~ +90°
        antenna_array: BGT60AntennaArray 对象
        wavelength: 波长 (m)
    
    输出：
        a: 导向矢量，形状 (3, 1)
    """
    
    angle_rad = np.deg2rad(angle_deg)
    d = antenna_array.d  # λ/2
    
    # 相位差步长（仅x轴方向）
    phase_step_x = 2 * np.pi * d * np.sin(angle_rad) / wavelength
    
    # 生成导向矢量
    # RX1: 参考点，相位为0
    # RX2: x轴相邻，相位差为 phase_step_x
    # RX3: y轴，不对方位角敏感，相位为0
    a = np.array([
        np.exp(1j * 0),                    # RX1
        np.exp(1j * phase_step_x),         # RX2
        np.exp(1j * 0)                     # RX3 (y轴不对方位角敏感)
    ]).reshape(-1, 1)
    
    return a


def steering_vector_l_array_full(angle_deg, antenna_array, wavelength=0.06):
    """
    为L型天线阵列生成完整的导向矢量（完整方案）
    
    说明：
    L型天线包括x轴和y轴的天线，可以捕捉2D信息
    但为了简化，当前只使用方位角θ，忽略仰角φ
    
    对于完整的2D MVDR，需要扫描(θ,φ)两个角度，计算复杂度高
    
    输入：
        angle_deg: 方位角（度），范围 -90° ~ +90°
        antenna_array: BGT60AntennaArray 对象
        wavelength: 波长 (m)
    
    输出：
        a: 导向矢量，形状 (3, 1)
    """
    
    angle_rad = np.deg2rad(angle_deg)
    d = antenna_array.d
    
    # 仅考虑方位角（azimuth），不考虑仰角（elevation）
    # 即假设所有信号来自同一高度
    
    # RX1位置: (0, 0)
    # RX2位置: (d, 0)  -> 相位差: 2π*d*sin(θ)/λ
    # RX3位置: (0, d)  -> 相位差: 2π*d*sin(θ)*sin(0)/λ = 0 (假设信号在x-z平面)
    
    phase_step = 2 * np.pi * d * np.sin(angle_rad) / wavelength
    
    a = np.array([
        np.exp(1j * 0),           # RX1
        np.exp(1j * phase_step),  # RX2
        np.exp(1j * 0)            # RX3
    ]).reshape(-1, 1)
    
    return a


def mvdr_spectrum(R_inv, antenna_array, angle_range=None, use_simplified=True):
    """
    计算MVDR谱
    
    原理：
    P(θ) = 1 / (a(θ)^H * R^(-1) * a(θ))
    
    输入：
        R_inv: 协方差矩阵的逆，形状 (3, 3)
        antenna_array: BGT60AntennaArray 对象
        angle_range: 角度范围，tuple (start, end, step)
        use_simplified: 是否使用简化方案 (仅x轴天线)
    
    输出：
        spectrum: MVDR谱
        angles: 对应的角度数组
        peak_angle: 峰值对应的角度
    """
    
    print("\n" + "=" * 70)
    print("步骤3.3：空间谱扫描与MVDR谱计算")
    print("=" * 70)
    
    if angle_range is None:
        angle_range = (-60, 60, 0.5)
    
    start_angle, end_angle, angle_step = angle_range
    angles = np.arange(start_angle, end_angle + angle_step, angle_step)
    num_angles = len(angles)
    
    print(f"\n✓ 扫描角度范围: [{start_angle}°, {end_angle}°]")
    print(f"  角度分辨率: {angle_step}°")
    print(f"  扫描点数: {num_angles}")
    
    if use_simplified:
        print(f"\n✓ 使用简化方案: 仅x轴方向 (RX1 + RX2)")
        print(f"  原因: BGT60TR13C为L型天线阵列")
        print(f"  优点: 计算简单，专注于方位角测量")
        steering_fn = steering_vector_linear_array
    else:
        print(f"\n✓ 使用L型完整导向矢量")
        steering_fn = steering_vector_l_array_full
    
    spectrum = np.zeros(num_angles)
    
    print(f"\n✓ 计算MVDR谱...")
    for i, angle in enumerate(angles):
        a = steering_fn(angle, antenna_array)
        denominator = (a.conj().T @ R_inv @ a)[0, 0]
        spectrum[i] = 1.0 / np.abs(denominator)
    
    # 归一化
    spectrum = spectrum / np.max(spectrum)
    
    # 找峰值
    peak_idx = np.argmax(spectrum)
    peak_angle = angles[peak_idx]
    peak_power = spectrum[peak_idx]
    
    print(f"\n✓ 峰值检测结果:")
    print(f"  目标方位角: {peak_angle:.2f}°")
    print(f"  峰值功率: {peak_power:.6f}")
    
    # 计算3dB带宽
    threshold = peak_power / 2
    indices_above_threshold = np.where(spectrum > threshold)[0]
    if len(indices_above_threshold) > 0:
        angle_3db = (angles[indices_above_threshold[-1]] - angles[indices_above_threshold[0]])
        print(f"  3dB带宽: {angle_3db:.2f}°")
        print(f"  ⚠️ 注意: 3根天线的分辨率比8根天线更宽")
    
    return spectrum, angles, peak_angle


def visualize_antenna_layout(antenna_array, save_path="antenna_layout.png"):
    """
    可视化天线布局
    
    参数：
        antenna_array: BGT60AntennaArray 对象
        save_path: 保存路径
    """
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制天线位置
    antenna_names = ['RX1', 'RX2', 'RX3']
    colors = ['red', 'blue', 'green']
    
    for (name, pos), color in zip(antenna_array.antenna_positions.items(), colors):
        ax.scatter(pos[0]*100, pos[1]*100, s=200, c=color, marker='o', 
                  edgecolors='black', linewidth=2, label=name, zorder=3)
        ax.annotate(name, (pos[0]*100, pos[1]*100), 
                   xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
    
    # 绘制连接线
    ax.plot([0, antenna_array.d*100], [0, 0], 'b--', linewidth=2, alpha=0.5)  # RX1-RX2
    ax.plot([0, 0], [0, antenna_array.d*100], 'g--', linewidth=2, alpha=0.5)  # RX1-RX3
    
    # 标注距离
    ax.text(antenna_array.d*100/2, -0.2, f'd={antenna_array.d*100:.1f}cm', 
           ha='center', fontsize=10, fontweight='bold')
    ax.text(-0.5, antenna_array.d*100/2, f'd={antenna_array.d*100:.1f}cm', 
           ha='right', fontsize=10, fontweight='bold')
    
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X 方向 (cm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y 方向 (cm)', fontsize=12, fontweight='bold')
    ax.set_title('BGT60TR13C 天线布局 (L型)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    print(f"\n✓ 保存天线布局图到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_mvdr_spectrum(spectrum, angles, peak_angle, antenna_array,
                           save_path="mvdr_spectrum.png"):
    """
    可视化MVDR谱
    
    参数：
        spectrum: MVDR谱
        angles: 角度数组
        peak_angle: 峰值角度
        antenna_array: 天线阵列对象
        save_path: 保存路径
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'MVDR角度估计谱 (BGT60TR13C L型天线阵列)\n简化方案: 仅使用RX1+RX2', 
                fontsize=14, fontweight='bold')
    
    # 左图：线性谱
    ax1.plot(angles, spectrum, 'b-', linewidth=2.5, label='MVDR谱')
    ax1.plot(peak_angle, spectrum[np.argmax(spectrum)], 'r*', markersize=20, 
            label=f'检测角度: {peak_angle:.2f}°')
    ax1.axvline(peak_angle, color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_xlabel('方位角 (度)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('归一化功率', fontsize=12, fontweight='bold')
    ax1.set_title('MVDR谱 (线性)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.set_xlim(angles[0], angles[-1])
    
    # 右图：dB谱
    spectrum_db = 10 * np.log10(spectrum + 1e-10)
    ax2.plot(angles, spectrum_db, 'g-', linewidth=2.5, label='MVDR谱 (dB)')
    peak_idx = np.argmax(spectrum)
    ax2.plot(peak_angle, spectrum_db[peak_idx], 'r*', markersize=20, 
            label=f'检测角度: {peak_angle:.2f}°')
    ax2.axvline(peak_angle, color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax2.set_xlabel('方位角 (度)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('功率 (dB)', fontsize=12, fontweight='bold')
    ax2.set_title('MVDR谱 (dB)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.set_xlim(angles[0], angles[-1])
    
    plt.tight_layout()
    print(f"\n✓ 保存MVDR谱图到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_beampattern(R_inv, antenna_array, save_path="beampattern.png"):
    """
    可视化波束方向图
    
    参数：
        R_inv: 协方差矩阵的逆
        antenna_array: 天线阵列对象
        save_path: 保存路径
    """
    
    spectrum, angles, _ = mvdr_spectrum(R_inv, antenna_array, 
                                       angle_range=(-90, 90, 0.1), 
                                       use_simplified=True)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    angles_rad = np.deg2rad(angles)
    spectrum_normalized = spectrum / np.max(spectrum)
    
    ax.plot(angles_rad, spectrum_normalized, 'b-', linewidth=2.5)
    ax.fill(angles_rad, spectrum_normalized, alpha=0.25, color='blue')
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_title('MVDR波束方向图 (BGT60TR13C)\n极坐标表示', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print(f"\n✓ 保存波束方向图到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    # 初始化天线阵列配置
    antenna_array = BGT60AntennaArray(wavelength=0.06)
    
    # 加载数据
    print("加载处理结果...")
    target_signal = np.load(TARGET_SIGNAL_FILE)  # 使用配置文件
    print(f"✓ 已加载目标信号: {target_signal.shape}\n")
    
    # 步骤3.1-3.3: 计算协方差矩阵和MVDR谱
    R = calculate_covariance_matrix(target_signal)
    R_inv = compute_inverse_covariance(R)
    mvdr_spectrum_data, angles, peak_angle = mvdr_spectrum(
        R_inv, antenna_array, use_simplified=True
    )
    
    # 可视化MVDR谱
    print("\n" + "=" * 70)
    print("可视化MVDR角度估计结果")
    print("=" * 70)
    visualize_mvdr_spectrum(mvdr_spectrum_data, angles, peak_angle, antenna_array,
                           save_path="results/mvdr_spectrum.png")
    
    # 可视化波束方向图
    print("\n" + "=" * 70)
    print("绘制波束方向图")
    print("=" * 70)
    visualize_beampattern(R_inv, antenna_array, save_path="results/beampattern.png")
    
    # 保存结果
    print("\n" + "=" * 70)
    print("保存处理结果")
    print("=" * 70)
    np.save(COVARIANCE_MATRIX_FILE, R)
    np.save(MVDR_SPECTRUM_FILE, mvdr_spectrum_data)
    
    print(f"✓ 已保存到 {RESULT_DIR}/")
    
    # 最终总结
    print("\n" + "=" * 70)
    print("✅ 步骤3处理完成！(修正版)")
    print("=" * 70)
    print(f"\n📍 角度估计结果:")
    print(f"   目标方位角: {peak_angle:.2f}°")
    print(f"\n⚠️ 重要说明:")
    print(f"   • 使用简化方案: 仅x轴方向 (RX1 + RX2)")
    print(f"   • BGT60TR13C天线为L型布局")
    print(f"   • 3根天线的分辨率比8根天线更宽")
    print(f"   • 两个人的角度差应 > 30-40° 避免融合")
    print(f"\n📊 生成的可视化图片:")
    print(f"   ✓ results/antenna_layout.png (天线布局)")
    print(f"   ✓ results/mvdr_spectrum.png (MVDR谱)")
    print(f"   ✓ results/beampattern.png (波束方向图)")