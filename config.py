"""
全局配置文件 - 管理所有数据和结果路径
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
import warnings

# ============================================================================
# Matplotlib 字体配置 - 在最开始处理（防止字体警告）
# ============================================================================

# 忽略所有字体和字形相关的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Glyph.*missing.*')
warnings.filterwarnings('ignore', message='.*does not have a glyph.*')
warnings.filterwarnings('ignore', message='.*Substituting.*')

# 使用 Agg 后端（避免 Tkinter 问题）
try:
    matplotlib.use('Agg')
except:
    pass

# 尝试使用 Windows 内置的 Microsoft YaHei 字体
try:
    import matplotlib.font_manager as fm
    font_path = r'C:\Windows\Fonts\msyh.ttc'  # Microsoft YaHei
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'DejaVu Sans']
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

# 关键配置 - 解决 Unicode 减号警告
plt.rcParams['axes.unicode_minus'] = False              # 禁用 Unicode 减号
plt.rcParams['mathtext.fontset'] = 'dejavusans'        # 数学文本字体
plt.rcParams['mathtext.default'] = 'regular'           # 数学文本样式
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# 数据配置
# ============================================================================

# 原始数据文件名和路径
DATA_NAME = "test_1000_1m"
INPUT_FILE = f"{DATA_NAME}.npy"  # 原始数据文件

# 结果文件夹路径
RESULT_BASE_DIR = "results"
RESULT_DIR = os.path.join(RESULT_BASE_DIR, DATA_NAME)

# 确保结果文件夹存在
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================================
# 文件路径定义（步骤1）
# ============================================================================
MTI_RESHAPED_FILE = os.path.join(RESULT_DIR, "mti_reshaped.npy")
DATA_ORIGINAL_FILE = os.path.join(RESULT_DIR, "data_original_reshaped.npy")

# ============================================================================
# 文件路径定义（步骤2）
# ============================================================================
TARGET_SIGNAL_FILE = os.path.join(RESULT_DIR, "target_signal.npy")
RANGE_SPECTRUM_FILE = os.path.join(RESULT_DIR, "range_spectrum.npy")
RANGE_FFT_FILE = os.path.join(RESULT_DIR, "range_fft.npy")

# ============================================================================
# 文件路径定义（步骤3）
# ============================================================================
COVARIANCE_MATRIX_FILE = os.path.join(RESULT_DIR, "covariance_matrix.npy")
COVARIANCE_MATRIX_INV_FILE = os.path.join(RESULT_DIR, "covariance_matrix_inv.npy")
MVDR_SPECTRUM_FILE = os.path.join(RESULT_DIR, "mvdr_spectrum.npy")

# ============================================================================
# 文件路径定义（步骤4）
# ============================================================================
BEAMFORMING_WEIGHTS_FILE = os.path.join(RESULT_DIR, "beamforming_weights.npy")
BEAMFORMED_SIGNAL_FILE = os.path.join(RESULT_DIR, "beamformed_signal.npy")

# ============================================================================
# 文件路径定义（步骤5）
# ============================================================================
PHASE_WRAPPED_FILE = os.path.join(RESULT_DIR, "phase_wrapped.npy")
PHASE_UNWRAPPED_FILE = os.path.join(RESULT_DIR, "phase_unwrapped.npy")
PHASE_DIFF_FILE = os.path.join(RESULT_DIR, "phase_diff.npy")
RESPIRATION_FILE = os.path.join(RESULT_DIR, "respiration.npy")
HEARTBEAT_FILE = os.path.join(RESULT_DIR, "heartbeat.npy")

# ============================================================================
# 图片保存路径
# ============================================================================
def get_image_path(filename):
    """获取图片保存路径"""
    return os.path.join(RESULT_DIR, filename)

# ============================================================================
# 硬件和信号处理参数
# ============================================================================

# BGT60TR13C 雷达参数
RADAR_FREQ = 5e9  # 5 GHz
SPEED_OF_LIGHT = 3e8
WAVELENGTH = SPEED_OF_LIGHT / RADAR_FREQ  # 0.06 m

# 天线配置
NUM_RX_ANTENNAS = 3
NUM_TX_ANTENNAS = 1

# 帧率和时间参数
FRAME_RATE = 10.0  # Hz
FRAME_PERIOD = 1.0 / FRAME_RATE  # 秒

# 生命体征参数
RESPIRATION_FREQ_RANGE = (0.2, 0.5)  # Hz (12-30 breaths/min)
HEARTBEAT_FREQ_RANGE = (0.8, 2.5)     # Hz (48-150 bpm)

# ============================================================================
# 打印配置信息
# ============================================================================
print(f"""
╔════════════════════════════════════════════════════════════════╗
║                   雷达信号处理配置信息                          ║
╚════════════════════════════════════════════════════════════════╝

📊 数据配置:
   • 数据名称: {DATA_NAME}
   • 输入文件: {INPUT_FILE}
   • 结果文件夹: {RESULT_DIR}/

📡 硬件参数:
   • 工作频率: {RADAR_FREQ/1e9:.1f} GHz
   • 波长: {WAVELENGTH*1000:.1f} mm
   • RX天线数: {NUM_RX_ANTENNAS}
   • TX天线数: {NUM_TX_ANTENNAS}

⏱️  时间参数:
   • 帧率: {FRAME_RATE} Hz
   • 帧周期: {FRAME_PERIOD*1000:.1f} ms

💓 生命体征参数:
   • 呼吸频率范围: {RESPIRATION_FREQ_RANGE[0]:.2f}-{RESPIRATION_FREQ_RANGE[1]:.2f} Hz
   • 心跳频率范围: {HEARTBEAT_FREQ_RANGE[0]:.2f}-{HEARTBEAT_FREQ_RANGE[1]:.2f} Hz
""")