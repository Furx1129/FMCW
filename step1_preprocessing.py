import numpy as np
import matplotlib.pyplot as plt
from config import *  # 导入配置文件

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 步骤 1：数据整形与预处理 (Data Shaping & MTI)
# ============================================================================

def load_and_preprocess_data(npy_file_path):
    """
    加载原始.npy数据并进行MTI预处理
    
    输入：
        npy_file_path: .npy文件路径
    
    输出：
        data_mti_reshaped: MTI处理后的整形数据 (frames, samples, rx)
    """
    
    # 1. 加载数据
    print("=" * 70)
    print("步骤1：加载原始数据")
    print("=" * 70)
    raw_data = np.load(npy_file_path)
    print(f"✓ 数据加载成功")
    print(f"  原始形状: {raw_data.shape}")
    print(f"  数据类型: {raw_data.dtype}")
    print(f"  取值范围: [{raw_data.min():.6f}, {raw_data.max():.6f}]")
    
    # 2. 数据整形
    print("\n" + "=" * 70)
    print("步骤1.2：数据整形")
    print("=" * 70)
    
    # 根据实际维度处理
    if raw_data.ndim == 4:
        frames, rx, chirps, samples = raw_data.shape
        print(f"  帧数(Slow Time): {frames}")
        print(f"  接收天线(RX): {rx}")
        print(f"  每帧chirp数: {chirps}")
        print(f"  每chirp样本数: {samples}")
        print(f"  数据维度: (frames, rx, chirps, samples)")
    else:
        frames, tx, rx, chirps, samples = raw_data.shape
        print(f"  帧数(Slow Time): {frames}")
        print(f"  发射天线(TX): {tx}")
        print(f"  接收天线(RX): {rx}")
        print(f"  每帧chirp数: {chirps}")
        print(f"  每chirp样本数: {samples}")
        raw_data = raw_data.squeeze(axis=1)  # 删除TX维度（如果只有1个TX）
        frames, rx, chirps, samples = raw_data.shape
    
    data_reshaped = raw_data.reshape(frames, rx, chirps * samples)
    data_reshaped = np.transpose(data_reshaped, (0, 2, 1))
    
    print(f"✓ 整形完成: {data_reshaped.shape}")
    print(f"  形式: (帧数=Slow Time, 采样点数=Fast Time, 接收天线)") 
    
    # 3. MTI处理 (去静止杂波)
    print("\n" + "=" * 70)
    print("步骤1.3：去静止杂波处理 (MTI)")
    print("=" * 70)
    
    mean_clutter = np.mean(data_reshaped, axis=0, keepdims=True)
    data_mti_reshaped = data_reshaped - mean_clutter
    
    print(f"✓ MTI处理完成")
    print(f"  原始功率: {np.mean(data_reshaped**2):.6f}")
    print(f"  MTI后功率: {np.mean(data_mti_reshaped**2):.6f}")
    suppression_ratio = (1 - np.mean(data_mti_reshaped**2) / np.mean(data_reshaped**2)) * 100
    print(f"  杂波抑制比: {suppression_ratio:.2f}%")
    
    return data_mti_reshaped, data_reshaped


def visualize_mti_effect(data_original, data_mti, channel=0, frame=0, save_path="mti_effect.png"):
    """
    可视化MTI去杂波效果
    
    参数：
        data_original: 整形后的原始数据 (frames, samples, rx)
        data_mti: MTI处理后的数据 (frames, samples, rx)
        channel: 要显示的接收通道 (0, 1, 2)
        frame: 要显示的帧数
        save_path: 保存图片的路径
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'MTI去杂波效果对比 (第{frame}帧, RX{channel+1}通道)', fontsize=14, fontweight='bold')
    
    # 1. 单帧原始数据
    ax = axes[0, 0]
    ax.plot(data_original[frame, :, channel], linewidth=1)
    ax.set_title('原始数据（单帧）', fontsize=12)
    ax.set_xlabel('采样点索引', fontsize=11)
    ax.set_ylabel('幅度', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. 单帧MTI处理后数据
    ax = axes[0, 1]
    ax.plot(data_mti[frame, :, channel], linewidth=1, color='orange')
    ax.set_title('MTI处理后（单帧）', fontsize=12)
    ax.set_xlabel('采样点索引', fontsize=11)
    ax.set_ylabel('幅度', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 3. 所有帧的原始数据热力图
    ax = axes[1, 0]
    im1 = ax.imshow(data_original[:, :, channel].T, aspect='auto', cmap='viridis', origin='lower')
    ax.set_title('原始数据热力图（所有帧）', fontsize=12)
    ax.set_xlabel('帧索引', fontsize=11)
    ax.set_ylabel('采样点索引', fontsize=11)
    cbar1 = plt.colorbar(im1, ax=ax)
    cbar1.set_label('幅度', fontsize=10)
    
    # 4. 所有帧的MTI处理后热力图
    ax = axes[1, 1]
    im2 = ax.imshow(data_mti[:, :, channel].T, aspect='auto', cmap='viridis', origin='lower')
    ax.set_title('MTI处理后热力图（所有帧）', fontsize=12)
    ax.set_xlabel('帧索引', fontsize=11)
    ax.set_ylabel('采样点索引', fontsize=11)
    cbar2 = plt.colorbar(im2, ax=ax)
    cbar2.set_label('幅度', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    print(f"\n✓ 保存MTI效果对比图到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 主程序

if __name__ == "__main__":
    print("=" * 70)
    print("开始处理雷达信号 - 步骤1：数据预处理与MTI去杂波")
    print("=" * 70 + "\n")
    
    # 使用配置文件中定义的路径
    npy_path = INPUT_FILE  # 从config.py读取
    
    print(f"✓ 使用配置文件中的数据路径: {npy_path}\n")
    
    data_mti_reshaped, data_original_reshaped = load_and_preprocess_data(npy_path)
    
    # 可视化MTI效果
    print("\n" + "=" * 70)
    print("可视化MTI去杂波效果")
    print("=" * 70)
    visualize_mti_effect(data_original_reshaped, data_mti_reshaped, channel=0, frame=0, 
                        save_path=get_image_path("mti_effect_rx1.png"))
    
    visualize_mti_effect(data_original_reshaped, data_mti_reshaped, channel=1, frame=0, 
                        save_path=get_image_path("mti_effect_rx2.png"))
    
    visualize_mti_effect(data_original_reshaped, data_mti_reshaped, channel=2, frame=0, 
                        save_path=get_image_path("mti_effect_rx3.png"))
    
    # 保存处理结果
    print("\n" + "=" * 70)
    print("保存处理结果")
    print("=" * 70)
    np.save(MTI_RESHAPED_FILE, data_mti_reshaped)  # 使用config.py中定义的路径
    print(f"✓ mti_reshaped.npy (MTI处理后的数据)")
    
    print(f"\n✓ 步骤1处理完成！")
    print(f"✓ 生成的文件已保存到 {RESULT_DIR}/ 文件夹")