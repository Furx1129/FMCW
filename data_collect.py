import os  # 确保已导入os模块
import pprint
import numpy as np
from ifxradarsdk import get_version
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import create_dict_from_sequence,FmcwSequenceChirp,FmcwSimpleSequenceConfig

# 获取代码文件所在的绝对目录, # __file__是代码文件的完整路径
CODE_DIR = os.path.dirname(os.path.abspath(__file__))  
# 拼接Data和保存文件的路径（基于代码目录）
file_path = os.path.join(CODE_DIR, "Data", "test1.npy") #修改每次保存的文件名
# 自动创建目录
os.makedirs(os.path.dirname(file_path), exist_ok=True)

frames = 100  # 采集100帧数据

config = FmcwSimpleSequenceConfig(
    frame_repetition_time_s=0.05,        # 帧间隔 50ms → 帧率 20Hz
    chirp_repetition_time_s=0.00296,     # chirp间隔 ~3ms
    num_chirps=16,                        # 每帧16个chirp
    tdm_mimo=False,                       # 不启用MIMO
    chirp=FmcwSequenceChirp(
        start_frequency_Hz=58_000_000_000,  # 起始频率 58GHz
        end_frequency_Hz=63_000_000_000,    # 结束频率 63GHz → 带宽5GHz
        sample_rate_Hz=2_000_000,           # ADC采样率 2MHz
        num_samples=128,                    # 每个chirp采128个点
        rx_mask=7,                          # 0b111 = 启用RX1,RX2,RX3三个接收天线
        tx_mask=1,                          # 0b001 = 只启用TX1发射天线
        tx_power_level=31,                  # 发射功率等级(0-31)
        lp_cutoff_Hz=500_000,              # 低通滤波器截止频率 500kHz
        hp_cutoff_Hz=80_000,               # 高通滤波器截止频率 80kHz
        if_gain_dB=23,                      # 中频增益 23dB
        config = FmcwSimpleSequenceConfig(
        frame_repetition_time_s=0.05000000074505806,  # Frame repetition time 0.05s (frame rate of 20Hz)
        # chirp_repetition_time_s=0.0029607180040329695,  # Chirp repetition time (or pulse repetition time) of 0.5ms
        chirp_repetition_time_s = 0.0005,
        num_chirps = 16,  # chirps per frame
        tdm_mimo=False,  # MIMO disabled
    )
    )
)


raw_data = []  # 存储所有帧数据

with DeviceFmcw() as device:  # 创建设备连接（with语句自动关闭）
    print("Radar SDK Version: " + get_version())
    print("UUID of board: " + device.get_board_uuid())  # 硬件唯一ID
    print("Sensor: " + str(device.get_sensor_type()))   # 传感器型号
    
    # 创建并设置采集序列
    sequence_element = device.create_simple_sequence(config)
    device.set_acquisition_sequence(sequence_element)
    
    # 循环采集100帧
    for frame_number in range(frames):
        frame_contents = device.get_next_frame()  # 获取一帧数据
        raw_data.append(frame_contents)
        
    np.save(file_path, raw_data)  # 保存为.npy文件