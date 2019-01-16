import numpy as np
from utils import hann, linear_2D_array


fft_size = 512         # fft size
fft_hop  = 8           # hop size in stft
fft_zp = 512           # zero padding in stft
analysis_window = np.concatenate((hann(fft_size), np.zeros(fft_zp)))
t_cut = 0.83           # length in [s] 用于消除无信号的片段

# 仿真参数
Fs = 8000
t0 = 1./(Fs*np.pi*1e-2)  # 开始时间
absorption = 0.1         # 墙面的吸收程度 0-1, 越小表述多径效应越明显
max_order_sim = 2
sigma2_n = 5e-7         # 环境噪声

# 麦克风阵列的几何结构
mic1 = np.array([2, 1.5])   # 坐标
M = 8                       # 麦克风数量
d = 0.08                    # 麦克风间隔 8cm
phi = 0.                    # 摆放角度为水平
max_order_design = 1        # 仿真相关的参数

Lg_t = 0.100                # 滤波器的窗口长度 时域上为0.1s
Lg = np.ceil(Lg_t*Fs)       # 滤波器的窗口长度 具体的阶数
delay = 0.050               # 仿真beamforming的延迟


N = 1024

R = linear_2D_array(mic1, M, phi, d) # 生成LUA形状麦克风阵列的坐标位置