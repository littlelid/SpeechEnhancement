import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from plot_util import *
#import IPython

import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("./3rdParty/pyroomacoustics/") # 用于仿真声场

from room import ShoeBox
from utils import hann, normalize, highpass, linear_2D_array
from stft import *
# Spectrogram figure properties
figsize=(15, 7)        # figure size

from config import *

path = "./data/"
#target source
rate1, signal1 = wavfile.read(path + 'singing_'+str(Fs)+'.wav')
signal1 = np.array(signal1, dtype=float)
signal1 = normalize(signal1)
#signal1 = highpass(signal1, Fs) # 滤掉低于50HZ的语音信号
delay1 = 0.

# interference
rate2, signal2 = wavfile.read(path + 'man1_'+str(Fs)+'.wav')
signal2 = signal2[1000:]
signal2 = signal2[:len(signal1)] ###
signal2 = np.array(signal2, dtype=float)
signal2 = normalize(signal2)
#signal2 = highpass(signal2, Fs)  # 滤掉低于50HZ的语音信号
delay2 = 1.

room_dim = [4, 6]   # 4m X 6m 的房间
room1 = ShoeBox(
    room_dim,
    absorption=absorption,
    fs=Fs,
    t0=t0,
    max_order=max_order_sim,
    sigma2_awgn=sigma2_n)

# Add sources to room
good_source = np.array([1, 4.5])           # 目标信号
normal_interferer = np.array([2.8, 4.3])   # 干扰信号
room1.add_source(good_source, signal=signal1, delay=delay1)
room1.add_source(normal_interferer, signal=signal2, delay=delay2)

from beamforming import Beamformer
bf = Beamformer(R, Fs, N=N, Lg=Lg)
room1.add_microphone_array(bf)
room1.compute_rir()
room1.simulate()


# Delay and Sum
bf.delay_and_sum_weights(room1.sources[0][0:1],
                    room1.sources[1][0:1],
                    sigma2_n*np.eye(int(Lg)*int(M)) )
bf.filters_from_weights()
output_ds = bf.filter_out()
output_ds = normalize(highpass(output_ds, Fs))

# MSINR
bf.max_sinr_filters(room1.sources[0].get_images(max_order=max_order_design),
                    room1.sources[1].get_images(max_order=max_order_design),
                    sigma2_n*np.eye(int(Lg)*int(M)))
output_sinr = bf.filter_out()
output_sinr = normalize(highpass(output_sinr, Fs))


#MVDR
bf.mvdr_beamformer(room1.sources[0][0:1],
                    room1.sources[1][0:1],
                    sigma2_n*np.eye(int(Lg)*int(M) ), delay=delay)
output_mvdr = bf.filter_out()
output_mvdr = normalize(highpass(output_mvdr, Fs))


# 下面画的是报告中 图5：不同beamforming效果对比图

figsize=(28, 22)
fig, ax = plt.subplots(figsize=figsize, nrows=2, ncols=2)
ax = plt.subplot(5,2,2)
len_orign = len(signal1)
F1 = stft(signal1, fft_size, fft_hop,
          win=analysis_window,
          zp_back=fft_zp)
plot_s(F1, 'clean')
ax.set_title("Origin", fontsize=18)

ax = plt.subplot(5,2,4)
F2 = stft(normalize(highpass(bf.signals[bf.M//2], Fs))[:len_orign], fft_size, fft_hop,
          win=analysis_window,
          zp_back=fft_zp)
plot_s(F2, 'micro')
ax.set_title("Microphone", fontsize=18)

ax = plt.subplot(5,2,6)
F3 = stft(output_ds[:len_orign], fft_size, fft_hop,
          win=analysis_window,
          zp_back=fft_zp)
plot_s(F3, 'Delay and Sum')
ax.set_title("Delay and Sum", fontsize=18)

ax = plt.subplot(5,2,8)
F4 = stft(output_sinr[:len_orign], fft_size, fft_hop,
          win=analysis_window,
          zp_back=fft_zp)
plot_s(F4, 'MSNR')
ax.set_title("MSNR", fontsize=18)

ax = plt.subplot(5,2,10)
F5 = stft(output_mvdr[:len_orign], fft_size, fft_hop,
          win=analysis_window,
          zp_back=fft_zp)
plot_s(F5, 'MVDR')
ax.set_title("MVDR", fontsize=18)


ax = plt.subplot(5,2,1)
len_orign = len(signal1)
plot_t(signal1, 'clean')
ax.set_title("Origin", fontsize=18)

ax = plt.subplot(5,2,3)
plot_t(normalize(highpass(bf.signals[bf.M//2], Fs))[:len_orign], 'micro')
ax.set_title("Microphone", fontsize=18)

ax = plt.subplot(5,2,5)
plot_t(output_ds[:len_orign], 'Delay and Sum')
ax.set_title("Delay and Sum", fontsize=18)

ax = plt.subplot(5,2,7)
plot_t(output_sinr[:len_orign], 'MSNR')
ax.set_title("MSNR", fontsize=18)

ax = plt.subplot(5,2,9)
plot_t(output_mvdr[:len_orign], 'MVDR')
ax.set_title("MVDR", fontsize=18)

plt.tight_layout()

fig.savefig("./temp/time_spec.png")
wavfile.write("./temp/orginal.wav", Fs, signal1)
wavfile.write("./temp/interference.wav", Fs, signal2[:len_orign])
wavfile.write("./temp/MIC.wav", Fs, normalize(highpass(bf.signals[bf.M//2], Fs))[:len_orign])
wavfile.write("./temp/DS.wav", Fs, output_ds[:len_orign])
wavfile.write("./temp/MVDR.wav", Fs, output_mvdr[:len_orign])
wavfile.write("./temp/MSNR.wav", Fs, output_sinr[:len_orign])

