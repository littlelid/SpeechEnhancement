# -*- coding: utf-8 -*-

import numpy as np
import utils
from scipy.io import wavfile
import pyroomacoustics as pra

fft_size = 512
fft_hop  = 8           # step size
fft_zp = 512           # zero padding
analysis_window = np.concatenate((utils.hann(fft_size), np.zeros(fft_zp)))
t_cut = 0.83           # length in [s] to remove at end of signal (no sound)

Fs = 8000
t0 = 1./(Fs*np.pi*1e-2)
absorption = 0.1  #模拟
max_order_sim = 2
sigma2_n = 5e-7


# Microphone array design parameters
mic1 = np.array([2, 1.5])   # position
M = 8                       # number of microphones
d = 0.08                    # distance between microphones
phi = 0.                    # angle from horizontal
max_order_design = 1        # maximum image generation used in design
Lg_t = 0.100                # Filter size in seconds
Lg = np.ceil(Lg_t*Fs)       # Filter size in samples
delay = 0.050               # Beamformer delay in seconds

# Define the FFT length
N = 1024

# Create a microphone array
R = pra.linear_2D_array(mic1, M, phi, d)

rate1, signal1 = wavfile.read('data/singing_8000.wav')
signal1 = np.array(signal1, dtype=float)
signal1 = pra.normalize(signal1)
signal1 = pra.highpass(signal1, Fs) # 滤掉低于50HZ的语音信号
delay1 = 0

rate2, signal2 = wavfile.read('data/german_speech_8000.wav')
signal2 = np.array(signal2, dtype=float)
signal2 = pra.normalize(signal2)
signal2 = pra.highpass(signal2, Fs)  # 滤掉低于50HZ的语音信号
delay2 = 1.

room_dim = [4, 6]   # 4m X 6m 的房间
room1 = pra.ShoeBox(
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

mics = pra.Beamformer(R, Fs, N=N, Lg=Lg)
room1.add_microphone_array(mics)
room1.compute_rir()
room1.simulate()

mics.rake_delay_and_sum_weights(room1.sources[0][0:1],
                    room1.sources[1][0:1],
                    sigma2_n*np.eye(mics.Lg*mics.M), delay=delay)

output = mics.process()

input_mic = pra.normalize(pra.highpass(mics.signals[mics.M//2], Fs))
out_delay_and_sum = pra.normalize(pra.highpass(output, Fs))
wavfile.write('data/output/1.wav', Fs, out_DirectMVDR)
wavfile.write('data/output/2.wav', Fs, out_DirectMVDR)
