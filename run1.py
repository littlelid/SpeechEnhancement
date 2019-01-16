import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

import IPython

import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("./3rdParty/pyroomacoustics/") # 用于仿真声场

from room import ShoeBox
from utils import hann, normalize, highpass, linear_2D_array
from stft import *
from beamforming import Beamformer
from config import *

from plot_util import *


path = './data/'
rate1, signal1 = wavfile.read(path + 'singing_'+str(Fs)+'.wav')
signal1 = np.array(signal1, dtype=float)
signal1 = normalize(signal1)

delay1 = 0.

# second signal is interferer
rate2, signal2 = wavfile.read(path + 'man1_'+str(Fs)+'.wav')
signal2 = signal2[1000:]
signal2 = signal2[:len(signal1)] ###
signal2 = np.array(signal2, dtype=float)
signal2 = normalize(signal2)
delay2 = 1.

sigma2_n = 5e-7
room_dim = [4, 6]
mvdr_snrs = []
for sig2 in [add_db(sigma2_n, i) for i in range(0, 30, 3)]:
    print(sig2)
    room1 = ShoeBox(
        room_dim,
        absorption=absorption,
        fs=Fs,
        t0=t0,
        max_order=max_order_sim,
        sigma2_awgn=sig2)

    good_source = np.array([1, 4.5])  # 目标信号
    normal_interferer = np.array([2.8, 4.3])  # 干扰信号
    room1.add_source(good_source, signal=signal1, delay=delay1)
    room1.add_source(normal_interferer, signal=signal2, delay=delay2)

    bf = Beamformer(R, Fs, N=N, Lg=Lg)
    room1.add_microphone_array(bf)
    room1.compute_rir()
    room1.simulate()

    mvdr_snr = bf.mvdr_beamformer(room1.sources[0][0:1],
                                  room1.sources[1][0:1],
                                  sig2 * np.eye(int(Lg) * int(M)), delay=delay)
    mvdr_snrs.append(mvdr_snr)

sigma2_n = 5e-7
room_dim = [4, 6]
msnr_snrs = []
for sig2 in [add_db(sigma2_n, i) for i in range(0, 30, 3)]:
    print(sig2)
    room1 = ShoeBox(
        room_dim,
        absorption=absorption,
        fs=Fs,
        t0=t0,
        max_order=max_order_sim,
        sigma2_awgn=sig2)

    good_source = np.array([1, 4.5])  # 目标信号
    normal_interferer = np.array([2.8, 4.3])  # 干扰信号
    room1.add_source(good_source, signal=signal1, delay=delay1)
    room1.add_source(normal_interferer, signal=signal2, delay=delay2)

    bf = Beamformer(R, Fs, N=N, Lg=Lg)
    room1.add_microphone_array(bf)
    room1.compute_rir()
    room1.simulate()

    msnr_snr = bf.max_sinr_filters(room1.sources[0].get_images(max_order=max_order_design),
                                   room1.sources[1].get_images(max_order=max_order_design),
                                   sig2 * np.eye(int(Lg) * int(M)))

    msnr_snrs.append(msnr_snr)

print("saving fig to ./temp/snr.pdf")
plt.rcParams["font.family"] = "Times New Roman"
plt.subplots(figsize=(6,4))
x = list(range(0,30,3))

plt.plot(x,10*np.log10(msnr_snrs) )
plt.plot(x, 10*np.log10(mvdr_snrs) )
plt.legend(["MSNR", "MVDR"], fontsize=18)
plt.xlabel("Noise Gain (dB)", fontsize=18)
plt.ylabel("Output SNR (dB)", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
#plt.ylim([5,52])
#plt.xticks([], x)
#ax1.set_xticklabels(['0', '0.05', '0.1', '0.2', '0.4', '0.6', '0.8', '1.0'],fontsize=20)
plt.tight_layout()
plt.savefig("")
plt.savefig("./temp/snr.pdf")