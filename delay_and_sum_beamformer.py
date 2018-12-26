# -*- coding: utf-8 -*-

import numpy as np
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