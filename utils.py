import numpy as np
import scipy
from scipy.signal import iirfilter, lfilter, freqz

def highpass(siganl, Fs):
    '''高通滤波器'''

    fc = 300
    rp = 5  # minimum ripple in dB in pass-band
    rs = 60   # minimum attenuation in dB in stop-band
    n = 4    # order of the filter

    # normalized cut-off frequency
    wc = 2. * fc / Fs

    # define
    b, a = iirfilter(n, Wn=wc, rp=rp, rs=rs, btype='highpass', ftype='butter')

    # apply
    out = lfilter(b, a, signal.copy())
    return out

def normalize(signal):
    return signal / np.max(np.abs(signal))


def hann(N):
    ''' STFT 窗口函数 hann window '''

    t = np.arange(0, N)
    t = t / float(N)
    w = 0.5 * (1 - np.cos(2 * np.pi * t))
    return w