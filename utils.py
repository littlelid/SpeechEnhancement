import numpy as np
import scipy
from scipy.signal import iirfilter, lfilter, freqz

_constants = {}
_constants_default = {
    'c': 343.0,
    'ffdist': 10.,
    'fc_hp': 300.,
    'frac_delay_length': 81,
}

class Constants:
    def set(self, name, val):
        # add constant to dictionnary
        _constants[name] = val

    def get(self, name):

        try:
            v = _constants[name]
        except KeyError:
            try:
                v = _constants_default[name]
            except KeyError:
                raise NameError(name + ': no such constant')

        return v
constants = Constants()

def highpass(signal, Fs):
    '''高通滤波器'''

    fc = 300
    rp = 5  # minimum ripple in dB in pass-band
    rs = 60   # minimum attenuation in dB in stop-band
    n = 4    # order of the filter

    # normalized cut-off frequency
    wc = 2. * fc / Fs
    #import numpy as np
    #import scipy
    from scipy.signal import iirfilter, lfilter, freqz
    b, a = iirfilter(n, Wn=wc, rp=rp, rs=rs, btype='highpass', ftype='butter')

    # apply
    out = lfilter(b, a, signal.copy())
    return out

def normalize(signal):
    '''归一化'''
    return signal / np.max(np.abs(signal))

def hann(N):
    ''' STFT 窗口函数 hann window '''

    t = np.arange(0, N)
    t = t / float(N)
    w = 0.5 * (1 - np.cos(2 * np.pi * t))
    return w

def distance(x, y):
    '''
    计算两点之间距离.

    x and y are DxN ndarray containing N D-dimensional vectors.
    '''

    # Assume x, y are arrays, *not* matrices
    x = np.array(x)
    y = np.array(y)

    # return np.sqrt((x[0,:,np.newaxis]-y[0,:])**2 + (x[1,:,np.newaxis]-y[1,:])**2)

    return np.sqrt(np.sum((x[:, :, np.newaxis] - y[:, np.newaxis, :]) ** 2, axis=0))

def convmtx(x, n):

    c = np.concatenate((x, np.zeros(n-1)))
    r = np.zeros(n)
    return scipy.linalg.toeplitz(c, r)

def low_pass_dirac(t0, alpha, Fs, N):
    return alpha*np.sinc(np.arange(N) - Fs*t0)

def build_rir_matrix(mics, sources, Lg, Fs, epsilon=5e-3, unit_damping=False):
    '''
    计算 Room Impulse Response
    '''
    d_min = np.inf
    d_max = 0.
    dmp_max = 0.
    for s in range(len(sources)):
        dist_mat = distance(mics, sources[s].images)
        if unit_damping is True:
            dmp_max = np.maximum((1. / (4 * np.pi * dist_mat)).max(), dmp_max)
        else:
            dmp_max = np.maximum((sources[s].damping[np.newaxis, :] / (4 * np.pi * dist_mat)).max(), dmp_max)
        d_min = np.minimum(dist_mat.min(), d_min)
        d_max = np.maximum(dist_mat.max(), d_max)

    t_max = d_max / constants.get('c')
    t_min = d_min / constants.get('c')

    offset = dmp_max / (np.pi * Fs * epsilon)

    # RIR length
    Lh = int((t_max - t_min + 2 * offset) * float(Fs))

    # build the channel matrix
    L = Lg + Lh - 1
    H = np.zeros((Lg * mics.shape[1], len(sources) * L))

    for s in range(len(sources)):
        for r in np.arange(mics.shape[1]):

            dist = sources[s].distance(mics[:, r])
            time = dist / constants.get('c') - t_min + offset
            if unit_damping == True:
                dmp = 1. / (4 * np.pi * dist)
            else:
                dmp = sources[s].damping / (4 * np.pi * dist)

            h = low_pass_dirac(time[:, np.newaxis], dmp[:, np.newaxis], Fs, Lh).sum(axis=0)
            H[r * Lg:(r + 1) * Lg, s * L:(s + 1) * L] = convmtx(h, Lg).T
    return H

def unit_vec2D(phi):
    '''用于构建麦克风阵列,计算方向'''

    return np.array([[np.cos(phi), np.sin(phi)]]).T

def linear_2D_array(center, M, phi, d):
    '''
    计算二维平面内麦克风阵列的坐标点
    :param center: array
        中心麦克风的坐标
    :param M: int
        number of microphone
    :param phi: float
        摆放角度
    :param d:  float
        间距

    :return: ndarray (2, M)
        M个麦克风的坐标点
    '''

    u = unit_vec2D(phi)
    return np.array(center)[:, np.newaxis] + d * \
           (np.arange(M)[np.newaxis, :] - (M - 1.) / 2.) * u

