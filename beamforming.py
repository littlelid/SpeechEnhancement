import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from utils import build_rir_matrix, distance, constants
from scipy.signal import fftconvolve

def H(A, **kwargs):
    '''Hermitian'''
    return np.transpose(A, **kwargs).conj()

class MicrophoneArray(object):
    '''Microphone array class.'''

    def __init__(self, R, fs):

        R = np.array(R)
        self.dim = R.shape[0] # 只考虑2维平面, 所以目前 dim=2
        self.M = R.shape[1]  # 麦克风数量
        self.R = R  # 噪声自相关函数， 暂时只考虑高斯白噪声
        self.fs = fs  # sampling frequency
        self.signals = None

        self.center = np.mean(R, axis=1, keepdims=True)

    def record(self, signals, fs):
        '''定义接口,给仿真程序调用，仿真程序会麦克模拟声音,并保存在self.signal'''
        if fs != self.fs:
            try:
                import samplerate

                fs_ratio = self.fs / float(fs)
                newL = int(fs_ratio * signals.shape[1]) - 1
                self.signals = np.zeros((self.M, newL))
                # samplerate resample function considers columns as channels (hence the transpose)
                for m in range(self.M):
                    self.signals[m] = samplerate.resample(signals[m], fs_ratio, 'sinc_best')
            except ImportError:
                return ImportError("To Be Done")
        else:
            self.signals = signals

class Beamformer(MicrophoneArray):
    '''
    Beamformer类, 继承MicrophoneArray
    ----------
    R: numpy.ndarray
        麦克风位置
    fs: int
        采样频率
    N: int
        FFT 个数
    '''

    def __init__(self, R, fs, N=1024, Lg=None):
        MicrophoneArray.__init__(self, R, fs)

        assert N % 2 == 0

        self.N = int(N)  # FFT length
        if Lg is None:
            self.Lg = N  # TD filters length
        else:
            self.Lg = int(Lg)

        self.frequencies = np.arange(0, self.N // 2 + 1) / self.N * float(self.fs)
        self.filters = None # time domain filters (M x Lg）

    def filter_out(self, FD=False):
        output = fftconvolve(self.filters[0], self.signals[0])
        for i in range(1, len(self.signals)):
            output += fftconvolve(self.filters[i], self.signals[i])

        return output


    def mvdr_beamformer(self, source, interferer, R_n, delay=0.03, epsilon=5e-3):
        '''
        计算MVDR的滤波器
        '''

        H = build_rir_matrix(self.R, (source, interferer), self.Lg, self.fs, epsilon=epsilon, unit_damping=True)
        L = H.shape[1] // 2

        # the constraint vector
        kappa = int(delay * self.fs)
        h = H[:, kappa]

        # We first assume the sample are uncorrelated
        R_xx = np.dot(H[:, :L], H[:, :L].T)
        K_nq = np.dot(H[:, L:], H[:, L:].T) + R_n

        # Compute the TD filters
        C = la.cho_factor(R_xx + K_nq, check_finite=False)
        g_val = la.cho_solve(C, h)  #现在 g_val  = K^{-1} h

        g_val /= np.inner(h, g_val) #现在  g_val  = (K^{-1} h) / (h^H K^{-1} h))
        self.filters = g_val.reshape((self.M, self.Lg))

        # compute and return SNR
        sig = np.inner(g_val.T, np.dot(R_xx, g_val))
        noise = np.inner(np.dot(g_val.T, K_nq), g_val)

        return sig / noise

    def get_Rx_and_Knq(self, source, interferer, R_n, delay=0.03, epsilon=5e-3):
        '''
        计算信号相关函数 用于估算SNR
        '''

        H = build_rir_matrix(self.R, (source, interferer), self.Lg, self.fs, epsilon=epsilon, unit_damping=True)
        L = H.shape[1] // 2

        R_xx = np.dot(H[:, :L], H[:, :L].T)
        K_nq = np.dot(H[:, L:], H[:, L:].T) + R_n
        return R_xx, K_nq


    def filters_from_weights(self, non_causal=0.):
        '''
            将频域的滤波器转化成时域的滤波器
        '''

        if self.weights is None:
            raise TypeError('Weights null.')

        self.filters = np.zeros((self.M, self.Lg))

        if self.N <= self.Lg:

            # go back to time domain and shift DC to center
            tw = np.fft.irfft(np.conj(self.weights), axis=1, n=self.N)
            self.filters[:, :self.N] = np.concatenate((tw[:, -self.N // 2:], tw[:, :self.N // 2]), axis=1)

        elif self.N > self.Lg:

            # Least-square projection
            for i in np.arange(self.M):
                Lgp = np.floor((1 - non_causal) * self.Lg)
                Lgm = self.Lg - Lgp
                # the beamforming weights in frequency are the complex conjugates of the FT of the filter
                w = np.concatenate((np.conj(self.weights[i]), self.weights[i, -2:0:-1]))

                # create partial Fourier matrix
                k = np.arange(self.N)[:, np.newaxis]
                l = np.concatenate((np.arange(self.N - Lgm, self.N), np.arange(Lgp)))
                F = np.exp(-2j * np.pi * k * l / self.N)

                self.filters[i] = np.real(np.linalg.lstsq(F, w)[0])


    def steering_vector_2D_from_point(self, frequency, source, attn=True, ff=False):
        '''
        计算steering vector, 用于DSbeamforming
        '''

        X = np.array(source)
        if X.ndim == 1:
            X = source[:, np.newaxis]

        omega = 2 * np.pi * frequency

        if (ff):
            p = (X - self.center)
            p /= np.linalg.norm(p)
            D = np.dot(self.R.T, p)
            D -= np.min(D)
        else:

            D = distance(self.R, X)

        phase = np.exp(-1j * omega * D / constants.get('c'))

        if attn:
            return 1. / (4 * np.pi) / D * phase
        else:
            return phase

    def delay_and_sum_weights(self, source, interferer=None, R_n=None, attn=True, ff=False):
        '''
        计算DS beamforming的滤波器
        '''
        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        K = source.images.shape[1] - 1
        #K = 1
        for i, f in enumerate(self.frequencies):
            W = self.steering_vector_2D_from_point(f, source.images, attn=attn, ff=ff)
            self.weights[:, i] = 1.0 / self.M / (K + 1) * np.sum(W, axis=1)

    def rake_max_sinr_filters(self, source, interferer, R_n, epsilon=5e-3, delay=0.):
        '''
        '''

        H = build_rir_matrix(self.R, (source, interferer), self.Lg, self.fs, epsilon=epsilon, unit_damping=True)
        L = H.shape[1] / 2


        K_s = np.dot(H[:, :L], H[:, :L].T)
        K_nq = np.dot(H[:, L:], H[:, L:].T) + R_n

        # 本质上就是generalized Rayleigh问题，直接特征值分解即可~
        SINR, v = la.eigh(K_s, b=K_nq, eigvals=(self.M * self.Lg - 1, self.M * self.Lg - 1), overwrite_a=True,
                          overwrite_b=True, check_finite=False)
        g_val = np.real(v[:, 0])

        self.filters = g_val.reshape((self.M, self.Lg))


        return SINR[0]
