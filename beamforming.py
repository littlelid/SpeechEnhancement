import numpy as np
import scipy.linalg as la

from utils import build_rir_matrix, distance
from scipy.signal import fftconvolve

def H(A, **kwargs):
    '''Hermitian'''
    return np.transpose(A, **kwargs).conj()

class MicrophoneArray(object):
    '''Microphone array class.'''

    def __init__(self, R, fs):

        R = np.array(R)
        self.dim = R.shape[0]
        self.M = R.shape[1]  # number of microphones
        self.R = R  # array geometry
        self.fs = fs  # sampling frequency of microphones
        self.signals = None

        self.center = np.mean(R, axis=1, keepdims=True)

    def record(self, signals, fs):
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
                raise ImportError('The samplerate package must be installed for resampling of the signals.')

        else:
            self.signals = signals

class Beamformer(MicrophoneArray):
    '''
    At some point, in some nice way, the design methods
    should also go here. Probably with generic arguments.
    Parameters
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
        Compute the time-domain filters of the minimum variance distortionless
        response beamformer.
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
        g_val = la.cho_solve(C, h)

        g_val /= np.inner(h, g_val)
        self.filters = g_val.reshape((self.M, self.Lg))

        # compute and return SNR
        num = np.inner(g_val.T, np.dot(R_xx, g_val))
        denom = np.inner(np.dot(g_val.T, K_nq), g_val)

        return num / denom


    def filters_from_weights(self, non_causal=0.):
        '''
        Compute time-domain filters from frequency domain weights.

        Parameters
        ----------
        non_causal: float, optional
            ratio of filter coefficients used for non-causal part
        '''

        if self.weights is None:
            raise NameError('Weights must be defined.')

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
