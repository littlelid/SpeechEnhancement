import numpy as np
from numpy.lib.stride_tricks import as_strided as _as_strided
import sys

def stft(x, L, hop, transform=np.fft.fft, win=None, zp_back=0, zp_front=0):


    N = L + zp_back + zp_front

    if (win is not None and len(win) != N):
        print('Window length error.')
        sys.exit(-1)

    new_strides = (hop * x.strides[0], x.strides[0])
    new_shape = ((len(x) - L) // hop + 1, L)
    y =  _as_strided(x, shape=new_shape, strides=new_strides)
    y = np.concatenate(
            (
                np.zeros( (y.shape[0], zp_front), dtype=x.dtype),
                y,
                np.zeros( (y.shape[0], zp_back), dtype=x.dtype)
            ), axis=1)

    if (win is not None):
        y = win.astype(x.dtype) * y
    Z = transform(y, axis=1)

    return Z


def istft(X, L, hop, transform=np.fft.ifft, win=None, zp_back=0, zp_front=0):

    N = L + zp_back + zp_front

    if (win is not None and len(win) != N):
        print('Window length error.')
        sys.exit(-1)

    iX = transform(X, axis=1)
    if (iX.dtype == 'complex128'):
        iX = np.real(iX)

    if (win is not None):
        iX *= win

    # create output signal
    x = np.zeros(X.shape[0] * hop + (L - hop) + zp_back + zp_front)

    # overlap add
    for i in range(X.shape[0]):
        x[i * hop:i * hop + N] += iX[i]

    return x
