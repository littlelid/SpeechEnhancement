import matplotlib.pyplot as plt
import numpy as np
from config import *

plt.rcParams["font.family"] = "Times New Roman"

def spectroplot(Z, N, hop, fs, filename, fdiv=None, vmin=None, vmax=None, cmap=None, interpolation='none', ):
    plt.imshow(
        20 * np.log10(np.abs(Z[:N // 2 + 1, :])),
        aspect='auto',
        origin='lower',
        vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)

    plt.ylabel('Freq / Hz', fontsize=18)
    yticks = plt.getp(plt.gca(), 'yticks')
    plt.setp(plt.gca(), 'yticklabels', np.array(np.round(yticks / float(N) * fs), dtype=np.int))
    print(np.array(np.round(yticks / float(N) * fs), dtype=np.int))

    plt.xlabel('Time / s', fontsize=18)
    xticks = plt.getp(plt.gca(), 'xticks')
    plt.setp(plt.gca(), 'xticklabels', xticks / float(fs) * hop)
    # plt.savefig(filename + ".pdf")


def plot_s(F, filename):
    spectroplot(F.T, fft_size + fft_zp, fft_hop, Fs, filename=filename,
                interpolation='none', cmap=plt.get_cmap('winter'))


def plot_t(signal, title=None):
    plt.plot(signal)
    plt.ylabel('Amplitude', fontsize=18)
    plt.xlabel('Time / s', fontsize=18)
    plt.ylim([-1, 1])
    xticks = plt.getp(plt.gca(), 'xticks')
    plt.setp(plt.gca(), 'xticklabels', xticks / Fs)



def add_db(org, gain_db=3):
    assert org != 0
    org = np.abs(org)
    return org * np.power(10, gain_db / 10)