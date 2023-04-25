import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfreqz
import matplotlib.pyplot as plt
from scipy import signal


def low_pass_filter(xs, alpha=0.8, order=5):
    # low_cut = band[0] / (len(xs) / 2)

    xs = np.concatenate((xs, xs[::-1]))
    sig = signal.butter(N=order, Wn=alpha, btype='lowpass')
    b = sig[0]
    a = sig[1]
    y1 = signal.filtfilt(b, a, xs)

    xs = y1[:int(len(xs) / 2)]
    return xs


xdf = pd.read_csv('15_gold.csv',
                  index_col='time',
                  parse_dates=True)
print(xdf)
x = xdf.close.to_numpy()
# x_s = low_pass_filter(x, 0.0001, 4)
plt.figure(1)
plt.plot(x)
# x_1 = x - x_s
x_s = low_pass_filter(x, 0.002, 4)
plt.plot(x_s)

plt.figure(2)
plt.plot( x - x_s)

plt.show()
