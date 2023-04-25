import numpy as np
from PyEMD import EMD , EEMD
from statsmodels.tsa.stattools import adfuller
import antropy as ant


class Decompose:
    def __init__(self,
                 time_series: np.ndarray,
                 ):
        #self.time_series = time_series
        #self.lin_trend = self._get_lin_trend(time_series)
        self.ts = time_series# - self.lin_trend
        self.modes_1 = self._get_modes(self.ts)
        self.trend = self._get_trend(self.ts, self.modes_1)
        self.de_trend_ts = self.ts - self.trend

        # self.modes_2 = self._get_modes(self.de_trend_ts)
        self.fractal = self._get_fractal(self.ts, self.modes_1)

        self.de_trend_de_fractal = self.de_trend_ts - self.fractal

    def get_decomposition(self, ):
        return np.vstack([self.trend, self.de_trend_de_fractal, self.fractal])  # .T

    @staticmethod
    def _get_modes(ts):
        time_vector = np.linspace(0, 1, len(ts))
        eemd = EMD()
        all_modes = eemd.emd(ts, time_vector)
        return all_modes

    @staticmethod
    def _get_lin_trend(x):
        y = np.arange(0, len(x))
        m_b = np.polyfit(y, x, 1)
        m, b = m_b[0], m_b[1]
        trend_hat = m * y + b
        return trend_hat
    @staticmethod
    def _get_trend(ts, modes):
        trend = np.zeros_like(ts)
        ad_test = adfuller(ts, autolag="AIC")

        for k in range(1, len(modes)):
            if ad_test[1] >= 0.05:
                trend += modes[-k]
                ad_test = adfuller(ts - trend, autolag="AIC")

            elif ad_test[1] < 0.05:
                return trend

        return trend

    @staticmethod
    def _get_fractal(ts, modes):
        fractal = np.zeros_like(ts)
        fractal_test = ant.higuchi_fd(ts)
        for k in range(len(modes)):
            if fractal_test >= 1.15:
                fractal += modes[k]
                fractal_test = ant.higuchi_fd(ts - fractal)

            elif fractal_test <= 1.15:
                return fractal

        return fractal

# import matplotlib.pyplot as plt
# import pandas as pd
# xdf = pd.read_csv('C:\\Users\\Administrator\\PycharmProjects\\pythonProject3\\data\\database\\D_gold.csv',
#                   index_col='time',
#                   parse_dates=True)
# a = 1001
# windowx = 300
# ohlc = xdf#.iloc[a * 3:a * 3 + windowx, :]
# x = ohlc.mean(axis=1).to_numpy()
# import time
# t_start = time.time()
# decom = Decompose(x)
# print(time.time() - t_start)
# print(decom.get_decomposition())
# print(decom.get_decomposition().shape)
#
#
#
#
# plt.figure(1)
#
# plt.plot(decom.trend)  # + decom.de_fractal)
# plt.plot(decom.trend + decom.de_trend_de_fractal)
# plt.plot(decom.trend + decom.de_trend_de_fractal + decom.fractal)
#
#
# plt.figure(2)
# plt.plot(decom.de_trend_de_fractal )
# plt.figure(3)
# plt.plot(decom.fractal)
#
# plt.show()
