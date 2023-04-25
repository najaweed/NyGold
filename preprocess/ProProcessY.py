import pandas as pd
from preprocess.Decompose import Decompose
import numpy as np


class PreProcess:
    def __init__(self,
                 df: pd.DataFrame,
                 step_prediction: int = 1,
                 step_share: int = 0,
                 ):
        self.df = df[['open', 'high', 'low', 'close']]
        self.step_prediction = step_prediction
        self.step_share = step_share

        self.raw_obs_df, self.raw_target_df = self._split_obs_target()
        self.c_normal = Normalizer({})
        self.normal_obs, self.normal_target = self.normal_obs_target()

    def _split_obs_target(self):
        obs = self.df.iloc[:-self.step_prediction, :]
        target = self.df.iloc[-(self.step_share + self.step_prediction):, :]
        return obs, target

    def obs(self):
        ts = self.normal_obs
        return Decompose(ts).get_decomposition()

    def target(self):
        return self.normal_target

    def normal_obs_target(self, ):
        _obs = self.raw_obs_df.mean(axis=1).to_numpy()
        _target = self.raw_target_df.mean(axis=1).to_numpy()
        self.c_normal.fit_transform(_obs)
        obs = self.c_normal.transform(_obs)
        target = self.c_normal.transform(_target)
        return obs, target


class Normalizer:
    def __init__(self,
                 config,
                 ):
        self.config = config
        self.scale = None
        self.drift = None

    @staticmethod
    def _normalizer(x):
        scale = (x.max() - x.min())
        drift = x.min()
        return (x - drift) / scale, scale, drift

    def fit_transform(self, x):
        x, scale, drift = self._normalizer(x)
        self.scale = scale
        self.drift = drift

    def transform(self, x):
        return (x - self.drift) / self.scale

    def inverse_transform(self, x):
        return x * self.scale + self.drift

#
# import matplotlib.pyplot as plt
#
# xdf = pd.read_csv('C:\\Users\\Administrator\\PycharmProjects\\pythonProject3\\data\\database\\D_gold.csv',
#                   index_col='time',
#                   parse_dates=True)
# a = 111
# windowx = 300
# ohlc = xdf.iloc[a * 3:a * 3 + windowx, :]
#
# pre = PreProcess(ohlc, step_prediction=20, step_share=3)
#
#
# # print(pre.c_normal.inverse_transform(pre.target()))
#
# obs = np.diff(pre.obs(), axis=1)
# plt.plot(np.diff(pre.normal_obs))
# plt.plot(obs[0, :])
# plt.plot(obs[1, :])
# plt.plot(obs[2, :])
#
# plt.show()
# plt.plot(pre.normal_obs.to_numpy())
# plt.show()
# obs = pre.obs()
# plt.figure(1)
# plt.plot(obs[0, :])
# plt.figure(2)
# plt.plot(obs[1, :])
# plt.figure(3)
# plt.plot(pre.target())
# plt.show()
