import pandas as pd
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
        self.drift = None
        self.scale = None
        self.raw_obs_df, self.raw_target_df = self._split_obs_target()

        self.normal_obs_df, self.normal_target_df = self.trend_normal_df()

    def _split_obs_target(self):
        if self.step_prediction !=0:
            obs = self.df.iloc[:-self.step_prediction, :][['high', 'low', ]]
        else:
            obs = self.df[['high', 'low', ]]
        target = self.df.iloc[-(self.step_share + self.step_prediction):, :][['high', 'low', ]]
        return obs, target

    def obs(self):
        normal_obs_df = self.normal_obs_df.copy().to_numpy()
        return normal_obs_df

    def target(self, ):
        ts_target = self.normal_target_df[['high', 'low', ]].to_numpy().copy()
        high = self.raw_target_df.high.to_numpy()
        low = self.raw_target_df.low.to_numpy()
        return np.vstack([ts_target[:, 0].T, ts_target[:, 1].T, self.scale * np.ones_like(high), self.drift, high, low])

    def trend_normal_df(self, ):
        x = self.raw_obs_df.mean(axis=1).to_numpy()
        y = np.arange(0, len(x))
        m_b = np.polyfit(y, x, 1)
        m, b = m_b[0], m_b[1]
        trend_hat = m * y + b

        obs_detrend_df = self.raw_obs_df.copy()
        obs_detrend_df = obs_detrend_df.sub(trend_hat, axis='index')

        target = self.raw_target_df.copy()
        t_ = np.arange(0, len(target))
        if self.step_share == 0:
            trend_target = m * t_ + trend_hat[-1]
        else:
            trend_target = m * t_ + trend_hat[-self.step_share]
        tar_detrend_df = target.sub(trend_target, axis='index')
        self.drift = trend_target
        self.scale = (obs_detrend_df.high.max()-obs_detrend_df.low.min())
        return obs_detrend_df / self.scale, tar_detrend_df / self.scale

    def inverse_normal(self, nn_prediction: np.ndarray):
        # inverse normalizer
        normal_prediction = (nn_prediction * self.scale) + self.drift
        return normal_prediction.T

    def inverse_normal_df(self, nn_prediction: np.ndarray):
        # inverse normalizer
        normal_prediction = (nn_prediction * self.scale) + self.drift
        # print(normal_prediction.shape)
        # print(normal_prediction)
        # if self.step_share > 0:
        #     normal_prediction = normal_prediction[:, -1]
        p_df = self.raw_target_df.copy()
        normal_prediction = normal_prediction.flatten()
        p_df['high'] = normal_prediction[0]
        p_df['low'] = normal_prediction[1]
        # p_df['close'] = normal_prediction[2]
        return p_df


# import matplotlib.pyplot as plt
#
# xdf = pd.read_csv('gold.csv', index_col='time', parse_dates=True)
# a = 1
# window = 200 + 1
# ohlc = xdf.iloc[a * 3:a * 3 + window, :]
#
# pre_process = PreProcess(ohlc, step_prediction=3, step_share=3)
# obs = pre_process.raw_obs_df
# # print(pre_process.obs())
#
# tar = pre_process.target()[:2, :] - 0.2 * pre_process.target()[:2, :]
# raw_target = pre_process.raw_target_df.to_numpy()
# print(raw_target.shape)
# plt.figure(1)
# # plt.plot(obs)
# plt.plot(raw_target)
# # plt.figure(2)
# plt.plot(pre_process.inverse_normal(tar),'.-')
# plt.show()

#
# print(pre_process.raw_target_df)
# print(pre_process.target())
# tar = pre_process.target() + [0.1,0.1]
# print(tar)
# print(pre_process.x_inverse_normal(tar))
#
# # print(pre_process.obs().shape)
# # print(pre_process.obs().min())
# # print(pre_process.obs().max())
