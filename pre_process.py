import pandas as pd
import numpy as np


class NyDiffNormalizer:
    def __init__(self,
                 df: pd.DataFrame,
                 step_share=0,
                 ):
        self.df = df
        self.step_share = step_share
        # print(df)
        self.scale_normal = None
        self.min = None
        self.trend_speed = None
        self.ndf = self.ohlc_normal_df()
        self.df_dtrend = self.df_de_trend()
        self.x_df = self.df_dtrend.copy()

    def obs(self):
        ts_obs = self.x_df.to_numpy().copy()
        # ts_diff_1 = self.diff_ts(self.df[['open', 'high', 'low', 'close']].iloc[:-1, :].to_numpy().copy())
        # ts_diff_2 = self.diff_ts(ts_diff_1)
        # ts_diff_3 = self.diff_ts(ts_diff_2)

        # ts_obs = np.hstack([ts_obs,  ]) #  ts_diff_1,ts_diff_2, ts_diff_3,
        return ts_obs

    def target(self, ):
        ts_target = self.df[['high', 'low', 'close']].to_numpy()[-1, :].copy()

        ts_target -= self.min_trendi
        ts_target /= self.scale_normal

        if self.step_share > 0:
            ts_obs_share = self.x_df.to_numpy().copy()[-self.step_share:]
            ts_target = np.vstack([ts_obs_share, ts_target])
        return ts_target

    def ohlc_normal_df(self, ):
        n_df = self.df[['high', 'low', 'close']].iloc[:-1, :].copy()
        # print(n_df)
        self.min = n_df.low.min()
        self.scale_normal = (n_df.high.max() - n_df.low.min())
        n_df = (n_df - self.min) / self.scale_normal
        return n_df

    def diff_ts(self, time_series):
        diff_ts = np.diff(time_series, axis=0, )
        diff_ts = np.vstack([diff_ts[0], diff_ts])
        # print(np.min(diff_ts,axis=0,keepdims=True))
        min = np.min(diff_ts, axis=0, keepdims=True)
        max_min = np.max(diff_ts, axis=0, keepdims=True) - np.min(diff_ts, axis=0, keepdims=True)
        diff_ts = (diff_ts - min) / max_min
        return diff_ts

    def prediction_to_ohlc_df(self, nn_prediction: np.ndarray):
        # inverse normalizer
        print(nn_prediction)
        print(self.scale_normal)
        print(self.min)

        normal_prediction = (nn_prediction * self.scale_normal) + self.min_trendi
        p_df = self.df.iloc[-1, :].copy()
        # p_df[['high', 'low', 'close']] = normal_prediction.reshape(-1, len(normal_prediction))
        # if diff
        normal_prediction = normal_prediction.flatten()
        p_df['high'] = normal_prediction[0]  # + self.ohlc_obs['high'][-1]
        p_df['low'] = normal_prediction[1]  # + self.ohlc_obs['low'][-1]
        p_df['close'] = normal_prediction[2]  # + self.ohlc_obs['close'][-1]

        return p_df

    def df_de_trend(self, ):
        x = self.df[['open', 'high', 'low', 'close']].mean(axis=1).to_numpy()
        # print(x)
        y = np.arange(0, len(x))
        m_b = np.polyfit(y, x, 1)
        m, b = m_b[0], m_b[1]
        self.trend_speed = m
        # print(m, b)
        y_hat = m * y + b
        x_df = self.df[['open', 'high', 'low', 'close']].copy()
        x_df = x_df.sub(y_hat, axis='index')
        self.min_trendi = y_hat[-1] + m
        x_df /= self.scale_normal
        return x_df


# import matplotlib.pyplot as plt
#
# xdf = pd.read_csv('gold.csv', index_col='time', parse_dates=True)
# ohlc = xdf.iloc[:5 * 40, :]
# nydiff = NyDiffNormalizer(ohlc)
#
# x = nydiff.obs()
# y = nydiff.target()
# print(x)
# print(y)
# df_r = nydiff.df_de_trend()
# plt.figure(1)
# plt.plot(df_r.iloc[:, 1].to_numpy())
# plt.figure(2)
# plt.plot(nydiff.ndf.iloc[:, 1].to_numpy())
# plt.show()
