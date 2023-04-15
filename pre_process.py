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

        self.raw_obs_df = df.iloc[:-1, :]
        self.raw_target_df = df.iloc[-1, :]

        self.normal_obs_df, scales = self.trend_normal_df(self.raw_obs_df)
        self.scale, self.drift = scales[0], scales[1]

    def _split_obs_target(self):
        obs = self.df.iloc[:-self.step_prediction, :]
        target = self.df.iloc[-(self.step_share + self.step_prediction):, :]
        return obs, target

    def obs(self):
        normal_obs_df = self.normal_obs_df.copy().to_numpy()
        return normal_obs_df

    def target(self, ):
        ts_target = self.raw_target_df[['high', 'low', ]].to_numpy().copy()
        ts_target -= self.drift
        ts_target /= self.scale

        return np.array(
            [ts_target[0], ts_target[1], self.scale, self.drift, self.raw_target_df.high, self.raw_target_df.low])

    @staticmethod
    def trend_normal_df(df):
        x = df[['open', 'high', 'low', 'close']].mean(axis=1).to_numpy()
        y = np.arange(0, len(x))
        m_b = np.polyfit(y, x, 1)
        m, b = m_b[0], m_b[1]
        y_hat = m * y + b

        x_df = df[['open', 'high', 'low', 'close']].copy()
        x_df = x_df.sub(y_hat, axis='index')
        drift_trend = y_hat[-1]
        scale_normal = 100  # (x_df.high.max() - x_df.low.min())
        x_df /= scale_normal
        return x_df, (scale_normal, drift_trend)

    def inverse_normal(self, nn_prediction: np.ndarray):
        # inverse normalizer
        normal_prediction = (nn_prediction * self.scale) + self.drift
        return normal_prediction

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
#
# xdf = pd.read_csv('training/hyper_params_data/gold.csv', index_col='time', parse_dates=True)
# a = 1
# window = 200 + 1
# ohlc = xdf.iloc[a * 3:a * 3 + window, :]
#
# pre_process = PreProcess(ohlc,step_prediction=4,step_share=4 )
# print(pre_process._split_obs_target()[0])
#
# print(pre_process._split_obs_target()[1])
# #
# print(pre_process.raw_target_df)
# print(pre_process.target())
# tar = pre_process.target() + [0.1,0.1]
# print(tar)
# print(pre_process.x_inverse_normal(tar))
#
# # print(pre_process.obs().shape)
# # print(pre_process.obs().min())
# # print(pre_process.obs().max())
