import pandas as pd
import numpy as np
import pwlf


class PreProcess:
    def __init__(self,
                 df: pd.DataFrame,
                 step_share=0,
                 ):
        self.df = df
        self.raw_obs_df = df.iloc[:-1, :]
        self.raw_target_df = df.iloc[-1, :]
        self.step_share = step_share
        self.normal_obs_df, scales = self.trend_normal_df(self.raw_obs_df)
        self.scale, self.drift = scales[0], scales[1]

    def obs(self):
        normal_obs_df = self.normal_obs_df.to_numpy().copy()
        return normal_obs_df

    def target(self, ):
        ts_target = self.raw_target_df[['high', 'low', ]].to_numpy().copy()
        ts_target -= self.drift
        ts_target /= self.scale
        if self.step_share > 0:
            ts_obs_share = self.normal_obs_df.to_numpy().copy()[-self.step_share:]
            ts_target = np.vstack([ts_obs_share, ts_target])
        return ts_target


    @staticmethod
    def trend_normal_df(df):
        x = df[['high', 'low', 'close']].mean(axis=1).to_numpy()
        y = np.arange(0, len(x))
        m_b = np.polyfit(y, x, 1)
        m, b = m_b[0], m_b[1]
        # print(m, b)
        y_hat = m * y + b
        x_df = df[['high', 'low', 'close']].copy()
        x_df = x_df.sub(y_hat, axis='index')
        drift_trend = y_hat[-1]
        scale_normal = (df.high.max() - df.low.min())
        x_df /= scale_normal
        return x_df, (scale_normal, drift_trend)

    @staticmethod
    def max_min_normal_df(df):
        n_df = df[['high', 'low', 'close']].copy()
        x_min = n_df.low.min()
        scale_normal = (n_df.high.max() - n_df.low.min())
        n_df = (n_df - x_min) / scale_normal
        return n_df, (scale_normal, x_min)

    def inverse_normal(self, nn_prediction: np.ndarray):
        # inverse normalizer
        normal_prediction = (nn_prediction * self.scale) + self.drift
        print(normal_prediction.shape)
        print(normal_prediction)
        if self.step_share > 0:
            normal_prediction = normal_prediction[:, -1]
        p_df = self.raw_target_df.copy()
        normal_prediction = normal_prediction.flatten()
        p_df['high'] = normal_prediction[0]
        p_df['low'] = normal_prediction[1]
        # p_df['close'] = normal_prediction[2]
        return p_df


# xdf = pd.read_csv('training/gold.csv', index_col='time', parse_dates=True)
# ohlc = xdf.iloc[5 * 40 - 2:8 * 40 - 2, :]
# pre_process = PreProcess(ohlc, )
# # print(pre_process.raw_obs_df)
# print(pre_process.obs())
# print(pre_process.target())

# print(pre_process.raw_target_df)
# print(pre_process.inverse_normal(np.array([-0.23989629, -0.32803092, -0.30920604])))
# print(pre_process.raw_target_df)
