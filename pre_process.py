import math
from datetime import datetime
import pandas as pd
import numpy as np
import pwlf


class PreProcess:
    def __init__(self,
                 df: pd.DataFrame,
                 task='predict_classify',
                 ):
        self.df = df
        # d = datetime.strptime(str(df.index[-1]), '%Y-%m-%d %H:%M:%S')
        # if d.hour != 2:
        #     print("ERORR in target date time , shift time to new york session ( @hour ==2)")
        #     print(df)
        #     breakpoint()
        self.task = task
        self.raw_obs_df = df.iloc[:-1, :]
        self.raw_target_df = df.iloc[-1, :]
        self.normal_obs_df, scales = self.trend_normal_df(self.raw_obs_df)
        self.scale, self.drift = scales[0], scales[1]

    def obs(self):
        #normal_obs_df = self.normal_obs_df.copy()

        #normal_obs_df['inner_momentum'] =(normal_obs_df.close - normal_obs_df.open) / (normal_obs_df.high - normal_obs_df.low)
        return self.normal_obs_df.to_numpy()

    def target(self, ):
        if self.task == 'prediction':
            ts_target = self.raw_target_df[['high', 'low', ]].to_numpy().copy()
            ts_target -= self.drift
            ts_target /= self.scale
            # if self.step_share > 0:
            #     ts_obs_share = self.normal_obs_df.to_numpy().copy()[-self.step_share:]
            #     ts_target = np.vstack([ts_obs_share, ts_target])
            return ts_target
        elif self.task == 'predict_classify':
            # print(self.raw_target_df)
            price_target = self.raw_target_df.signal_price
            # print(price_target)
            # print(self.raw_target_df.index)
            if not math.isnan(price_target):
                price_target -= self.drift
                price_target /= self.scale

                side = None
                if self.raw_target_df.signal_side == 'sell':
                    side = [price_target, 1]
                elif self.raw_target_df.signal_side == 'buy':
                    side = [price_target, 0]

                return side


            # if self.raw_target_df[['signal_price']] is not None:
            else:
                # print(price_target, self.raw_target_df[['signal_side']])

                return None, None

    @staticmethod
    def trend_normal_df(df):
        x = df[['open','high', 'low', 'close']].mean(axis=1).to_numpy()
        y = np.arange(0, len(x))
        m_b = np.polyfit(y, x, 1)
        m, b = m_b[0], m_b[1]
        # print(m, b)
        y_hat = m * y + b
        x_df = df[['open','high', 'low', 'close']].copy()
        x_df = x_df.sub(y_hat, axis='index')
        drift_trend = y_hat[-1]
        scale_normal = (df.high.max() - df.low.min())
        #print((df.high.max() - df.low.min()))
        #print((x_df.high.max() - x_df.low.min()))
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
# a = 0
# for a in range(1):
#     window = 200+1
#     ohlc = xdf.iloc[a * 3 :a * 3 + window, :]
#     pre_process = PreProcess(ohlc, task='predict_classify')
#     # print(pre_process.raw_obs_df)
#     print(pre_process.obs())
#     #print(pre_process.target())


