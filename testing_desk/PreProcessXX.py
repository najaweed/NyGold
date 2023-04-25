import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


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
        self.diff_raw_obs_df = self.raw_obs_df.copy().diff().dropna()
        self.diff_raw_target_df = self.raw_target_df.copy().diff().dropna()
        self.transform = StandardScaler()
        self.transform.fit_transform(self.diff_raw_obs_df.mean(axis=1).to_numpy().reshape(-1, 1))

    def _split_obs_target(self):
        if self.step_prediction != 0:
            obs = self.df.iloc[:-self.step_prediction, :]
        else:
            obs = self.df[['high', 'low', ]]
        target = self.df.iloc[-(self.step_share + self.step_prediction):, :]
        return obs, target

    def obs(self):
        normal_obs_df = self.transform.transform(self.diff_raw_obs_df.mean(axis=1).to_numpy().reshape(-1, 1))
        return normal_obs_df.flatten()

    def target(self, ):
        ts_target = self.transform.transform(self.diff_raw_target_df.mean(axis=1).to_numpy().reshape(-1, 1))
        return ts_target.flatten()

    @staticmethod
    def inverse_diff(x_ts, x0):
        inv_diff = [x0]
        xt = x0
        for x in x_ts:
            xt += x

            inv_diff.append(xt)
        return np.asarray(inv_diff)


# #
# import matplotlib.pyplot as plt
#
# #
# xdf = pd.read_csv('D_gold.csv', index_col='time', parse_dates=True)
# a = 1
# window = 300
# ohlc = xdf.iloc[a * 3:a * 3 + window, :]
#
# pre_process = PreProcess(ohlc, step_prediction=20, step_share=3)
# tar = pre_process.target()  # [0] + 0.2*np.random.random(len(pre_process.target()[0]))
# inv_norm_tar = pre_process.transform.inverse_transform(np.expand_dims(tar, axis=0)).flatten()
# inv_diff = pre_process.inverse_diff(inv_norm_tar, x0=pre_process.raw_target_df.open[0])
# print(inv_diff)

# plt.figure(1)
# plt.plot(inv_diff)
# plt.plot(pre_process.raw_target_df.mean(axis=1).to_numpy())
# plt.show()
#
# def inverse_normal(normal_data,scale,drift,):
#     inv_normal_prediction = (normal_data * scale) + drift
#     return inv_normal_prediction
# print(inverse_normal(tar[0],tar[1],tar[2]))
# print(pre_process.inverse_normal(tar[0]))
# plt.plot(pre_process.inverse_normal(tar))
# plt.plot(pre_process.raw_target_df[['high', 'low','close' ]].mean(axis=1).to_numpy())
# plt.show()
# print(pre_process.obs())
# plt.figure(1)
# plt.plot(pre_process.raw_target_df[['high', 'low','close' ]].mean(axis=1).to_numpy())
# plt.figure(2)
# plt.plot(pre_process.target()[0])
# plt.figure(3)
# plt.plot(pre_process.obs())
# plt.figure(4)
# plt.plot(pre_process.raw_obs_df[['high', 'low','close' ]].mean(axis=1).to_numpy())
#
# plt.show()
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
