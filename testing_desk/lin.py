import pandas as pd
import numpy as np
import datetime as dt

# READ DATA
xdf = pd.read_csv('15_gold.csv',
                  index_col='time',
                  parse_dates=True)


class LinearPrediction:
    def __init__(self,
                 df: pd.DataFrame
                 ):
        self.df = df
        # print(df.last('7D'))
        # print(self._linear_reg(df.loc[-(df.index[-1]-dt.timedelta(days=7)):,:]))
        self.trend_df()

    def trend_df(self):
        all_df = self.df.copy()
        tfs = ['4H', '12H', '1D', '3D', '7D', '15D']
        for tf in tfs:
            df = self.df.last(tf).copy()
            trend_df = self._linear_reg(df, tf)
            all_df = all_df.join(trend_df)
        return all_df.last(tfs[-1])

    @staticmethod
    def _linear_reg(df, col_name):
        x = df.mean(axis=1).to_numpy()
        y = np.arange(0, len(x))
        m_b = np.polyfit(y, x, 1)
        m, b = m_b[0], m_b[1]
        y_hat = m * y + b
        lin_df = pd.DataFrame({f'{col_name}': y_hat}, index=df.index)
        return lin_df


df = LinearPrediction(xdf).trend_df()
import matplotlib.pyplot as plt

plt.plot(df[['high']])  # .bfill().to_numpy())
plt.show()
