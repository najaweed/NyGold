import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime


class GoldData:
    def __init__(self,
                 start_year: int = 2019
                 ):
        self.raw_df = self.get_ohlc(start_year=start_year)
        self.ny_df = self.group_data('1D', self.session_transform(self.raw_df))

    def save_csv(self, path: str = 'gold.csv'):
        self.ny_df = self.ny_df.iloc[:-3,:]
        self.ny_df.to_csv(path, index_label='time')

    @staticmethod
    def get_ohlc(start_year: int = 2019, symbol: str = 'XAUUSD', ):
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            quit()
        utc_from = datetime(start_year, 1, 1, )
        utc_to = datetime.now()
        ticks = pd.DataFrame(mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, utc_from, utc_to))
        # ticks = pd.DataFrame(mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, s, shift_5m))
        ticks['time'] = pd.to_datetime(ticks['time'], unit='s')
        ticks = ticks.set_index('time')
        return ticks

    @staticmethod
    def session_transform(x_df, sessions=None):
        if sessions is None:
            sessions = [('00:01', '7:00'), ('07:01', '13:00'), ('13:01', '23:59')]
        ny_df = []
        ohlc = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'tick_volume': 'sum'
        }

        for i, session in enumerate(sessions):
            df = x_df.copy().between_time(*session)
            df = df.resample('1D').apply(ohlc)
            df['time'] = df.index
            df['time'] = df['time'] + pd.Timedelta(hours=i)
            df = df.set_index('time')
            ny_df.append(df)

        return pd.DataFrame(pd.concat(ny_df, axis=0).dropna().sort_index())

    @staticmethod
    def group_data(freq, x_df):
        df = x_df.copy()
        df['time'] = df.index
        df_list = []
        for group_name, df_group in df.groupby(pd.Grouper(freq=freq, key='time')):
            g_df = df_group.drop(columns=['time'])
            if len(g_df) == 3:  # 5days*5weeks
                df_list.append(g_df)
        return pd.DataFrame(pd.concat(df_list, axis=0).sort_index())


gold = GoldData(start_year=2017)
print(gold.ny_df)
gold.save_csv('training/gold.csv')
