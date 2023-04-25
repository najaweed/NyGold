import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime


class NyGoldData:
    def __init__(self,
                 start_year: int = 2019
                 ):
        self.raw_df = self.get_ohlc(start_year=start_year)
        self.target_df = None
        # self.ny_df = self.group_data('1D', self.session_transform(self.raw_df))
        # self.labels = self.gen_label()
        # self.join_labels()

    def join_labels(self):
        self.ny_df.join(self.labels)
        labels_df = self.labels
        # print(labels_df.tail())
        # labels_df = labels_df.set_index('time')
        # labels_df.index = labels_df.index.date
        labels_df['date_time'] = pd.to_datetime(labels_df['time'].dt.date)  #
        labels_df['date_time'] += pd.Timedelta(hours=2)
        labels_df = labels_df.set_index('date_time')
        self.ny_df = self.ny_df.join(labels_df)
        pass

    def save_csv(self, path: str = 'gold'):
        self.ny_df.to_csv(path + '.csv', index_label='time')
        # self.labels.to_csv(path+'_label'+'.csv', index_label='time')

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

    def session_transform(self, x_df, sessions=None):
        if sessions is None:
            sessions = [('00:01', '7:00'), ('07:01', '13:00'), ('13:01', '18:00'), ('18:01', '23:59')]
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
            if i == 2:
                self.target_df = df
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
            if len(g_df) == 4:  # 5days*5weeks
                df_list.append(g_df)
        return pd.DataFrame(pd.concat(df_list, axis=0).sort_index())

    def gen_label(self):
        df = self.target_df.copy()
        df['time'] = df.index
        h = 0
        df_labels = {'time': [],
                     'signal_price': [],
                     'signal_side': [],
                     }

        for group_name, df_group in df.groupby(pd.Grouper(freq='1D', key='time')):
            g_df = df_group.drop(columns=['time'])
            if len(g_df) > 0:
                if g_df.high.idxmax() < g_df.low.idxmin():
                    # print('sell')
                    # print(g_df.high.idxmax(), g_df.high.max())
                    # print(g_df.low.idxmin(), g_df.low.min())
                    df_labels['time'].append(g_df.high.idxmax())
                    df_labels['signal_price'].append(g_df.high.max())
                    df_labels['signal_side'].append('sell')

                elif g_df.high.idxmax() > g_df.low.idxmin():
                    # print('buy')
                    # print(g_df.high.idxmax(), g_df.high.max())
                    # print(g_df.low.idxmin(), g_df.low.min())
                    # print(g_df.loc[g_df.low.idxmin(),:].low,g_df.low.min())
                    df_labels['time'].append(g_df.low.idxmin())
                    df_labels['signal_price'].append(g_df.low.min())
                    df_labels['signal_side'].append('buy')
                elif g_df.high.idxmax() == g_df.low.idxmin():
                    # print('HOLD')
                    # print(g_df.high.idxmax(), g_df.high.max())
                    # print(g_df.low.idxmin(), g_df.low.min())
                    # h += 1
                    # sample time is low , need under-15 min candle from 2019
                    pass
        label_df = pd.DataFrame.from_dict(df_labels)
        # label_df.set_index('time', inplace=True)
        # print(label_df)
        return label_df


class ImbalancedGold:
    def __init__(self,
                 start_year: int = 2019
                 ):
        self.raw_df = self.get_ohlc(start_year=start_year)
        print(self._imb_df())
        self.imb_df = self._imb_df()

    def save_csv(self, path: str = 'imb_gold'):
        self.imb_df.to_csv(path + '.csv', index_label='time')
        # self.labels.to_csv(path+'_label'+'.csv', index_label='time')

    @staticmethod
    def get_ohlc(start_year: int = 2019, symbol: str = 'EURUSD', ):
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            quit()
        utc_from = datetime(start_year, 1, 1, )
        utc_to = datetime.now()
        ticks = pd.DataFrame(mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, utc_from, utc_to))
        # ticks = pd.DataFrame(mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, s, shift_5m))
        ticks['time'] = pd.to_datetime(ticks['time'], unit='s')
        ticks = ticks.set_index('time')
        return ticks

    def _imb_df(self):
        df = self.raw_df
        imb_max = df[['tick_volume']].max()[0]
        imb_vol = 0
        last_index = df.index[0]
        list_df_imb = []
        for i in df.index:
            if imb_vol <= imb_max:
                imb_vol += df.loc[i, 'tick_volume']

            elif imb_vol > imb_max:
                chunk_df = df.loc[last_index:i, :]
                chunk_df = self.__gen_ohlc(chunk_df)
                list_df_imb.append(chunk_df)
                imb_vol = 0
                last_index = i
        return pd.DataFrame(pd.concat(list_df_imb, axis=1).sort_index()).transpose()

    @staticmethod
    def __gen_ohlc(df):
        x_df = df.iloc[-1, :].copy()
        x_df['open'] = df.open[0]
        x_df['high'] = df.high.max()
        x_df['low'] = df.low.min()
        x_df['close'] = df.close[-1]
        # x_df['imb_volume'] = df.tick_volume.sum()
        x_df['inner_momentum'] = (df.close[-1] - df.open[0]) / (df.high.max() - df.low.min())
        return x_df[['open', 'high', 'low', 'close', 'inner_momentum']]


# gold = NyGoldData(start_year=2016)
# print(gold.ny_df)
# print(gold.gen_label())
# gold.save_csv('training/hyper_params_data/4_gold')
def get_ohlc(start_year: int = 2021, symbol: str = 'XAUUSD', ):
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    utc_from = datetime(start_year, 1, 1, )
    utc_to = datetime.now()
    ticks = pd.DataFrame(mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, utc_from, utc_to))
    # ticks = pd.DataFrame(mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, s, shift_5m))
    ticks['time'] = pd.to_datetime(ticks['time'], unit='s')
    ticks = ticks.set_index('time')
    ticks = ticks.fillna(method='bfill')
    ticks = ticks.fillna(method='ffill')
    return ticks[['open', 'high', 'low', 'close']]


df = get_ohlc(start_year=2018)
print(df)
print(df.isnull().values.any())
if not df.isnull().values.any():
    df.to_csv('database/D_gold.csv', index_label='time')
# df.to_csv('training/hyper_params_data/D_gold.csv', index_label='time')
# imb_gold = ImbalancedGold(start_year=2021)
# imb_gold.save_csv('training/imb_eurusd')
