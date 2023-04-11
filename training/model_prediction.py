import torch

from model.ResCnn import ResForcast
from pre_process import PreProcess
import pickle
import numpy as np
import pandas as pd


def load_model(c_model, c_config, path_params):
    x_model = c_model(c_config)
    checkpoint = torch.load(path_params)
    model_weights = checkpoint["state_dict"]

    # update keys by dropping `nn_model.`
    for key in list(model_weights):
        model_weights[key.replace("nn_model.", "")] = model_weights.pop(key)
    x_model.load_state_dict(model_weights)
    x_model.eval()
    print('model loaded with learned parameters, ready for predict')
    return x_model


with open('hyper_params_data/config_ResForcast.pkl', 'rb') as f:
    config = pickle.load(f)
    print(config)

model = load_model(ResForcast, config, "hyper_params_data/ResForcast_params.ckpt")
xdf = pd.read_csv('hyper_params_data/gold.csv', index_col='time', parse_dates=True)
w_window = config['number_days'] * config['tick_per_day']


#
# bt_logs = []
# for a in range(20):
#     # print(a)
#
#     if a == 0:
#         input_to_predict = df.iloc[-1 * window - a * config['tick_per_day']:].copy()
#
#     else:
#         input_to_predict = df.iloc[-1 * window - a * config['tick_per_day']:-a * config['tick_per_day']].copy()
#
#     # SingleDayBackTest
#     proc = PreProcess(input_to_predict, task=config['task_network'])
#     nn_input_torch = torch.from_numpy(np.expand_dims(proc.obs(), 0)).type(torch.float32)
#     nn_input_torch = torch.permute(nn_input_torch, (0, 2, 1))
#     x = model(nn_input_torch)
#
#     _target = proc.raw_target_df[['high', 'low']]
#     _predict = proc.inverse_normal_df(x.detach().numpy())
#     # print(_target.high,_target.low,)
#     # print(_predict.high,_predict.low,)
#     # print(abs(_target.high - _predict.high))
#     # print(abs(_target.low - _predict.low))
#     # print(_target)
#     backtest_log = {'index': _target.name,
#                     'buy': 0,
#                     'buy_price': 0,
#                     'sell': 0,
#                     'sell_price': 0,
#                     'missed_buy': 0,
#                     'missed_sell': 0,
#                     'sl_buy': 0,
#                     'sl_sell': 0,
#                     'max_profit': 0,
#                     'value_sl': 0,
#                     }
#     # BUY Condition
#     if abs(_predict.low - _target.low) < sl_buy:
#         if _predict.low >= _target.low:
#             # Valid Buy
#             backtest_log['buy'] += 1
#             backtest_log['buy_price'] = _predict.low
#             backtest_log['max_profit'] = _target.high - _predict.low
#
#         elif _predict.low < _target.low:
#             # Missed Buy Position
#             backtest_log['missed_buy'] += 1
#     # Sell Condition
#     if abs(_predict.high - _target.high) < sl_sell:
#         if _predict.high <= _target.high:
#             # Valid Sell
#             backtest_log['sell'] += 1
#             backtest_log['sell_price'] = _predict.high
#             backtest_log['max_profit'] = _predict.high - _target.low
#         elif _predict.high > _target.high:
#             # Missed Sell Position
#             backtest_log['missed_sell'] += 1
#
#     # Missed Condition
#     if _predict.low < _target.low:
#         if abs(_predict.low - _target.low) > sl_buy:
#             # Missed Buy Position
#             backtest_log['missed_buy'] += 1
#             pass
#     if _predict.high > _target.high:
#         if abs(_predict.high - _target.high) > sl_sell:
#             # Missed Sell Position
#             backtest_log['missed_sell'] += 1
#             pass
#     # Stop Loss Condition
#     if abs(_predict.low - _target.low) >= sl_buy:
#         if _predict.low > _target.low:
#             # Valid SL Buy
#             backtest_log['sl_buy'] += 1
#             backtest_log['value_sl'] += sl_buy
#
#             pass
#
#     if abs(_predict.high - _target.high) >= sl_sell:
#         if _predict.high < _target.high:
#             # Valid SL Sell
#             backtest_log['sl_sell'] += 1
#             backtest_log['value_sl'] += sl_sell
#
#             pass
#     bt_logs.append(backtest_log)
#     # print(backtest_log['index'])
# print(len(bt_logs), 'of days backtest')
# # Report Backtest
# final_report = {
#     'max_profit': 0,
#     'value_sl': 0,
#     'num_sl_sell': 0,
#     'num_sl_buy': 0,
#     'num_missed': 0,
#     'num_buy': 0,
#     'num_sell': 0,
# }
#
# for bt_log in bt_logs:
#     final_report['max_profit'] += bt_log['max_profit']
#     final_report['value_sl'] += bt_log['value_sl']
#     final_report['num_sl_sell'] += bt_log['sl_sell']
#     final_report['num_sl_buy'] += bt_log['sl_buy']
#     final_report['num_missed'] += bt_log['missed_buy']
#     final_report['num_missed'] += bt_log['missed_sell']
#     final_report['num_buy'] += bt_log['buy']
#     final_report['num_sell'] += bt_log['sell']
#
# print(final_report)

class BackTester:
    def __init__(self,
                 df,
                 window,
                 stop_loss,
                 loaded_model,
                 num_days=20,
                 ):
        self.df = df
        self.window = window
        self.num_days = num_days
        self.model = loaded_model
        self.buy_sl = stop_loss
        self.sell_sl = stop_loss

    def _one_day_backtest(self, step_df):
        # SingleDayBackTest
        proc = PreProcess(step_df, task=config['task_network'])
        nn_input_torch = torch.from_numpy(np.expand_dims(proc.obs(), 0)).type(torch.float32)
        nn_input_torch = torch.permute(nn_input_torch, (0, 2, 1))
        x = self.model(nn_input_torch)

        _target = proc.raw_target_df[['high', 'low']]
        _predict = proc.inverse_normal_df(x.detach().numpy())
        # print(_target.high,_target.low,)
        # print(_predict.high,_predict.low,)
        # print(abs(_target.high - _predict.high))
        # print(abs(_target.low - _predict.low))
        # print(_target)
        backtest_log = {'index': _target.name,
                        'buy': 0,
                        'buy_price': 0,
                        'sell': 0,
                        'sell_price': 0,
                        'missed_buy': 0,
                        'missed_sell': 0,
                        'sl_buy': 0,
                        'sl_sell': 0,
                        'max_profit': 0,
                        'value_sl': 0,
                        }
        # BUY Condition
        if abs(_predict.low - _target.low) < self.buy_sl:
            if _predict.low >= _target.low:
                # Valid Buy
                backtest_log['buy'] += 1
                backtest_log['buy_price'] = _predict.low
                backtest_log['max_profit'] = _target.high - _predict.low

            elif _predict.low < _target.low:
                # Missed Buy Position
                backtest_log['missed_buy'] += 1
        # Sell Condition
        if abs(_predict.high - _target.high) < self.sell_sl:
            if _predict.high <= _target.high:
                # Valid Sell
                backtest_log['sell'] += 1
                backtest_log['sell_price'] = _predict.high
                backtest_log['max_profit'] = _predict.high - _target.low
            elif _predict.high > _target.high:
                # Missed Sell Position
                backtest_log['missed_sell'] += 1

        # Missed Condition
        if _predict.low < _target.low:
            if abs(_predict.low - _target.low) > self.buy_sl:
                # Missed Buy Position
                backtest_log['missed_buy'] += 1
                pass
        if _predict.high > _target.high:
            if abs(_predict.high - _target.high) > self.sell_sl:
                # Missed Sell Position
                backtest_log['missed_sell'] += 1
                pass
        # Stop Loss Condition
        if abs(_predict.low - _target.low) >= self.buy_sl:
            if _predict.low > _target.low:
                # Valid SL Buy
                backtest_log['sl_buy'] += 1
                backtest_log['value_sl'] += self.buy_sl

                pass

        if abs(_predict.high - _target.high) >= self.sell_sl:
            if _predict.high < _target.high:
                # Valid SL Sell
                backtest_log['sl_sell'] += 1
                backtest_log['value_sl'] += self.sell_sl

                pass
        return backtest_log

    def backtest(self):
        bt_logs = []
        for a in range(self.num_days):
            # print(a)

            if a == 0:
                input_to_predict = self.df.iloc[-1 * self.window - a * config['tick_per_day']:].copy()

            else:
                input_to_predict = self.df.iloc[
                                   -1 * self.window - a * config['tick_per_day']:-a * config['tick_per_day']].copy()

            bt_log = self._one_day_backtest(input_to_predict)
            bt_logs.append(bt_log)
        return bt_logs

    def report_backtest(self):
        final_report = {
            'max_profit': 0,
            'value_sl': 0,
            'num_sl_sell': 0,
            'num_sl_buy': 0,
            'num_missed': 0,
            'num_buy': 0,
            'num_sell': 0,
        }
        bt_logs = self.backtest()
        for bt_log in bt_logs:
            final_report['max_profit'] += bt_log['max_profit']
            final_report['value_sl'] += bt_log['value_sl']
            final_report['num_sl_sell'] += bt_log['sl_sell']
            final_report['num_sl_buy'] += bt_log['sl_buy']
            final_report['num_missed'] += bt_log['missed_buy']
            final_report['num_missed'] += bt_log['missed_sell']
            final_report['num_buy'] += bt_log['buy']
            final_report['num_sell'] += bt_log['sell']

        return final_report


for stop_loss in np.linspace(1.5, 5, 30):
    # stop_loss = 2.5
    bt = BackTester(xdf, w_window, stop_loss, model, 10)
    rep = bt.report_backtest()
    print(rep)
    print(rep['max_profit'] / rep['value_sl'], 'stop loss  = ', stop_loss)
