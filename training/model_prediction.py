import torch

from model.ResCnn import ResidualStacks
from pre_process import PreProcess
import pickle
import numpy as np
import pandas as pd


def load_model(c_model, c_config, path_params="conv_params.ckpt"):
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


with open('gold_config_CasualRnn.pkl', 'rb') as f:
    config = pickle.load(f)
    print(config)

model = load_model(ResidualStacks, config, "conv_params.ckpt")
df = pd.read_csv('gold.csv', index_col='time', parse_dates=True)

window = config['number_days'] * config['tick_per_day']
for sl in np.linspace(1, 4, 30):
    print(sl, 'stop loss')
    num_correct_high = 0
    num_correct_low = 0
    num_correct_high_low = 0

    num_sl = 0
    max_val_tp = 0
    val_tp = 0
    val_sl = 0
    for a in range(20):
        # print(a)
        if a == 0:
            input_to_predict = df.iloc[-1 * window - a * config['tick_per_day']:].copy()

        else:
            input_to_predict = df.iloc[-1 * window - a * config['tick_per_day']:-a * config['tick_per_day']].copy()

        proc = PreProcess(input_to_predict, step_share=config['step_share'])
        nn_input_torch = torch.from_numpy(np.expand_dims(proc.obs(), 0)).type(torch.float32)
        nn_input_torch = torch.permute(nn_input_torch, (0, 2, 1))
        x = model(nn_input_torch)
        _target = proc.raw_target_df[['high', 'low']].to_numpy()
        _predict = proc.inverse_normal(x.detach().numpy())[0]
        #BACK TEST
        _l1_diff = abs(_predict - _target)
        if _l1_diff[0] < sl < _l1_diff[1]:
            # print('sell sl buy')
            num_correct_high += 1
            max_val_tp += abs(_target[0] - _target[1])
            val_tp += abs(_predict[0] - _predict[1])

        elif _l1_diff[1] < sl < _l1_diff[0]:
            # print('buy sl sell')
            num_correct_low += 1
            max_val_tp += abs(_target[0] - _target[1])
            val_tp += abs(_predict[0] - _predict[1])
        elif _l1_diff[1] > sl and _l1_diff[0] > sl:
            # print('SL')
            num_sl += 1
            val_sl += sl
        elif (_l1_diff[0] and _l1_diff[1]) < sl:
            num_correct_high_low += 1
            max_val_tp += abs(_target[0] - _target[1])
            val_tp += abs(_predict[0] - _predict[1])
            # print('hittt')
            pass
        # print(proc.df.iloc[-3:-1, :])
        # print(_target, 'raw target')
        # print(_predict, 'model prediction')
        # print(_l1_diff, 'diff')
        # print(x.data, 'model prediction', )
        # print(torch.from_numpy(proc.target()).data, 'target', )
        # print(abs(x.detach().numpy() - proc.target()), 'diff')
        # print(x.detach().numpy() - proc.target(), 'diff')

    print(num_correct_high, 'number of sell', )
    print(num_correct_low, 'number of buy', )
    print(num_correct_high_low, 'number of buy sell', )
    print(num_sl, 'number of sl',)
    # print(num_tp / num_sl)
    # print(val_tp / val_sl)
    print(val_tp - val_sl, 'max', max_val_tp - val_sl)
    print(val_tp/np.sqrt(sl))
    print('-----')
