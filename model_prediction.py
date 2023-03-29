import torch

from model.ResCnn import ResidualStacks
from pre_process import NyDiffNormalizer
import pickle
import numpy as np
import pandas as pd


with open('gold_config_CasualRnn.pkl', 'rb') as f:
    config = pickle.load(f)
    print(config)
window = config['number_days'] * config['tick_per_day']

model = ResidualStacks(config)
checkpoint = torch.load("conv_params.ckpt")
model_weights = checkpoint["state_dict"]

# update keys by dropping `nn_model.`
for key in list(model_weights):
    model_weights[key.replace("nn_model.", "")] = model_weights.pop(key)
model.load_state_dict(model_weights)
model.eval()
print('model loaded with learned parameters, ready for predict')


df = pd.read_csv('gold.csv', index_col='time', parse_dates=True)
#print(df)

window = config['number_days'] * config['tick_per_day']
a = 8 # -a*config['tick_per_day']
input_to_predict = df.iloc[-1 * window-a*config['tick_per_day']:].copy()
proc = NyDiffNormalizer(input_to_predict.iloc[:, :])
nn_input_numpy = proc.obs()  # add dummy dim for batch
nn_input_torch = torch.from_numpy(np.expand_dims(nn_input_numpy, 0)).type(torch.float32)
nn_input_torch = torch.permute(nn_input_torch, (0, 2, 1))
x = model(nn_input_torch)

x_loss = torch.nn.MSELoss()
x_1, x_2 = x[0], torch.from_numpy(proc.target())

nn_output_numpy = x.detach().numpy()
nn_output_numpy = nn_output_numpy[0].T
x_in = nn_input_torch.detach().numpy()[0].T
df_predict = proc.prediction_to_ohlc_df(nn_output_numpy)
print(proc.df.iloc[-3:-1, :])
print(proc.df.iloc[-1, :])
print(df_predict)
print(x_1, x_2)
print(torch.sqrt(x_loss(x_1, x_2)))
