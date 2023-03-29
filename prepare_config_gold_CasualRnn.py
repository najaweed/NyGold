import numpy as np
import pandas as pd
import pickle
from ny_data_loader import LitNyData

# # READ DATA
df = pd.read_csv('gold.csv', index_col='time', parse_dates=True)
df = df.iloc[:1000, :]

# READ DATA
config_data_loader = {
    # config dataset and dataloader
    'batch_size': 4,
    'step_share': 0,
    'tick_per_day': 3,
    'number_days': 80,
    'split': (9, 1),  # make a function for K-fold validation
}
# https://yanglin1997.github.io/files/TCAN.pdf
kernel = 2
num_layer_dail = int(np.log2((80 * 3 / (kernel - 1)) + 1))

# FIND MODEL CONFIG  DATALOADER
in_shape, out_shape = None, None
config_CasualRnn = {
    'hidden_channels': 3,
    'num_stack_layers': 2,
    'dropout': 0.4,
    'out_channels': 3,
    'dilation': [2 ** i for i in range(num_layer_dail)],
}
lit_data = LitNyData(df, config_data_loader)
lit_val = lit_data.val_loader
for i, (a, b) in enumerate(lit_data.train_dataloader()):
    in_shape, out_shape = a.size(), b[0].size()
    config_CasualRnn['in_channels'] = in_shape[-1]
    config_CasualRnn['output_size'] = out_shape[-1]
    print(a.shape)
    print(b[0].shape)
    print('input_shape ', in_shape, ',', 'output_shape ', out_shape)
    print(i)
    break
config = config_data_loader | config_CasualRnn  # == {**config_data_loader , **config_conv}
print(config)

with open('gold_config_CasualRnn.pkl', 'wb') as f:
    pickle.dump(config, f)
