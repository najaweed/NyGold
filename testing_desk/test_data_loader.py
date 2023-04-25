from testing_desk.ny_data_loader import NyDataset
import pandas as pd
import torch
from torch.utils.data import DataLoader

# # READ DATA
df = pd.read_csv('D_gold.csv', index_col='time', parse_dates=True)
df = df.iloc[:400, :]
config_data_loader = {
    # config dataset and dataloader
    'batch_size': 1,
    'learning_rate': 1e-3,
    'window_temporal': 200,
    'split': (9, 1),
    'in_channels': 1,
    'step_prediction': 5,
    'step_share': 1,
    'out_channels': 1,
}
dataset = NyDataset(df, config_data_loader)
t_Dataset = torch.utils.data.TensorDataset(dataset.obs, dataset.target)
print(t_Dataset.tensors)
