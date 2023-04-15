from ny_data_loader import LitNyData, NyDataset
import pandas as pd
import pickle
from torch.utils.data import DataLoader

# # READ DATA
df = pd.read_csv('gold.csv', index_col='time', parse_dates=True)
df = df.iloc[:1000,:]

with open('../training/hyper_params_data/config_ResForcast.pkl', 'rb') as f:
    config = pickle.load(f)
config['is_newyork'] = False
dataset = NyDataset(df, config)

print(dataset.obs)