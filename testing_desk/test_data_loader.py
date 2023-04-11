from ny_data_loader import LitNyData, NyDataset
import pandas as pd
import pickle
from torch.utils.data import DataLoader

# # READ DATA
df = pd.read_csv('imb_eurusd.csv', index_col='time', parse_dates=True)


with open('../training/hyper_params_data/config_ResForcast.pkl', 'rb') as f:
    config = pickle.load(f)
config['is_newyork'] = False
dataloader = LitNyData(df, config)

for i,(obs, target) in enumerate(dataloader.train_dataloader()):
    print(i)
    #print(obs,target)
    #print(obs.shape,target.shape)
    if i ==0:
        break
