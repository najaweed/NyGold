from ny_data_loader import LitNyData, NyDataset
import pandas as pd
import pickle
from torch.utils.data import DataLoader

# # READ DATA
df = pd.read_csv('../training/gold.csv', )  # , parse_dates=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

with open('../training/gold_config_CasualRnn.pkl', 'rb') as f:
    config = pickle.load(f)

dataset = NyDataset(df, config)

dataloader = DataLoader(dataset=dataset,
                        batch_size=config['batch_size'],
                        # num_workers=8,
                        drop_last=False,
                        # pin_memory=True,
                        shuffle=False,
                        )
for i, batch in dataloader.dataset:
    print(i)
    print(batch)
    if i ==1:
        break
