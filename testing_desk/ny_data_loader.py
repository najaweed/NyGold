import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from testing_desk.PreProcessX import PreProcess
from preprocess.ProProcessY import PreProcess


class NyDataset(Dataset):
    def __init__(self,
                 data_temporal: pd.DataFrame,
                 config: dict,
                 ):
        self.time_series = data_temporal
        self.config = config
        self.window_temporal = config['window_temporal']
        self.obs, self.target = self.split_observation_prediction()

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, index):
        return self.obs[index, ...], self.target[index]

    def split_observation_prediction(self):
        x, y = [], []
        # print(self.time_series)
        in_shape, target_shape = None, None
        for t in range(0, len(self.time_series) - self.window_temporal):

            x_df = self.time_series.iloc[t:(t + self.window_temporal), :].copy()
            ny_normal = PreProcess(x_df, self.config['step_prediction'], self.config['step_share'])
            obs = ny_normal.obs()
            target = ny_normal.target()
            if t == 0:
                in_shape = obs.shape
            if obs.shape == in_shape:
                x.append(obs)
                y.append(target)

        x = torch.tensor(np.array(x, dtype=np.float32), dtype=torch.float32)
        y = torch.tensor(np.array(y, dtype=np.float32), dtype=torch.float32)

        return x, y


class LitNyData(pl.LightningDataModule, ):

    def __init__(self,
                 df: pd.DataFrame,
                 config: dict,
                 ):
        super().__init__()
        self.config = config
        self.df = df
        self.train_loader, self.val_loader = self._gen_data_loaders()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def _gen_data_loaders(self):
        # split_index = [int(self.df.shape[0] * self.config['split'][i] / 10) for i in range(len(self.config['split']))]
        split_index = int(self.df.shape[0] * self.config['split'][0] / 10)

        data_loaders = []
        # print('df train', self.df.iloc[:split_index + 1, :])
        dataset = NyDataset(self.df.iloc[:split_index + 1, :], self.config)
        data_loaders.append(DataLoader(dataset=dataset,
                                       batch_size=self.config['batch_size'],
                                       # num_workers=8,
                                       drop_last=False,
                                       # pin_memory=True,
                                       shuffle=self.config['shuffle'],

                                       ))
        # print('df valid', self.df.iloc[split_index + 1:, :])
        dataset = NyDataset(self.df.iloc[split_index + 1:, :], self.config)
        data_loaders.append(DataLoader(dataset=dataset,
                                       batch_size=self.config['batch_size'],
                                       # num_workers=1,
                                       drop_last=False,
                                       # pin_memory=True,
                                       shuffle=self.config['shuffle'],

                                       ))
        return data_loaders
