import torch
from torch.utils.data import Dataset, TensorDataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader


class PredictDataLoader:

    def __init__(self,
                 config: dict,
                 ):
        self.config = config

    @staticmethod
    def _load_dataset(path: str):
        x = np.load(path)
        return torch.tensor(x, dtype=torch.float32)

    def val_dataloader(self):
        obs = self._load_dataset(self.config['path_valid_obs'])
        predict = self._load_dataset(self.config['path_valid_predict'])
        dataset = TensorDataset(obs, predict)
        return DataLoader(dataset=dataset,
                          batch_size=self.config['batch_size'],
                          # num_workers=8,
                          drop_last=False,
                          # pin_memory=True,
                          shuffle=False,
                          )


# data_config = {
#     'batch_size': 1,
#     'shuffle': True,
#     'path_train_obs': 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject3\\data\\database\\train_obs.npy',
#     'path_train_predict': 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject3\\data\\database\\train_predict.npy',
#     'path_valid_obs': 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject3\\data\\database\\valid_obs.npy',
#     'path_valid_predict': 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject3\\data\\database\\valid_predict.npy',
# }

# lit_data = PredictDataLoader(data_config)
# data_loader = lit_data.val_dataloader()

# import matplotlib.pyplot as plt
#
# for i, batch in enumerate(data_loader):
#     obs = batch[0].squeeze(0)
#     plt.plot(obs[0, :])
#     plt.plot(obs[0, :] + obs[1, :])
#     plt.plot(obs[0, :] + obs[1, :] + obs[2, :])
#
#     plt.show()
