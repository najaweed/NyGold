import pandas as pd
import numpy as np
from preprocess.ProProcessY import PreProcess


class GenDataSet:
    def __init__(self,
                 df,
                 config
                 ):
        self.df = df
        self.config = config

    def save_datasets(self):
        train_data, valid_data = self.gen_datasets()

        np.save('database/train_obs.npy', train_data[0])
        np.save('database/train_predict.npy', train_data[1])
        np.save('database/valid_obs.npy', valid_data[0])
        np.save('database/valid_predict.npy', valid_data[1])

    def gen_datasets(self):
        df_train, df_valid = self.split_train_valid_df()
        train_data = self.split_observation_prediction(df_train)
        valid_data = self.split_observation_prediction(df_valid)
        return train_data, valid_data

    def split_observation_prediction(self, df: pd.DataFrame):
        x, y = [], []
        in_shape, target_shape = None, None
        for t in range(0, len(df) - self.config['window_temporal']):
            x_df = df.iloc[t:(t + self.config['window_temporal']), :].copy()
            ny_normal = PreProcess(x_df, self.config['step_prediction'], self.config['step_share'])
            obs = ny_normal.obs()
            target = ny_normal.target()
            if t == 0:
                in_shape = obs.shape
            if obs.shape == in_shape:
                x.append(obs)
                y.append(target)

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return x, y

    def split_train_valid_df(self):
        train_indexes, valid_indexes = self._get_split_indexes()
        df_valid = self.df.iloc[valid_indexes[0]:valid_indexes[1], :].copy()
        if isinstance(train_indexes, tuple):
            df_train = []
            for train_index in train_indexes:
                df_train.append(self.df.iloc[train_index[0]:train_index[1], :].copy)
        else:
            df_train = self.df.iloc[train_indexes[0]:train_indexes[1], :].copy()
        return df_train, df_valid

    def _get_split_indexes(self):
        len_valid = int(self.df.shape[0] * self.config['split'][1] / 100)
        valid_indexes = self.__get_index_fold_valid(len_valid)
        train_indexes = self.__get_index_fold_train(valid_indexes)
        return train_indexes, valid_indexes

    def __get_index_fold_valid(self, len_valid):
        start_index = len(self.df) - len_valid - self.config['fold'] * len_valid
        end_index = len(self.df) - self.config['fold'] * len_valid
        return [start_index, end_index]

    def __get_index_fold_train(self, valid_indexes):
        if self.config['fold'] == 0:
            return [0, valid_indexes[0]]
        elif self.config['fold'] == 9:
            return [valid_indexes[1], len(self.df)]
        else:
            train_indexes_1 = [0, valid_indexes[0]]
            train_indexes_2 = [valid_indexes[1], len(self.df)]
            return train_indexes_1, train_indexes_2


x_config = {
    'window_temporal': 60,
    'split': (95, 5),
    'step_prediction': 5,
    'step_share': 0,
    'fold': 0,
}

# READ DATA
df = pd.read_csv('database/D_gold.csv',
                 index_col='time',
                 parse_dates=True)
dataset = GenDataSet(df, x_config)
dataset.save_datasets()
