import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
# https://stackoverflow.com/questions/47985835/tensorboard-is-not-recognized-as-an-internal-or-external-command
from model_loader import LitNetModel
from ny_data_loader import LitNyData
from model.ResCnn import ResForcast
from model.WaveNet import WaveNet
from model.InceptionClassify import InceptionClassify
import pandas as pd
import pickle

# # READ DATA
df = pd.read_csv('hyper_params_data/imb_gold.csv', index_col='time', parse_dates=True)
# df = df.loc['2019-01-01':]

with open('hyper_params_data/config_ResForcast.pkl', 'rb') as f:
    config = pickle.load(f)

logger = TensorBoardLogger("tb_logs", name="gold_model_predict")
trainer = pl.Trainer(
    # gpus=0, ii
    logger=logger,
    max_epochs=300,
    #log_every_n_steps=10,
)
print(config)
config['learning_rate'] = 5e-4
config['batch_size'] = 1
config['hidden_channels'] = 4
config['is_newyork'] = False
config['number_days'] = 200
x_config = {
    'res_channels': 128,
    'skip_channels': 32,
    'len_seq_out': 1,
    'num_wave_layer': 6,
    'num_stack_wave_layer': 1,
}
config = config | x_config
if __name__ == '__main__':
    data_module = LitNyData(df, config)
    model = LitNetModel(WaveNet, config, mode='predict')
    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint("checkpoints/ResForcast_params.ckpt")
