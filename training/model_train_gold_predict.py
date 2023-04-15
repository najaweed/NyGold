import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
# https://stackoverflow.com/questions/47985835/tensorboard-is-not-recognized-as-an-internal-or-external-command
from model_loader import LitNetModel
from ny_data_loader import LitNyData
from model.ResCnn import ResForcast
from model.WaveNet import WaveNet
from model.ConvNet import ConvNet

from model.InceptionClassify import InceptionClassify
import pandas as pd
import pickle

logger = TensorBoardLogger("tb_logs", name="gold_model_predict")
trainer = pl.Trainer(
    # gpus=0, ii
    logger=logger,
    max_epochs=200,
    log_every_n_steps=1,
    precision=64,
)

# READ DATA
df = pd.read_csv('hyper_params_data/D_gold.csv',
                 index_col='time',
                 parse_dates=True)
# with open('hyper_params_data/config_ResForcast.pkl', 'rb') as f:
#     config = pickle.load(f)
config_data_loader = {
    # config dataset and dataloader
    'batch_size': 16,
    'learning_rate': 1e-4,
    'window_temporal': 128,
    'split': (9, 1),
    'in_channels': 2,
    'step_prediction': 1,
    'step_share': 0,
    'out_channels': 2,
}
model_config = {
    'hidden_channels': 8,
    'kernel_size': 2,
    'res_channels': 32,
    'skip_channels': 2,
    'out_channels': 2,
    'num_wave_layer': 6,
    'num_stack_wave_layer': 1,
}
config = config_data_loader | model_config

if __name__ == '__main__':
    data_module = LitNyData(df, config)
    model = LitNetModel(WaveNet, config, mode='predict')
    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint("checkpoints/ResForcast_params.ckpt")
