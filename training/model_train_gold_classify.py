import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
# https://stackoverflow.com/questions/47985835/tensorboard-is-not-recognized-as-an-internal-or-external-command
from model_loader import LitNetModel
from ny_data_loader import LitNyData
from model.InceptionClassify import InceptionClassify
import pandas as pd
import pickle

# # READ DATA
df = pd.read_csv('hyper_params_data/gold.csv', )  # , parse_dates=True)
df['time'] =  pd.read_csv('hyper_params_data/gold.csv', index_col='time', parse_dates=True)

with open('gold_config_CasualRnn.pkl', 'rb') as f:
    config = pickle.load(f)

logger = TensorBoardLogger("tb_logs", name="gold_model_classify")
trainer = pl.Trainer(
    # gpus=0, ii
    logger=logger,
    max_epochs=300,
    # log_every_n_steps=50,
)
print(config)
config['learning_rate'] = 5e-5
config['batch_size'] = 4
config['num_classes'] = 1
config['hidden_channels'] = 4
config['out_channels'] = 1
config['in_channels'] = 3
config['bottleneck_channels'] = 4
if __name__ == '__main__':
    data_module = LitNyData(df, config)
    # model = LitNetModel(ResidualStacks, config,mode = 'predict')
    model = LitNetModel(InceptionClassify, config, mode='classifier')

    trainer.fit(model, datamodule=data_module)
    #trainer.save_checkpoint("conv_paramsa.ckpt")
