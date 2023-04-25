import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger  # on cmd ::: python -m tensorboard.main --logdir=

from model_loader import LitNetModel
from testing_desk.ny_data_loader import LitNyData

from model.DLinear import DDLinear


logger = TensorBoardLogger("tb_logs", name="gold_model_predict")
trainer = pl.Trainer(
    logger=logger,
    max_epochs=200,
    log_every_n_steps=1,
)

# READ DATA
df = pd.read_csv('hyper_params_data/D_gold.csv',
                 index_col='time',
                 parse_dates=True)

config = {
    # config dataset and dataloader
    'batch_size': 8,
    'learning_rate': 1e-4,
    'window_temporal': 300,
    'split': (9, 1),
    'step_prediction': 5,
    'step_share': 5,
    'shuffle': True,
}

d_config = {
    'seq_len': config['window_temporal'] - config['step_prediction'],
    'predict_len': config['step_prediction'] + config['step_share'],
}
config = config | d_config
if __name__ == '__main__':
    data_module = LitNyData(df, config)
    model = LitNetModel(DDLinear, config, mode='predict')
    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint("checkpoints/ResForcast_params.ckpt")

    #pd.DataFrame(config).to_csv('checkpoints/config.csv')
