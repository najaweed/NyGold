import pickle
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger  # on cmd ::: python -m tensorboard.main --logdir=

from dataloader.LitData import LitData
from model_loader import LitNetModel
from model.DLinear import DDLinear
from model.ResLin import ResLin
logger = TensorBoardLogger("tb_logs", name="gold")
trainer = pl.Trainer(
    logger=logger,
    max_epochs=200,
    log_every_n_steps=1,
)

data_config = {
    'batch_size': 1,
    'shuffle': True,
    'path_train_obs': 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject3\\data\\database\\train_obs.npy',
    'path_train_predict': 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject3\\data\\database\\train_predict.npy',
    'path_valid_obs': 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject3\\data\\database\\valid_obs.npy',
    'path_valid_predict': 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject3\\data\\database\\valid_predict.npy',
    'window_temporal': 60,
    'split': (95, 5),
    'step_prediction': 5,
    'step_share': 0,
    'fold': 0,
}


model_config = {
    'in_channels': data_config['window_temporal'] - data_config['step_prediction'],
    'out_channels': data_config['step_prediction'] + data_config['step_share'],
    'num_res_layer': 1,
    'bias': True,
}
config = data_config | model_config
config['learning_rate'] = 3e-4
if __name__ == '__main__':
    data_module = LitData(data_config)
    model = LitNetModel(ResLin, config, mode='predict')
    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint("checkpoints/ResForcast_params.ckpt")

    with open('checkpoints/config.pkl', 'wb') as f:
        pickle.dump(config, f)
