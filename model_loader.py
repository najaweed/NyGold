import torch
import pytorch_lightning as pl


class LitNetModel(pl.LightningModule, ):

    def __init__(self,
                 net_model,
                 config: dict,
                 mode='predict'
                 ):
        super().__init__()

        # configuration
        self.config = config
        self.mode = mode
        self.lr = config['learning_rate']
        # model initialization
        self.nn_model = net_model(config)
        self.loss_mse = torch.nn.MSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, )
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose=True)
        return optimizer  # [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        # prepare inputs
        x_in = train_batch[0]
        # process model
        x = self.nn_model(x_in)
        # criterion
        target = train_batch[1]
        loss = self.loss_mse(x, target)  # torch.sqrt(self.loss_mse(x, target))
        # logger
        metrics = {'loss': loss, }
        self.log_dict(metrics)
        return metrics

    def validation_step(self, val_batch, batch_idx):
        # prepare inputs
        x_in = val_batch[0]
        # process model
        x = self.nn_model(x_in)
        # criterion
        target = val_batch[1]
        loss = self.loss_mse(x, target)  # torch.sqrt(self.loss_mse(x, target))

        # logger
        self.log_dict({'val_loss': loss, })
        return {'val_loss': loss, }
