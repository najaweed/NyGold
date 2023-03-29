import torch
import pytorch_lightning as pl


class LitNetModel(pl.LightningModule, ):

    def __init__(self,
                 net_model,
                 config: dict,
                 is_encoder=False,
                 ):
        super().__init__()

        # configuration
        self.config = config
        self.lr = config['learning_rate']
        self.is_encoder = is_encoder
        # model initialization
        self.nn_model = net_model(config)
        self.loss = torch.nn.MSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # prepare inputs
        x_in = train_batch[0]
        x_in = torch.permute(x_in, (0, 2, 1))
        # process model
        x = self.nn_model(x_in)
        # criterion
        target = train_batch[1]
        if self.config['step_share'] > 0:
            target = torch.permute(target, (0, 2, 1))
        if self.is_encoder:
            loss = torch.sqrt(self.loss(x, x_in))
        else:
            loss = torch.sqrt(self.loss(x, target))
        # logger
        metrics = {'loss': loss, }
        self.log_dict(metrics)
        return metrics

    def validation_step(self, val_batch, batch_idx):
        # prepare inputs
        x_in = val_batch[0]
        x_in = torch.permute(x_in, (0, 2, 1))
        # process model
        x = self.nn_model(x_in)
        # criterion
        target = val_batch[1]
        if self.config['step_share'] > 0:
            target = torch.permute(target, (0, 2, 1))
        if self.is_encoder:
            loss = torch.sqrt(self.loss(x, x_in))
        else:
            loss = torch.sqrt(self.loss(x, target))
        # logger
        metrics = {'val_loss': loss, }
        print('val_loss', loss)
        self.log_dict(metrics)
        return metrics
