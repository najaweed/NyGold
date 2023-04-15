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
        self.l1_loss = torch.nn.L1Loss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, )
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose=True)
        return optimizer  # [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        # prepare inputs
        x_in = train_batch[0]
        x_in = torch.permute(x_in, (0, 2, 1))
        # process model
        x = self.nn_model(x_in)
        # criterion
        target = train_batch[1]
        loss = torch.sqrt(self.loss_mse(x, target[:, :2]))  # self._calculate_loss(x, target,)
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
        # print(x[-1,...] ,'',target[-1,...])
        loss = torch.sqrt(self.loss_mse(x, target[:, :2]))  # self._calculate_loss(x, target,)

        # logger
        ac_loss =self._inv_normal_target(x, target)
        ac_metric = {'price_loss':ac_loss}
        #print('val loss', loss)
        self.log_dict({'val_loss': loss,'price_loss':ac_metric })
        return {'val_loss': loss,'price_loss':ac_metric }

    def _inv_normal_target(self, nn_output, target):
        high_low = target[:, 4:]
        #print('out net', nn_output)
        #print('target', target[:, :2])

        nn_output *= target[:, 2:3]
        nn_output += target[:, 3:4]
        #print('inv net', nn_output, nn_output.shape)
        #print('inv target', high_low, high_low.shape)

        #print('diff', nn_output - high_low)
        #print('l1diff', self.l1_loss(nn_output, high_low))
        return self.l1_loss(nn_output, high_low)
