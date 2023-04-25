import torch
import torch.nn as nn


class PredictorBlock(torch.nn.Module):
    def __init__(self,
                 config,
                 ):
        super(PredictorBlock, self).__init__()
        self.config = config
        self.lin_1 = nn.Linear(in_features=config['in_channels'] * config['window_obs'],
                               out_features=config['out_channels'] * config['window_predict'],
                               bias=False)
        # Block(in_channels=self.config['hidden_channels'],
        #                        out_channels=self.config['out_channels'],
        #                        dilation=self.config['dilation'][0])
        self.lin = nn.Sequential(
            nn.Linear(in_features=config['in_channels'] * config['window_obs'],
                      out_features=config['in_channels'] * config['window_predict'],
                      bias=False),
            nn.ReLU(),
            nn.Linear(in_features=config['in_channels'] * config['window_predict'],
                      out_features=config['out_channels'] * config['window_predict'],
                      bias=False)
        )
        self.init_weights()
        self.batch_norm = nn.BatchNorm1d(config['in_channels'])

    def init_weights(self):
        for name, param in self.lin.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, x):
        # print(x.shape)
        x = self.batch_norm(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.lin(x)
        # print(x.shape)

        return x


# net_config = {
#     'in_channels': 1,
#     'out_channels': 1,
#     'window_obs': 200,
#     'window_predict': 20,
# }
# model = PredictorBlock(net_config)
# x_in = torch.rand(1, 1, net_config['window_obs'])
# print(model(x_in))
