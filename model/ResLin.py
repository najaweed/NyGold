import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, is_zero_init=True):
        super(Block, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias)
        self.tanh = nn.Tanh()
        self.init_weights(is_zero_init)

    def init_weights(self, is_zero_init):
        for name, param in self.linear.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                if is_zero_init:
                    nn.init.constant_(param, 0.0)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, x):
        x = self.linear(x)
        return x


class ResBlock(torch.nn.Module):
    def __init__(self,
                 config,
                 ):
        super(ResBlock, self).__init__()
        self.config = config
        self.first_block = Block(in_channels=config['in_channels'],
                                 out_channels=config['out_channels'],
                                 bias=config['bias'],
                                 is_zero_init=False)
        self.res_blocks = nn.ModuleList(self.stack_res_block())

    def stack_res_block(self):
        res_blocks = []
        for _ in range(self.config['num_res_layer']):
            block = Block(in_channels=self.config['out_channels'],
                          out_channels=self.config['out_channels'],
                          bias=self.config['bias'],
                          is_zero_init=True)
            res_blocks.append(block)
        return res_blocks

    def forward(self, x):
        x_hat = self.first_block(x)
        for res_block in self.res_blocks:
            x_hat = x_hat + res_block(x_hat)
        return x_hat


class ResLin(nn.Module):

    def __init__(self,
                 configs: dict,
                 ):
        super(ResLin, self).__init__()
        self.net_trend = ResBlock(configs)
        self.net_seasonal = ResBlock(configs)
        self.net_fractal = ResBlock(configs)

    def forward(self, x):
        trend_in = x[:, 0, :]
        seasonal_in = x[:, 1, :]
        fractal_in = x[:, 2, :]

        trend = self.net_trend(trend_in)
        seasonal = self.net_seasonal(seasonal_in)
        fractal = self.net_fractal(fractal_in)
        x = trend + seasonal + fractal
        return x


# res_config = {
#     'in_channels': 250,
#     'out_channels': 50,
#     'num_res_layer': 3,
#     'bias': True,
# }
# net = ResLin(res_config)
# x_in = torch.rand(1, 3, 250)
# print(net(x_in))
# print(net(x_in).shape)
