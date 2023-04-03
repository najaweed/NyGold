import math

import torch
import torch.nn as nn
from model.CausalConv1d import CausalConv1d


class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilation, dropout=0.0):
        super(Block, self).__init__()
        # self.batch_norm = nn.BatchNorm1d(in_channels)
        self.is_dropout = dropout
        self.dilated = CausalConv1d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=2,
                                    dilation=dilation,
                                    bias=True)

        self.gate_tanh = torch.nn.Tanh()
        # self.gate_sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x):
        # x = self.batch_norm(x)
        x = self.dilated(x)
        gated_tanh = self.gate_tanh(x)
        # gated_sigmoid = self.gate_sigmoid(x)
        x = gated_tanh  # * gated_sigmoid
        if self.is_dropout > 0:
            x = self.dropout(x)
        return x


class ResidualStack(torch.nn.Module):
    def __init__(self,
                 config,
                 ):
        super(ResidualStack, self).__init__()
        self.config = config
        self.first_block = Block(in_channels=self.config['in_channels'],
                                 out_channels=self.config['hidden_channels'],
                                 dilation=self.config['dilation'][0])
        self.res_blocks = nn.ModuleList(self.stack_res_block())
        self.last_block = Block(in_channels=self.config['hidden_channels'],
                                out_channels=self.config['out_channels'],
                                dilation=self.config['dilation'][0])
        self.init_weights()

    def init_weights(self):
        for name, param in self.first_block.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                # nn.init.normal_(param, mean=0, std=1/math.sqrt(self.config['in_channels']))
                nn.init.uniform_(param, -0.1, 0.1)
        for name, param in self.res_blocks.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                # nn.init.normal_(param, mean=0, std=0.001)
                nn.init.uniform_(param, -.001, .001)
        for name, param in self.last_block.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                # nn.init.normal_(param, mean=0, std=1/math.sqrt(self.config['hidden_channels']))
                nn.init.uniform_(param, -0.1, 0.1)

    def stack_res_block(self):
        res_blocks = []
        for dilation in self.config['dilation']:
            block = Block(in_channels=self.config['hidden_channels'],  # self.config['in_channels'],
                          out_channels=self.config['hidden_channels'],
                          dilation=dilation,
                          dropout=self.config['dropout'])
            res_blocks.append(block)
        return res_blocks

    def forward(self, x):
        x_hat = self.first_block(x)
        for res_block in self.res_blocks:
            x_hat = x_hat + res_block(x_hat)
        x_hat = self.last_block(x_hat)
        return x_hat  # [:, :,-1]


class ResidualStacks(torch.nn.Module):
    def __init__(self,
                 config,
                 ):
        super(ResidualStacks, self).__init__()
        self.config = config
        self.res_1 = ResidualStack(config)
        #elf.res_2 = ResidualStack(config)
        # self.res_3 = ResidualStack(config)
        # self.res_4 = ResidualStack(config)

    def forward(self, x):
        x_1 = self.res_1(x)
        #x_1 = x_1 + self.res_2(x)
        # x_1 = self.res_3(x_1)
        # x_1 = self.res_4(x_1)
        # x_c = self.callif(x_1)
        if self.config['step_share'] > 0:
            return x_1[:, :, -(self.config['step_share'] + 1):]
        else:
            return x_1[:, :, -1]#,x_c

# x_in = torch.rand(1, 3, 17)
# x_in[-1, -1, -1] = 1
# x_in[-1, -1, 0] = 2
# # x_in[-1, -1, 6] = 1
#
# # print(x_in)
# x_config = {
#     'in_channels': 3,
#     'out_channels': 3,
#     'hidden_channels': 64,
#     'kernels': 2,
#     'dilation': [2 ** i for i in range(6)],
#     'step_share':2,
#     'dropout':0.1,
# }
# model = ResidualStack(x_config)
# # print(model.state_dict().keys())
# y_hat = model(x_in)
# # print(model.res_blocks.named_parameters())
# # print(y_hat.shape)
# print(y_hat)
# print(y_hat.shape)
