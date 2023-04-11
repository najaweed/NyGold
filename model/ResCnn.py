import math

import torch
import torch.nn as nn
from model.CausalConv1d import CausalConv1d


class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel,):
        super(Block, self).__init__()
        # self.batch_norm = nn.BatchNorm1d(in_channels)
        self.dilated = CausalConv1d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel,
                                    dilation=dilation,
                                    bias=True)

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.dilated(x)
        gated_tanh = self.gate_tanh(x)
        gated_sigmoid = self.gate_sigmoid(x)
        x = gated_tanh * gated_sigmoid
        return x


class ResidualStack(torch.nn.Module):
    def __init__(self,
                 config,
                 ):
        super(ResidualStack, self).__init__()
        self.config = config
        self.first_block = Block(in_channels=self.config['in_channels'],
                                 out_channels=self.config['hidden_channels'],
                                 dilation=self.config['dilation'][0],
                                 kernel=config['kernel'])
        self.res_blocks = nn.ModuleList(self.stack_res_block())

        #self.init_weights()

    def init_weights(self):
        for name, param in self.first_block.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -0.6, 0.6)
        for name, param in self.res_blocks.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -.1, .1)

    def stack_res_block(self):
        res_blocks = []
        for dilation in self.config['dilation']:
            block = Block(in_channels=self.config['hidden_channels'],  # self.config['in_channels'],
                          out_channels=self.config['hidden_channels'],
                          dilation=dilation,
                          kernel=self.config['kernel'])
            # dropout=self.config['dropout'])
            res_blocks.append(block)
        return res_blocks

    def forward(self, x):
        x_hat = self.first_block(x)
        for res_block in self.res_blocks:
            x_hat = x_hat + res_block(x_hat)
        return x_hat


class PredictorBlock(torch.nn.Module):
    def __init__(self,
                 config,
                 ):
        super(PredictorBlock, self).__init__()
        self.config = config
        self.predictor = nn.Linear(in_features=config['hidden_channels'],
                                   out_features=config['out_channels'],
                                   bias=True)
        # Block(in_channels=self.config['hidden_channels'],
        #                        out_channels=self.config['out_channels'],
        #                        dilation=self.config['dilation'][0])

        #self.init_weights()

    def init_weights(self):
        for name, param in self.predictor.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, x):
        x = x[:, :, -1]
        x = self.predictor(x)
        return x


class ResForcast(torch.nn.Module):
    def __init__(self,
                 config,
                 ):
        super(ResForcast, self).__init__()
        self.config = config
        self.res_1 = ResidualStack(config)
        # self.res_2 = ResidualStack(config)
        # self.res_3 = ResidualStack(config)
        # self.res_4 = ResidualStack(config)
        self.predictor = PredictorBlock(config)
        # self.classifier = ClassifyBlock(config)
        self.drop = nn.Dropout(.5)

    def forward(self, x):
        x = self.res_1(x)
        x_predict = self.predictor(x)
        # x_classify = self.classifier(x[:, :, -1])
        # print(x_classify)
        return x_predict  # , x_classify

#
# x_in = torch.rand(12, 3, 16)
#
# # print(x_in)
# x_config = {
#     'in_channels': 3,
#     'out_channels': 2,
#     'hidden_channels': 64,
#     'kernel': 3,
#     'dilation': [2 ** i for i in range(8)],
#     'step_share': 2,
#     'dropout': 0.1,
#     'num_classes': 2,
# }
# model = ResForcast(x_config)
# # model = ClassifyBlock(x_config)
#
# # print(model.state_dict().keys())
# y_hat = model(x_in)
# # print(model.res_blocks.named_parameters())
# # print(y_hat.shape)
# print(y_hat)
# print(y_hat.shape)
