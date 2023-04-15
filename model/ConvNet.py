import torch
import torch.nn as nn
from model.CausalConv1d import CausalConv1d


class ConvNet(torch.nn.Module):
    def __init__(self,
                 config
                 ):
        super(ConvNet, self).__init__()
        self.len_seq_out = config['step_prediction']+config['step_share']
        #self.batch_norm = nn.BatchNorm1d(config['in_channels'])
        self.causal = CausalConv1d(config['in_channels'], config['hidden_channels'], config['kernel_size'])
        self.predict = CausalConv1d(config['hidden_channels'], config['out_channels'], config['kernel_size'])
        # self.predict = PredictorBlock(config['hidden_channels'] * config['len_seq_out'],
        #                               config['out_channels'] * config['len_seq_out'], )

    def forward(self, x):
        #x = self.batch_norm(x)
        x = self.causal(x)
        x = self.predict(x)
        x = x[:, :, -self.len_seq_out:]

        return x

# x_ = torch.rand(1, 5, 400)
# x_config = {
#     'in_channels': 5,
#     'hidden_channels': 16,
#     'out_channels': 2,
#     'len_seq_out': 4,
#     'kernel_size': 8,
# }
# net =ConvNet(x_config)
# skip_ = net(x_)  # skip_size=3)
# print(skip_.shape)
# print(net.state_dict().keys())
