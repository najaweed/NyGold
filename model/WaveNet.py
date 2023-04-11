import torch
import torch.nn as nn
from model.CausalConv1d import CausalConv1d


class DilatedCausalConv1d(torch.nn.Module):
    """Dilated Causal Convolution for WaveNet"""

    def __init__(self, channels, dilation=1):
        super(DilatedCausalConv1d, self).__init__()

        self.conv = torch.nn.Conv1d(channels, channels,
                                    kernel_size=2, stride=1,  # Fixed for WaveNet
                                    dilation=dilation,
                                    padding=0,  # Fixed for WaveNet dilation
                                    bias=False)  # Fixed for WaveNet but not sure

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        return output


class ResidualBlock(torch.nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        """
        Residual block
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :param dilation:
        """
        super(ResidualBlock, self).__init__()
        super(ResidualBlock, self).__init__()

        self.dilated = DilatedCausalConv1d(res_channels, dilation=dilation)
        self.conv_res = torch.nn.Conv1d(res_channels, res_channels, 1)
        self.conv_skip = torch.nn.Conv1d(res_channels, skip_channels, 1)

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x, skip_size):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output = self.dilated(x)

        # PixelCNN gate
        gated_tanh = self.gate_tanh(output)
        gated_sigmoid = self.gate_sigmoid(output)
        gated = gated_tanh * gated_sigmoid

        # Residual network
        output = self.conv_res(gated)
        input_cut = x[:, :, -output.size(2):]
        output += input_cut

        # Skip connection
        skip = self.conv_skip(gated)
        skip = skip[:, :, -skip_size:]

        return output, skip


class ResidualStack(torch.nn.Module):
    def __init__(self,
                 config,
                 ):

        super(ResidualStack, self).__init__()

        self.layer_size = config['num_wave_layer']
        self.stack_size = config['num_stack_wave_layer']

        self.res_blocks = self.stack_res_block(config['res_channels'], config['skip_channels'])

    @staticmethod
    def _residual_block(res_channels, skip_channels, dilation):
        block = ResidualBlock(res_channels, skip_channels, dilation)

        if torch.cuda.device_count() > 1:
            block = torch.nn.DataParallel(block)

        if torch.cuda.is_available():
            block.cuda()

        return block

    def build_dilations(self):
        dilations = []
        # 5 = stack[layer1, layer2, layer3, layer4, layer5]
        for s in range(0, self.stack_size):
            # 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
            for l_ in range(0, self.layer_size):
                dilations.append(2 ** l_)
        return dilations

    def stack_res_block(self, res_channels, skip_channels):
        res_blocks = []
        dilations = self.build_dilations()
        for dilation in dilations:
            block = self._residual_block(res_channels, skip_channels, dilation)
            res_blocks.append(block)
        return res_blocks

    def forward(self, x, skip_size):
        output = x
        skip_connections = []
        for res_block in self.res_blocks:
            output, skip = res_block(output, skip_size)
            skip_connections.append(skip)

        return torch.stack(skip_connections)


class PredictorBlock(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 ):
        super(PredictorBlock, self).__init__()
        self.predictor = nn.Linear(in_features=in_features,
                                   out_features=out_features,
                                   bias=True)

    def init_weights(self):
        for name, param in self.predictor.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, x):
        x = self.predictor(x)
        return x


class WaveNet(torch.nn.Module):
    def __init__(self,
                 config
                 ):
        super(WaveNet, self).__init__()
        self.len_seq_out = config['len_seq_out']

        self.causal = CausalConv1d(config['in_channels'], config['res_channels'], kernel_size=1)
        self.res_stack = ResidualStack(config)
        self.predict = PredictorBlock(config['skip_channels'] * config['len_seq_out'],
                                      config['out_channels'] * config['len_seq_out'],
                                      )
        self.config = config

    def forward(self, x):
        x = self.causal(x)
        skip_connections = self.res_stack(x, self.len_seq_out)
        x = torch.sum(skip_connections, dim=0)
        x = torch.flatten(x, start_dim=1)
        x = self.predict(x)
        if self.config['len_seq_out'] > 1:
            x = x.view(x.shape[0], self.config['out_channels'], self.config['len_seq_out'], )
        return x

#
# x_ = torch.rand(1, 5, 200)
# x_config = {
#     'in_channels': 5,
#     'res_channels': 16,
#     'skip_channels': 8,
#     'out_channels': 2,
#     'len_seq_out': 1,
#     'num_wave_layer': 4,
#     'num_stack_wave_layer': 3,
# }
# net = WaveNet(x_config)
# skip_ = net(x_)  # skip_size=3)
# print(skip_.shape)
