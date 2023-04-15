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

        self.dilated = CausalConv1d(res_channels, res_channels, dilation=dilation, kernel_size=2)
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
        self.res_blocks = nn.ModuleList(self.stack_res_block(config['res_channels'], config['skip_channels']))

    @staticmethod
    def _residual_block(res_channels, skip_channels, dilation):
        block = ResidualBlock(res_channels, skip_channels, dilation)

        if torch.cuda.device_count() > 1:
            block = torch.nn.DataParallel(block)

        if torch.cuda.is_available():
            block.cuda()

        return block

    def build_dilations(self):
        stacks = []
        # 5 = stack[layer1, layer2, layer3, layer4, layer5]
        for s in range(0, self.stack_size):
            dilations = []
            # 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
            for l_ in range(0, self.layer_size):
                dilations.append(2 ** l_)
            stacks.append(dilations)
        return stacks

    def stack_res_block(self, res_channels, skip_channels):
        res_blocks = []
        stacks = self.build_dilations()
        for stack in stacks:
            for dilation in stack:
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
        self.linear_1 = nn.Linear(in_features=in_features,
                                  out_features=out_features,
                                  bias=True)
        self.predictor = nn.Sequential(
            # nn.Linear(in_features=in_features,
            #           out_features=2*in_features,
            #           bias=True),
            # nn.ReLU(),
            # nn.Linear(in_features=2*in_features,
            #           out_features=in_features,
            #           bias=True),
            # nn.ReLU(),
            nn.Linear(in_features=in_features,
                      out_features=out_features,
                      bias=False),
        )

    def forward(self, x):
        x = self.predictor(x)
        return x


class WaveNet(torch.nn.Module):
    def __init__(self,
                 config
                 ):
        super(WaveNet, self).__init__()
        self.len_seq_out = config['step_prediction']+config['step_share']
        self.causal = CausalConv1d(config['in_channels'], config['res_channels'], config['kernel_size'])
        self.res_stack = ResidualStack(config)

        self.config = config

    def forward(self, x):
        x = self.causal(x)
        skip_connections = self.res_stack(x, self.len_seq_out)
        x = torch.sum(skip_connections, dim=0)
        return x[:, :, -self.len_seq_out:]
#
# x_ = torch.rand(1, 5, 400)
# x_config = {
#     'in_channels': 5,
#     'res_channels': 16,
#     'skip_channels': 2,
#     'out_channels': 2,
#     'num_wave_layer': 8,
#     'num_stack_wave_layer': 8,
#     'step_prediction': 1,
#     'step_share': 3,
#     'kernel_size': 8,
# }
#
# net = WaveNet(x_config)
# skip_ = net(x_)  # skip_size=3)
# print(skip_.shape)
# print(net.state_dict().keys())
