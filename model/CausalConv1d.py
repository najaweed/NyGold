import torch


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, x):
        x = super(CausalConv1d, self).forward(x)
        if self.__padding != 0:
            return x[:, :, :-self.__padding]
        return x


# x = torch.ones(1, 6, 16)
# net = CausalConv1d(in_channels=6, out_channels=1, kernel_size=1)
# y = net(x)
#
# print(y.shape)
