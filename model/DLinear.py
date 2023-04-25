import torch
import torch.nn as nn
from model.CausalConv1d import CausalConv1d


class SeasonalNet(nn.Module):

    def __init__(self,
                 configs: dict,
                 ):
        super(SeasonalNet, self).__init__()

        self.net_seasonal = nn.Sequential(CausalConv1d(1,  # configs['in_channels'],
                                                       configs['hidden_channels'],
                                                       configs['kernel']),
                                          CausalConv1d(configs['hidden_channels'],
                                                       1,  # configs['out_channels'],
                                                       configs['kernel'])
                                          )

        self.net_seasonal = nn.Sequential(nn.Linear(configs['seq_len'],
                                                    configs['seq_len'],
                                                    ),
                                          nn.ReLU(),
                                          nn.Linear(configs['seq_len'],
                                                    configs['predict_len'],
                                                    )
                                          )

    def forward(self, x):
        x = self.net_seasonal(x)
        return x


class DDLinear(nn.Module):

    def __init__(self,
                 configs: dict,
                 ):
        super(DDLinear, self).__init__()
        self.seq_len = configs['seq_len']
        self.predict_len = configs['predict_len']

        self.net_trend = nn.Sequential(
            nn.Linear(self.seq_len, 2*self.seq_len, False),
            nn.Linear(2*self.seq_len, self.seq_len, False),
            nn.Linear(self.seq_len, self.predict_len, False),

        )
        self.net_seasonal = nn.Sequential(
            nn.Linear(self.seq_len, 2*self.seq_len, False),
            nn.Linear(2*self.seq_len, self.seq_len, False),
            nn.Linear(self.seq_len, self.predict_len, False),

        )

        self.net_fractal = nn.Sequential(
            nn.Linear(self.seq_len, 2*self.seq_len, False),
            nn.Linear(2*self.seq_len, self.seq_len, False),
            nn.Linear(self.seq_len, self.predict_len, False),
        )

    def forward(self, x):
        trend_in = x[:, 1, :]
        seasonal_in = x[:, 0, :]
        fractal_in = x[:, 2, :]

        trend = self.net_trend(trend_in)
        seasonal = self.net_seasonal(seasonal_in)
        fractal = self.net_fractal(fractal_in)
        x = trend + seasonal + fractal
        return x

# d_config = {
#     'seq_len': 200,
#     'predict_len': 20,
#     'hidden_channels': 64,
#     'kernel': 5,
# }
# model = DDLinear(d_config)
# x_in = torch.rand(2, 3, 200)
# print(model(x_in).shape)
