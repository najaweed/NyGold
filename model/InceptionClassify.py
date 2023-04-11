import torch
import torch.nn as nn
from model.InceptionTime import InceptionBlock


class InceptionClassify(torch.nn.Module):
    def __init__(self,
                 config,
                 ):
        super(InceptionClassify, self).__init__()
        self.config = config
        self.classier = InceptionBlock(in_channels=config['in_channels'],
                                       n_filters=config['hidden_channels'],
                                       bottleneck_channels=config['bottleneck_channels'],
                                       )
        self.avg_pool = nn.AvgPool1d(kernel_size=4)
        self.lin = nn.LazyLinear(1)
        self.sig = nn.Sigmoid()
        self.init_weights()
        self.drop = nn.Dropout(0.6)

    def init_weights(self):
        for name, param in self.classier.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, x):
        x = self.classier(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.drop(x)
        x = self.lin(x)
        x = self.sig(x)
        print(x)
        return x

# x_in = torch.rand(12,3,43)
# x_config = {'in_channels':3,
#             'hidden_channels':16,
#             'bottleneck_channels':32}
# model = ClassifyBlock(x_config)
# x_out = model(x_in)
# print(x_out)
# print(x_out.shape)

class ClassifyBlock(torch.nn.Module):
    def __init__(self,
                 config,
                 ):
        super(ClassifyBlock, self).__init__()
        self.config = config
        self.classier = nn.Sequential(
            nn.LazyLinear(2*512, bias=False),
            nn.ReLU(),
            nn.LazyLinear(512, bias=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LazyLinear(256, bias=False),
            nn.ReLU(),
            nn.LazyLinear(128, bias=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LazyLinear(config['num_classes'], bias=True),
            nn.Sigmoid(),
        )
        self.soft = nn.Softmax(dim=1)

        # self.init_weights()

    # def init_weights(self):
    #     for name, param in self.classier.named_parameters():
    #         if 'bias' in name:
    #             nn.init.constant_(param, 0.0)
    #         else:
    #             nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, x):
        x = self.classier(x)
        #x = self.soft(x)
        return x


class ClassifierResidualStacks(torch.nn.Module):
    def __init__(self,
                 config,
                 ):
        super(ClassifierResidualStacks, self).__init__()
        self.config = config
        self.encoder = self.load_encoder(config)

        self.classifier = ClassifyBlock(config)
        self.avg_pool_1 = nn.AvgPool1d(kernel_size=8)
        self.avg_pool_2 = nn.AvgPool1d(kernel_size=8)

    @staticmethod
    def load_encoder(config, path: str ):
        model = ResidualStacks(config)
        model_weights = torch.load(path)["state_dict"]
        # update keys by dropping `nn_model.`
        for key in list(model_weights):
            model_weights[key.replace("nn_model.", "")] = model_weights.pop(key)
        model.load_state_dict(model_weights)
        model.eval()
        x_encoder = model.res_1
        return x_encoder

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        x = self.avg_pool_1(x)
        #x = self.avg_pool_2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
