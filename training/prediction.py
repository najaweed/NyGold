import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from dataloader.PredictDataLoader import PredictDataLoader
from model.DLinear import DDLinear
from model.ResLin import ResLin


def load_model(c_model, c_config, path_params):
    x_model = c_model(c_config)
    checkpoint = torch.load(path_params)
    model_weights = checkpoint["state_dict"]

    # update keys by dropping `nn_model.`
    for key in list(model_weights):
        model_weights[key.replace("nn_model.", "")] = model_weights.pop(key)
    x_model.load_state_dict(model_weights)
    x_model.eval()
    print('model loaded with learned parameters, ready for predict')
    return x_model


with open('checkpoints/config.pkl', 'rb') as f:
    config = pickle.load(f)
config['batch_size'] = 1

model = load_model(ResLin, config, "checkpoints/ResForcast_params.ckpt")
lit_data = PredictDataLoader(config)
data_loader = lit_data.val_dataloader()

for i, batch in enumerate(data_loader):
    x_in = batch[0]
    target = batch[1].detach().numpy()
    x = model(x_in)
    x = x.detach().numpy()

    if config['step_share'] > 0:
        plt.axvline(config['step_share'])
    plt.plot(x.flatten(), c='r')
    plt.plot(target.flatten(), '--', c='g')

    plt.show()
