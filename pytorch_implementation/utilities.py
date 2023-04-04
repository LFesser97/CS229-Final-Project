import torch
from torch import nn


def init_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)


def get_model(input_size, output_size, n_hidden=5, hidden_width=100, activation=nn.Tanh()):
    layers = nn.ModuleList()
    layer_dims = [input_size, *(n_hidden*[hidden_width])]
    for i, j in zip(layer_dims, layer_dims[1:]):
        layers.append(nn.Linear(i, j))
        layers.append(activation)
    layers.append(nn.Linear(layer_dims[-1], output_size))

    model = nn.Sequential(*layers)
    model.apply(init_xavier)
    return model

