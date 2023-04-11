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


def get_grads(model): 
    return [p.grad for p in model.parameters()]

def set_grads(model, grads): 
    for p, g in zip(model.parameters(), grads): 
        p.grad = g

def grad_dot(grads1, grads2): 
    return torch.dot(
        torch.cat([t.flatten() for t in grads1]),
        torch.cat([t.flatten() for t in grads2]),
    )


def calc_dpm_grad(overall_grads, domain_grads, delta): 
    numerator = -grad_dot(overall_grads, domain_grads) + delta
    denominator = grad_dot(domain_grads, domain_grads)
    factor = numerator / denominator
    return [factor * t_f + t_o for (t_f, t_o) in zip(domain_grads, overall_grads)]