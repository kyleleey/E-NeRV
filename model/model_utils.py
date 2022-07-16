from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, pe_embed_b, pe_embed_l):
        super(PositionalEncoding, self).__init__()
        if pe_embed_b == 0:
            self.embed_length = 1
            self.pe_embed = False
        else:
            self.lbase = float(pe_embed_b)
            self.levels = float(pe_embed_l)
            self.levels = int(self.levels)
            self.embed_length = 2 * self.levels
            self.pe_embed = True
    
    def __repr__(self):
        return f"Positional Encoder: pos_b={self.lbase}, pos_l={self.levels}, embed_length={self.embed_length}, to_embed={self.pe_embed}"

    def forward(self, pos):
        if self.pe_embed is False:
            return pos[:,None]
        else:
            pe_list = []
            for i in range(self.levels):
                temp_value = pos * self.lbase **(i) * math.pi
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
            return torch.stack(pe_list, 1)


class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(30 * input)  # see SIREN paper for the factor 30


def sin_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class SIREN(nn.Module):
    def __init__(self, dim_list, bias=True, first_layer_init=True):
        super(SIREN, self).__init__()

        nl = Sin()
        weight_init = sin_init
        if first_layer_init:
            first_init = first_layer_sine_init
        else:
            first_init = None

        fc_list = []
        for i in range(len(dim_list) - 1):
            dim_in = dim_list[i]
            dim_out = dim_list[i+1]
            fc_list += [nn.Linear(dim_in, dim_out, bias=bias), nl]
        self.fc = nn.Sequential(*fc_list)
        # init
        self.fc.apply(weight_init)
        if first_init is not None:
            self.fc[0].apply(first_init)
    
    def forward(self, x):
        return self.fc(x)


def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        # act_layer = torch.sin
        act_layer = Sin()
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    elif act_type == 'non':
        act_layer = nn.Identity()
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):    
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


def gradient(y, t):
    grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad([y], [t], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad