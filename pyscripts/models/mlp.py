import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_mlp(n_hidden_layers, input_dim, hidden_dim, output_dim, dropout=0):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    for i in range(n_hidden_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)
