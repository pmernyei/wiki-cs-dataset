import argparse
import torch.nn as nn

import load_graph_data
from load_graph_data import register_data_args
from train import train_and_eval
from train import register_general_args


def mlp_model_fn(args, data):
    layers = []
    layers.append(nn.Linear(data.n_feats, args.n_hidden))
    for i in range(args.n_layers - 1):
        layers.append(nn.Linear(args.n_hidden, args.n_hidden))
        if dropout > 0:
            layers.append(nn.Dropout(p=args.dropout))
    layers.append(nn.Linear(args.n_hidden, data.n_classes))
    return nn.Sequential(*layers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    register_general_args(parser)
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    args = parser.parse_args()
    print('Parsed args:', args)

    train_and_eval(mlp_model_fn, load_graph_data.load(args), args)
