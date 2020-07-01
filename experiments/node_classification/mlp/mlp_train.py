import argparse
import torch.nn as nn

from .. import load_graph_data
from ..train import train_and_eval
from ..train import register_general_args


def mlp_model_fn(args, data):
    layers = []
    layers.append(nn.Linear(data.n_feats, args.n_hidden))
    for i in range(args.n_hidden_layers - 1):
        layers.append(nn.Linear(args.n_hidden, args.n_hidden))
        layers.append(nn.ReLU())
        if args.dropout > 0:
            layers.append(nn.Dropout(p=args.dropout))
    layers.append(nn.Linear(args.n_hidden, data.n_classes))
    return nn.Sequential(*layers)


def register_mlp_args(parser):
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--n-hidden-layers", type=int, default=1,
            help="number of hidden layers")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP')
    register_general_args(parser)
    register_mlp_args(parser)
    args = parser.parse_args()
    print('Parsed args:', args)

    train_and_eval(mlp_model_fn, load_graph_data.load(args), args)
