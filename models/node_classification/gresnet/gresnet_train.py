import argparse

from .. import load_graph_data
from ..train import train_and_eval
from ..train import register_general_args
from .gresnet import GCN_GResNet
from .gresnet import GAT_GResNet


def gresnet_model_fn(args, data):
    if args.base_conv == 'gcn':
        return GCN_GResNet(data.graph,
                    args.graph_res,
                    args.raw_res,
                    args.n_layers,
                    data.n_feats,
                    args.n_hidden,
                    data.n_classes,
                    args.dropout)
    elif args.base_conv == 'gat':
        return GAT_GResNet(data.graph,
                    args.graph_res,
                    args.raw_res,
                    args.n_layers,
                    data.n_feats,
                    args.n_hidden,
                    data.n_classes,
                    args.dropout)


def register_gresnet_args(parser):
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--n-layers", type=int, default=4,
            help="number of hidden gresnet layers")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--graph-res", action='store_true',
            help="pass residual across graph edges from neighbours instead of "
                 "the same node")
    parser.add_argument("--raw-res", action='store_true',
            help="pass residual from initial features instead of previous "
                 "layer")
    parser.add_argument("--base-conv", default="gcn",
            help="what underlying graph convolution to use, must be gcn or gat")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GResNet')
    register_general_args(parser)
    register_gresnet_args(parser)
    args = parser.parse_args()
    print('Parsed args:', args)

    train_and_eval(gresnet_model_fn, load_graph_data.load(args), args)
