import argparse
import torch.nn.functional as F

from .. import load_graph_data
from ..train import train_and_eval
from ..train import register_general_args
from .gat import GAT


def gat_model_fn(args, data):
    heads = ([args.n_heads] * args.n_layers) + [args.n_out_heads]
    return GAT(data.graph,
                args.n_hidden_layers,
                data.n_feats,
                args.n_hidden_units,
                data.n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)


def register_gat_args(parser):
    parser.add_argument("--n-hidden-units", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-hidden-layers", type=int, default=1,
            help="number of hidden gat layers")
    parser.add_argument("--n-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--n-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--residual", action="store_true", default=False,
                            help="use residual connection")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    register_general_args(parser)
    register_gat_args(parser)
    args = parser.parse_args()
    print('Parsed args:', args)

    train_and_eval(gat_model_fn, load_graph_data.load(args), args)
