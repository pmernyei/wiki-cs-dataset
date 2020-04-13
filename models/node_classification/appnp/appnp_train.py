import argparse
import torch.nn.functional as F

from .. import load_graph_data
from ..train import train_and_eval
from ..train import register_general_args
from .appnp import APPNP


def appnp_model_fn(args, data):
    return APPNP(data.graph,
                  data.n_feats,
                  args.hidden_sizes,
                  data.n_classes,
                  F.relu,
                  args.in_drop,
                  args.edge_drop,
                  args.alpha,
                  args.k)


def register_appnp_args(parser):
    parser.add_argument("--in-drop", type=float, default=0.5,
                        help="input feature dropout")
    parser.add_argument("--edge-drop", type=float, default=0.5,
                        help="edge propagation dropout")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of propagation steps")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Teleport Probability")
    parser.add_argument("--hidden_sizes", type=int, nargs='+', default=[64],
                        help="hidden unit sizes for appnp")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='APPNP')
    register_general_args(parser)
    register_appnp_args(parser)
    args = parser.parse_args()
    print('Parsed args:', args)

    train_and_eval(appnp_model_fn, load_graph_data.load(args), args)
