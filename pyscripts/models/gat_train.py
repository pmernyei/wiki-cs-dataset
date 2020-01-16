import argparse
import torch.nn.functional as F

from train import train_and_eval
from train import register_general_args
from gat import GAT


def gat_model_fn(args, data):
    heads = ([args.num_heads] * args.n_layers) + [args.num_out_heads]
    return GAT(data.graph,
                args.n_layers,
                data.n_feats,
                args.n_hidden,
                data.n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_general_args(parser)
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gat layers")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--residual", action="store_true", default=False,
                            help="use residual connection")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    args = parser.parse_args()
    print('Parsed args:', args)

    train_and_eval(gat_model_fn, args)
