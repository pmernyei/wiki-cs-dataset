import argparse, time
import numpy as np
import json
import itertools
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.data

import load_dgl_data
from gcn import GCN
from gat import GAT
from appnp import APPNP
from mlp import create_mlp

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # load and preprocess dataset
    if args.dataset.startswith('file:'):
        data = load_dgl_data.load_file(args.dataset[5:])
    else:
        data = load_dgl_data.load_builtin(args)
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (data.n_edges, data.n_classes,
              data.train_mask.int().sum().item(),
              data.val_mask.int().sum().item(),
              data.test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        data.features = data.features.cuda()
        data.labels = data.labels.cuda()
        data.train_mask = data.train_mask.cuda()
        data.val_mask = data.val_mask.cuda()
        data.test_mask = data.test_mask.cuda()

    # graph normalization
    degs = data.graph.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    data.graph.ndata['norm'] = norm.unsqueeze(1)

    # create appropriate model
    if args.model == 'gcn':
        model = GCN(data.graph,
                    data.n_feats,
                    args.n_hidden,
                    data.n_classes,
                    args.n_layers,
                    F.relu,
                    args.dropout)
    elif args.model == 'gat':
        heads = ([args.num_heads] * args.n_layers) + [args.num_out_heads]
        model = GAT(data.graph,
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
    elif args.model == 'appnp':
        model = APPNP(data.graph,
                      data.n_feats,
                      args.hidden_sizes,
                      data.n_classes,
                      F.relu,
                      args.in_drop,
                      args.edge_drop,
                      args.alpha,
                      args.k)
    elif args.model == 'mlp':
        model = create_mlp(args.n_layers,
                            data.n_feats,
                            args.n_hidden,
                            data.n_classes,
                            args.dropout)
    else:
        raise Exception('Unknown model name {} given'.format(args.model))

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(data.features)
        loss = loss_fcn(logits[data.train_mask], data.labels[data.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = evaluate(model, data.features, data.labels, data.train_mask)
        val_acc = evaluate(model, data.features, data.labels, data.val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Train acc {:.4f} | "
              "Val acc {:.4f} | ETputs(KTEPS) {:.2f}". format(
                epoch, np.mean(dur), loss.item(), train_acc, val_acc,
                data.n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, data.features, data.labels, data.test_mask)
    print("Test accuracy {:.2%}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNNs')
    dgl.data.register_data_args(parser)

    parser.add_argument("--model", help="model to train")

    # Arguments applicable for multiple (not necessarily all) models
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.set_defaults(self_loop=False)

    # GAT arguments
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")

    # APPNP arguments
    parser.add_argument("--edge-drop", type=float, default=0.5,
                        help="edge propagation dropout")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of propagation steps")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Teleport Probability")
    parser.add_argument("--hidden_sizes", type=int, nargs='+', default=[64],
                        help="hidden unit sizes for appnp")

    args = parser.parse_args()
    print(args)

    main(args)
