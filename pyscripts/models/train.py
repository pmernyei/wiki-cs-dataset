import argparse, time
import numpy as np
import seaborn as sns
import json
import itertools
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

import load_graph_data
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


def create_model(args, data):
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
        raise ValueError('Unknown model name {}'.format(args.model))
    if args.gpu >= 0:
        model.cuda()
    return model


def train_and_eval(data, model, split_idx, stopping_patience, lr, weight_decay,
                    test=False, preds_out=None, log_details=False):
    dur = []
    max_acc = 0
    patience_left = stopping_patience
    best_vars = None
    epoch = 0

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    while patience_left > 0:
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(data.features)
        loss = loss_fcn(logits[data.train_masks[split_idx]],
                        data.labels[data.train_masks[split_idx]])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = evaluate(model, data.features, data.labels, data.train_masks[split_idx])
        stopping_acc = evaluate(model, data.features, data.labels, data.stopping_masks[split_idx])
        if stopping_acc > max_acc:
            max_acc = stopping_acc
            patience_left = stopping_patience
            best_vars = { key: value.cpu() for key, value in model.state_dict().items()}
        else:
            patience_left -= 1

        if log_details:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Train acc {:.4f} | "
                  "Val acc {:.4f} | ETputs(KTEPS) {:.2f}". format(
                    epoch, np.mean(dur), loss.item(), train_acc, stopping_acc,
                    data.n_edges / np.mean(dur) / 1000))
        epoch += 1

    model.load_state_dict(best_vars)

    if test:
        result_acc = evaluate(model, data.features, data.labels, data.test_mask)
    else:
        result_acc = evaluate(model, data.features, data.labels, data.val_masks[split_idx])

    if preds_out is not None:
        mask = 1 - data.test_mask
        logits = model(data.features)

        _, preds = torch.max(logits, dim=1)
        preds = preds*(1 - data.test_mask) - data.test_mask
        json.dump(preds.tolist(), open(preds_out,'w'))

    return result_acc, epoch


def main(args):
    data = load_graph_data.load(args)

    if args.full_eval:
        results = []
        epoch_counts = []
        for split_idx in range(len(data.train_masks)):
            for run_idx in range(args.runs_per_split):
                model = create_model(args, data)
                acc, epochs = train_and_eval(data, model, split_idx, args.patience,
                    args.lr, args.weight_decay, args.test)
                results.append(acc)
                epoch_counts.append(epochs)
                print('Split {} run {} accuracy: {:.4f} in {} epochs'.format(split_idx, run_idx, acc, epochs))
        results = np.array(results)
        avg = results.mean()
        bootstrap = sns.algorithms.bootstrap(
            results, func=np.mean, n_boot=args.n_boot)
        conf_int = sns.utils.ci(bootstrap, args.conf_int)
        uncertainty = np.max(np.abs(conf_int - avg))
        print('{} accuracy: {:.4f} ± {:.4f}'.format(
            'Test' if args.test else 'Validation', avg, uncertainty))
        print('avg epochs: {}'.format(np.array(epoch_counts).mean()))
    else:
        model = create_model(args, data)
        acc = train_and_eval(data, model, args.split_idx, args.patience, args.lr,
            args.weight_decay, args.test,
            log_details=True, preds_out=args.preds_output_file)
        print('Single split {} accuracy: {:.2f}'.format(
            'test' if args.test else 'validation', acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNNs')
    load_graph_data.register_data_args(parser)

    # Arguments for high level training loop behaviour
    parser.add_argument("--model", help="model to train")
    parser.add_argument("--patience", type=int, default=100,
            help="epochs to train before giving up if accuracy doesn't improve")
    parser.add_argument("--test", action='store_true',
            help="evaluate on test set after training (default=False)")
    parser.set_defaults(test=False)
    parser.add_argument("--full-eval", action='store_true',
            help="evaluate all splits and calculate confidence interval (default=False)")
    parser.set_defaults(full_eval=False)
    parser.add_argument("--runs-per-split", type=int, default=5,
            help="how many times to train and eval on each split in full eval")
    parser.add_argument("--n-boot", type=int, default=1000,
            help="resampling count for bootstrap confidence interval calculation in full eval")
    parser.add_argument("--conf-int", type=int, default=95,
            help="confidence interval probability for full eval")
    parser.add_argument("--split-idx", type=int, default=0,
            help="split id to run if only running one")
    parser.add_argument("--preds-output-file",
            help="file for writin predictions on train/validation set for analysis")

    # Arguments applicable for multiple (not necessarily all) models
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")

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
