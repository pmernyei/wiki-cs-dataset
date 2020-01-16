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

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train_and_eval_once(data, model, split_idx, stopping_patience, lr,
                weight_decay, test=False, preds_out=None, log_details=False):
    dur = []
    max_acc = 0
    patience_left = stopping_patience
    best_vars = None
    epoch = 0

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)

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
            best_vars = { key: value.clone() for key, value in model.state_dict().items()}
        else:
            patience_left -= 1

        if log_details:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Train acc {:.2%} | "
                  "Val acc {:.2%} | ETputs(KTEPS) {:.2f}". format(
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

    return result_acc


def train_and_eval(model_fn, args):
    data = load_graph_data.load(args)
    results = []
    for split_idx in range(len(data.train_masks)):
        for run_idx in range(args.runs_per_split):
            model = model_fn(args, data)
            if args.gpu >= 0:
                model.cuda()
            acc = train_and_eval_once(data, model, split_idx, args.patience,
                args.lr, args.weight_decay, args.test, log_details=args.verbose)
            results.append(acc)
            print('Split {} run {} accuracy: {:.2%}'
                    .format(split_idx, run_idx, acc))
    results = np.array(results)
    avg = results.mean()
    bootstrap = sns.algorithms.bootstrap(
        results, func=np.mean, n_boot=args.n_boot)
    conf_int = sns.utils.ci(bootstrap, args.conf_int)
    uncertainty = np.max(np.abs(conf_int - avg))
    print('{} accuracy: {:.2%} ± {:.2%}'.format(
        'Test' if args.test else 'Validation', avg, uncertainty))


def register_general_args(parser):
    load_graph_data.register_data_args(parser)
    parser.add_argument('--patience', type=int, default=100,
            help='epochs to train before giving up if accuracy does not '
                 'improve')
    parser.add_argument('--test', action='store_true',
            help='evaluate on test set after training (default=False)')
    parser.add_argument('--runs-per-split', type=int, default=5,
            help='how many times to train and eval on each split in full eval')
    parser.add_argument('--n-boot', type=int, default=1000,
            help='resampling count for bootstrap confidence interval '
                 'calculation in full eval')
    parser.add_argument('--conf-int', type=int, default=95,
            help='confidence interval probability for full eval')
    parser.add_argument('--max-splits', type=int,
            help='maximum number of different training splits to evaluate on. '
                 'Unbounded by default so all splits in dataset will be used')
    parser.add_argument('--preds-output-file',
            help='file for writin predictions on train/validation set for'
                 'analysis')
    parser.add_argument('--lr', type=float, default=1e-2,
            help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
            help='Weight for L2 loss')
    parser.add_argument('--verbose', action='store_true',
            help='Print performance after each epoch')
