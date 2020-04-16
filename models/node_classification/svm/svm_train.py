import numpy as np
from sklearn.svm import SVC
import json
import sys
import argparse
import itertools

from node_classification import load_graph_data


def fit_svm(data, C=200, kernel='rbf', gamma='scale', test=False):
    acc_sum = 0
    for i in range(len(data.train_masks)):
        svm = SVC(C=C, kernel=kernel, gamma=gamma)
        tr = data.train_masks[i]
        val = data.val_masks[i]
        svm.fit(data.features[tr], data.labels[tr])
        acc = svm.score(data.features[val], data.labels[val])
        print('Validation accuracy:', acc)
        if test:
            acc = svm.score(data.features[data.test_mask],
                            data.labels[data.test_mask])
            print('Test accuracy:', acc)
        acc_sum += acc
        if acc < 0.7:
            print('Low accuracy, returning early estimate')
            return acc_sum/(i+1)
    acc = acc_sum/len(data.train_masks)
    print('Avg {} accuracy: {:.4f}'.format('test' if test else 'val', acc))
    return acc


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def hparam_search(data):
    kernels = ['linear', 'rbf', 'poly']
    gamma = ['auto', 'scale']
    c = ([mul*order for order in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
        for mul in range(1,10)])
    configs = list(product_dict(kernel=kernels, gamma=gamma, C=c))
    best_acc = 0
    best_config = {}
    for config in configs:
        print('Evaluating config {}'.format(str(config)))
        acc = fit_svm(data, kernel=config['kernel'], gamma=config['gamma'],
                C=config['C'])
        if acc > best_acc:
            best_acc = acc
            best_config = config
    print('Avg acc {} with config {}'.format(best_acc, str(best_config)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVM training')
    load_graph_data.register_data_args(parser)
    parser.add_argument("--hparam-search", action='store_true',
            help="run hparam search (ignores other arguments)")
    parser.add_argument("--test", action='store_true',
            help="evaluate on test set after training (default=False)")
    parser.add_argument("--c", type=int, default=200)
    parser.add_argument("--kernel", default='rbf')
    parser.add_argument("--gamma", default='scale')
    args = parser.parse_args()
    data = load_graph_data.load(args)
    if args.hparam_search:
        hparam_search(data)
    else:
        fit_svm(data, C=args.c, kernel=args.kernel, gamma=args.gamma,
            test=args.test)
