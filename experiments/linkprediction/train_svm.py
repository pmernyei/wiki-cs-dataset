"""
Training, evaluation and and hyperparameter search for link prediction with
an SVM.
"""
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE
import torch_geometric.utils as gutils
import numpy as np
import load_wiki
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
import itertools
import random
import torch

def sample_negative(count, nodes, avoid):
    avoid_set = set(zip(avoid[0], avoid[1]))
    result = np.array([[],[]], dtype=np.int32)
    while(result.shape[1] < count):
        candidates = np.random.randint(low=0, high=nodes, size=(2*count, 2))
        for u,v in candidates:
            if u != v and (u,v) not in avoid_set:
                avoid_set.add((u,v))
                result = np.concatenate((result, [[u],[v]]), axis=1)
            if result.shape[1] == count:
                break
    return result


def combine_node_pair_features(features, pos_edge_index, neg_edge_index):
    x_pos = np.concatenate((features[pos_edge_index[0]],
                            features[pos_edge_index[1]]), axis=1)
    x_neg = np.concatenate((features[neg_edge_index[0]],
                            features[neg_edge_index[1]]), axis=1)
    x = np.concatenate((x_pos, x_neg), axis=0)
    y = np.concatenate((np.ones(len(x_pos)), np.zeros(len(x_pos))), axis=0)
    return x, y


def load_data(dataset_name):
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
            '.', 'data', dataset_name)
        data = Planetoid(path, dataset_name)[0]
    else:
        data = load_wiki.load_data()

    data.edge_index = gutils.to_undirected(data.edge_index)
    data = GAE.split_edges(GAE, data)

    features = data.x.numpy()
    train_pos_edges = data.train_pos_edge_index.numpy()
    train_neg_edges = sample_negative(
        count = train_pos_edges.shape[1],
        avoid = train_pos_edges,
        nodes = features.shape[0]
    )

    x_tr, y_tr = combine_node_pair_features(features,
        train_pos_edges, train_neg_edges)
    x_val, y_val = combine_node_pair_features(features,
        data.val_pos_edge_index.numpy(), data.val_neg_edge_index.numpy())
    x_test, y_test = combine_node_pair_features(features,
        data.test_pos_edge_index.numpy(), data.test_neg_edge_index.numpy())
    return x_tr, y_tr, x_val, y_val, x_test, y_test


def eval(x_tr, y_tr, x_eval, y_eval, C=200, kernel='rbf', gamma='scale'):
    svm = SVC(C=C, kernel=kernel, gamma=gamma)
    svm.fit(x_tr, y_tr)
    preds = svm.decision_function(x_eval)
    print('pred mean', np.mean(preds))
    acc = svm.score(x_eval, y_eval)
    auc = roc_auc_score(y_eval, preds)
    ap = average_precision_score(y_eval, preds)
    return acc, auc, ap


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def hparam_search(x_tr, y_tr, x_eval, y_eval):
    kernels = ['rbf', 'poly']
    c = [5, 10, 20, 40, 80, 160, 320]
    configs = list(product_dict(kernel=kernels, C=c))
    best_res = 0
    best_config = {}
    for config in configs:
        print('Evaluating config {}'.format(str(config)))
        acc, auc, ap = eval(x_tr, y_tr, x_eval, y_eval,
            C=config['C'], kernel=config['kernel'])
        res = (auc + ap) / 2
        if res > best_res:
            best_res = res
            best_config = config
        print('res {} with config {}'.format(res, str(config)))
    print('Best acc {} with config {}'.format(best_res, str(best_config)))


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description='SVM link prediction')
    parser.add_argument('--dataset')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--kernel', default='rbf')
    parser.add_argument('--hparam-search', action='store_true', default=False)
    parser.add_argument('--cap-training-samples', type=int, default=10000)
    parser.add_argument('--c', type=float, default=200)
    args = parser.parse_args()

    x_tr, y_tr, x_val, y_val, x_test, y_test = load_data(args.dataset)

    if len(x_tr) > args.cap_training_samples:
        perm = np.random.permutation(len(x_tr))
        x_tr = x_tr[perm]
        y_tr = y_tr[perm]

        x_tr = x_tr[:args.cap_training_samples]
        y_tr = y_tr[:args.cap_training_samples]
        print(sum(y_tr), len(y_tr)-sum(y_tr))

    if args.test:
        acc, auc, ap = eval(x_tr, y_tr, x_test, y_test, args.c, args.kernel)
    else:
        if args.hparam_search:
            hparam_search(x_tr, y_tr, x_val, y_val)
        acc, auc, ap = eval(x_tr, y_tr, x_val, y_val, args.c, args.kernel)
    print('Accuracy: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(acc, auc, ap))
