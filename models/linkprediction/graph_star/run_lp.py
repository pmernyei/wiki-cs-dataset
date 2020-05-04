import sys

from torch_geometric.datasets import Planetoid
from utils.gsn_argparse import str2bool, str2actication
import os.path as osp
import torch_geometric.transforms as T
import torch_geometric.utils as gutils

import ssl
import torch
from torch_geometric.nn import GAE
import trainer
import utils.gsn_argparse as gap
import load_wiki_data
import numpy as np
import seaborn as sns
import json

ssl._create_default_https_context = ssl._create_unverified_context


def load_data(dataset_name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset_name)
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, dataset_name, T.TargetIndegree())
        num_features = dataset.num_features
        data = GAE.split_edges(GAE, dataset[0])
    else:
        data = load_wiki_data.load_data(dataset_name, T.TargetIndegree())
        data = GAE.split_edges(GAE, data)
        num_features = data.x.shape[1]

    data.train_pos_edge_index = gutils.to_undirected(data.train_pos_edge_index)
    data.val_pos_edge_index = gutils.to_undirected(data.val_pos_edge_index)
    data.test_pos_edge_index = gutils.to_undirected(data.test_pos_edge_index)

    data.edge_index = torch.cat([data.train_pos_edge_index, data.val_pos_edge_index, data.test_pos_edge_index], dim=1)

    data.edge_train_mask = torch.cat([torch.ones((data.train_pos_edge_index.size(-1))),
                                      torch.zeros((data.val_pos_edge_index.size(-1))),
                                      torch.zeros((data.test_pos_edge_index.size(-1)))], dim=0).byte()
    data.edge_val_mask = torch.cat([torch.zeros((data.train_pos_edge_index.size(-1))),
                                    torch.ones((data.val_pos_edge_index.size(-1))),
                                    torch.zeros((data.test_pos_edge_index.size(-1)))], dim=0).byte()
    data.edge_test_mask = torch.cat([torch.zeros((data.train_pos_edge_index.size(-1))),
                                     torch.zeros((data.val_pos_edge_index.size(-1))),
                                     torch.ones((data.test_pos_edge_index.size(-1)))], dim=0).byte()
    data.edge_type = torch.zeros(((data.edge_index.size(-1)),)).long()

    data.batch = torch.zeros((1, data.num_nodes), dtype=torch.int64).view(-1)
    data.num_graphs = 1
    return data, num_features


def mean_with_uncertainty(values, n_boot=10000, conf_threshold=95):
    values = np.array(values)
    avg = values.mean()
    bootstrap = sns.algorithms.bootstrap(
        values, func=np.mean, n_boot=n_boot)
    conf_int = sns.utils.ci(bootstrap, conf_threshold)
    return avg, np.max(np.abs(conf_int - avg))



def main(_args):
    args = gap.parser.parse_args(_args)

    data, num_features = load_data(args.dataset)

    aucs = []
    aps = []
    for i in range(10):
        print("===========================================")
        auc, ap = trainer.trainer(args, args.dataset, [data], [data], [data], transductive=True,
                        num_features=num_features, max_epoch=args.epochs,
                        num_node_class=0,
                        link_prediction=True, test_per_epoch=10, val_per_epoch=10)
        aucs.append(auc)
        aps.append(ap)
        json.dump(aucs, open('aucs.txt', 'w'))
        json.dump(aps, open('aps.txt', 'w'))
    auc_mean, auc_ci = mean_with_uncertainty(aucs)
    ap_mean, ap_ci = mean_with_uncertainty(aps)
    print('AUC-ROC:', auc_mean, '+-', auc_ci)
    print('AP:',      ap_mean,  '+-', ap_ci)


if __name__ == '__main__':
    main(sys.argv[1:])
