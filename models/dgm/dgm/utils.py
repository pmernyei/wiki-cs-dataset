import numpy as np
import torch
import random
import os

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import load_wiki_data

def load_dataset(dataset):
    if dataset == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        num_classes = dataset.num_classes
    elif dataset == 'pubmed':
        dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
        num_classes = dataset.num_classes
    elif dataset == 'citeseer':
        dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
        num_classes = dataset.num_classes
    elif dataset == 'spam':
        dataset = load_spam_dataset()
        num_classes = 2
    elif dataset == 'wiki':
        dataset = [load_wiki_data.load_data(
            os.path.join('..','..','results','dataset','vectors.json'))]
        num_classes = 10
    else:
        raise ValueError('Unsupported dataset {}'.format(dataset))

    return dataset[0], num_classes


def name_from_args(args, labeled):
    return "{}_{}_{}_{}_{}".format(
        args.dataset,
        "sdgm" if args.sdgm else "dgm",
        args.train_mode,
        args.reduce_method,
        "labeled" if labeled else "pred")


def load_spam_dataset():
    """Code for the spam dataset from:
    https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Deep_GCN_Spam.ipynb"""
    labels = []
    N = 1000
    nodes = range(0, N)
    node_features = []
    edge_features = []

    for node in nodes:

        # spammer
        if random.random() > 0.5:
            # more likely to have many connections with a maximum of 1/5 of the nodes in the graph
            nb_nbrs = int(random.random() * (N / 5))
            # more likely to have sent many bytes
            node_features.append((random.random() + 1) / 2.)
            # more likely to have a high trust value
            edge_features += [(random.random() + 2) / 3.] * nb_nbrs
            # associate a label
            labels.append(1)

        # non-spammer
        else:
            # at most connected to 10 nbrs
            nb_nbrs = int(random.random() * 10 + 1)
            # associate more bytes and random bytes
            node_features.append(random.random())
            edge_features += [random.random()] * nb_nbrs
            labels.append(0)

        # connect to some random nodes
        nbrs = np.random.choice(nodes, size=nb_nbrs)
        nbrs = nbrs.reshape((1, nb_nbrs))

        # add the edges of nbrs
        node_edges = np.concatenate([np.ones((1, nb_nbrs), dtype=np.int32) * node, nbrs], axis=0)

        # add the overall edges
        if node == 0:
            edges = node_edges
        else:
            edges = np.concatenate([edges, node_edges], axis=1)

    x = torch.tensor(np.expand_dims(node_features, 1), dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    edge_index = torch.tensor(edges, dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[:int(0.8 * data.num_nodes)] = 1  # train only on the 80% nodes
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)  # test on 20 % nodes
    data.test_mask[- int(0.2 * data.num_nodes):] = 1

    return [data]
