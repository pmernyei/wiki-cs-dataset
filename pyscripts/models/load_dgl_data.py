import numpy as np
import json
import itertools
import torch
import networkx as nx
import dgl.data
from dgl import DGLGraph

class DGLGraphDataset:
    def __init__(self, graph, features, labels, train_mask, val_mask,
                           test_mask, n_edges, n_classes, n_feats):
        self.graph = graph
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.n_edges = n_edges
        self.n_classes = n_classes
        self.n_feats = n_feats



def load_file(filename):
    data = json.load(open(filename))
    features = torch.FloatTensor(np.array(data['features']))
    labels = torch.LongTensor(np.array(data['labels']))
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(np.array(data['splits']) == 0)
        val_mask = torch.BoolTensor(np.array(data['splits']) == 1)
        test_mask = torch.BoolTensor(np.array(data['splits']) == 2)
    else:
        train_mask = torch.ByteTensor(np.array(data['splits']) == 0)
        val_mask = torch.ByteTensor(np.array(data['splits']) == 1)
        test_mask = torch.ByteTensor(np.array(data['splits']) == 2)
    n_feats = features.shape[1]
    n_classes = len(set(data['labels']))

    g = DGLGraph()
    g.add_nodes(len(data['features']))
    edge_list = list(itertools.chain(*[[(i, nb) for nb in nbs] for i,nbs in enumerate(data['links'])]))
    n_edges = len(edge_list)
    # add edges two lists of nodes: src and dst
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(dst, src)
    return DGLGraphDataset(g, features, labels, train_mask, val_mask,
                           test_mask, n_edges, n_classes, n_feats)


def load_builtin(args):
    data = dgl.data.load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    n_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    # graph preprocess
    g = data.graph
    # add self loop
    if args.self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = DGLGraph(g)
    return DGLGraphDataset(
            g, features, labels, train_mask, val_mask,
            test_mask, n_edges, n_classes, n_feats)
