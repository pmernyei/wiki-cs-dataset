"""
Calculate and plot the distribution among nodes of what ratio of neighbours has
the same label in a given dataset.
"""
import numpy as np
import json
import itertools
import networkx as nx
import dgl.data
import argparse
import os
from dgl import DGLGraph
import queue
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

graph = None
dists = dict()


def load_builtin(args):
    ds = dgl.data.load_data(args)
    return ds.graph, ds.labels

def load_wiki(path=os.path.join('..','..','dataset','data.json')):
    global graph
    data = json.load(open(path))
    edge_list = list(itertools.chain(*[[(i, nb) for nb in nbs]
                                      for i,nbs in enumerate(data['links'])]))
    reverse_edge_list = [(b,a) for (a,b) in edge_list]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(data['features'])))
    graph.add_edges_from(edge_list)
    graph.add_edges_from(reverse_edge_list)
    return graph, data['labels']


def calc_ratios(graph, ys):
    rs = []
    for node in tqdm(list(graph.nodes())):
        nbs = list(graph.adj[node].keys())
        if len(nbs) > 0:
            r = sum((1 for nb in nbs if ys[nb] == ys[node]))/len(nbs)
        rs.append(r)
    return rs


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 24})
    for dataset in ['wiki', 'cora', 'citeseer', 'pubmed']:
        if dataset == 'wiki':
            g,y = load_wiki()
        else:
            args = argparse.Namespace()
            args.dataset = dataset
            g,y = load_builtin(args)
        rs = pd.Series(calc_ratios(g,y))
        sns.distplot(rs,
                    bins=25,
                    norm_hist=False,
                    kde=False,
                    hist_kws={'weights': np.ones(len(rs))/len(rs)})
        plt.savefig('{}.png'.format(dataset))
        plt.clf()
