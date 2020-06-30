"""
Calculate the average length of shortest path between two nodes in the given
graph dataset.
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

graph = None
dists = dict()


def load_builtin(args):
    global graph
    graph = dgl.data.load_data(args).graph
    return graph

def load_wiki(path=os.path.join('..','..','dataset','data.json')):
    global graph
    data = json.load(open(path))
    edge_list = list(itertools.chain(*[[(i, nb) for nb in nbs] for i,nbs in enumerate(data['links'])]))
    reverse_edge_list = [(b,a) for (a,b) in edge_list]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(data['features'])))
    graph.add_edges_from(edge_list)
    graph.add_edges_from(reverse_edge_list)
    return graph


def bfs_component(node):
    global dists
    global graph
    dists[node] = 0
    q = queue.Queue()
    q.put(node)
    component_size = 0
    while not q.empty():
        component_size += 1
        n = q.get()
        for nb in graph.adj[n]:
            if nb not in dists:
                dists[nb] = dists[n]+1
                q.put(nb)
    return component_size


def component_sizes():
    global dists
    global graph
    dists.clear()
    sizes = []
    for node in graph.nodes():
        if node not in dists:
            print('component root', node)
            sizes.append(bfs_component(node))
            print('size', sizes[-1])
    return sizes


def avg_sp():
    global dists
    sp_lens = []
    for node in graph.nodes():
        print('calculating paths from node', node, '...')
        dists.clear()
        bfs_component(node)
        sp_lens += (d for d in dists.values() if d > 0)
    return np.mean(sp_lens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    args = parser.parse_args()
    if args.dataset == 'wiki':
        load_wiki()
    else:
        load_builtin(args)
    print(avg_sp())
