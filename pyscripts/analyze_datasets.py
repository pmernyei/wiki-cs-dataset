from wiki_node import WikiDataNode
from itertools import product
import numpy as np
import random


def calculate_connectivity_stats(nodes, label_list):
    ids = {lab: i for lab,i in zip(label_list, range(len(label_list)))}
    link_counts = np.zeros((len(label_list), len(label_list)))
    nodes_for_label = np.zeros(len(label_list))

    for node in nodes.values():
        for lab in node.labels:
            nodes_for_label[ids[lab]] += 1
        for id in node.outlinks:
            for (lab1, lab2) in product(node.labels, nodes[id].labels):
                link_counts[ids[lab1]][ids[lab2]] += 1

    connectivities = link_counts / (nodes_for_label*np.reshape(nodes_for_label, (len(label_list), 1)))
    inside_links = sum([link_counts[i][i] for i in range(len(label_list))])
    inside_link_bound = sum([s*s for s in nodes_for_label])
    inside_connectivity = inside_links / inside_link_bound
    inter_links = sum([link_counts[i][j] for (i,j) in product(range(len(label_list)),range(len(label_list))) if i != j])
    inter_link_bound = sum([nodes_for_label[i]*nodes_for_label[j] for (i,j) in product(range(len(label_list)),range(len(label_list))) if i != j])
    inter_connectivity = inter_links / inter_link_bound
    return connectivities, inside_connectivity, inter_connectivity


def calculate_multilabel_stats(nodes, label_list):
    ids = {lab: i for lab,i in zip(label_list, range(len(label_list)))}
    overlap_counts = np.zeros((len(label_list), len(label_list)))
    nodes_for_label = np.zeros(len(label_list))

    for node in nodes.values():
        for lab in node.labels:
            nodes_for_label[ids[lab]] += 1
        for (lab1, lab2) in product(node.labels, node.labels):
            overlap_counts[ids[lab1]][ids[lab2]] += 1

    overlap_ratios = overlap_counts / np.reshape(nodes_for_label, (-1,1))
    return overlap_ratios


def calculate_avg_cosine_similarities(nodes, label_list, sample_count=1000):
    totals = np.zeros((len(label_list), len(label_list)))
    valids = np.zeros((len(label_list), len(label_list)))
    nodes_per_label = [[node for node in nodes.values() if lab in node.labels] for lab in label_list]

    for i,j in product(range(len(label_list)), range(len(label_list))):
        left = random.sample(nodes_per_label[i], sample_count)
        right = random.sample(nodes_per_label[j], sample_count)
        for node1, node2 in zip(left, right):
            v1 = node1.words_binary
            v2 = node2.words_binary
            l1 = np.sum(v1*v1)
            l2 = np.sum(v2*v2)
            if l1 > 0 and l2 > 0:
                valids[i][j] += 1
                totals[i][j] += np.sum(v1*v2)/(l1*l2)
    totals /= valids
    return totals, valids


def print_node_data(node, all_nodes_map):
    print('Title:', node.title, '; id:', node.id)
    print('Labels:', node.labels)
    print('Text:', node.tokens[:10], '...')
    print('Linked pages:', [all_nodes_map[id].title for id in node.outlinks][:5], 'total', len(node.outlinks))
    print()


def print_sample_pages(nodes, sample_count=5):
    nodes_list = list(nodes.values())
    samples = random.sample(nodes_list, sample_count)
    for sample in samples:
        print_node_data(sample, nodes)
