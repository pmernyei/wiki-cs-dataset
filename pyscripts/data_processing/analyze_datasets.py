"""
Calculate statistics about an extracted dataset.
"""

from wiki_node import WikiDataNode
from itertools import product
from datetime import datetime
import numpy as np
import random
import pickle
import os
import json
import sys

import process_dataset

def calculate_connectivity_stats(nodes, label_list):
    """
    Calculate the following stats:
    - connectivity_matrix: matrix of conditional probabilities that a specific
        node with label A links to a specific other node with label B.
    - inside_connectivity: overall probability that two nodes with the same
        label are connected
    - inter_connectivity: overall probability that two nodes with different
        labels are connected
    """
    C = len(label_list)
    ids = {lab: i for lab,i in zip(label_list, range(C))}

    link_counts = np.zeros((C, C))
    nodes_for_label = np.zeros(C)

    for node in nodes.values():
        nodes_for_label[ids[node.label]] += 1
        for id in node.outlinks:
            link_counts[ids[node.label]][ids[nodes[id].label]] += 1

    connectivities = (link_counts /
        (nodes_for_label*np.reshape(nodes_for_label, (C, 1))))
    inside_links = sum([link_counts[i][i] for i in range(C)])
    inside_link_bound = sum([s*s for s in nodes_for_label])
    inside_connectivity = inside_links / inside_link_bound
    inter_links = sum(
        [link_counts[i][j] for (i,j) in product(range(C),range(C)) if i != j]
    )
    inter_link_bound = sum(
        [nodes_for_label[i]*nodes_for_label[j]
            for (i,j) in product(range(C),range(C)) if i != j]
    )
    inter_connectivity = inter_links / inter_link_bound
    return {
        'inside_connectivity': float(inside_connectivity),
        'inter_connectivity': float(inter_connectivity),
        'connectivity_matrix': connectivities.tolist()
    }


def calculate_avg_cosine_similarities(nodes, label_list, sample_count=1000):
    """
    Calculate the average cosine similarities between vectors of nodes from
    every pair of classes.
    """
    C = len(label_list)
    totals = np.zeros((C, C))
    valids = np.zeros((C, C))
    nodes_per_label = [[node for node in nodes.values() if lab == node.label]
        for lab in label_list]

    for i,j in product(range(C), range(C)):
        samples = min(sample_count,
                      min(len(nodes_per_label[i]), len(nodes_per_label[j])))
        left = random.sample(nodes_per_label[i], samples)
        right = random.sample(nodes_per_label[j], samples)
        for node1, node2 in zip(left, right):
            v1 = node1.vector
            v2 = node2.vector
            l1 = np.sum(v1*v1)
            l2 = np.sum(v2*v2)
            if l1 > 0 and l2 > 0:
                valids[i][j] += 1
                totals[i][j] += np.sum(v1*v2)/(l1*l2)
    totals /= valids
    return totals

def cosine_similarity_classification_accuracy(nodes):
    """
    Calculate how accurately we could classify nodes by calculating the average
    vector of each class and mapping each example to the closest of those class
    averages. No train/test split so just serves as a quick ballpark, not a
    rigorous measure.
    """
    labels = process_dataset.label_set(nodes)
    label_ids = {lab: i for lab,i in zip(labels, range(len(labels)))}
    id_to_label = {i: lab for lab,i in label_ids.items()}
    words_dim = len(next(iter(nodes.values())).vector)
    label_freqs = np.zeros(len(labels))
    avg_vecs = np.zeros((len(labels), words_dim))
    for node in nodes.values():
        avg_vecs[label_ids[node.label]] += node.vector
        label_freqs[label_ids[node.label]] += 1

    # Normalize row vectors
    avg_vecs /= np.sqrt(np.sum(avg_vecs*avg_vecs, axis=1)).reshape((-1,1))
    correct_count = 0
    for node in nodes.values():
        closest_id = np.argmax(np.sum(avg_vecs*node.vector, axis=1))
        if id_to_label[closest_id] == node.label:
            correct_count += 1

    return correct_count / len(nodes)


def analyze_nodes(nodes):
    labels = list(process_dataset.label_set(nodes))
    sizes = {
        'total': len(nodes),
        'label_sizes': {lab: sum(lab == node.label for node in nodes.values())
            for lab in labels}
    }
    all_stats = {
        'labels': labels,
        'sizes': sizes,
        'connectivity': calculate_connectivity_stats(nodes, labels),
    }
    return all_stats


def analyze(data_dir):
    data = pickle.load(open(os.path.join(data_dir, 'fulldata.pickle'), 'rb'))
    stats = analyze_nodes(data)
    json.dump(stats, open(os.path.join(data_dir, 'analysis.txt'), 'w'),
                indent=4)

if __name__ == '__main__':
    analyze(sys.argv[1])
