from wiki_node import WikiDataNode
from itertools import product
from datetime import datetime
import numpy as np
import random
import pickle
import os
import json

import process_dataset




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
    return {
        'inside_connectivity': float(inside_connectivity),
        'inter_connectivity': float(inter_connectivity),
        'connectivity_matrix': connectivities.tolist()
    }

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
    multilabel_ratio = sum(len(node.labels) > 1 for node in nodes.values()) / \
        len(nodes)
    return {
        'multilabel_node_ratio': multilabel_ratio,
        'overlap_ratios': overlap_ratios.tolist()
    }


def calculate_avg_cosine_similarities(nodes, label_list, sample_count=1000):
    totals = np.zeros((len(label_list), len(label_list)))
    valids = np.zeros((len(label_list), len(label_list)))
    nodes_per_label = [[node for node in nodes.values() if lab in node.labels] for lab in label_list]

    for i,j in product(range(len(label_list)), range(len(label_list))):
        samples = min(sample_count, min(len(nodes_per_label[i]), len(nodes_per_label[j])))
        left = random.sample(nodes_per_label[i], samples)
        right = random.sample(nodes_per_label[j], samples)
        for node1, node2 in zip(left, right):
            v1 = node1.words_binary
            v2 = node2.words_binary
            l1 = np.sum(v1*v1)
            l2 = np.sum(v2*v2)
            if l1 > 0 and l2 > 0:
                valids[i][j] += 1
                totals[i][j] += np.sum(v1*v2)/(l1*l2)
    totals /= valids
    return totals

def cosine_similarity_classification_accuracy(nodes):
    labels = process_dataset.label_set(nodes)
    label_ids = {lab: i for lab,i in zip(labels, range(len(labels)))}
    id_to_label = {i: lab for lab,i in label_ids.items()}
    words_dim = len(next(iter(nodes.values())).words_binary)
    label_freqs = np.zeros(len(labels))
    avg_vecs = np.zeros((len(labels), words_dim))
    for node in nodes.values():
        for lab in node.labels:
            avg_vecs[label_ids[lab]] += node.words_binary
            label_freqs[label_ids[lab]] += 1

    # Normalize row vectors
    avg_vecs /= np.sqrt(np.sum(avg_vecs*avg_vecs, axis=1)).reshape((-1,1))
    correct_count = 0
    for node in nodes.values():
        closest_id = np.argmax(np.sum(avg_vecs*node.words_binary, axis=1))
        if id_to_label[closest_id] in node.labels:
            correct_count += 1

    return correct_count / len(nodes)


def word_selection_stats(nodes, label_list, multipliers, thresholds):
    print(datetime.now().strftime('%H:%M:%S'), 'Starting experiments...')
    results = {multiplier: {threshold: {} for threshold in thresholds} for multiplier in multipliers}
    for multiplier, threshold in product(multipliers, thresholds):
        print(datetime.now().strftime('%H:%M:%S'), 'Trying', multiplier, threshold, '...')
        words = process_dataset.get_significant_words(nodes, multiplier, threshold)
        print(datetime.now().strftime('%H:%M:%S'), 'Selected', len(words), 'words')
        process_dataset.add_binary_word_vectors(nodes, words)
        zeros = int(sum(np.sum(node.words_binary) == 0 for node in nodes.values()))
        results[multiplier][threshold] = {
            'words_dim': len(words),
            'zero_vectors': zeros,
            'baseline_accuracy': cosine_similarity_classification_accuracy(nodes),
            'avg_class_similarities': calculate_avg_cosine_similarities(nodes, label_list).tolist()
        }
        print(datetime.now().strftime('%H:%M:%S'), 'Baseline acc', results[multiplier][threshold]['baseline_accuracy'])
    return results


def full_analysis(nodes):
    labels = list(process_dataset.label_set(nodes))
    sizes = {
        'total': len(nodes),
        'label_sizes': {lab: sum(lab in node.labels for node in nodes.values()) for lab in labels}
    }
    all_stats = {
        'labels': labels,
        'sizes': sizes,
        'connectivity': calculate_connectivity_stats(nodes, labels),
        'multi_labels': calculate_multilabel_stats(nodes, labels),
        'word_selections': word_selection_stats(nodes, labels, [2, 5, 10, 20, 50, 100], [100, 200, 500, 1000, 2000, 5000, 10000])
    }
    return all_stats


def print_node_data(node, all_nodes_map):
    print('Title:', node.title, '; id:', node.id)
    print('Labels:', node.labels)
    print('Text:', ' '.join(node.tokens[:20])+'...')
    print('Linked pages:', [all_nodes_map[id].title for id in node.outlinks][:5], 'total', len(node.outlinks))
    print()


def print_sample_pages(nodes, sample_count=5):
    samples = random.sample(nodes.values(), sample_count)
    for sample in samples:
        print_node_data(sample, nodes)

def sample_and_validate(dataset_dir, sample_count=20):
    nodes = pickle.load(open(os.path.join(dataset_dir, 'data'), 'rb'))
    labels = process_dataset.label_set(nodes)
    nodes_for_labels = {lab:[] for lab in labels}
    verdicts = {}
    for node in nodes.values():
        if len(node.labels) > 1:
            continue
        for lab in node.labels:
            nodes_for_labels[lab].append(node)
    for lab in labels:
        sample = random.sample(nodes_for_labels[lab], sample_count)
        sample_verdicts = {}
        print('Listing sample for label', lab)
        print()
        for node in sample:
            print_node_data(node, nodes)
            sample_verdicts[node.title] = input('Any problems? ')
        verdicts[lab] = sample_verdicts
        print('Finished label', lab)
        print()
    json.dump(verdicts, open(os.path.join(dataset_dir, 'sample_manual_check.txt'), 'wb'), indent=2)
    return verdicts
