import numpy as np
import json
import random
import sys
import pickle
import os
import word_frequencies

def label_set(nodes):
    return {n.label for n in nodes.values()}

def get_significant_words(nodes, multiplier = 2.0, min_occurrences = 100):
    labels = label_set(nodes)
    labels.add('ALL')
    total_tokens = {label: 0 for label in labels}
    word_counts = {label: {} for label in labels}
    for node in nodes.values():
        node_labels = [node.label, 'ALL']
        for label in node_labels:
            total_tokens[label] += len(node.tokens)
            for token in node.tokens:
                word_counts[label][token] = word_counts[label].get(token, 0) + 1

    significant_words = set()
    for word in word_counts['ALL'].keys():
        freqs = {lab: word_counts[lab].get(word, 0) / total_tokens[lab] for lab in labels}
        if word_counts['ALL'][word] > min_occurrences and \
            (any([freq > freqs['ALL']*multiplier for lab,freq in freqs.items()]) \
            or any([freq < freqs['ALL']/multiplier for lab,freq in freqs.items()])):
            significant_words.add(word)

    return significant_words


def add_binary_word_vectors(nodes, words):
    for id,node in nodes.items():
        token_set = set(node.tokens)
        node.vector = np.concatenate((node.vector, np.array([bool(word in token_set) for word in words])))

def add_glove_word_vectors(nodes, glove_dict, words_whitelist=None):
    zeros = []
    for id,node in nodes.items():
        sum = np.zeros(len(next(iter(glove_dict.values()))))
        for t in node.tokens:
            if t in glove_dict and (words_whitelist is None or t in words_whitelist):
                sum += glove_dict[t]
        if np.linalg.norm(sum) == 0.0:
            zeros += [node.title];
        node.vector = np.concatenate((node.vector, sum/len(node.tokens)))
    print(len(zeros), 'nodes with no words in glove dict:', zeros)

def load_glove_dict(filename, relevant_words=None):
    result = {}
    with open(filename, 'r', encoding='utf8') as input:
        for line in input:
            l = line.split(' ')
            word = l[0]
            if relevant_words is None or word in relevant_words:
                weights = np.array([float(x) for x in l[1:]])
                result[word] = weights
    return result

def raw_data_dict(node):
    return {
        'id': node.id,
        'title': node.title,
        'label': node.label,
        'outlinks': node.outlinks,
        'tokens': node.tokens
    }

def output_data(nodes, vectors_outfile, raw_data_outfile, train_ratio=0.05,
                test_ratio=0.5, stopping_ratio = 0.3, n_train_splits = 20):
    labels = list(label_set(nodes))
    node_ids_for_labels = {lab: [] for lab in labels}
    all_ids_list = []
    for node in nodes.values():
        node_ids_for_labels[node.label].append(node.id)
        all_ids_list.append(node.id)

    test_ids = set()
    train_sets = [set() for _ in range(n_train_splits)]
    stopping_sets = [set() for _ in range(n_train_splits)]
    val_sets = [set() for _ in range(n_train_splits)]
    for lab in labels:
        ids = node_ids_for_labels[lab]
        random.shuffle(ids)
        n_train = int(train_ratio*len(ids))
        n_test = int(test_ratio*len(ids))
        n_stopping = int(stopping_ratio*len(ids))

        test_ids.update(ids[:n_test])
        visible_ids = ids[n_test:]
        for i in range(n_train_splits):
            random.shuffle(visible_ids)
            train_sets[i].update(visible_ids[:n_train])
            stopping_sets[i].update(visible_ids[n_train : (n_train+n_stopping)])
            val_sets[i].update(visible_ids[n_train+n_stopping:])

    remap_node_ids = {old_id: new_id for new_id, old_id in enumerate(all_ids_list)}

    test_mask = [(id in test_ids) for id in all_ids_list]
    train_masks = [[id in train_sets[i] for id in all_ids_list] for i in range(n_train_splits)]
    stopping_masks = [[id in stopping_sets[i] for id in all_ids_list] for i in range(n_train_splits)]
    val_masks = [[id in val_sets[i] for id in all_ids_list] for i in range(n_train_splits)]

    node_features = [nodes[id].vector.tolist() for id in all_ids_list]
    label_ids = {lab: i for i,lab in enumerate(labels)}
    labels_vec = [label_ids[nodes[id].label] for id in all_ids_list]
    links = [[remap_node_ids[nb] for nb in nodes[id].outlinks] for id in all_ids_list]

    vector_data = {
        'features': node_features,
        'labels': labels_vec,
        'links': links,
        'train_masks': train_masks,
        'stopping_masks': stopping_masks,
        'val_masks': val_masks,
        'test_mask': test_mask
    }
    raw_metadata = {
        'labels': {i: lab for i,lab in enumerate(labels)},
        'nodes': [raw_data_dict(nodes[id]) for id in all_ids_list]
    }
    json.dump(vector_data, open(vectors_outfile, 'w'))
    json.dump(raw_metadata, open(raw_data_outfile, 'w'))


if __name__ == '__main__':
    data_dir = sys.argv[1]
    glove_file = sys.argv[2]
    data = pickle.load(open(os.path.join(data_dir, 'data'), 'rb'))
    freqs = word_frequencies.dataset_word_frequencies(data)
    words = freqs.keys()
    glove = load_glove_dict(glove_file, relevant_words=words)
    add_glove_word_vectors(data, glove)
    output_data(data,
                os.path.join(data_dir, 'vectors.json'),
                os.path.join(data_dir, 'rawdata.json'))
