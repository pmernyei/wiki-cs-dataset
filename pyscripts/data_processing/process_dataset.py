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

def output_data(nodes, vectors_outfile, raw_data_outfile, train_split=0.05):
    labels = list(label_set(nodes))
    node_ids_for_labels = {lab: [] for lab in labels}
    for node in nodes.values():
        node_ids_for_labels[node.label].append(node.id)
    training_ids = []
    validation_ids = []
    test_ids = []
    for lab in labels:
        ids = node_ids_for_labels[lab]
        train_cutoff = int(train_split*len(ids))
        validate_cutoff = int((1-((1-train_split)/2))*len(ids))
        training_ids += ids[:train_cutoff]
        validation_ids += ids[train_cutoff:validate_cutoff]
        test_ids += ids[validate_cutoff:]
    random.shuffle(training_ids)
    random.shuffle(validation_ids)
    random.shuffle(test_ids)
    splits = len(training_ids)*[0] + len(validation_ids)*[1] + len(test_ids)*[2]
    final_id_list = training_ids + validation_ids + test_ids
    remap_node_ids = {old_id: new_id for new_id, old_id in enumerate(final_id_list)}
    node_features = [nodes[id].vector.tolist() for id in final_id_list]
    label_ids = {lab: i for i,lab in enumerate(labels)}
    labels_vec = [label_ids[nodes[id].label] for id in final_id_list]
    links = [[remap_node_ids[nb] for nb in nodes[id].outlinks] for id in final_id_list]
    vector_data = {
        'features': node_features,
        'labels': labels_vec,
        'links': links,
        'splits': splits
    }
    raw_metadata = {
        'labels': {i: lab for i,lab in enumerate(labels)},
        'nodes': [raw_data_dict(nodes[id]) for id in final_id_list]
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
