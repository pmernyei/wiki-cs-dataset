import numpy as np

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

def add_glove_word_vectors(nodes, glove_dict):
    zeros = 0
    for id,node in nodes.items():
        sum = np.zeros(len(next(iter(glove_dict.values()))))
        for t in node.tokens:
            if t in glove_dict:
                sum += glove_dict[t]
        if np.linalg.norm(sum) == 0.0:
            zeros += 1;
        node.vector = np.concatenate((node.vector, sum/len(node.tokens)))
    print(zeros, 'nodes with no words in glove dict')

def load_glove_dict(filename):
    result = {}
    with open(filename, 'r') as input:
        for line in input:
            l = line.split(' ')
            word = l[0]
            weights = np.array([float(x) for x in l[1:]])
            result[word] = weights
    return result


def output_data(out_filename):
    ## TODO: list IDs, matrix from node vectors, vector from labels,
    ## map linked ids and output ragged matrix, choose training set??
    pass
