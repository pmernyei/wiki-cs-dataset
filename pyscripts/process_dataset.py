def get_significant_words(nodes, multiplier = 2.0, min_occurrences = 100):
    labels = set().union(*[set(n.labels) for n in nodes.values()])
    labels.add('ALL')
    total_tokens = {label: 0 for label in labels}
    word_counts = {label: {} for label in labels}
    for node in nodes.values():
        labels = node.labels.union({'ALL'})
        for label in labels:
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


def add_binary_word_vectors(nodes, significant_words):
    for id,node in nodes.items():
        token_set = set(node.tokens)
        node.words_binary = np.array([bool(word in token_set) for word in significant_words])
