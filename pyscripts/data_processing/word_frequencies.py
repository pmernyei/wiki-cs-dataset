"""
Functions to calculate and process word frequencies across either the entire
Wikipedia data or an extracted subgraph dataset, and plotting this data.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import json
import string
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def get_entire_wiki_word_frequencies(text_extractor_data_dir, output=None):
    """
    Get frequencies of all words from extracted Wikipedia text.
    """
    freqs = {}
    for root, dirs, files in os.walk(text_extractor_data_dir):
        for file in files:
            for line in open(os.path.join(root, file), "r", encoding='utf8'):
                entry = json.loads(line)
                tokens = [t.lower() for t in nltk.word_tokenize(entry['text'])
                                            if t not in string.punctuation]
                for t in tokens:
                    freqs[t] = freqs.get(t, 0) + 1
    print('Calculated frequency of', len(freqs), 'words')
    if output is not None:
        with open(output, 'w', encoding='utf8') as outfile:
            json.dump(freqs, outfile, indent=1)
    return freqs


def dataset_word_frequencies(nodes):
    """
    Get frequency of words from an extracted dataset.
    """
    freqs = {}
    for node in nodes.values():
        for t in node.tokens:
            freqs[t.lower()] = freqs.get(t.lower(), 0) + 1
    return freqs


def plot_frequencies(freqs, words, title=None):
    """
    Plot frequencies of the listed words, looking up from the given freqs dict.
    """
    ys = np.array([freqs[w] for w in words])

    fig, ax = plt.subplots()
    index = np.arange(len(words))
    rects1 = plt.bar(index, ys, 0.8)

    if title is not None:
        plt.title(title)
    plt.yscale('log')
    plt.xticks(index, words, rotation=60, ha='right')
    plt.tight_layout()
    plt.show()


def desc_frequency_list(freqs):
    l = sorted(freqs.items(), reverse=True, key=lambda x: x[1])
    return [w for (w,c) in l]
