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
    freqs = {}
    for root, dirs, files in os.walk(text_extractor_data_dir):
        for file in files:
            for line in open(os.path.join(root, file), "r", encoding='utf8'):
                entry = json.loads(line)
                tokens = [t.lower() for t in nltk.word_tokenize(entry['text']) \
                                            if t not in string.punctuation]
                for t in tokens:
                    freqs[t] = freqs.get(t, 0) + 1
    print('Calculated frequency of', len(freqs), 'words')
    if output is not None:
        with open(output, 'w', encoding='utf8') as outfile:
            json.dump(freqs, outfile, indent=1)
    return freqs

def dataset_word_frequencies(nodes):
    freqs = {}
    for node in nodes.values():
        for t in node.tokens:
            freqs[t.lower()] = freqs.get(t.lower(), 0) + 1
    return freqs

def plot_frequencies(freqs, words):
    ys = np.array([freqs[w] for w in words])

    fig, ax = plt.subplots()
    index = np.arange(len(words))
    rects1 = plt.bar(index, ys, 0.8)

    plt.yscale('log')
    plt.xticks(index, words, rotation=60, ha='right')
    plt.tight_layout()
    plt.show()
