"""
Sample a set of pages from each class for interactive inspection by the user
and write the inspection results to file.
"""
import pickle
import os
import json
import random

import process_dataset

def print_node_data(node, all_nodes_map):
    """Print details of a single page."""
    print('Title:', node.title, '; id:', node.id)
    print('Label:', node.label)
    print('Text:', ' '.join(node.tokens[:20])+'...')
    print('Linked pages:',
        [all_nodes_map[id].title for id in node.outlinks][:5], 'total',
        len(node.outlinks))
    print()


def sample_and_validate(dataset_dir, sample_count=20):
    """
    Sample sample_count pages per class from the dataset in the given directory,
    ask user to evaluate correctness of labels, write results to file in that
    directory.
    """
    nodes = pickle.load(
        open(os.path.join(dataset_dir, 'fulldata.pickle'), 'rb')
    )
    labels = process_dataset.label_set(nodes)
    nodes_for_labels = {lab:[] for lab in labels}
    verdicts = {}
    for node in nodes.values():
        nodes_for_labels[node.label].append(node)
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
    json.dump(verdicts,
              open(os.path.join(dataset_dir, 'sample_manual_check.txt'), 'w'),
              indent=2)
    return verdicts


if __name__ == '__main__':
    sample_and_validate(sys.argv[1])
