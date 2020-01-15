import numpy as np
from sklearn.svm import SVC
import json
import sys
import argparse

import load_graph_data


def fit_svm(data, C=1.0, test=False):
    acc_sum = 0
    for i in range(len(data.train_masks)):
        svm = SVC(C=C, kernel='linear')
        tr = data.train_masks[i]
        val = data.val_masks[i]
        svm.fit(data.features[tr], data.labels[tr])
        acc = svm.score(data.features[val], data.labels[val])
        print('Validation accuracy:', acc)
        if test:
            acc = svm.score(data.features[data.test_mask],
                            data.labels[data.test_mask])
            print('Test accuracy:', acc)
        acc_sum += acc
    print('Avg {} accuracy: {:.4f}'.format('test' if test else 'val', acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVM training')
    load_graph_data.register_data_args(parser)
    parser.add_argument("--test", action='store_true',
            help="evaluate on test set after training (default=False)")
    args = parser.parse_args()
    data = load_graph_data.load(args)
    fit_svm(data, test=args.test)
