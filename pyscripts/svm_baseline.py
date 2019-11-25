import numpy as np
from sklearn.svm import SVC
import json
import sys

def fit_svm(vector_file, C=1.0, test=False):
    data = json.load(open(vector_file))
    all_x = np.array(data['features'])
    all_y = np.array(data['labels'])
    splits = np.array(data['splits'])
    train_x = all_x[splits == 0]
    train_y = all_y[splits == 0]
    validate_x = all_x[splits == 1]
    validate_y = all_y[splits == 1]
    svm = SVC(C=C, kernel='linear')
    svm.fit(train_x, train_y)

    acc = svm.score(validate_x, validate_y)
    print('Validation accuracy:', acc)
    if test:
        test_x = all_x[splits == 2]
        test_y = all_y[splits == 2]
        acc = svm.score(test_x, test_y)
        print('Test accuracy:', acc)


if __name__ == '__main__':
    fit_svm(sys.argv[1], test=('--test' in sys.argv))
