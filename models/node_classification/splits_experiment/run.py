import argparse
import numpy as np
import pandas as pd
import os

from .. import load_graph_data
from .. import train
from ..mlp import mlp_train
from ..gcn import gcn_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='splits-experiment')
    parser.add_argument('--model', help='choice of model to run')
    parser.add_argument('--results-file', help='file to write or append results to')

    train.register_general_args(parser)
    mlp_train.register_mlp_args(parser)
    args = parser.parse_args()
    print('Parsed args:', args)

    if args.model == 'mlp':
        model_fn = mlp_train.mlp_model_fn
    elif args.model == 'gcn':
        model_fn = gcn_train.gcn_model_fn
    else:
        raise ValueError('Model not supported')

    if not os.path.exists(args.results_file):
        dtypes = np.dtype([('model', str), ('split', int), ('accuracy', float)])
        empty_data = np.empty(0, dtype=dtypes)
        df = pd.DataFrame(empty_data)
        df.to_pickle(args.results_file)


    data = load_graph_data.load(args)

    start_split = args.start_split
    splits = args.max_splits
    end = min(len(data.train_masks), start_split + splits)
    args.max_splits = 1
    evals_output_dir = args.output_dir

    for split in range(start_split, end):
        print('Starting runs on split', split)
        args.start_split = split
        args.output_dir = os.path.join(evals_output_dir, str(split))
        os.mkdir(args.output_dir)

        def callback(objective, context):
            results = pd.read_pickle(args.results_file)
            print(context)
            for acc in context['split_details']['test_accs']:
                results = results.append({
                    'model': args.model,
                    'split': split,
                    'accuracy': acc
                }, ignore_index=True)
            results.to_pickle(args.results_file)

        train.train_and_eval(model_fn, data, args, callback)
