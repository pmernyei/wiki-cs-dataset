import sherpa
import argparse
import os
import json

from .. import load_graph_data
from ..train import train_and_eval
from ..train import register_general_args
from .mlp_train import mlp_model_fn
from .mlp_train import register_mlp_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP hparam search')
    register_general_args(parser)
    register_mlp_args(parser)
    parser.add_argument('--study-dir', help='file to write study results to')
    parser.add_argument('--n-trials', type=int, default=100,
        help='number of trials to run')
    args = parser.parse_args()
    print('Parsed args:', args)
    with open(os.path.join(args.study_dir, 'args.json'), 'w') as out:
        json.dump(vars(args), out)
    parent_out_dir = args.output_dir

    parameters = [
        sherpa.Continuous(name='dropout', range=[0.01, 0.6]),
        sherpa.Continuous(name='lr', range=[1e-4, 1e-1], scale='log'),
        sherpa.Discrete(name='n_hidden_layers', range=[1, 6]),
        sherpa.Discrete(name='n_hidden_units', range=[10, 60])
    ]
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=args.n_trials)
    study = sherpa.Study(parameters=parameters,
                 algorithm=algorithm,
                 lower_is_better=False,
                 disable_dashboard=True,
                 output_dir=args.study_dir)
    data = load_graph_data.load(args)

    for trial in study:
        print('Starting trial {} with params {}'.format(
            trial.id, trial.parameters))
        args.output_dir = os.path.join(parent_out_dir, str(trial.id))
        args.lr = trial.parameters['lr']
        args.dropout = trial.parameters['dropout']
        args.n_layers = trial.parameters['n_hidden_layers']
        args.n_hidden = trial.parameters['n_hidden_units']
        callback = (lambda objective, context:
                        study.add_observation(trial=trial,
                                              objective=objective,
                                              context=context))
        train_and_eval(mlp_model_fn, data, args, callback)
        study.finalize(trial)
        study.save()
