import sherpa
import argparse
import os
import json

from .. import load_graph_data
from ..train import train_and_eval
from ..train import register_general_args
from .gat_train import gat_model_fn
from .gat_train import register_gat_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN hparam search')
    register_general_args(parser)
    register_gat_args(parser)
    parser.add_argument('--study-dir', help='file to write study results to')
    parser.add_argument('--n-trials', type=int, default=100,
        help='number of trials to run')
    args = parser.parse_args()
    print('Parsed args:', args)
    with open(os.path.join(args.study_dir, 'args.json'), 'w') as out:
        json.dump(vars(args), out)

    parameters = [
        sherpa.Continuous(name='lr', range=[1e-3, 1e-1], scale='log'),
        sherpa.Continuous(name='dropout', range=[0.2, 0.8]),
        sherpa.Discrete(name='num_hidden_units', range=[12, 20], scale='log'),
        sherpa.Choice(name='residual', range=[True, False]),
        sherpa.Discrete(name='num_heads', range=[4,8], scale='log'),
        sherpa.Discrete(name='num_layers', range=[1,2])
    ]
    algorithm = sherpa.algorithms.GPyOpt(max_num_trials=args.n_trials)
    study = sherpa.Study(parameters=parameters,
                 algorithm=algorithm,
                 lower_is_better=False,
                 disable_dashboard=True,
                 output_dir=args.study_dir)
    data = load_graph_data.load(args)

    for trial in study:
        print('Starting trial {} with params {}'.format(
            trial.id, trial.parameters))
        args.lr = trial.parameters['lr']
        args.n_hidden = int(trial.parameters['num_hidden_units'])
        args.attn_drop = trial.parameters['dropout']
        args.in_drop = trial.parameters['dropout']
        args.residual = trial.parameters['residual']
        args.num_heads = int(trial.parameters['num_heads'])
        args.n_layers = int(trial.parameters['num_layers'])
        callback = (lambda objective, context:
                        study.add_observation(trial=trial,
                                              objective=objective,
                                              context=context))
        train_and_eval(gat_model_fn, data, args, callback)
        study.finalize(trial)
        study.save()
