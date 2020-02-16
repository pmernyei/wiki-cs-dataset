import sherpa
import argparse
import os
import json

import load_graph_data
from train import train_and_eval
from train import register_general_args
from appnp_train import appnp_model_fn
from appnp_train import register_appnp_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='APPNP hparam search')
    register_general_args(parser)
    register_appnp_args(parser)
    parser.add_argument('--study-dir', help='file to write study results to')
    parser.add_argument('--n-trials', type=int, default=100,
        help='number of trials to run')
    args = parser.parse_args()
    print('Parsed args:', args)
    with open(os.path.join(args.study_dir, 'args.json'), 'w') as out:
        json.dump(vars(args), out)
    parent_out_dir = args.output_dir

    parameters = [
        sherpa.Continuous(name='lr', range=[5e-4, 5e-2], scale='log'),
        sherpa.Continuous(name='dropout', range=[0.01, 0.6]),
        sherpa.Continuous(name='alpha', range=[0.05, 0.2]),
        sherpa.Discrete(name='k', range=[5, 15])
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
        args.output_dir = os.path.join(args.output_dir, str(trial.id))
        args.lr = trial.parameters['lr']
        args.in_drop = trial.parameters['dropout']
        args.edge_drop = trial.parameters['dropout']
        args.alpha = trial.parameters['alpha']
        args.k = trial.parameters['k']
        callback = (lambda objective, context:
                        study.add_observation(trial=trial,
                                              objective=objective,
                                              context=context))
        train_and_eval(appnp_model_fn, data, args, callback)
        study.finalize(trial)
        study.save()
