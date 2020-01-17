import sherpa
import argparse

import load_graph_data
from load_graph_data import register_data_args
from train import train_and_eval
from train import register_general_args
from gcn_train import gcn_model_fn
from gcn_train import register_gcn_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN hparam search')
    register_general_args(parser)
    register_gcn_args(parser)
    parser.add_argument('--study-dir', help='file to write study results to')
    parser.add_argument('--n-trials', type=int, default=100,
        help='number of trials to run')
    args = parser.parse_args()
    print('Parsed args:', args)

    parameters = [
        sherpa.Continuous(name='lr', range=[1e-3, 1e-1], scale='log'),
        sherpa.Continuous(name='dropout', range=[0.2, 0.8]),
        sherpa.Discrete(name='num_hidden_units', range=[5, 50], scale='log')
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
        args.lr = trial.parameters['lr']
        args.n_hidden = trial.parameters['num_hidden_units']
        args.dropout = trial.parameters['dropout']
        callback = (lambda objective, context:
                        study.add_observation(trial=trial,
                                              objective=objective,
                                              context=context))
        train_and_eval(gcn_model_fn, data, args, callback)
        study.finalize(trial)
