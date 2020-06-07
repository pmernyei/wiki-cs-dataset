import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='splits-experiment-plot')
    parser.add_argument('--results-file', help='file to read results from')
    parser.add_argument('--start-split', type=int, default=0)
    parser.add_argument('--total-splits', type=int, default=5)
    parser.add_argument('--width', type=float, default=16)
    parser.add_argument('--height', type=float, default=8)
    parser.add_argument("--relative", action="store_true", default=False,
                            help="show accuracy difference from model avgs")
    parser.add_argument('--out', help='file to read results from',
        default='violinplot.png')
    args = parser.parse_args()

    results = pd.read_pickle(args.results_file)
    results = results[(results.model != 'mlp')]
    results = results[(results.split >= args.start_split) & (results.split < args.start_split + args.total_splits)]
    results['accuracy_delta'] = results.accuracy
    for model in results.model.unique():
        mask = (results.model == model)
        avg = results[mask].accuracy.mean()
        results.loc[mask, 'accuracy_delta'] -= avg

    sns.set(style="whitegrid")
    plt.figure(figsize=(args.width, args.height))
    plot = sns.violinplot(
        x='split',
        y=('accuracy_delta' if args.relative else 'accuracy'),
        hue='model',
        data=results,
        palette='muted'
    )
    fig = plot.get_figure()
    fig.savefig(args.out)
