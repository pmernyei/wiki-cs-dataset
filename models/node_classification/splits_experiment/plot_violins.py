import seaborn as sns
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='splits-experiment-plot')
    parser.add_argument('--results-file', help='file to read results from')
    parser.add_argument('--out', help='file to read results from',
        default='violinplot.png')
    args = parser.parse_args()

    results = pd.read_pickle(args.results_file)
    results['accuracy_delta'] = results.accuracy
    for model in results.model.unique():
        mask = (results.model == model)
        avg = results[mask].accuracy.mean()
        results.loc[mask, 'accuracy_delta'] -= avg

    sns.set(style="whitegrid")
    plot = sns.violinplot(x='split', y='accuracy_delta', hue='model',
        data=results, palette='muted')
    fig = plot.get_figure()
    fig.savefig(args.out)
