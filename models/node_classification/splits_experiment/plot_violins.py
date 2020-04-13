import seaborn as sns
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='splits-experiment-plot')
    parser.add_argument('--results-file', help='file to read results from')
    parser.add_argument('--out', help='file to read results from', default='violinplot.png')
    args = parser.parse_args()

    results = pd.read_pickle(args.results_file)
    sns.set(style="whitegrid")
    plot = sns.violinplot(x='split', y='accuracy', hue='model', data=results,
        palette='muted')
    fig = plot.get_figure()
    fig.savefig(args.out)
