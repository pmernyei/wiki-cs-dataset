Wiki-CS
=======

Wiki-CS is a Wikipedia-based dataset for benchmarking Graph Neural Networks. This repository contains the data pipeline used to create it as well as experiments performed to benchmark node classification and link prediction methods.

The dataset
-----------

The `dataset/data.json` file contains the vectorised representation of the dataset. It includes the node features, adjacency lists, labels and dataset splits. Note that we provide 20 different training splits to avoid overfitting to a specific small set of training nodes for semi-supervised node classification. See our paper for more details.

See `experiments/linkprediction/load_wiki.py` and `experiments/node_classification/load_graph_data.py` for reference data loaders in PyTorch Geometric and DGL, respectively.

Metadata about the nodes and labels can be found in `dataset/metadata.json`, with the same ordering of nodes and labels as the vectorised data file. For nodes, this describes what page of Wikipedia it was derived from and what textual content was used for the features. For labels, the corresponding category is named.

Experiments
-----------
Our experiments were performed using Python 3.5, CUDA 10.1 and the dependencies noted in `requirements.txt`.

### Node classification
For node classification models, run the following, with `#MODEL_NAME#` one of `mlp`, `gcn`, `gat` and `appnp`:
```
cd experiments
python -m node_classification.#MODEL_NAME#.#MODEL_NAME#_train --dataset=wiki
```

Add the hyperparameters as described in the paper to replicate results. For example:
```
cd experiments
python -m node_classification.gcn.train_gcn --dataset=wiki --self-loop --n-hidden-units 33 --dropout 0.25 --lr 0.02 --test
```

### Link prediction
The SVM and VGAE benchmarks for link prediction are included in this repository:
```
cd experimeents/linkprediction
python train_vgae.py --dataset=wiki --test
python train_svm.py --dataset=wiki --c=10 --test
```

Software used
-------------
* The dataset pipeline includes a modified version of the [Wikipedia category sanitizer](https://github.com/corradomonti/wikipedia-categories) by Boldi and Monti for extracting and sanitizing category labels.
* [`wikiextractor`](https://github.com/attardi/wikiextractor) was used to extract article text data.
* [`mysqldump-to-csv`](https://github.com/jamesmishra/mysqldump-to-csv) was used for processing hyperlink data.
