Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks
=======

Wiki-CS is a Wikipedia-based dataset for benchmarking Graph Neural Networks. This repository contains the dataset files, the data pipeline used to create it as well as experiments performed to benchmark node classification and link prediction methods.

The dataset
-----------

### Loading via PyTorch Geometric
You can load the dataset easily using the [`torch_geometric.datasets.WikiCS` class](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/wikics.html#WikiCS) in PyTorch Geometric. Note that the `train_mask`, `val_mask`, `stopping_mask` fields are matrices with rows corresponding to 20 different dataset splits, please average over all of them for evaluation. (The `test_mask` has a single version.)

### Loading from file
The `dataset/data.json` file contains the vectorised representation of the dataset. It includes the node features, adjacency lists, labels and dataset splits. Note that we provide 20 different training splits to avoid overfitting to a specific small set of training nodes for semi-supervised node classification. See our paper for more details.

See `experiments/linkprediction/load_wiki.py` and `experiments/node_classification/load_graph_data.py` for reference data loaders in PyTorch Geometric and DGL, respectively.

Metadata about the nodes and labels can be found in `dataset/metadata.json`, with the same ordering of nodes and labels as the vectorised data file. For nodes, this describes what page of Wikipedia it was derived from and what textual content was used for the features. For labels, the corresponding category is named.

Experiments
-----------
Our experiments were performed using Python 3.5, CUDA 10.1 and the dependencies noted in `requirements.txt`.

### Node classification
For node classification models, run the following, with `#MODEL_NAME#` one of `svm`, `mlp`, `gcn`, `gat` and `appnp`:
```
cd experiments
python -m node_classification.#MODEL_NAME#.#MODEL_NAME#_train --dataset=wiki
```

Add the hyperparameters as follows to replicate our results:
```
cd experiments
python -m node_classification.svm.svm_train --dataset=wiki --self-loop --kernel rbf --c 8 --test
python -m node_classification.mlp.mlp_train --dataset=wiki --self-loop --n-hidden-layers 1 --n-hidden-units 35 --dropout 0.35 --lr 0.003 --test
python -m node_classification.gcn.gcn_train --dataset=wiki --self-loop --n-hidden-layers 1 --n-hidden-units 33 --dropout 0.25 --lr 0.02 --test
python -m node_classification.gat.gat_train --dataset=wiki --self-loop --n-hidden-layers 1 --n-hidden-units 14 --in-drop 0.5 --attn-drop 0.5 --n-heads 5 --lr 0.007 --test
python -m node_classification.appnp.appnp_train --dataset=wiki --self-loop --n-hidden-units 14 --k 2 --alpha 0.11 --in-drop 0.4 --edge-drop 0.4 --lr 0.02 --test
```

### Link prediction
The SVM and VGAE benchmarks for link prediction are included in this repository:
```
cd experiments/linkprediction
python train_vgae.py --dataset=wiki --test
python train_svm.py --dataset=wiki --c=10 --test
```

Citing
-----
If you use our dataset, please cite our paper (Bibtex below).
```
@article{mernyei2020wiki,
  title={Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks},
  author={Mernyei, P{\'e}ter and Cangea, C{\u{a}}t{\u{a}}lina},
  journal={arXiv preprint arXiv:2007.02901},
  year={2020}
}
```

Software used
-------------
* The dataset pipeline includes a modified version of the [Wikipedia category sanitizer](https://github.com/corradomonti/wikipedia-categories) by Boldi and Monti for extracting and sanitizing category labels.
* [`wikiextractor`](https://github.com/attardi/wikiextractor) was used to extract article text data.
* [`mysqldump-to-csv`](https://github.com/jamesmishra/mysqldump-to-csv) was used for processing hyperlink data.
* The GCN, GAT and APPNP implementations were taken from the [DGL examples repository](https://github.com/dmlc/dgl/tree/master/examples/pytorch/).
