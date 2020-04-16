#!/usr/bin/env bash

# ================================================= SDGM ===============================================================

## Plot the SDGM graphs with embeddings computed by a supervised GCN network
python -u run_dgm.py --dataset=spam --reduce_method=binary_prob --intervals=20 --overlap=0 --eps=0.01 \
                     --min_component_size=4 --sdgm --dir=wiki_sdgm_supervised
python -u run_dgm.py --dataset=wiki --reduce_method=tsne --intervals=10 --overlap=0 --eps=0.10 \
                     --min_component_size=20 --sdgm --dir=wiki_sdgm_supervised
python -u run_dgm.py --dataset=wiki --reduce_method=tsne --intervals=20 --overlap=0 --eps=0.01 \
                     --min_component_size=10 --sdgm --dir=wiki_sdgm_supervised
python -u run_dgm.py --dataset=wiki --reduce_method=tsne --intervals=5 --overlap=0 --eps=0.01 \
                     --min_component_size=10 --sdgm --dir=wiki_sdgm_supervised

# Plot the SDGM graph with embeddings computed with unsupervised lens (DGI network)
python -u run_dgm.py --dataset=wiki --train_mode=unsupervised --reduce_method=tsne --intervals=20 \
                     --overlap=0 --eps=0.05 --min_component_size=8 --sdgm --dir=wiki_sdgm_unsupervised \

python -u run_dgm.py --dataset=wiki --train_mode=unsupervised --reduce_method=tsne --intervals=20 \
                     --overlap=0 --eps=0.03 --min_component_size=8 --sdgm --dir=wiki_sdgm_unsupervised \

# SDGM ablation for dimensionality reduction method
python -u run_dgm.py --dataset=wiki --train_mode=unsupervised --reduce_method=tsne --intervals=20 \
                     --overlap=0 --eps=0.05 --min_component_size=8 --sdgm --reduce_method=tsne --dir=wiki_sdgm_reduce_ablation

python -u run_dgm.py --dataset=wiki --train_mode=unsupervised --reduce_method=tsne --intervals=15 \
                     --overlap=0 --eps=0.04 --min_component_size=10 --sdgm --reduce_method=pca --dir=wiki_sdgm_reduce_ablation

python -u run_dgm.py --dataset=wiki --train_mode=unsupervised --reduce_method=tsne --intervals=15 \
                     --overlap=0 --eps=0.04 --min_component_size=10 --sdgm --reduce_method=isomap --dir=wiki_sdgm_reduce_ablation

python -u run_dgm.py --dataset=wiki --train_mode=unsupervised --reduce_method=tsne --intervals=20 \
                     --overlap=0 --eps=0.04 --min_component_size=10 --sdgm --reduce_method=umap --dir=wiki_sdgm_reduce_ablation


# ================================================== DGM ===============================================================

# Plot the DGM graphs with embeddings computed by a supervised GCN network
python -u run_dgm.py --dataset=spam --reduce_method=binary_prob --intervals=20 --overlap=0.1 \
                     --min_component_size=4 --dir=wiki_dgm_supervised
python -u run_dgm.py --dataset=wiki --reduce_method=tsne --intervals=8 --overlap=0.2 \
                     --min_component_size=15 --dir=wiki_dgm_supervised
python -u run_dgm.py --dataset=wiki --reduce_method=umap --intervals=3 --overlap=0.05 \
                     --min_component_size=50 --dir=wiki_dgm_supervised
python -u run_dgm.py --dataset=wiki --reduce_method=tsne --intervals=6 --overlap=0.2 \
                     --min_component_size=20 --dir=wiki_dgm_supervised

# Plot the DGM graph with embeddings computed with unsupervised lens (DGI network)
python -u run_dgm.py --dataset=wiki --train_mode=unsupervised --reduce_method=tsne --intervals=8 \
                     --overlap=0.2 --min_component_size=15 --dir=wiki_dgm_unsupervised \

python -u run_dgm.py --dataset=wiki --train_mode=unsupervised --reduce_method=tsne --intervals=8 \
                     --overlap=0.2 --min_component_size=20 --dir=wiki_dgm_unsupervised \


# DGM ablation for dimensionality reduction method
python -u run_dgm.py --dataset=wiki --train_mode=unsupervised --reduce_method=tsne --intervals=9 \
                     --overlap=0.2 --min_component_size=15 --reduce_method=tsne --dir=wiki_dgm_reduce_ablation

python -u run_dgm.py --dataset=wiki --train_mode=unsupervised --reduce_method=tsne --intervals=9 \
                     --overlap=0.1 --min_component_size=15 --reduce_method=pca --dir=dgm_reduce_ablation

python -u run_dgm.py --dataset=wiki --train_mode=unsupervised --reduce_method=tsne --intervals=9 \
                     --overlap=0.1 --min_component_size=15 --reduce_method=isomap --dir=dgm_reduce_ablation

python -u run_dgm.py --dataset=wiki --train_mode=unsupervised --reduce_method=tsne --intervals=9 \
                     --overlap=0.2 --min_component_size=15 --reduce_method=umap --dir=dgm_reduce_ablation
