import numpy as np
import umap
import os
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib import cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from scipy.special import softmax


def plot_graph(graph, node_color, node_size, edge_weight, node_list, figsize=(12, 10), colorbar=True,
               save_dir='', name='plot'):
    """Example function for plotting the Mapper graph using networkx."""
    # Set color map
    if colorbar:
        cmap = cm.coolwarm
        cmap = cm.get_cmap(cmap, 100)
        plt.set_cmap(cmap)
    else:
        plt.set_cmap(cm.Accent)

    # Set figures size
    plt.figure(figsize=figsize)

    # Compute edge width
    edges = graph.edges()
    weights = np.array([edge_weight[(min(u, v), max(u, v))] for u, v in edges], dtype=np.float32)

    # Make width visually pleasing
    width = weights * 20

    # Compute visually pleasing node size
    node_size = np.sqrt(node_size) * 90

    # Draw the graph
    pos = nx.nx_pydot.graphviz_layout(graph)
    nx.draw(graph, pos=pos, node_color=node_color, width=width, node_size=node_size,
            node_list=node_list)

    if colorbar:
        sm = cm.ScalarMappable(cmap=cmap)
        sm._A = []
        plt.colorbar(sm)

    dir_path = os.path.join('plots/', save_dir, "")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = os.path.join(dir_path, "{}.png".format(name))
    plt.savefig(file_path, dgi=300)


def color_mnodes_with_labels(mnode_to_nodes, labels, binary=True):
    """Colors the nodes of the Mapper graph using the true labels of the nodes."""
    label_color = []
    for mnode, cc in mnode_to_nodes.items():
        nodes = np.array(list(cc))
        cc_labels = labels[nodes]
        unique_labels, freq = np.unique(cc_labels, return_counts=True)

        if binary:
            # For binary labels, add the proportion of class one inside the cluster
            if len(freq) == 1:
                label_color.append(unique_labels[0])
            else:
                label_color.append(freq[1] / np.sum(freq))
        else:
            # For multi categorical labels, add the most frequent class inside the node
            label_color.append(unique_labels[np.argmax(freq)])
    return np.array(label_color)


def color_from_bivariate_data(mnode_to_color, cmap1=plt.cm.cool, cmap2=plt.cm.coolwarm):
    """Produces a 2D colormap for visualising the 2D lens functions.

    Code adapted from https://stackoverflow.com/questions/49871436/scatterplot-with-continuous-bivariate-color-palette-in-python
    """
    Z1, Z2 = mnode_to_color[:, 0], mnode_to_color[:, 1]
    # Rescale values to fit into colormap range (0->255)
    Z1_plot = np.array(255 * (Z1 - Z1.min()) / (Z1.max() - Z1.min()), dtype=np.int)
    Z2_plot = np.array(255 * (Z2 - Z2.min()) / (Z2.max() - Z2.min()), dtype=np.int)

    Z1_color = cmap1(Z1_plot)
    Z2_color = cmap2(Z2_plot)

    # Color for each point
    Z_color = np.sum([Z1_color, Z2_color], axis=0) / 2.0
    return Z_color


def reduce_embedding(embed, reduce_dim, method):
    print('Reducing the embedding dimension using {}...'.format(method))
    if method == 'tsne':
        embed = TSNE(n_components=reduce_dim, n_jobs=-1).fit_transform(embed)
    elif method == 'isomap':
        embed = Isomap(n_components=reduce_dim, n_jobs=-1).fit_transform(embed)
    elif method == 'pca':
        embed = PCA(n_components=reduce_dim).fit_transform(embed)
    elif method == 'umap':
        embed = umap.UMAP(n_components=reduce_dim).fit_transform(embed)
    elif method == 'binary_prob':
        assert embed.shape[1] == 2
        embed = softmax(embed, axis=1)
        embed = embed[:, 1][:, None]
    else:
        raise ValueError()

    return embed
