import numpy as np
import networkx as nx

from collections import defaultdict, OrderedDict
from dgm.plotting import color_from_bivariate_data


class DGM:
    def __init__(self, num_intervals, overlap, eps=0.05, min_component_size=15, sdgm=True):
        """Initialises DGM (https://arxiv.org/abs/2002.03864) or SDGM (Recommended - Appendix B).

        Args:
            num_intervals (int): The number of intervals to use along each axis to produce the overlapping grid.
            overlap (float): The overlap between consecutive intervals along each axis expressed as a number in [0, 1].
            eps (float): The filtration value for SDGM (used only when SDGM is true)
            min_component_size (int): The minimum size of a newly formed clusters. Small clusters are filtered out to reduce
                potential clutter (see the filter_mapper_graph method for more details).
            sdgm (bool): Use Structural Deep Graph Mapper (recommended). The refined pull back elements are connected
                with edges weighted by the number of edges in the original graph between the elements.
        """
        self.num_intervals = num_intervals
        self.overlap = overlap
        self.eps = eps
        self.min_component_size = min_component_size
        self.sdgm = sdgm

    def generate_1d_grid(self, vmin=0, vmax=1):
        """Generates overlapping intervals in [0, 1] with the specified overlap.

        Args:
            vmin (float): Lower end of the interval
            vmax (float): Higher end of the interval

        Returns:
            intervals (np.array): Numpy array of shape (num_intervals, 2) with last dimension specifying the start and end
                of each interval
        """
        assert 0.0 <= self.overlap <= 1.0  # Overlap must be a percentage

        self.overlap *= 1. / (self.num_intervals + 1)
        mid = np.linspace(vmin, vmax, self.num_intervals + 1)
        high = mid + self.overlap
        low = mid - self.overlap
        intervals = np.concatenate((low[:-1, None], high[1:, None]), axis=1)
        return intervals

    def generate_2d_grid(self, vmin=0, vmax=1):
        """Generates a grid of overlapping cells in [0, 1].

        Args:
            vmin (float): Lower end of the interval
            vmax (float): Higher end of the interval

        Returns:
            xx (np.array): Coordinates of the top left corner of each cell
            yy (np.array): Coordinates of the bottom right corner of each cell
        """
        x = np.linspace(vmin, vmax, self.num_intervals + 1)
        y = np.linspace(vmin, vmax, self.num_intervals + 1)

        self.overlap *= 1. / (self.num_intervals + 1)
        xx1, yy1 = np.meshgrid(x - self.overlap, y - self.overlap, indexing='ij')
        xx1, yy1 = xx1[:-1, :-1], yy1[:-1, :-1]

        xx2, yy2 = np.meshgrid(x + self.overlap, y + self.overlap, indexing='ij')
        xx2, yy2 = xx2[1:, 1:], yy2[1:, 1:]

        xx1 = np.ravel(xx1)
        yy1 = np.ravel(yy1)
        xx = np.concatenate((xx1[:, None], yy1[:, None]), axis=-1)

        xx2 = np.ravel(xx2)
        yy2 = np.ravel(yy2)
        yy = np.concatenate((xx2[:, None], yy2[:, None]), axis=-1)

        return xx, yy

    def generate_1d_pull_back(self, f, intervals):
        """Computes the pull back of the function 1-valued f using the specified intervals.

        Besides the pre-images of the intervals, the function also computes the associated colour of each pre-image based
        on the average value of the lens.

        Args:
            f (np.array): The output of the lens function f specified by a one-dimensional numpy array
            intervals (np.array): Array of shape (num_nodes, 2) specifying the overlapping intervals over [0, 1]

        Returns:
            pull_back (list): List of tuples containing a colour and the pre-image.
        """
        if len(f.shape) != 1:
            raise ValueError('The lens must be one-dimensional but has shape {}'.format(f.shape))

        nodes = np.arange(len(f))
        pull_back = []

        for interv in intervals:
            # Compute f^-1(U), where U is an interval
            preimage = nodes[np.logical_and(interv[0] < f, f < interv[1])]

            # Skip empty preimages
            if not len(preimage):
                continue

            # Set colour to average lens value
            color = np.mean(f[preimage])

            pull_back.append((color, preimage))
        return pull_back

    def generate_2d_pull_back(self, f, xx, yy):
        """Computes the pull back of the function 2-valued f using the specified grid.

        Besides the pre-images of the intervals, the function also computes the associated colour of each pre-image based
        on the average value of the lens.

        Args:
            f (np.array): The output of the lens function f specified by a one-dimensional numpy array
            xx (np.array): Array of shape (num_nodes, 2) specifying the top left corner of each cell in the grid
            yy (np.array): Array of shape (num_nodes, 2) specifying the bottom right corner of each cell in the grid

        Returns:
            pull_back (list): List of tuples containing a colour and the pre-image.
        """

        nodes = np.arange(len(f))
        pull_back = []

        for i in range(len(xx)):
            x_and = np.logical_and(xx[i][0] <= f[:, 0], yy[i][0] >= f[:, 0])
            y_and = np.logical_and(xx[i][1] <= f[:, 1], yy[i][1] >= f[:, 1])
            preimage = nodes[np.logical_and(x_and, y_and)]

            # Skip empty preimages
            if not len(preimage):
                continue

            # Set colour to average lens value
            color = np.mean(f[preimage], axis=0)

            pull_back.append((color, preimage))
        return pull_back

    def construct_dgm_graph(self, graph, pull_back, bivariate_color=False):
        """Given a graph and the pull back of the function f it computes the refined pullback and builds the Mapper graph.

        Args:
            graph (networkx.Graph): The graph to be visualized.
            pull_back (list): A pull_back cover over the nodes of the graph.
            bivariate_color (bool): Whether to use 1D or 2D colouring.
        Returns:
            mapper_graph (networkx.Graph): The Mapper visualisation graph.
            aux (dict): Auxiliary results containing metadata in the form of a dictionary:
                mnode_to_nodes (OrderedDict): A mapping from the nodes of the output graph to the nodes in the original graph.
                mnode_to_color (np.array): The color of the new nodes in the order specified by node_list.
                edge_weight (dict): A dictionary from edges to the weight of the edge.
                node_list (np.array): A list of the nodes the graph in non-decreasing order.

        """
        mnode = 0

        # Mapper nodes to old nodes.
        mnode_to_nodes = OrderedDict()
        # Mapper nodes to colour.
        mnode_to_color = []
        # Old node to new node.
        node_to_mnode = defaultdict(set)
        # Edges between clusters.
        edge_weight = defaultdict(float)

        for colored_preimages in pull_back:
            color = colored_preimages[0]
            pre_img = colored_preimages[1]

            # Skip if the preimage is empty
            if not len(pre_img):
                continue

            # Build the subgraph of the preimage and determine the connected components.
            G_pre = graph.subgraph(pre_img)
            connected_components = list(nx.connected_components(G_pre))

            for cc in connected_components:
                # Make each connected component a node and assign its color.
                cc_set = set(cc)

                mnode_to_nodes[mnode] = cc_set
                mnode_to_color.append(color)

                # Build a map from the old nodes to the new nodes.
                for node in cc_set:
                    node_to_mnode[node].add(mnode)

                # Increment the node index in the new graph.
                mnode += 1

        # Initialise Mapper graph
        mapper_graph = nx.Graph()
        mapper_graph.add_nodes_from(np.arange(mnode))

        new_edges = defaultdict(float)
        if self.sdgm:
            # Check edges between the connected components.
            for edge in graph.edges:
                n1, n2 = edge[0], edge[1]
                for mn1 in node_to_mnode[n1]:
                    for mn2 in node_to_mnode[n2]:
                        if mn1 != mn2:
                            new_edges[(min(mn1, mn2), max(mn1, mn2))] += 1
        else:
            # DGM: Check common nodes between the connected components.
            for node in node_to_mnode:
                for mn1 in node_to_mnode[node]:
                    for mn2 in node_to_mnode[node]:
                        if mn1 != mn2:
                            new_edges[(min(mn1, mn2), max(mn1, mn2))] += 1

        if len(new_edges) > 0:
            maxw = np.max([w for _, w in new_edges.items()])

        # Add the normalised edges to the graph and the edge_weight dictionary.
        for edge, edgew in new_edges.items():
            # Normalise the weight.
            assert 0 <= edgew
            norm_weight = edgew / (float(maxw) + 1e-8)
            if norm_weight >= self.eps or not self.sdgm:
                mapper_graph.add_edge(edge[0], edge[1], weight=norm_weight)
                edge_weight[edge] = norm_weight

        node_sizes = np.array([len(cc) for _, cc in mnode_to_nodes.items()])

        mnode_to_color = np.array(mnode_to_color)
        if bivariate_color:
            mnode_to_color = color_from_bivariate_data(mnode_to_color)

        aux = {
            'mnode_to_nodes': mnode_to_nodes,
            'mnode_to_color': mnode_to_color,
            'edge_weight': edge_weight,
            'node_list': np.arange(mnode),
            'node_sizes': node_sizes,
        }

        print("Mapper graph nodes", mapper_graph.number_of_nodes())
        print("Mapper graph edges", mapper_graph.number_of_edges())

        return mapper_graph, aux

    def normalise_embed(self, embed):
        """Brings each embedding dimension in [0, 1]."""
        embed -= np.min(embed, axis=0, keepdims=True)
        embed /= (np.max(embed, axis=0, keepdims=True) + 0.000001)
        return embed

    def build_1d_dgm(self, graph, embed):
        """Computes the DGM visualisation for one-dimensional embeddings (lens)."""
        embed = np.ravel(embed)

        # Generate the cover \mathcal(U) defined by a series of overlapping intervals
        intervals = self.generate_1d_grid()

        # Compute the pull back cover of \mathcal(U)
        pull_back = self.generate_1d_pull_back(embed, intervals)

        # Construct the Mapper graph
        return self.construct_dgm_graph(graph, pull_back)

    def build_2d_dgm(self, graph, embed):
        """Computes the DGM visualisation for two-dimensional embeddings (lens)."""

        # Generate the cover \mathcal(U) defined by a series of overlapping 2D grid cells
        xx, yy = self.generate_2d_grid()

        # Compute the pull back cover of \mathcal(U)
        pull_back = self.generate_2d_pull_back(embed, xx, yy)

        # Construct the Mapper graph
        return self.construct_dgm_graph(graph, pull_back, bivariate_color=True)

    def filter_mapper_graph(self, mg, aux):
        """Filters out from the mapper graph the small connected components.

        Args:
            mg (networkx.Graph): The Mapper graph to be filtered.
            aux (dict): The metadata dictionary associated with the graph
            min_component_size (int): The minimum number of nodes from the original graph that each new node should contain.
                Connected components formed from fewer (original) nodes than this are removed to reduce clutter.

        Returns:
            fmg (networkx.Graph): The filtered Mapper graph.
            filtered_aux (dict): The filtered auxiliary data.
        """
        fmg = mg.copy()
        # Initialise a mask over the nodes
        mask = np.ones(fmg.number_of_nodes(), dtype=np.bool)

        connected_components = nx.connected_components(mg)
        for cc in connected_components:
            component_size = np.sum([len(aux['mnode_to_nodes'][mnode]) for mnode in cc])
            if component_size < self.min_component_size:
                mask[np.array(list(cc))] = 0.0
                fmg.remove_nodes_from(cc)

        new_mnode_to_color = aux['mnode_to_color'][mask]
        new_node_list = aux['node_list'][mask]
        new_node_sizes = aux['node_sizes'][mask]

        new_mnode_to_nodes = OrderedDict()
        for mnode in new_node_list:
            new_mnode_to_nodes[mnode] = aux['mnode_to_nodes'][mnode]

        new_edge_weight = defaultdict(float)
        for edge, edgew in aux['edge_weight'].items():
            if mask[edge[0]] and mask[edge[1]]:
                new_edge_weight[edge] = edgew

        filtered_aux = {
            'mnode_to_nodes': new_mnode_to_nodes,
            'mnode_to_color': new_mnode_to_color,
            'edge_weight': new_edge_weight,
            'node_list': new_node_list,
            'node_sizes': new_node_sizes,
        }

        return fmg, filtered_aux

    def fit_transform(self, graph, f):
        """Constructs a graph using DGM (https://arxiv.org/abs/2002.03864) or SDGM (Recommended - Appendix B).

        Args:
            graph (networkx.Graph): The graph to be visualised.
            f (np.array): The 1D or 2D node embeddings produced by the chosen lens function.

        Returns:
            mg_graph (networkx.Graph): The Mapper graph
            aux (dict): The graph metadata to be used for visualisation.
        """
        if len(f.shape) != 2 or f.shape[1] > 2:
            raise ValueError('SDGM supports only 1D and 2D dimensional parametrization spaces.')

        f = self.normalise_embed(f)

        if f.shape[1] == 1:
            mg_graph, aux = self.build_1d_dgm(graph, f)
        else:
            mg_graph, aux = self.build_2d_dgm(graph, f)

        if self.min_component_size > 0:
            return self.filter_mapper_graph(mg_graph, aux)
        return mg_graph, aux
