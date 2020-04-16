import argparse

from dgm.dgm import DGM
from dgm.plotting import *
from dgm.utils import *
from dgm.models import GraphClassifier, DGILearner

from torch_geometric.utils.convert import to_networkx

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Dataset to use (spam, cora)', type=str, default='cora')
parser.add_argument('--sdgm', help='Whether to use SDGM or not', action="store_true")
parser.add_argument('--train_mode', help='Supervised or unsupervised training', default='supervised')
parser.add_argument('--reduce_method', help='Method to use for dimensionality reduction', default='tsne')
parser.add_argument('--reduce_dim', help='The final embedding dimension after dimensionality reduction', type=int,
                    default=2)
parser.add_argument('--intervals', help='Number of intervals to use across each axis for the grid', type=int,
                    required=True)
parser.add_argument('--overlap', help='Overlap percentage between consecutive intervals on each axis', type=float,
                    required=True)
parser.add_argument('--eps', help='Edge filtration value for SDGM', type=float, default=0.0)
parser.add_argument('--min_component_size', help='Minimum connected component size to be included in the visualisation',
                    type=int, default=0.0)
parser.add_argument('--dir', help='Directory inside plots where to save the results', default='')


def train_model(dataset, train_mode, num_classes, device):
    if train_mode == 'supervised':
        model = GraphClassifier(dataset.num_node_features, num_classes, device)
    elif train_mode == 'unsupervised':
        model = DGILearner(dataset.num_node_features, 512, device)
    else:
        raise ValueError('Unsupported train mode {}'.format(train_mode))

    train_epochs = 81 if train_mode == 'supervised' else 201
    for epoch in range(0, train_epochs):
        train_loss = model.train(dataset)

        if epoch % 5 == 0:
            if train_mode == 'unsupervised':
                test_loss = model.test(dataset)
                log = 'Epoch: {:03d}, train_loss: {:.3f}, test_loss:{:.3f}'
                print(log.format(epoch, train_loss, test_loss))
            else:
                log = 'Epoch: {:03d}, train_loss: {:.3f}, test_loss:{:.3f}, train_acc: {:.2f}, test_acc: {:.2f}'
                print(log.format(epoch, train_loss, *model.test(dataset)))

    return model.embed(dataset).detach().cpu().numpy()


def plot_dgm_graph(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Plotting the {} graph".format(args.dataset))

    data, num_classes = load_dataset(args.dataset)
    data = data.to(device)
    graph = to_networkx(data).to_undirected()

    print("Graph nodes", graph.number_of_nodes())
    print("Graph edges", graph.number_of_edges())

    embed_path = "./data/{}_{}.npy".format(args.dataset, args.train_mode)
    if os.path.isfile(embed_path):
        print('Using existing embedding')
        embed = np.load(embed_path)
    else:
        print('No embedding found. Training a new model...')
        embed = train_model(data, args.train_mode, num_classes, device)
        np.save(embed_path, embed)

    embed = reduce_embedding(embed, reduce_dim=args.reduce_dim, method=args.reduce_method)

    print('Creating visualisation...')
    out_graph, res = DGM(num_intervals=args.intervals, overlap=args.overlap, eps=args.eps,
                         min_component_size=args.min_component_size, sdgm=args.sdgm).fit_transform(graph, embed)

    binary = args.reduce_method == 'binary_prob'
    plot_graph(out_graph, node_color=res['mnode_to_color'], node_size=res['node_sizes'], edge_weight=res['edge_weight'],
               node_list=res['node_list'], name=name_from_args(args, False), save_dir=args.dir, colorbar=binary)

    print("Filtered Mapper Graph nodes", out_graph.number_of_nodes())
    print("Filtered Mapper Graph edges", out_graph.number_of_edges())

    labeled_colors = color_mnodes_with_labels(res['mnode_to_nodes'], data.y.cpu().numpy(), binary=binary)
    plot_graph(out_graph, node_color=labeled_colors, node_size=res['node_sizes'], edge_weight=res['edge_weight'],
               node_list=res['node_list'], name=name_from_args(args, True), save_dir=args.dir, colorbar=binary)


if __name__ == "__main__":
    random.seed(444)
    np.random.seed(444)
    torch.manual_seed(444)
    plot_dgm_graph(parser.parse_args())
