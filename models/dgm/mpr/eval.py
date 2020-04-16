import argparse
from collections import defaultdict
import datetime
import os
import os.path as osp
import random

import networkx as nx
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.nn import Linear, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import to_networkx, dense_to_sparse

from utils import dense_diff_pool, dense_mincut_pool, to_dense_adj

EPS = 1e-5


parser = argparse.ArgumentParser()

# Model
parser.add_argument('--mode', type=str, choices=['mpr', 'diffpool', 'mincut'])
parser.add_argument('--pagerank_pooling', type=bool, default=True)
parser.add_argument('--cluster_dims', type=int, nargs='+')
parser.add_argument('--hidden_dims', type=int, nargs='+')
parser.add_argument('--interval_overlap', type=float, default=0.1)
parser.add_argument('--pooling_ratio', type=float, default=0.25)
parser.add_argument('--std_hidden_dim', type=int, default=32)

# Optimization
parser.add_argument('--dataset', type=str, default='DD',
                    choices=['COLLAB', 'DD', 'PROTEINS', 'REDDIT-BINARY'])
parser.add_argument('--fold', type=int, choices=range(11), default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lrate', type=float, default=1e-3)
parser.add_argument('--lrate_anneal_coef', type=float, default=0)
parser.add_argument('--sim_batch_size', type=int, default=32)

# Logs etc
parser.add_argument('--log_dir', type=str, default='./logs/')

args = parser.parse_args()
log_name = '%s_%s_%d_%s_%d_%.4f_%.2f' % (str(args.cluster_dims),
                                         str(args.hidden_dims),
                                         args.interval_overlap,
                                         args.dataset,
                                         args.epochs,
                                         args.lrate,
                                         args.lrate_anneal_coef)


def generate_intervals(num=5, overlap=0.10, vmin=0, vmax=1):
  """Generates the Mapper intervals in [0, 1] with the specified overlap.

  params:
    num: Number of intevals
    overlap: Percentage specifying overlap between consecutive segments.
  """
  assert 0.0 <= overlap <= 1.0 # Overlap must be a percentage

  overlap *= 1. / (num + 1)
  mid = np.linspace(vmin, vmax, num+1)
  high = mid + overlap
  low = mid - overlap
  intervals = np.concatenate((low[:-1, None], high[1:, None]), axis=1)
  return intervals


def compute_preimages(G, f, intervals=generate_intervals()):
  """Computes the preimages of the function f using the specified intervals."""
  preimages = []
  for interv in intervals:
    preimages.append(np.arange(G.number_of_nodes())[np.logical_and(
        interv[0] <= f, f <= interv[1])])
  return preimages


def build_mapper_graph(G, preimages, cluster=False, return_nx=False):
  """Given a graph G and the preimages of the function f on G it builds the
    mapper graph.

  params:
    G: The graph to be visualized.
    preimages: The preimages of a function f on the nodes of graph G.
  """
  mnode = 0
  mnode_to_nodes = []
  mnode_to_color = []
  edge_weight = defaultdict(int)
  current_adj = torch.tensor(
      nx.to_numpy_matrix(G, nodelist=np.arange(G.number_of_nodes()),
                         dtype=np.float32)).to(device)

  # Each preimage has a corresponding colour given by its index.
  for color, pre_img in enumerate(preimages):
    # Skip if the preimage is empty
    if not len(pre_img):
      continue

    # Build the subgraph of the preimage and determine the connected components.
    if cluster:
        G_pre = G.subgraph(pre_img)
        connected_components = list(nx.connected_components(G_pre))
    else:
        connected_components = [pre_img]

    for cc in connected_components:
        # Make each connected component a node and assign its color.
        mnode_to_node = torch.zeros(G.number_of_nodes()).to(device)
        mnode_to_node[np.fromiter(cc, int, len(cc))] = 1.0
        mnode_to_nodes.append(mnode_to_node)
        mnode_to_color.append(color)
        mnode += 1

  # Initialise Mapper graph
  mnode_to_nodes = torch.stack(mnode_to_nodes)
  mnode_to_nodes = mnode_to_nodes / torch.sum(mnode_to_node, dim=0)
  adj = mnode_to_nodes.mm(current_adj).mm(mnode_to_nodes.t())

  if return_nx:
    MG = nx.from_numpy_matrix(adj.cpu().numpy())
  else:
    MG = None

  return MG, [mnode_to_nodes.cpu().numpy(),
              mnode_to_color,
              mnode_to_nodes.t().cpu().numpy(),
              adj]


class GEmbedNet(torch.nn.Module):
    def __init__(self, input_dim=89, hidden_dim=128):
        super(GEmbedNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x


class GClassifier(torch.nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(GClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.mlp = Linear(hidden_dim, dataset.num_classes)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        x = F.dropout(x, training = self.training)
        x = torch.mean(x, dim=0)
        x = self.mlp(x)
        return x


class StandardPoolingModel(torch.nn.Module):
    def __init__(self, mode='diffpool', hidden_dim=32):
        assert mode in ['diffpool', 'mincut']
        super(StandardPoolingModel, self).__init__()
        self.graph_convs = ModuleList([
            GCNConv(dataset.num_node_features, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)])
        self.graph_skips = ModuleList([
            Linear(dataset.num_node_features, hidden_dim),
            Linear(hidden_dim, hidden_dim),
            Linear(hidden_dim, hidden_dim)])

        max_num_nodes = np.max(dataset.data.num_nodes)
        pool_dims = [int(max_num_nodes * args.pooling_ratio),
                     int(max_num_nodes * args.pooling_ratio**2)]
        self.mode = mode
        self.pooling_fn = (dense_diff_pool if mode == 'diffpool'
                           else dense_mincut_pool)
        if mode == 'mincut':
            self.assignment_ws = ModuleList([
                Linear(dataset.num_node_features, hidden_dim),
                Linear(hidden_dim, pool_dims[0]),
                Linear(hidden_dim, hidden_dim),
                Linear(hidden_dim, pool_dims[1])])
        else:
            self.pool_convs = ModuleList([
                GCNConv(dataset.num_node_features, pool_dims[0]),
                GCNConv(hidden_dim, pool_dims[1])])

        self.classifier = Linear(hidden_dim, dataset.num_classes)

    def forward(self, x, edge_index):
        total_loss1 = 0
        total_loss2 = 0
        edge_attr = None

        for i in range(len(self.graph_convs)):
            if i < len(self.graph_convs) - 1:
                if self.mode == 'mincut':
                    s = self.assignment_ws[2*i+1](F.relu(
                        self.assignment_ws[2*i](x)))
                else:
                    s = self.pool_convs[i](x, edge_index, edge_attr)

            x = F.relu((self.graph_convs[i](x, edge_index, edge_attr) +
                        self.graph_skips[i](x)))

            if i < len(self.graph_convs) - 1:
                x, adj, loss1, loss2 = self.pooling_fn(
                    x,
                    to_dense_adj(edge_index, x.size(0), edge_attr=edge_attr),
                    s)
                edge_index, edge_attr = dense_to_sparse(adj.squeeze(0))
                x = x.squeeze(0)
                total_loss1 += loss1
                total_loss2 += loss2

        x_avg = torch.mean(x, dim=0).unsqueeze(0)
        out = self.classifier(x_avg)

        return out, total_loss1, total_loss2


def get_graph_classification_dataset(dataset):
  node_transform = None
  if dataset in ['COLLAB', 'REDDIT-BINARY']:
    node_transform = OneHotDegree(max_degree=64)

  path = osp.join(osp.dirname('/tmp/'), dataset)
  dataset = TUDataset(path, name=dataset, pre_transform=node_transform)

  return dataset


def evaluate_loss():
  loss = F.cross_entropy(y_pred, data.y)
  return loss


def update_lrate(optimizer, epoch):
  if args.lrate_anneal_coef and epoch >= args.epochs // 2:
    optimizer.param_groups[0]['lr'] = (optimizer.param_groups[0]['lr'] *
                                       args.lrate_anneal_coef)


def mpr_forward(data, pool=True):
  # Get node embeddings f(G)
  data = data.to(device)
  max_num_nodes = np.max(dataset.data.num_nodes)

  data.edge_attr = None
  for i in range(len(args.hidden_dims)):
    # Get node embeddings (first part of the lens)
    x, edge_index = data.x, data.edge_index
    x = g_embed_nets[i](x, edge_index)
    vv = np.zeros(len(x))

    if pool:
      # Compute the PageRank function of node embeddings (second part of lens)
      G = nx.to_undirected(to_networkx(data))
      r_dict = nx.pagerank_scipy(G)
      r = np.zeros(G.number_of_nodes())

      for node, rval in r_dict.items():
          r[node] = rval
      ff = r
      ff -= np.min(ff)
      ff /= max(np.max(ff), EPS)
      ff = np.ravel(ff)
      vv = ff

      # Generate intervals
      curr_cluster_dim = args.cluster_dims[i]
      intervals = generate_intervals(num=curr_cluster_dim,
                                     overlap=args.interval_overlap)

      # Compute pre-image f^-1(G)
      preimages = compute_preimages(G, vv, intervals)
      _, [mnode_to_nodes, mnode_to_color, node_to_mnode, adj] = build_mapper_graph(
          G, preimages)

      mnode_features = torch.empty(size=(len(mnode_to_color), x.size(1)),
                                   dtype=torch.float).to(device)

      for mn in range(len(mnode_to_color)):
        mnode_features[mn] = torch.sum(x[torch.BoolTensor(mnode_to_nodes[mn])],
                                       dim=0)

      x = mnode_features
      edge_index, edge_attr = dense_to_sparse(adj)

    # Update data for new embedding/pooling step
    data.x = x
    data.num_nodes = x.size(0)
    data.edge_index = edge_index
    data.edge_attr = edge_attr

  # Classify resulting graph
  y_pred = g_classifier(x, edge_index, edge_attr)

  return vv, y_pred




# Load dataset
dataset = get_graph_classification_dataset(args.dataset)
print(args.dataset, dataset[0],
      dataset.num_classes, 'classes',
      dataset.num_features, 'features')

# Determine runtime device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare log file
log_name += '_' + str(datetime.datetime.now()) + '.txt'
if not osp.exists(args.log_dir):
    os.makedirs(args.log_dir)
f = open(osp.join(args.log_dir, log_name), 'w')

# Set seeds (need to maintain same splits and shuffling across models/runs)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# 10-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=False)

curr_fold = 0
test_accs = []

for train_val_idxs, test_idxs in kf.split(dataset, dataset.data.y):
  curr_fold += 1
  if args.fold and curr_fold != args.fold:
    continue
  s = '>>> 10-fold cross-validation --- fold %d' % curr_fold
  print(s)
  f.write(s + '\n')

  # Split into train-val and test
  train_val_dataset = dataset[torch.LongTensor(train_val_idxs)]
  test_dataset = dataset[torch.LongTensor(test_idxs)]

  # Split first set into train and val
  kf2 = StratifiedKFold(n_splits=9, shuffle=False)
  for train_idxs, val_idxs in kf2.split(train_val_dataset,
                                        train_val_dataset.data.y):
      train_dataset = train_val_dataset[torch.LongTensor(train_idxs)]
      val_dataset = train_val_dataset[torch.LongTensor(val_idxs)]
      break

  # Shuffle the training data
  shuffled_idx = torch.randperm(len(train_dataset))
  train_dataset = train_dataset[shuffled_idx]

  if args.mode == 'mpr':
      g_embed_nets = []
      prev_dim = dataset.num_node_features
      for i in range(len(args.hidden_dims)):
        curr_dim = args.hidden_dims[i]
        g_embed_nets.append(GEmbedNet(input_dim=prev_dim,
                                      hidden_dim=curr_dim).to(device))
        print(g_embed_nets[-1])
        f.write(str(g_embed_nets[-1]) + '\n')
        prev_dim = curr_dim

      g_classifier = GClassifier(input_dim=args.hidden_dims[-1],
                                 hidden_dim=args.hidden_dims[-1]).to(device)
      print(g_classifier)
      f.write(str(g_classifier) + '\n')

      params_list = list(g_classifier.parameters())
      for g_embed_net in g_embed_nets:
        params_list += list(g_embed_net.parameters())
      optimizer = torch.optim.Adam(params_list, lr=args.lrate)
  else:
      model = StandardPoolingModel(mode=args.mode,
                                   hidden_dim=args.std_hidden_dim).to(device)
      print(model)
      f.write(str(model) + '\n')
      optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)

  max_val_acc = 0.0
  for epoch in range(args.epochs):
    # Train model
    train_loss = 0
    if args.mode == 'mpr':
        for g_embed_net in g_embed_nets:
          g_embed_net.train()
        g_classifier.train()
    else:
        train_loss1 = 0
        train_loss2 = 0
        model.train()

    optimizer.zero_grad()

    for i, data in enumerate(train_dataset):
      data = data.to(device)
      if args.mode == 'mpr':
          _, y_pred = mpr_forward(data, args.pagerank_pooling)
          y_pred = y_pred.unsqueeze(0)
      else:
          y_pred, loss1, loss2 = model(data.x, data.edge_index)

      loss = evaluate_loss()
      train_loss += loss
      if args.mode == 'mpr':
          (loss / args.sim_batch_size).backward()
          if i % args.sim_batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
      else:
          train_loss1 += loss1
          train_loss2 += loss2
          total_loss = loss + loss1 + loss2
          total_loss.backward()
          optimizer.step()
          optimizer.zero_grad()

    train_loss /= len(train_dataset)
    if args.mode in ['diffpool', 'mincut']:
        train_loss1 /= len(train_dataset)
        train_loss2 /= len(train_dataset)

    # Run validation set
    val_acc = 0
    val_loss = 0
    if args.mode == 'mpr':
        for g_embed_net in g_embed_nets:
          g_embed_net.eval()
        g_classifier.eval()
    else:
        model.eval()
        val_loss1 = 0
        val_loss2 = 0

    with torch.no_grad():
      for _, data in enumerate(val_dataset):
        data = data.to(device)
        if args.mode == 'mpr':
            _, y_pred = mpr_forward(data, args.pagerank_pooling)
            y_pred = y_pred.unsqueeze(0)
        else:
            y_pred, loss1, loss2 = model(data.x, data.edge_index)

        loss = evaluate_loss().detach().cpu().numpy()
        val_loss += loss
        if args.mode in ['diffpool', 'mincut']:
            val_loss1 += loss1
            val_loss2 += loss2
        val_acc += y_pred.max(1)[1].eq(data.y)

      val_acc = float(val_acc) / len(val_dataset)
      val_loss /= len(val_dataset)
      if args.mode == 'mpr':
          s = ('Epoch %d - train loss %.4f, val loss %.4f, val accuracy %.4f' %
               (epoch, train_loss, val_loss, val_acc))
      else:
          val_loss1 /= len(dataset)
          val_loss2 /= len(dataset)
          s = (('Epoch %d - train loss %.4f, loss1 %.4f, loss2 %.4f, '
                'val loss %.4f, loss1 %.4f, loss2 %.4f, val accuracy %.4f') %
               (epoch,
                train_loss, train_loss1, train_loss2,
                val_loss, val_loss1, val_loss2,
                val_acc))
      print(s)
      f.write(s + '\n')

      if val_acc > max_val_acc:
          s = 'New best validation accuracy at epoch %d: %.4f' % (epoch, val_acc)
          max_val_acc = val_acc
          print(s)
          f.write(s + '\n')

          # Run test set
          test_acc = 0
          test_loss = 0
          if args.mode == 'mpr':
              for g_embed_net in g_embed_nets:
                g_embed_net.eval()
              g_classifier.eval()
          else:
              model.eval()
              test_loss1 = 0
              test_loss2 = 0

          for _, data in enumerate(test_dataset):
            data = data.to(device)
            if args.mode == 'mpr':
                _, y_pred = mpr_forward(data, args.pagerank_pooling)
                y_pred = y_pred.unsqueeze(0)
            else:
                y_pred, loss1, loss2 = model(data.x, data.edge_index)

            loss = evaluate_loss().detach().cpu().numpy()
            test_loss += loss
            if args.mode in ['diffpool', 'mincut']:
                test_loss1 += loss1
                test_loss2 += loss2
            test_acc += y_pred.max(1)[1].eq(data.y)

          test_acc = float(test_acc) / len(test_dataset)
          test_loss /= len(test_dataset)
          if args.mode == 'mpr':
              s = ('Epoch %d - test loss %.4f, test accuracy %.4f' %
                   (epoch, test_loss, test_acc))
          else:
              test_loss1 /= len(dataset)
              test_loss2 /= len(dataset)
              s = ((('Epoch %d - test loss %.4f, loss1 %.4f, loss2 %.4f,'
                    'accuracy %.4f') %
                    (epoch, test_loss, test_loss1, test_loss2, test_acc)))
          print(s)
          f.write(s + '\n')

    update_lrate(optimizer, epoch)
  test_accs.append(test_acc)

s = 'Test accuracies: %s, %.4f +- %.4f' % (str(test_accs),
                                           np.mean(np.array(test_accs)),
                                           np.std(np.array(test_accs)))
print(s)
f.write(s + '\n')

f.close()
