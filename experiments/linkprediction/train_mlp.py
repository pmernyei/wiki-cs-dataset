"""
Training and evaluation search for link prediction with an MLP.
"""
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE
import torch_geometric.utils as gutils
import numpy as np
import load_wiki
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
import itertools
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from train_vgae import mean_with_uncertainty

def make_mlp(hidden_layers, hidden_size, dropout=0.5, in_dim=600):
    layers = []
    layers.append(nn.Linear(in_dim, hidden_size))
    for i in range(hidden_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(hidden_size, 1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def make_loader(x,y,batch_size=64):
    input = list(zip(x,y))
    return DataLoader(input, batch_size=batch_size, shuffle=True)


def train(model, loader, optimizer):
    model.train()
    losses = []
    for x,y in loader:
        optimizer.zero_grad()
        preds = model(x).view(-1)
        loss = F.binary_cross_entropy(preds, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def eval(model,x,y):
    model.eval()
    preds = model(x).view(-1).detach().cpu()
    auc = roc_auc_score(y.cpu(), preds)
    ap = average_precision_score(y.cpu(), preds)
    return auc,ap


def sample_negative(count, nodes, avoid):
    avoid_set = set(zip(avoid[0], avoid[1]))
    result = np.array([[],[]], dtype=np.int32)
    while(result.shape[1] < count):
        candidates = np.random.randint(low=0, high=nodes, size=(2*count, 2))
        for u,v in candidates:
            if u != v and (u,v) not in avoid_set:
                avoid_set.add((u,v))
                result = np.concatenate((result, [[u],[v]]), axis=1)
            if result.shape[1] == count:
                break
    return torch.LongTensor(result)


def combine_node_pair_features(features, pos_edge_index, neg_edge_index):
    x_pos = torch.cat((features[pos_edge_index[0]],
                      features[pos_edge_index[1]]), dim=1)
    x_neg = torch.cat((features[neg_edge_index[0]],
                       features[neg_edge_index[1]]), dim=1)
    x = torch.cat((x_pos, x_neg), dim=0)
    y = torch.cat((torch.ones(len(x_pos)), torch.zeros(len(x_pos))), dim=0)
    return x, y


def load_data(dataset_name, device):
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
            '.', 'data', dataset_name)
        data = Planetoid(path, dataset_name)[0]
    else:
        data = load_wiki.load_data()

    data.edge_index = gutils.to_undirected(data.edge_index)
    data = GAE.split_edges(GAE, data)

    features = data.x
    train_pos_edges = data.train_pos_edge_index
    train_neg_edges = sample_negative(
        count = train_pos_edges.shape[1],
        avoid = train_pos_edges,
        nodes = features.shape[0]
    )

    x_tr, y_tr = combine_node_pair_features(features,
        train_pos_edges, train_neg_edges)
    x_val, y_val = combine_node_pair_features(features,
        data.val_pos_edge_index.numpy(), data.val_neg_edge_index)
    x_test, y_test = combine_node_pair_features(features,
        data.test_pos_edge_index.numpy(), data.test_neg_edge_index)
    x_tr = x_tr.to(device)
    y_tr = y_tr.to(device)
    x_val = x_val.to(device)
    x_test = x_test.to(device)
    return x_tr, y_tr, x_val, y_val, x_test, y_test


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='MLP link prediction')
    parser.add_argument('--dataset')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--hidden-layers', type=int, default=1)
    parser.add_argument('--hidden-units', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    args = parser.parse_args()
    x_tr, y_tr, x_val, y_val, x_test, y_test = load_data(args.dataset, device)
    train_loader = make_loader(x_tr, y_tr)
    aucs = []
    aps = []
    for _ in range(args.runs):
        model = make_mlp(args.hidden_layers, args.hidden_units, args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay)
        best_val = 0
        test_auc = 0
        test_ap = 0
        for epoch in range(args.epochs):
            loss = train(model, train_loader, optimizer)
            tr_auc, tr_ap = eval(model, x_tr, y_tr)
            val_auc, val_ap = eval(model, x_val, y_val)
            print('Epoch {:03d}, loss {:.6f}, tr AUC {:.4f}, tr AP {:.4f}, val AUC {:.4f}, val AP {:.4f}'.format(epoch, loss, tr_auc, tr_ap, val_auc, val_ap))
            if args.test and val_ap + val_auc > best_val:
                best_val = val_ap + val_auc
                test_auc, test_ap = eval(model, x_test, y_test)
                print('test AUC {:.6f}, test AP {:.6f}'.format(test_auc, test_ap))

        if args.test:
            print('Final test AUC {:.6f}, test AP {:.6f}'.format(test_auc, test_ap))
            aucs.append(test_auc)
            aps.append(test_ap)
        else:
            aucs.append(val_auc)
            aps.append(val_ap)
        json.dump(aucs, open('aucs.txt', 'w'))
        json.dump(aucs, open('aps.txt', 'w'))
        auc_mean, auc_ci = mean_with_uncertainty(aucs)
        ap_mean, ap_ci = mean_with_uncertainty(aps)
        print('AUC-ROC:', auc_mean, '+-', auc_ci)
        print('AP:',      ap_mean,  '+-', ap_ci)
