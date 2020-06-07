import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as gutils
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, VGAE, GAE
import seaborn as sns
import numpy as np
import load_wiki
import json
import argparse

class VGAE_Encoder(nn.Module):
    """Two-layer GCN encoder as described in the VGAE paper."""
    def __init__(self, num_features, latent_dim=16):
        super(VGAE_Encoder, self).__init__()
        hidden_layer = 2*latent_dim
        self.conv_1 = GCNConv(num_features, hidden_layer)
        self.conv_2_mean = GCNConv(hidden_layer, latent_dim)
        self.conv_2_var = GCNConv(hidden_layer, latent_dim)

    def forward(self, features, edge_index):
        h = F.relu(self.conv_1(features, edge_index))
        mean = self.conv_2_mean(h, edge_index)
        var = self.conv_2_var(h, edge_index)
        return mean, var


def mean_with_uncertainty(values, n_boot=10000, conf_threshold=95):
    values = np.array(values)
    avg = values.mean()
    bootstrap = sns.algorithms.bootstrap(
        values, func=np.mean, n_boot=n_boot)
    conf_int = sns.utils.ci(bootstrap, conf_threshold)
    return avg, np.max(np.abs(conf_int - avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGAE')
    parser.add_argument('--dataset')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--val-freq', type=int, default=20)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
            '.', 'data', args.dataset)
        data = Planetoid(path, args.dataset)[0]
    else:
        data = load_wiki.load_data()

    data.edge_index = gutils.to_undirected(data.edge_index)
    data = GAE.split_edges(GAE, data)

    num_features = data.x.shape[1]
    aucs = []
    aps = []
    for run in range(args.runs):
        model = VGAE(VGAE_Encoder(num_features))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            z = model.encode(data.x, data.train_pos_edge_index)
            loss = model.recon_loss(z, data.train_pos_edge_index)#0.01*model.kl_loss()
            loss.backward()
            optimizer.step()

            # Log validation metrics
            if epoch % args.val_freq == 0:
                model.eval()
                with torch.no_grad():
                    z = model.encode(data.x, data.train_pos_edge_index)
                    auc, ap = model.test(z,
                        data.val_pos_edge_index, data.val_neg_edge_index)
                print('Train loss: {:.4f}, Validation AUC-ROC: {:.4f}, '
                      'AP: {:.4f} at epoch {:03d}'.format(loss, auc, ap, epoch))

        # Final evaluation
        model.eval()
        with torch.no_grad():
            if args.test:
                z = model.encode(data.x, data.train_pos_edge_index)
                auc, ap = model.test(z,
                    data.test_pos_edge_index, data.test_neg_edge_index)
            else:
                z = model.encode(data.x, data.train_pos_edge_index)
                auc, ap = model.test(z,
                    data.val_pos_edge_index, data.val_neg_edge_index)
        print(('Test' if args.test else 'Validation'), 'AUC-ROC:',
            auc, 'AP:', ap)
        aucs.append(auc)
        aps.append(ap)
        json.dump(aucs, open('aucs.txt', 'w'))
        json.dump(aucs, open('aps.txt', 'w'))

    auc_mean, auc_ci = mean_with_uncertainty(aucs)
    ap_mean, ap_ci = mean_with_uncertainty(aps)
    print('AUC-ROC:', auc_mean, '+-', auc_ci)
    print('AP:',      ap_mean,  '+-', ap_ci)
