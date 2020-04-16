import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GCNConv, DeepGraphInfomax


class GCNNet(torch.nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(inp_dim, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 128)
        self.conv4 = GCNConv(128, 128)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = F.relu(self.conv4(x, edge_index, edge_attr))

        x = self.dropout(x)
        x = self.fc(x)
        return x


class GraphClassifier:
    def __init__(self, inp_dim, out_dim, device):
        self.gcn = GCNNet(inp_dim, out_dim)
        self.gcn = self.gcn.to(device)
        self.optimizer = torch.optim.Adam(self.gcn.parameters())

    def evaluate_loss(self, data, mode):
        # use masking for loss evaluation
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if mode == 'train':
            loss = F.cross_entropy(self.gcn(x, edge_index, edge_attr)[data.train_mask], data.y[data.train_mask])
        else:
            loss = F.cross_entropy(self.gcn(x, edge_index, edge_attr)[data.test_mask], data.y[data.test_mask])
        return loss

    def embed(self, data):
        return self.gcn(data.x, data.edge_index, data.edge_attr)

    def train(self, data):
        # training
        self.gcn.train()
        self.optimizer.zero_grad()
        loss = self.evaluate_loss(data, mode='train')
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, data):
        # testing
        self.gcn.eval()
        logits, accs = self.gcn(data.x, data.edge_index, data.edge_attr), []
        loss = self.evaluate_loss(data, mode='test').item()

        for _, mask in data('train_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return [loss] + accs


class DGIEncoderNet(torch.nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(DGIEncoderNet, self).__init__()
        self.conv1 = GCNConv(inp_dim, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, out_dim)

    def forward(self, x, edge_index, edge_attr, msk=None):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x


class DGILearner:
    def __init__(self, inp_dim, out_dim, device):
        self.encoder = DGIEncoderNet(inp_dim, out_dim)
        self.dgi = DeepGraphInfomax(out_dim, encoder=self.encoder, summary=self.readout, corruption=self.corrupt)
        self.dgi = self.dgi.to(device)

        self.optimizer = torch.optim.Adam(self.dgi.parameters())

    def embed(self, data):
        pos_z, _, _ = self.dgi(data.x, data.edge_index, data.edge_attr, msk=None)
        return pos_z

    def readout(self, z, x, edge_index, edge_attr, msk=None):
        if msk is None:
            return torch.sigmoid(torch.mean(z, 0))
        else:
            return torch.sigmoid(torch.sum(z[msk], 0) / torch.sum(msk))

    def corrupt(self, x, edge_index, edge_attr, msk=None):
        shuffled_rows = torch.randperm(len(x))
        shuffled_x = x[shuffled_rows, :]
        return shuffled_x, edge_index, edge_attr

    def evaluate_loss(self, data, mode):
        # use masking for loss evaluation
        pos_z_train, neg_z_train, summ_train = self.dgi(data.x, data.edge_index, data.edge_attr, msk=data.train_mask)
        pos_z_test, neg_z_test, summ_test = self.dgi(data.x, data.edge_index, data.edge_attr, msk=data.test_mask)

        if mode == 'train':
            return self.dgi.loss(pos_z_train, neg_z_train, summ_train)
        else:
            return self.dgi.loss(pos_z_test, neg_z_test, summ_test)

    def train(self, data):
        # training
        self.dgi.train()
        self.optimizer.zero_grad()
        loss = self.evaluate_loss(data, mode='train')
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, data):
        # testing
        self.dgi.eval()
        return self.evaluate_loss(data, mode='test').item()
