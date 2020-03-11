import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphAttentionLayer import GraphAttentionLayer, SpGraphAttentionLayer
import numpy as np
from EvaluateAcc import EvaluateAcc
import torch.optim as optim
import time
import math
from torch.nn.parameter import Parameter
from utils import accuracy

class MethodDeepGATResNet(nn.Module):
    data = None
    lr = 0.01
    weight_decay = 5e-4
    epoch = 200
    depth = 1
    learning_record_dict = {}
    residual_type = None
    cuda_tag = None

    def __init__(self, nfeat, nhid, nclass, dropout, seed, alpha, nheads, depth, cuda):
        """Sparse version of GAT."""
        nn.Module.__init__(self)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.cuda_tag = cuda
        if self.cuda_tag:
            torch.cuda.manual_seed(seed)

        self.dropout = dropout
        self.depth = depth
        self.gat_list = [None] * self.depth
        self.residual_weight_list = [None] * self.depth

        if self.depth == 1:
            self.gat_list = []
            self.out_att = GraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha, concat=False)
            self.residual_weight_list[self.depth-1] = Parameter(torch.FloatTensor(nfeat, nclass))
        else:
            for depth_index in range(self.depth - 1):
                if depth_index == 0:
                    self.gat_list[depth_index] = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
                    self.residual_weight_list[depth_index] = Parameter(torch.FloatTensor(nfeat, nhid * nheads))
                else:
                    self.gat_list[depth_index] = [GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
                    self.residual_weight_list[depth_index] = Parameter(torch.FloatTensor(nhid * nheads, nhid * nheads))
                for i, attention in enumerate(self.gat_list[depth_index]):
                    self.add_module('attention_{}_{}'.format(depth_index, i), attention)
            self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
            self.residual_weight_list[self.depth-1] = Parameter(torch.FloatTensor(nhid * nheads, nclass))
        for i in range(self.depth):
            stdv = 1. / math.sqrt(self.residual_weight_list[i].size(1))
            self.residual_weight_list[i].data.uniform_(-stdv, stdv)

    #---- non residual ----
    def forward(self, raw_x, adj):
        if self.residual_type == 'naive':
            return self.forward_naive(raw_x, adj)
        elif self.residual_type == 'raw':
            return self.forward_raw(raw_x, adj)
        elif self.residual_type == 'graph_naive':
            return self.forward_graph_naive(raw_x, adj)
        elif self.residual_type == 'graph_raw':
            return self.forward_graph_raw(raw_x, adj)

    def forward_raw(self, raw_x, adj):
        x = raw_x
        for i in range(self.depth-1):
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.gat_list[i]], dim=1) + torch.mm(raw_x, self.residual_weight_list[0])
        x = F.dropout(x, self.dropout, training=self.training)
        if self.depth == 1:
            x = F.elu(self.out_att(x, adj)) + torch.mm(raw_x, self.residual_weight_list[self.depth - 1])
        else:
            x = F.elu(self.out_att(x, adj)) + torch.mm(torch.mm(raw_x, self.residual_weight_list[0]), self.residual_weight_list[self.depth-1])
        return F.log_softmax(x, dim=1)

    def forward_graph_raw(self, raw_x, adj):
        x = raw_x
        for i in range(self.depth-1):
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.gat_list[i]], dim=1) + torch.spmm(self.norm_adj, torch.mm(raw_x, self.residual_weight_list[0]))
        x = F.dropout(x, self.dropout, training=self.training)
        if self.depth == 1:
            x = F.elu(self.out_att(x, adj)) + torch.spmm(self.norm_adj, torch.mm(raw_x, self.residual_weight_list[self.depth - 1]))
        else:
            x = F.elu(self.out_att(x, adj)) + torch.spmm(self.norm_adj, torch.mm(torch.mm(raw_x, self.residual_weight_list[0]), self.residual_weight_list[self.depth-1]))
        return F.log_softmax(x, dim=1)

    def forward_naive(self, raw_x, adj):
        x = raw_x
        for i in range(self.depth-1):
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.gat_list[i]], dim=1) + torch.mm(x, self.residual_weight_list[i])
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj)) + torch.mm(x, self.residual_weight_list[self.depth-1])
        return F.log_softmax(x, dim=1)

    def forward_graph_naive(self, raw_x, adj):
        x = raw_x
        for i in range(self.depth-1):
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.gat_list[i]], dim=1) + torch.spmm(self.norm_adj, torch.mm(x, self.residual_weight_list[i]))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj)) + torch.spmm(self.norm_adj, torch.mm(x, self.residual_weight_list[self.depth-1]))
        return F.log_softmax(x, dim=1)

    def forward_non(self, raw_x, adj):
        x = raw_x
        for i in range(self.depth-1):
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.gat_list[i]], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

    def myparameters(self):
        parameter_list = list(self.parameters())
        for i in range(self.depth-1):
            for gat in self.gat_list[i]:
                parameter_list += gat.parameters()
        parameter_list += self.out_att.parameters()
        parameter_list += self.residual_weight_list
        return parameter_list

    def train_model(self, epoch_iter):
        t_begin = time.time()
        optimizer = optim.Adam(self.myparameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(epoch_iter):
            t_epoch_begin = time.time()
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj)
            loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
            acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj)

            loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
            acc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])

            loss_test = F.nll_loss(output[self.idx_test], self.labels[self.idx_test])
            acc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])

            self.learning_record_dict[epoch] = {'loss_train': loss_train.item(), 'acc_train': acc_train.item(),
                                                'loss_val': loss_val.item(), 'acc_val': acc_val.item(),
                                                'loss_test': loss_test.item(), 'acc_test': acc_test.item(),
                                                'time': time.time() - t_epoch_begin}

            if epoch % 10 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'loss_test: {:.4f}'.format(loss_test.item()),
                      'acc_test: {:.4f}'.format(acc_test.item()),
                      'time: {:.4f}s'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
        return acc_test

    def test_model(self):
        return {}

    def prepare_learning_settings(self):
        self.features = self.data['X']
        self.adj = self.data['A']
        self.norm_adj = self.data['norm_adj']
        self.labels = self.data['y']
        self.idx_test = self.data['idx_test']
        self.idx_train = self.data['idx_train']
        self.idx_val = self.data['idx_val']

        print(self.cuda_tag)
        if self.cuda_tag:
            self.cuda()
            self.features = self.features.cuda()
            self.adj = self.adj.cuda()
            self.labels = self.labels.cuda()
            self.idx_test = self.idx_test.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()

    def run(self):
        self.prepare_learning_settings()

        acc_test = self.train_model(self.epoch)
        result = self.test_model()
        result['learning_record'] = self.learning_record_dict
        return result, acc_test


