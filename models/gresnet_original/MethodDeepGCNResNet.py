'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from method import method
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from GraphConvolution import GraphConvolution
from EvaluateAcc import EvaluateAcc
import scipy.sparse as sp
import numpy as np
import time
import random
import math
from torch.nn.parameter import Parameter

class MethodDeepGCNResNet(nn.Module):
    data = None
    lr = 0.01
    weight_decay = 5e-4
    epoch = 200
    learning_record_dict = {}
    residual_type = 'none'

    def __init__(self, nfeat, nhid, nclass, dropout, seed, depth):
        nn.Module.__init__(self)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.depth = depth
        self.gc_list = [None] * self.depth
        self.residual_weight_list = [None] * self.depth
        if self.depth == 1:
            self.gc_list[self.depth-1] = GraphConvolution(nfeat, nclass)
            self.residual_weight_list[self.depth-1] = Parameter(torch.FloatTensor(nfeat, nclass))
        else:
            for i in range(self.depth):
                if i == 0:
                    self.gc_list[i] = GraphConvolution(nfeat, nhid)
                    self.residual_weight_list[i] = Parameter(torch.FloatTensor(nfeat, nhid))
                elif i == self.depth - 1:
                    self.gc_list[i] = GraphConvolution(nhid, nclass)
                    self.residual_weight_list[i] = Parameter(torch.FloatTensor(nhid, nclass))
                else:
                    self.gc_list[i] = GraphConvolution(nhid, nhid)
                    self.residual_weight_list[i] = Parameter(torch.FloatTensor(nhid, nhid))
        for i in range(self.depth):
            stdv = 1. / math.sqrt(self.residual_weight_list[i].size(1))
            self.residual_weight_list[i].data.uniform_(-stdv, stdv)
        self.dropout = dropout

    def myparameters(self):
        parameter_list = list(self.parameters())
        for i in range(self.depth):
            parameter_list += self.gc_list[i].parameters()
        parameter_list += self.residual_weight_list
        return parameter_list

    # ---- non residual ----
    def forward(self, raw_x, adj, eigen_adj=None):
        if self.residual_type == 'naive':
            return self.forward_naive(raw_x, adj)
        elif self.residual_type == 'raw':
            return self.forward_raw(raw_x, adj)
        elif self.residual_type == 'graph_naive':
            return self.forward_graph_naive(raw_x, adj)
        elif self.residual_type == 'graph_raw':
            return self.forward_graph_raw(raw_x, adj)

    #---- non residual ----
    def forward_none(self, raw_x, adj, eigen_adj=None):
        x = raw_x
        for i in range(self.depth-1):
            x = F.relu(self.gc_list[i](x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        y = self.gc_list[self.depth-1](x, adj)
        pred_y = F.log_softmax(y, dim=1)
        return pred_y

    #---- raw residual ----
    def forward_raw(self, raw_x, adj, eigen_adj=None):
        x = raw_x
        for i in range(self.depth-1):
            x = F.relu(self.gc_list[i](x, adj) + torch.mm(raw_x, self.residual_weight_list[0]))
            x = F.dropout(x, self.dropout, training=self.training)
        if self.depth == 1:
            y = self.gc_list[self.depth - 1](x, adj) + torch.mm(raw_x, self.residual_weight_list[0])
        else:
            y = self.gc_list[self.depth-1](x, adj) + torch.mm(torch.mm(raw_x, self.residual_weight_list[0]), self.residual_weight_list[self.depth-1])
        pred_y = F.log_softmax(y, dim=1)
        return pred_y

    #---- naive residual ----
    def forward_naive(self, raw_x, adj, eigen_adj=None):
        x = raw_x
        for i in range(self.depth-1):
            x = F.relu(self.gc_list[i](x, adj) + torch.mm(x, self.residual_weight_list[i]))
            x = F.dropout(x, self.dropout, training=self.training)
        y = self.gc_list[self.depth-1](x, adj) + torch.mm(x, self.residual_weight_list[self.depth-1])
        pred_y = F.log_softmax(y, dim=1)
        return pred_y

    #---- graph raw residual ----
    def forward_graph_raw(self, raw_x, adj, eigen_adj=None):
        x = raw_x
        for i in range(self.depth-1):
            x = F.relu(self.gc_list[i](x, adj) + torch.spmm(adj, torch.mm(raw_x, self.residual_weight_list[0])))
            x = F.dropout(x, self.dropout, training=self.training)
        if self.depth == 1:
            y = self.gc_list[self.depth - 1](x, adj) + torch.spmm(adj, torch.mm(raw_x, self.residual_weight_list[0]))
        else:
            y = self.gc_list[self.depth-1](x, adj) + torch.spmm(adj, torch.mm(torch.mm(raw_x, self.residual_weight_list[0]), self.residual_weight_list[self.depth-1]))
        pred_y = F.log_softmax(y, dim=1)
        return pred_y

    #---- graph naive residual ----
    def forward_graph_naive(self, raw_x, adj, eigen_adj=None):
        x = raw_x
        for i in range(self.depth-1):
            x = F.relu(self.gc_list[i](x, adj) + torch.spmm(adj, torch.mm(x, self.residual_weight_list[i])))
            x = F.dropout(x, self.dropout, training=self.training)
        y = self.gc_list[self.depth-1](x, adj) + torch.spmm(adj, torch.mm(x, self.residual_weight_list[self.depth-1]))
        pred_y = F.log_softmax(y, dim=1)
        return pred_y

    def train_model(self, epoch_iter):
        t_begin = time.time()
        optimizer = optim.Adam(self.myparameters(), lr=self.lr, weight_decay=self.weight_decay)
        accuracy = EvaluateAcc('', '')
        for epoch in range(epoch_iter):
            #self.myparameters()
            t_epoch_begin = time.time()
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data['X'], self.data['A'], self.data['B'])
            loss_train = F.nll_loss(output[self.data['idx_train']], self.data['y'][self.data['idx_train']])
            accuracy.data = {'true_y': self.data['y'][self.data['idx_train']], 'pred_y': output[self.data['idx_train']].max(1)[1]}
            acc_train = accuracy.evaluate()
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.data['X'], self.data['A'], self.data['B'])

            loss_val = F.nll_loss(output[self.data['idx_val']], self.data['y'][self.data['idx_val']])
            accuracy.data = {'true_y': self.data['y'][self.data['idx_val']], 'pred_y': output[self.data['idx_val']].max(1)[1]}
            acc_val = accuracy.evaluate()

            loss_test = F.nll_loss(output[self.data['idx_test']], self.data['y'][self.data['idx_test']])
            accuracy.data = {'true_y': self.data['y'][self.data['idx_test']],
                             'pred_y': output[self.data['idx_test']].max(1)[1]}
            acc_test = accuracy.evaluate()

            self.learning_record_dict[epoch] = {'loss_train': loss_train.item(), 'acc_train': acc_train.item(),
                                                'loss_val': loss_val.item(), 'acc_val': acc_val.item(),
                                                'loss_test': loss_test.item(), 'acc_test': acc_test.item(),
                                                'time': time.time() - t_epoch_begin}

            if epoch % 50 == 0:
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

    def test_model(self):
        self.eval()
        accuracy = EvaluateAcc()
        output = self.forward(self.data['X'], self.data['A'], self.data['B'])

        loss_test = F.nll_loss(output[self.data['idx_test']], self.data['y'][self.data['idx_test']])
        accuracy.data = {'true_y': self.data['y'][self.data['idx_test']], 'pred_y': output[self.data['idx_test']].max(1)[1]}
        acc_test = accuracy.evaluate()

        loss_train = F.nll_loss(output[self.data['idx_train']], self.data['y'][self.data['idx_train']])
        accuracy.data = {'true_y': self.data['y'][self.data['idx_train']], 'pred_y': output[self.data['idx_train']].max(1)[1]}
        acc_train = accuracy.evaluate()

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        return {'stat': {'test': {'loss': loss_test.item(), 'acc': acc_test}, 'train': {'loss': loss_train.item(), 'acc': acc_train}},
                'true_y': self.data['y'][self.data['idx_test']], 'pred_y': output[self.data['idx_test']].max(1)[1]}, acc_test.item()

    def run(self):
        time_cost = self.train_model(self.epoch)
        result, test_acc = self.test_model()
        result['stat']['time_cost'] = time_cost
        result['learning_record'] = self.learning_record_dict
        return result, test_acc



