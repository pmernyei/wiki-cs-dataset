'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv
import random

class DatasetLoader(dataset):
    c = 0.25

    data = None
    batch_size = None
    
    dataset_source_folder_path = None
    dataset_source_file_name = None
    method_type = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def load(self):
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(dataset))

        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        one_hot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path),
                                        dtype=np.int32)
        print(edges_unordered.shape)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(one_hot_labels.shape[0], one_hot_labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = self.normalize(features)
        #print(features, features.shape)

        raw_adj = adj
        #row_normlized_adj = self.normalize(adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))
        #eigen_adj = sp.coo_matrix(self.c*inv((sp.eye(adj.shape[0]) - (1-self.c)*adj).toarray()))

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(one_hot_labels)[1])
        if self.method_type == None or self.method_type == 'GCN':
            adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)
        elif self.method_type == 'GAT':
            adj = torch.FloatTensor(np.array(norm_adj.todense()))
        elif self.method_type == 'LoopyNet':
            adj = self.sparse_mx_to_torch_sparse_tensor(raw_adj)
        #eigen_adj = self.sparse_mx_to_torch_sparse_tensor(eigen_adj)
        for i in range(len(one_hot_labels)):
            if i not in idx_train:
                one_hot_labels[i] = [0.0] * len(one_hot_labels[i])
        one_hot_labels = torch.FloatTensor(one_hot_labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return {'X': features, 'A': adj, 'B': None, 'norm_adj': self.sparse_mx_to_torch_sparse_tensor(norm_adj), 'y': labels, 'one_hod_y': one_hot_labels, 'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
