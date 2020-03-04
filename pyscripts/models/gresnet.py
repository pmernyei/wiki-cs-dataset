import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv


class GResConv(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 out_dim,
                 graph_res,
                 raw_res,
                 activation,
                 base_conv):
        super(GResConv, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.graph_res = graph_res
        self.raw_res = raw_res
        self.conv = base_conv
        self.activation=activation

    def forward(self, prev, raw):
        res = raw if self.raw_res else prev
        if self.graph_res:
            norm = th.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (res.dim() - 1)
            norm = th.reshape(norm, shp).to(res.device)
            res = res * norm

            graph = self.g.local_var()
            graph.ndata['h'] = res
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            res = graph.ndata['h']
            res = res * norm
        next = self.conv(self.g, prev) + res
        if self.activation is not None:
            next = self.activation(next)
        return next


class GResNet(nn.Module):
    def __init__(self,
                 graph,
                 graph_res,
                 raw_res,
                 n_layers,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 dropout,
                 base_conv):
        super(GResNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hidden_dim, activation=F.relu))
        for i in range(n_layers - 1):
            self.layers.append(GResConv(graph, hidden_dim, hidden_dim,
                                        graph_res, raw_res, F.relu,
                                        GraphConv(hidden_dim,hidden_dim)))
        self.layers.append(GraphConv(hidden_dim, out_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.g = graph


        #elif base_conv == "gat":
        #    self.layers.append(GResConv(graph, in_dim, hidden_dim, graph_res,
        #                                raw_res, F.relu,
        #                                GATConv(in_dim, hidden_dim, 5,
        #                                        dropout, dropout,
        #                                        0.2, False, None)))
        #    for i in range(n_layers - 1):
        #        self.layers.append(GResConv(graph, hidden_dim, hidden_dim,
        #                                    graph_res, raw_res, F.relu,
        #                                    GATConv(5*hidden_dim, hidden_dim, 5,
        #                                            dropout, dropout,
        #                                            0.2, True, None)))
        #    self.layers.append(GResConv(graph, hidden_dim, out_dim, graph_res,
        #                                raw_res, F.relu,
        #                                GATConv(5*in_dim, out_dim, 1,
        #                                        dropout, dropout,
        #                                        0.2, False, None)))
        #    self.dropout = nn.Dropout(p=0)


    def forward(self, features):
        h = features
        h = self.layers[0](self.g, h)
        for i,layer in enumerate(self.layers[1:-1]):
            h = self.dropout(h)
            h = layer(h,features)
        h = self.dropout(h)
        h = self.layers[-1](self.g, h)
        return h
