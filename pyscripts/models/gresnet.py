import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
import dgl.function as fn


class GResConv(nn.Module):
    def __init__(self,
                 g,
                 graph_res,
                 raw_res,
                 activation,
                 base_conv):
        super(GResConv, self).__init__()
        self.g = g
        self.graph_res = graph_res
        self.raw_res = raw_res
        self.conv = base_conv
        self.activation=activation

    def forward(self, prev, raw):
        res = raw if self.raw_res else prev
        if self.graph_res:
            norm = torch.pow(self.g.in_degrees().float().clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (res.dim() - 1)
            norm = torch.reshape(norm, shp).to(res.device)
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


class GCN_GResNet(nn.Module):
    def __init__(self,
                 graph,
                 graph_res,
                 raw_res,
                 n_layers,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 dropout):
        super(GCN_GResNet, self).__init__()
        self.g = graph
        self.dropout = nn.Dropout(p=dropout)
        self.input_conv = GraphConv(in_dim, hidden_dim, activation=F.relu)

        self.gres_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.gres_layers.append(GResConv(graph, graph_res, raw_res, F.relu,
                                            GraphConv(hidden_dim, hidden_dim)))
        self.output_conv = GraphConv(hidden_dim, out_dim)
        self.raw_proj = nn.Linear(in_dim, hidden_dim)


    def forward(self, features):
        raw = self.raw_proj(features)
        h = features
        h = self.input_conv(self.g, h)
        for layer in self.gres_layers:
            h = self.dropout(h)
            h = layer(h,raw)
        h = self.dropout(h)
        return self.output_conv(self.g, h)


class FlattenedConv(nn.Module):
    def __init__(self, conv):
        super(FlattenedConv, self).__init__()
        self.conv = conv

    def forward(g, h):
        return self.conv(g, h).flatten(1)


class GAT_GResNet(nn.Module):
    def __init__(self,
                 graph,
                 graph_res,
                 raw_res,
                 n_layers,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 dropout):
        super(GAT_GResNet, self).__init__()
        num_heads = 5
        negative_slope = 0.2
        self.g = graph
        self.input_conv = GATConv(in_dim, hidden_dim, num_heads,
                                    dropout, dropout, negative_slope, False, F.relu)
        self.gres_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            flattened_conv = FlattenedConv(
                GATConv(num_heads*hidden_dim,
                        hidden_dim,
                        num_heads,dropout, dropout,
                        negative_slope,
                        residual=False,
                        activation=None),
            )
            self.gres_layers.append(GResConv(graph, graph_res, raw_res, F.relu,
                                             flattened_conv))
        self.output_conv = GATConv(num_heads*hidden_dim, out_dim, 1,dropout,
                                    dropout,negative_slope,
                                    residual=False,
                                    activation=None)
        self.raw_proj = nn.Linear(in_dim, num_heads*hidden_dim)


    def forward(self, features):
        raw = self.raw_proj(features)
        h = features
        h = self.input_conv(self.g, h).flatten(1)
        for layer in self.gres_layers:
            h = layer(h,raw).flatten(1)
        return self.output_conv(self.g, h).mean(1)
