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
                 in_dim,
                 out_dim,
                 activation,
                 base_conv):
        super(GResConv, self).__init__()
        self.g = g
        self.graph_res = graph_res
        self.raw_res = raw_res
        self.conv = base_conv
        self.activation=activation
        if not raw_res:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, prev, raw):
        res = raw if self.raw_res else self.res_proj(prev)
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
        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        self.layers.append(GResConv(graph, graph_res, raw_res, in_dim, hidden_dim,
                                    F.relu, GraphConv(in_dim, hidden_dim)))
        for i in range(n_layers - 1):
            self.layers.append(GResConv(graph, graph_res, raw_res, hidden_dim, hidden_dim,
                                        F.relu, GraphConv(hidden_dim, hidden_dim)))

        self.output_conv = GResConv(graph, graph_res, raw_res, hidden_dim, out_dim,
                                    None, GraphConv(hidden_dim, out_dim))

        self.raw_in_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.raw_out_proj = nn.Linear(hidden_dim, out_dim, bias=False)


    def forward(self, features):
        raw = self.raw_in_proj(features)
        h = features
        for layer in self.layers:
            h = layer(h, raw)
            h = self.dropout(h)
        return self.output_conv(h, self.raw_out_proj(raw))


class FlattenedConv(nn.Module):
    def __init__(self, conv):
        super(FlattenedConv, self).__init__()
        self.conv = conv

    def forward(self, g, h):
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
        num_heads = 8
        negative_slope = 0.2
        self.g = graph

        def create_layer(inp, out, heads, act):
            conv = FlattenedConv(GATConv(inp, out, heads,
                            dropout, dropout, negative_slope, False, None))
            return GResConv(graph, graph_res, raw_res, inp, out*heads, act, conv)

        self.layers = nn.ModuleList()
        self.layers.append(create_layer(in_dim, hidden_dim, num_heads, F.elu))
        for i in range(n_layers - 1):
            self.layers.append(
                create_layer(hidden_dim*num_heads, hidden_dim, num_heads, F.elu))

        self.output_conv = create_layer(hidden_dim*num_heads, out_dim, 1, None)

        self.raw_in_proj = nn.Linear(in_dim, hidden_dim*num_heads, bias=False)
        self.raw_out_proj = nn.Linear(hidden_dim*num_heads, out_dim, bias=False)


    def forward(self, features):
        raw = self.raw_in_proj(features)
        h = features
        for layer in self.layers:
            h = layer(h, raw)
        return self.output_conv(h, self.raw_out_proj(raw))
