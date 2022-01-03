import os.path
import numpy as np
import json
import itertools
import torch
from torch_geometric.data.data import Data

DATA_PATH = os.path.join('..', '..', 'dataset', 'data.json')

def load_data(filename=DATA_PATH, directed=False):
    raw = json.load(open(filename))
    features = torch.FloatTensor(np.array(raw['features']))
    labels = torch.LongTensor(np.array(raw['labels']))
    if hasattr(torch, 'BoolTensor'):
        train_masks = [torch.BoolTensor(tr) for tr in raw['train_masks']]
        val_masks = [torch.BoolTensor(val) for val in raw['val_masks']]
        stopping_masks = [torch.BoolTensor(st) for st in raw['stopping_masks']]
        test_mask = torch.BoolTensor(raw['test_mask'])
    else:
        train_masks = [torch.ByteTensor(tr) for tr in raw['train_masks']]
        val_masks = [torch.ByteTensor(val) for val in raw['val_masks']]
        stopping_masks = [torch.ByteTensor(st) for st in raw['stopping_masks']]
        test_mask = torch.ByteTensor(raw['test_mask'])

    if directed:
        edges = [[(i, j) for j in js] for i, js in enumerate(raw['links'])]
        edges = list(itertools.chain(*edges))
    else:
        edges = [[(i, j) for j in js] + [(j, i) for j in js]
                 for i, js in enumerate(raw['links'])]
        edges = list(set(itertools.chain(*edges)))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=features, edge_index=edge_index, y=labels)

    data.train_masks = train_masks
    data.val_masks = val_masks
    data.stopping_masks = stopping_masks
    data.test_mask = test_mask

    return data
