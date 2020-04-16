import os.path
import numpy as np
import json
import itertools
import torch
from torch_geometric.data.data import Data

def load_data(filename, transform=None):
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

    edge_list = list(itertools.chain(*[[(i, nb) for nb in nbs]
                    for i,nbs in enumerate(raw['links'])]))
    src, dst = tuple(zip(*edge_list))
    edges = torch.LongTensor([src, dst])
    data = Data(x=features, edge_index=edges, y=labels)

    data.train_masks = train_masks
    data.val_masks = val_masks
    data.stopping_masks = stopping_masks
    data.test_mask = test_mask

    if transform is not None:
        data = transform(data)

    return data
