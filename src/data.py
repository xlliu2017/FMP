"""
Code partially copied from 'Diffusion Improves Graph Learning' repo https://github.com/klicperajo/gdc/blob/master/data.py
"""

import os
import numpy as np
import copy

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected

# DGL imports for Chameleon
import dgl
from dgl.data import ChameleonDataset

from heterophilic import generate_random_splits
from utils import ROOT_DIR
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor

DATA_PATH = f'{ROOT_DIR}/data'


def choose1mask(data, i):
    data1 = copy.deepcopy(data)
    data1.train_mask = data1.train_mask[:, i]
    data1.val_mask   = data1.val_mask[:, i]
    data1.test_mask  = data1.test_mask[:, i]
    return data1


def get_dataset(opt: dict, data_dir, use_lcc: bool = False) -> InMemoryDataset:
    ds = opt['dataset']
    path = os.path.join(data_dir, ds)
    if ds in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(path, ds)
    elif ds in ['Computers', 'Photo']:
        dataset = Amazon(path, ds)
    elif ds == 'CoauthorCS':
        dataset = Coauthor(path, 'CS')
    elif ds in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root=path, name=ds, transform=T.NormalizeFeatures())
    elif ds in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=path, name=ds, transform=T.NormalizeFeatures())
    elif ds == 'film':
        dataset = Actor(root=path, transform=T.NormalizeFeatures())
    elif ds == 'ogbn-arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=ds, root=path, transform=T.ToSparseTensor())
        use_lcc = False  # never need LCC for ogb
    else:
        raise Exception('Unknown dataset.')

    # Possibly use largest connected component
    if use_lcc:
        print('use_lcc')
        lcc = get_largest_connected_component(dataset)
        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]
        row, col = dataset.data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))

        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        dataset.data = data

    # If you have a rewiring scheme, place it here:
    if opt.get('rewiring') is not None:
        dataset.data = rewire(dataset.data, opt, data_dir)

    # Check if the dataset already has train/val/test splits
    train_mask_exists = True
    try:
        dataset.data.train_mask
    except AttributeError:
        train_mask_exists = False

    # OGB special handling
    if ds == 'ogbn-arxiv':
        split_idx = dataset.get_idx_split()
        ei = to_undirected(dataset.data.edge_index)
        data = Data(
            x=dataset.data.x,
            edge_index=ei,
            y=dataset.data.y,
            train_mask=split_idx['train'],
            test_mask=split_idx['test'],
            val_mask=split_idx['valid']
        )
        dataset.data = data
        train_mask_exists = True

    # If we still need train/val/test splits:
    if (use_lcc or not train_mask_exists) and not opt['geom_gcn_splits']:
        dataset.data = set_train_val_test_split(
            seed=12345,
            data=dataset.data,
            dataset_name=ds
        )

    return dataset


def rewire(data, opt, data_dir):
    """Replace with your own rewiring logic if needed."""
    rw = opt['rewiring']
    if rw == 'two_hop':
        data = get_two_hop(data)
    elif rw == 'gdc':
        data = apply_gdc(data, opt)
    elif rw == 'pos_enc_knn':
        data = apply_pos_dist_rewire(data, opt, data_dir)
    return data


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = {start}
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.add(current_node)
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes.difference_update(comp)
    # Return the largest
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def set_train_val_test_split(
        seed: int,
        data: Data,
        dataset_name: str,
        num_development: int = 1500,
        num_per_class: int = 20
    ) -> Data:
    """
    If 'dataset_name' is 'chameleon', pull train/val/test masks directly
    from dgl.data.ChameleonDataset (which has 10 splits).
    Otherwise, use the old random-split logic.
    """
    if dataset_name == 'chameleon':
        print("Using DGL's default train/val/test splits for Chameleon.")
        # Load from DGL
        dgl_ds = ChameleonDataset()        # This automatically downloads (if needed)
        dgl_g = dgl_ds[0]                 # DGLGraph
        # Copy DGL’s masks (shape: [2277, 10]) into our PyG 'data'
        data.train_mask = dgl_g.ndata['train_mask']
        data.val_mask   = dgl_g.ndata['val_mask']
        data.test_mask  = dgl_g.ndata['test_mask']
        # Optionally, ensure the labels and features match exactly:
        data.y          = dgl_g.ndata['label']
        data.x          = dgl_g.ndata['feat']
        # Also replicate edges in PyG format
        src, dst = dgl_g.edges()
        data.edge_index = torch.stack([src, dst], dim=0)
        return data

    # For all other datasets, do the original random-split logic
    print(f"Using random splits for dataset {dataset_name}.")
    rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    for c in range(data.y.max() + 1):
        class_mask = (data.y[development_idx].cpu() == c).numpy()
        class_idx = development_idx[np.where(class_mask)[0]]
        # If not enough samples from this class to pick from, skip
        if len(class_idx) >= num_per_class:
            chosen = rnd_state.choice(class_idx, num_per_class, replace=False)
            train_idx.extend(chosen)

    val_idx = [i for i in development_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask   = get_mask(val_idx)
    data.test_mask  = get_mask(test_idx)

    return data