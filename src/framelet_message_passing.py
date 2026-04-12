import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg

import torch
import torch.nn as nn
from torch.nn import Dropout, ELU, Linear, Sequential as Seq

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import get_laplacian


def scipy_to_torch_sparse(matrix):
    matrix = sparse.coo_matrix(matrix)
    row = torch.tensor(matrix.row)
    col = torch.tensor(matrix.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(matrix.data)
    return torch.sparse_coo_tensor(index, value, matrix.shape)


def chebyshev_approx(function, degree):
    quad_points = 500
    coeffs = np.zeros(degree)
    scale = np.pi / 2
    x = np.linspace(0, np.pi, quad_points)
    for k in range(1, degree + 1):
        y = np.cos((k - 1) * x) * function(scale * (np.cos(x) + 1))
        coeffs[k - 1] = 2 / np.pi * np.trapz(y, x)
    return coeffs


def get_operator(laplacian, decomposition_filters, degree, scale, start_level, levels):
    filter_coeffs = [chebyshev_approx(filter_fn, degree) for filter_fn in decomposition_filters]
    domain_scale = np.pi / 2
    current = sparse.identity(laplacian.shape[0])
    operators = {}
    for level in range(1, levels + 1):
        for filter_index, coeffs in enumerate(filter_coeffs):
            t0 = current
            t1 = ((scale ** (-start_level + level - 1) / domain_scale) * laplacian) @ t0 - t0
            operators[filter_index, level - 1] = (0.5 * coeffs[0]) * t0 + coeffs[1] * t1
            for k in range(2, degree):
                tk = ((2 / domain_scale * scale ** (-start_level + level - 1)) * laplacian) @ t1 - 2 * t1 - t0
                t0 = t1
                t1 = tk
                operators[filter_index, level - 1] += coeffs[k] * tk
        current = operators[0, level - 1]
    return operators


def _get_framelet_filters(frame_type):
    if frame_type == 'Haar':
        d1 = lambda x: np.cos(x / 2)
        d2 = lambda x: np.sin(x / 2)
        return [d1, d2]
    if frame_type == 'Linear':
        d1 = lambda x: np.square(np.cos(x / 2))
        d2 = lambda x: np.sin(x) / np.sqrt(2)
        d3 = lambda x: np.square(np.sin(x / 2))
        return [d1, d2, d3]
    if frame_type == 'Quadratic':
        d1 = lambda x: np.cos(x / 2) ** 3
        d2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
        d3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
        d4 = lambda x: np.sin(x / 2) ** 3
        return [d1, d2, d3, d4]
    raise ValueError(f'Invalid frame_type: {frame_type}. Valid options are: Haar, Linear, Quadratic')


def framelets(data, frame_type='Haar', levels=2, scale=2, degree=2):
    num_nodes = data.num_nodes
    edge_index, edge_weight = get_laplacian(data.edge_index, num_nodes=num_nodes, normalization='sym')
    laplacian = sparse.coo_matrix(
        (edge_weight.numpy(), (edge_index[0, :].numpy(), edge_index[1, :].numpy())),
        shape=(num_nodes, num_nodes),
    )

    lobpcg_init = np.random.rand(num_nodes, 1)
    lambda_max, _ = lobpcg(laplacian, lobpcg_init, maxiter=50)
    start_level = np.log(lambda_max[0] / np.pi) / np.log(scale) + levels - 1
    decomposition_filters = _get_framelet_filters(frame_type)

    operators = get_operator(laplacian, decomposition_filters, degree, scale, start_level, levels)
    return [
        scipy_to_torch_sparse(operators[filter_index, level])
        for level in range(levels)
        for filter_index in range(len(decomposition_filters))
    ]


class UFGLevel(MessagePassing):
    def __init__(self, in_channels, out_channels, init_scale=None, dropout_prob=0.5, atten=False, if_filter=True,
                 channel_mix=False):
        super().__init__(aggr='add')

        self.atten = atten
        self.dropout_prob = dropout_prob
        self.init_scale = init_scale
        self.channel_mix = channel_mix
        if init_scale:
            self.filter = nn.Parameter(torch.Tensor(1, in_channels))
            nn.init.normal_(self.filter, mean=init_scale, std=0.1)

        self.linear = Linear(in_channels, out_channels)
        self.mlp = Seq(
            ELU(),
            Dropout(dropout_prob),
            Linear(in_channels, out_channels),
        )
        nn.init.xavier_normal_(self.mlp[2].weight)

    def forward(self, x, edge_index, edge_attr, edge_index_o=None):
        if self.channel_mix:
            x = self.linear(x)
        if self.init_scale:
            return self.propagate(edge_index, x=x, edge_attr=edge_attr) * self.filter
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return edge_attr.view(-1, 1) * x_j


__all__ = ['UFGLevel', 'chebyshev_approx', 'framelets', 'get_operator', 'scipy_to_torch_sparse']
