import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, Dropout, ReLU, BatchNorm1d
# from torch_geometric.nn.norm import PairNorm  # optional if you want PairNorm

from torch_geometric.utils import get_laplacian
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import MessagePassing, GATConv
from torch_geometric.datasets import Planetoid, WikipediaNetwork

import functools
import math
import os.path as osp

###############################################################################
#                            FRAMELET & CHEBYSHEV UTILS                       #
###############################################################################
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)  # ensure COO format
    row = torch.tensor(A.row, dtype=torch.long)
    col = torch.tensor(A.col, dtype=torch.long)
    val = torch.tensor(A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(
        indices=torch.stack((row, col), dim=0),
        values=val,
        size=A.shape
    )


def ChebyshevApprox(f, n, quad_points=500):
    c = np.zeros(n)
    x = np.linspace(0, np.pi, quad_points)
    for k in range(1, n + 1):
        integrand = np.cos((k - 1) * x) * f((np.pi / 2) * (np.cos(x) + 1))
        c[k - 1] = (2.0 / np.pi) * np.trapz(integrand, x)
    return c


def get_framelet_operator(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    coeff_list = []
    for j in range(r):
        coeff_list.append(ChebyshevApprox(DFilters[j], n))

    FD1 = sparse.identity(L.shape[0], dtype=np.float32, format='csr')
    d = dict()
    a = np.pi / 2

    for l in range(1, Lev + 1):
        for j in range(r):
            c_j = coeff_list[j]
            T0F = FD1
            T1F = ((s**(-J + l - 1) / a) * L).dot(T0F) - T0F

            Mtemp = (0.5 * c_j[0]) * T0F
            if n > 1:
                Mtemp += c_j[1] * T1F

            prev = T1F
            prev2 = T0F
            for k in range(2, n):
                TkF = ((2.0 / a) * (s**(-J + l - 1) * L)).dot(prev) - 2*prev - prev2
                Mtemp += c_j[k] * TkF
                prev2 = prev
                prev = TkF

            d[(j, l - 1)] = Mtemp
        FD1 = d[(0, l - 1)]
    return d


def build_framelet_ops(L, DFilters, n, s, Lev, device, threshold=1e-4):
    num_nodes = L.shape[0]
    lobpcg_init = np.random.rand(num_nodes, 1).astype(np.float32)
    lambda_max, _ = lobpcg(L, lobpcg_init, maxiter=50)
    lambda_max = float(lambda_max[0])

    J = math.log(lambda_max / np.pi, s) + Lev - 1
    d = get_framelet_operator(L, DFilters, n, s, J, Lev)

    d_list = []
    for l in range(Lev):
        for j in range(len(DFilters)):
            M = d[(j, l)]
            # optionally threshold small entries
            M.data[np.abs(M.data)<threshold] = 0.
            M.eliminate_zeros()
            d_sp = scipy_to_torch_sparse(M).coalesce().to(device)
            d_list.append(d_sp)
    return d_list


###############################################################################
#                      GRAPH REWIRING (OPTIONAL FOR HETEROPHILY)              #
###############################################################################
def rewire_graph(features, edge_index, threshold=0.2, add_similar=False, similar_K=5):
  """
  Simple trick: prune edges connecting nodes with a large feature distance.
  Optionally add edges for pairs that are very close in feature space.
  
  Args:
    features: (N, d) node features (torch.FloatTensor)
    edge_index: (2, E) existing edges
    threshold: float, distance threshold for pruning or adding
    add_similar: bool, if True, also add edges between highly similar nodes
    similar_K: int, for each node, add up to similar_K of its most similar neighbours
  Returns:
    new_edge_index: a possibly rewired adjacency
  """
  with torch.no_grad():
    x = features.cpu()
    row, col = edge_index[0].cpu(), edge_index[1].cpu()
    # Compute distances for existing edges
    dist = torch.norm(x[row] - x[col], dim=1)
    keep_mask = (dist < threshold)
    pruned_row = row[keep_mask]
    pruned_col = col[keep_mask]

    new_row = pruned_row.tolist()
    new_col = pruned_col.tolist()

    if add_similar:
      # For small graphs, we can compute full pairwise distances.
      # For larger graphs, consider approximate kNN.
      dists = torch.cdist(x, x, p=2)
      # For each node, add edges to its similar_K nearest neighbors (excluding itself)
      for i in range(x.size(0)):
        # Get the k smallest distances (ignoring self)
        dists_i = dists[i]
        sorted_vals, sorted_idx = torch.sort(dists_i)
        # Skip the self-distance (zero)
        for j in sorted_idx[1:similar_K+1]:
          if dists_i[j] < threshold:
            new_row.append(i)
            new_col.append(j.item())
    new_edge_index = torch.tensor([new_row, new_col], dtype=torch.long)
  return new_edge_index


###############################################################################
#                         FRAMELET CONV LAYER (UFGLevel)                      #
###############################################################################
class UFGLevel(MessagePassing):
    def __init__(self, in_channels, out_channels,
                 init_scale=1.0, dropout_prob=0.5,
                 atten=False, if_filter=False):
        super().__init__(aggr='add')
        self.atten = atten
        self.if_filter = if_filter
        self.filter = nn.Parameter(torch.Tensor(1, in_channels))
        nn.init.uniform_(self.filter, init_scale-0.2, init_scale+0.2)

        self.conv = GATConv(in_channels, out_channels, heads=1, dropout=dropout_prob)

    def forward(self, x, edge_index, edge_attr,
                edge_index_o=None):
        # If using GAT, we pass x+edge_index_o
        if self.atten and (edge_index_o is not None):
            x_atten = self.conv(x, edge_index_o)
            out = self.propagate(edge_index, x=x_atten, edge_attr=edge_attr)
        else:
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        if self.if_filter:
            out = out * self.filter
        return out

    def message(self, x_j, edge_attr):
        return edge_attr.view(-1,1) * x_j


###############################################################################
#                          FRAMELET-BASED GNN MODEL                           #
###############################################################################
class FrameletNet(nn.Module):
    """
    Tricks added:
      - A BatchNorm after combining partial results
      - Possibly more sub-bands (e.g., 'Linear' with 3 filters)
      - use_two_branch with deeper local MLP
    """
    def __init__(self,
                 num_features, hidden_dim, num_classes,
                 d_list,
                 dropout_prob=0.5,
                 levelMixer='sum',
                 alpha_init=0.4,
                 use_two_branch=True):
        super().__init__()
        self.levelMixer = levelMixer
        self.use_two_branch = use_two_branch

        # store wavelet ops
        self.edge_index_list, self.edge_attr_list = [], []
        for i, sp_op in enumerate(d_list):
            ei = sp_op.indices()
            ea = sp_op.values()
            self.register_buffer(f'edge_index_{i}', ei)
            self.register_buffer(f'edge_attr_{i}', ea)
            self.edge_index_list.append(ei)
            self.edge_attr_list.append(ea)

        # build UFG layers
        self.ufg_list = nn.ModuleList([
            UFGLevel(hidden_dim, hidden_dim, init_scale=1.0,
                     dropout_prob=dropout_prob, atten=False, if_filter=False)
            for _ in range(len(d_list))
        ])

        self.alpha_train = nn.Parameter(torch.tensor(alpha_init))

        self.linear_in = nn.Linear(num_features, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.bn_mid = BatchNorm1d(hidden_dim)  # batchnorm after partial merges
        self.linear_out = nn.Linear(hidden_dim, num_classes)

        # aggregator if levelMixer='mlp'
        self.mlp_agg = Seq(
            ReLU(),
            Dropout(dropout_prob),
            nn.Linear(len(d_list)*hidden_dim, hidden_dim),
            ReLU(),
            BatchNorm1d(hidden_dim)
        )

        # optional second branch for node-level MLP
        if self.use_two_branch:
            self.local_mlp = nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_prob)
            )
            self.merge_linear = nn.Linear(2*hidden_dim, num_classes)

        # optional PairNorm if you want
        # self.pairnorm = PairNorm()

    def forward(self, x):
        x_in = self.linear_in(x)

        # self.pairnorm(x_in) # if using PairNorm
        if self.levelMixer == 'sum':
            partials = []
            for i, (op_idx, op_attr, ufg) in enumerate(zip(self.edge_index_list,
                                                           self.edge_attr_list,
                                                           self.ufg_list)):
                part = ufg(x_in, op_idx, op_attr)
                partials.append(part)
            sum_part = functools.reduce(lambda a,b: a+b, partials)
            x_frame = x_in + torch.sigmoid(self.alpha_train)*sum_part
            # BN
            x_frame = self.bn_mid(x_frame)

        elif self.levelMixer == 'mlp':
            cat_list = []
            for i, (op_idx, op_attr, ufg) in enumerate(zip(self.edge_index_list,
                                                           self.edge_attr_list,
                                                           self.ufg_list)):
                cat_list.append(ufg(x_in, op_idx, op_attr))
            cat_out = torch.cat(cat_list, dim=1)
            x_frame = self.mlp_agg(cat_out)

        else:
            raise ValueError("Invalid levelMixer choice")

        if self.use_two_branch:
            x_local = self.local_mlp(x)  # deeper local MLP
            x_all = torch.cat([x_frame, x_local], dim=1)
            out = self.merge_linear(x_all)
        else:
            out = self.linear_out(x_frame)

        return F.log_softmax(out, dim=1)


###############################################################################
#                              TRAINING / DEMO                                #
###############################################################################
def run_demo(dataset_name='chameleon',
             levelMixer='mlp',
             Lev=4,         # more levels
             poly_degree=5, # bigger polynomial
             hidden_dim=64,
             dropout=0.6,
             lr=0.005,
             weight_decay=1e-3,
             epochs=500,
             use_two_branch=True,
             alpha_init=0.4,
             do_rewire=True,
             rewire_threshold=0.3,
             device='cuda'):

    ds_lower = dataset_name.lower()
    path = f'./data/{ds_lower}'

    if ds_lower in ['cora','citeseer','pubmed']:
        dataset = Planetoid(root=path, name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0]
    elif ds_lower in ['chameleon','squirrel']:
        dataset = WikipediaNetwork(root=path, name=ds_lower, transform=NormalizeFeatures())
        data = dataset[0]
        data.train_mask = data.train_mask[:, 1].bool()
        data.val_mask   = data.val_mask[:, 1].bool()
        data.test_mask  = data.test_mask[:, 1].bool()
    else:
        raise ValueError(f"Unsupported dataset_name {dataset_name}. Use Planetoid or chameleon/squirrel")

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Optional rewiring for heterophily
    if do_rewire:
        # prune edges with feature distance > threshold
        rewired_edge_index = rewire_graph(data.x, data.edge_index, threshold=rewire_threshold)
        data.edge_index = rewired_edge_index.to(device)

    num_nodes = data.num_nodes
    L_idx_val = get_laplacian(data.edge_index, num_nodes=num_nodes, normalization='sym')
    L_scipy = sparse.coo_matrix(
        (L_idx_val[1].cpu().numpy(),
         (L_idx_val[0][0,:].cpu().numpy(), L_idx_val[0][1,:].cpu().numpy())),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    )

    # Example "Linear" wavelets with 3 bands (for more sub-bands).
    # If you prefer Haar, comment out below lines and restore Haar
    D1 = lambda x: np.cos(x/2.)**2           # ~low
    D2 = lambda x: np.sqrt(2)*np.sin(x)/2.   # ~band
    D3 = lambda x: np.sin(x/2.)**2           # ~high
    DFilters = [D1, D2, D3]

    # If you want classic Haar, do:
    # D1 = lambda x: np.cos(x/2.)
    # D2 = lambda x: np.sin(x/2.)
    # DFilters = [D1, D2]

    d_list = build_framelet_ops(L_scipy, DFilters, n=poly_degree, s=2,
                                Lev=Lev, device=device, threshold=1e-5)

    model = FrameletNet(num_features=dataset.num_features,
                        hidden_dim=hidden_dim,
                        num_classes=dataset.num_classes,
                        d_list=d_list,
                        dropout_prob=dropout,
                        levelMixer=levelMixer,
                        alpha_init=alpha_init,
                        use_two_branch=use_two_branch).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = 0.
    best_test = 0.
    patience_count = 0
    max_patience = 80  # early stopping patience
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(data.x)

        def accuracy(mask):
            pred = out_eval[mask].max(dim=1)[1]
            return (pred == data.y[mask]).sum().item() / mask.sum().item()

        train_acc = accuracy(data.train_mask)
        val_acc   = accuracy(data.val_mask)
        test_acc  = accuracy(data.test_mask)

        # simple early stopping logic
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_model_state = model.state_dict()
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} "
                  f"| TrainAcc {train_acc:.2f} | ValAcc {val_acc:.2f} "
                  f"| TestAcc {test_acc:.2f} | Patience {patience_count}")

        if patience_count > max_patience:
            print("Early stopping triggered.")
            break

    # load best model if found
    if best_model_state:
        model.load_state_dict(best_model_state)

    print(f"\n==> {dataset_name} final best val acc = {best_val:.4f}, test acc = {best_test:.4f}\n")


if __name__ == "__main__":
    # Example usage on chameleon with additional tricks
    run_demo(dataset_name='chameleon',
             levelMixer='mlp',
             Lev=4,
             poly_degree=3,
             hidden_dim=80,
             dropout=0.6,
             lr=0.005,
             weight_decay=1e-2,
             epochs=500,
             use_two_branch=True,
             alpha_init=0.8,
             do_rewire=True,
             rewire_threshold=0.2,
             device='cuda')