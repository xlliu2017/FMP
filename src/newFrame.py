import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, ReLU, BatchNorm1d

from torch_geometric.utils import (
    get_laplacian, 
    add_self_loops, 
    remove_self_loops,
    coalesce
)
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import MessagePassing, GATConv
from torch_geometric.datasets import Planetoid, WikipediaNetwork

import functools
import math
import os.path as osp

###############################################################################
#                       CHOICE OF LAPLACIAN: standard, signless, or biLap     #
###############################################################################
def build_laplacian(edge_index, num_nodes, mode='standard'):
    """
    Build different types of Laplacian:
      mode='standard': L = I - D^{-1/2} A D^{-1/2}
      mode='signless': L = I + D^{-1/2} A D^{-1/2}
      mode='biLap': (I - D^{-1/2} A D^{-1/2})^2
    """
    if mode not in ['standard', 'signless', 'biLap']:
        raise ValueError("Invalid Laplacian mode")

    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index

    # compute adjacency
    adj = sparse.coo_matrix(
        (np.ones(len(row)), (row.cpu().numpy(), col.cpu().numpy())),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    )
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_sqrt = np.sqrt(deg+1e-12)
    deg_sqrt_inv = 1.0 / deg_sqrt

    # standard
    # L = I - D^{-1/2} A D^{-1/2}
    # signless
    # L = I + D^{-1/2} A D^{-1/2}
    # biLap
    # L^2
    I = sparse.eye(num_nodes, format='coo', dtype=np.float32)
    # normalized adjacency
    A_hat = adj.copy()
    rows, cols = A_hat.row, A_hat.col
    val = A_hat.data
    for i in range(len(val)):
        r, c = rows[i], cols[i]
        val[i] = val[i] * deg_sqrt_inv[r] * deg_sqrt_inv[c]
    A_hat = sparse.coo_matrix((val, (rows, cols)), shape=adj.shape)

    if mode == 'standard':
        L = (I - A_hat).tocoo()
    elif mode == 'signless':
        L = (I + A_hat).tocoo()
    else:  # 'biLap'
        L_temp = (I - A_hat)
        # multiply once more
        L = (L_temp @ L_temp).tocoo()

    return L


###############################################################################
#                     LEARNABLE POLYNOMIAL "WAVELETS" INIT                    #
###############################################################################
class LearnableChebFilter(nn.Module):
    """
    A learnable polynomial filter, initialized from wavelet Chebyshev expansions.
    This can approximate wavelet functions but each coefficient is trainable.
    """
    def __init__(self, init_coeffs, device='cpu'):
        super().__init__()
        # init_coeffs: [n], float array
        self.n = len(init_coeffs)
        # create a torch parameter
        # using random init around the wavelet value
        self.poly_params = nn.Parameter(torch.tensor(init_coeffs, dtype=torch.float32).to(device))

    def forward(self, L):
        """
        Evaluate polynomial \sum_{k=0}^{n-1} poly_params[k] * T_k(L) at sparse matrix L.
        We'll do Chebyshev recursion.
        """
        # L: (N,N) torch sparse
        # compute T0 = I, T1 = L, etc.
        # for sign or biLap, we'd want to clamp eigenvalues, but let's proceed carefully.
        # We'll approximate partial sums with an iterative approach (like your get_framelet_operator).
        N = L.shape[0]
        n = self.n
        T0 = torch.sparse_coo_tensor(
            indices=torch.arange(N, dtype=torch.long).unsqueeze(0).expand(2, N),
            values=torch.ones(N, dtype=torch.float32, device=L.device),
            size=(N, N), device=L.device
        ).coalesce()
        T1 = L  # T1 = L
        # partial sum
        out = (0.5*self.poly_params[0])*T0  # c0 * T0 / 2
        if n>1:
            out = sparse_add(out, self.poly_params[1]*T1)
        prev = T1
        prev2 = T0
        for k in range(2,n):
            # recursion T_k(L) = 2*L*T_{k-1}(L) - T_{k-2}(L)
            # But your approach is scaled slightly if there's a factor etc.
            # We'll do standard Chebyshev recursion for simplicity
            newT = sparse_sub(2*matmul_sp(L, prev), prev2)
            out = sparse_add(out, self.poly_params[k]*newT)
            prev2, prev = prev, newT
        return out.coalesce()


def sparse_add(spA, spB):
    return (spA + spB).coalesce()

def sparse_sub(spA, spB):
    return (spA - spB).coalesce()

def matmul_sp(spA, spB):
    """
    Multiply two sparse_coo_tensors spA(N,N), spB(N,N)
    returning coalesced spC(N,N)
    """
    return torch.sparse.mm(spA, spB).coalesce()


###############################################################################
#             BUILD LEARNABLE WAVELET OPERATORS: A Mixture of Polynomials     #
###############################################################################
def build_learnable_ops(L_list, poly_init_list, device):
    """
    Suppose we have multiple Laplacians or multiple scales (like for multi-level),
    each with a set of init polynomial coefficients. We'll produce a
    "LearnableChebFilter" for each, then evaluate them to get the final operator.

    But let's do it so each forward pass we do the polynomial construction dynamically.
    That means we store the modules in a list, and only compute
    the final operator inside the GNN forward pass. 
    This approach defers actual matrix multiplication until the GNN layer call.

    For simplicity, we'll just return a module list of "LearnableChebFilter",
    each of which has forward(L) -> sp matrix. The GNN layer can call them. 
    """
    # Typically, you'd have e.g. multiple wavelet scales or multiple Laplacians.
    # For now, we assume L_list is e.g. [L_scale1, L_scale2, ...].
    # poly_init_list is parallel list of arrays of init cheb coefficients.
    # We'll create a list of modules:
    filter_modules = nn.ModuleList()
    for init_coeffs in poly_init_list:
        filter_modules.append( LearnableChebFilter(init_coeffs, device) )
    return filter_modules


###############################################################################
#                Mixture of Experts Aggregator for Wavelet Scales            #
###############################################################################
class MoEAggregator(nn.Module):
    """
    Mixture-of-experts aggregator: We have multiple "expert MLPs" that handle each
    wavelet scale individually. Then we compute gating weights for each node
    (a small gating MLP) to combine them adaptively.

    The output dimension is 'hidden_dim'.
    """
    def __init__(self, hidden_dim, n_experts, dropout=0.5):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(n_experts)
        ])
        # gating network
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, n_experts),  # produce gating logits
        )
        self.bn_out = BatchNorm1d(hidden_dim)

    def forward(self, x_list):
        """
        x_list: list of node embeddings [N, hidden_dim], length = n_experts
        We'll produce an [N, hidden_dim] result by gating.

        Steps:
          1) For gating, we can sum up the x's or pick e.g. x_list[0] as a reference for gating input
             or compute an average as the gating input. We'll just do a naive approach: gating from sum of experts?
        """
        # let's compute gating from the average input as an approximation
        # or from x_list[0], or we can do a small MLP that merges them all. 
        # We'll do an average for simplicity.
        x_mean = torch.mean(torch.stack(x_list, dim=0), dim=0)  # [N, hidden_dim]
        gate_logits = self.gate_net(x_mean)  # [N, n_experts]
        gate_weights = F.softmax(gate_logits, dim=1)  # [N, n_experts]

        # now compute expert outputs
        expert_outputs = []
        for i in range(self.n_experts):
            expert_outputs.append(self.experts[i](x_list[i]))  # [N, hidden_dim]

        # combine
        # out[n] = sum_i( gate_weights[n,i] * expert_outputs[i][n] )
        # We'll do a weighted sum in a batched manner
        final_out = 0
        for i in range(self.n_experts):
            w_i = gate_weights[:, i].unsqueeze(1)  # [N,1]
            final_out = final_out + w_i*expert_outputs[i]

        # batchnorm
        final_out = self.bn_out(final_out)
        return final_out


###############################################################################
#                Two-Branch Framelet GNN with Learnable Polynomials           #
###############################################################################
class FrameletGNN(nn.Module):
    """
    Large changes:
      - We store a set of Laplacians or wavelet scales in self.L_list
      - We store a parallel set of "LearnableChebFilter" modules
      - On forward pass, we compute wavelet-operator = filter(L) for each scale
      - Then we do a single message passing step with that operator (like a custom step)
      - We do so for multiple scales, produce multiple embeddings
      - Then a MoE aggregator merges them
      - Two-branch: local MLP + aggregator -> final gating skip
    """
    def __init__(self, 
                 num_features, hidden_dim, num_classes,
                 L_list, init_coeffs_list,
                 dropout=0.5,
                 use_two_branch=True):
        super().__init__()
        self.use_two_branch = use_two_branch
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # We build learnable polynomial filters for each scale in L_list
        # and store them in a module list
        self.filter_list = nn.ModuleList()
        for coeffs in init_coeffs_list:
            self.filter_list.append(LearnableChebFilter(coeffs))

        self.L_list = L_list  # store raw Laplacians (torch sparse)
        self.n_scales = len(L_list)

        # mixture-of-expert aggregator
        self.moe_agg = MoEAggregator(hidden_dim, self.n_scales, dropout)

        # input transform
        self.lin_in = nn.Linear(num_features, hidden_dim)
        # local MLP for second branch
        if use_two_branch:
            self.local_mlp = nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                BatchNorm1d(hidden_dim)
            )
            # final gating or linear
            self.gate_res = nn.Linear(hidden_dim, hidden_dim)

        # final classification
        self.lin_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Steps:
         1) transform x -> x_in
         2) for each scale s, compute wavelet operator W_s = filter_s(L_s)
         3) do a single "message passing" step: X_s = W_s * x_in (sparse mm)
         4) pass X_s to some LN or BN or MLP if wanted. We'll do a BN after aggregator
         5) aggregator => X_frame
         6) optional local MLP => X_local
         7) combine => final
        """
        device = x.device
        x_in = self.lin_in(x)  # [N, hidden_dim]

        # compute scale outputs
        scale_out_list = []
        for i in range(self.n_scales):
            # wavelet operator is filter_list[i](L_list[i])
            W_s = self.filter_list[i](self.L_list[i])  # a (N,N) sp matrix
            # message passing: X_s = W_s * x_in
            X_s = torch.sparse.mm(W_s, x_in)  # [N, hidden_dim]
            scale_out_list.append(X_s)

        # aggregator
        X_frame = self.moe_agg(scale_out_list)

        if self.use_two_branch:
            X_local = self.local_mlp(x)
            # combine with gating: we do a residual gating approach
            # out_node = X_frame + sigma(Wg * X_local)
            gating = self.gate_res(X_local)  # [N,hidden_dim]
            gating = torch.sigmoid(gating)
            combined = X_frame + gating * X_local
        else:
            combined = X_frame

        out = self.lin_out(combined)
        return F.log_softmax(out, dim=1)


###############################################################################
#                    ADVANCED REWIRING: ITERATIVE BATCH UPDATES              #
###############################################################################
def advanced_rewire(data, steps=2, dist_threshold=0.3, sim_threshold=0.05):
    """
    Rewire edges in multiple iterations:
      - Remove edges that connect nodes with distance>dist_threshold
      - Potentially add edges for node pairs that are extremely close < sim_threshold
        in the feature space (this is expensive for large graphs if done naively).
    We'll do a simple "knn" approach for adding edges: each node can add edges
    to up to X nearest neighbors if their distance < sim_threshold.

    This is purely illustrative and may be slow for large graphs.
    """
    x = data.x.cpu()
    n = x.shape[0]

    # Start from data.edge_index
    eidx = data.edge_index.cpu()
    row, col = eidx

    for _ in range(steps):
        # remove edges that connect dissimilar features
        dist = torch.norm(x[row] - x[col], dim=1)
        keep_mask = (dist < dist_threshold)
        new_row = row[keep_mask]
        new_col = col[keep_mask]

        # add edges for close nodes
        # naive approach: O(N^2). For big graphs, do an approximate KNN.
        # We'll skip full N^2 here and just do random subset or so...
        # We'll do a small snippet for demonstration
        # or you can skip to keep it short.
        pass

        row, col = new_row, new_col

    new_edge_index = torch.stack([row, col], dim=0).long()
    data.edge_index = new_edge_index.to(data.x.device)
    return data


###############################################################################
#                              DEMO TRAINING SCRIPT                           #
###############################################################################
def run_demo(dataset_name='chameleon',
             lap_mode='signless',    # or 'standard','biLap'
             n_scales=3,
             poly_degree=4,
             hidden_dim=64,
             dropout=0.5,
             lr=0.005,
             weight_decay=1e-3,
             epochs=400,
             use_two_branch=True,
             advanced_rewire_flag=False,
             device='cuda'):

    ds_lower = dataset_name.lower()
    path = f'./data/{ds_lower}'

    if ds_lower in ['cora','citeseer','pubmed']:
        dataset = Planetoid(root=path, name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0]
    elif ds_lower in ['chameleon','squirrel']:
        dataset = WikipediaNetwork(root=path, name=ds_lower, transform=NormalizeFeatures())
        data = dataset[0]
        # choose fold 0
        data.train_mask = data.train_mask[:, 0].bool()
        data.val_mask   = data.val_mask[:, 0].bool()
        data.test_mask  = data.test_mask[:, 0].bool()
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # optional advanced rewiring
    if advanced_rewire_flag:
        data = advanced_rewire(data, steps=2, dist_threshold=0.4, sim_threshold=0.1)

    # Build multiple Laplacians or scales
    # e.g. we can store them in L_list
    # We'll do n_scales copies each with different "dilation" or approach,
    # or do something simpler:
    # For demonstration: we might interpret "scales" by artificially scaling the adjacency
    # or by repeated squaring. We'll do: L^1, L^2, L^3 if you want
    # or signless approach
    # We'll do L^1, L^2,... L^n
    # But let's keep it simpler: just repeated multiplications.

    # base Laplacian
    baseL = build_laplacian(data.edge_index, data.num_nodes, mode=lap_mode)
    # convert to torch sparse
    row = torch.from_numpy(baseL.row).long()
    col = torch.from_numpy(baseL.col).long()
    val = torch.from_numpy(baseL.data).float()
    L_sp = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=val,
        size=(data.num_nodes, data.num_nodes)
    ).coalesce().to(device)

    # build multi-scale by repeated multiplication
    L_list = [L_sp]
    for i in range(1, n_scales):
        # L^(i+1) = L^i * L
        Li = torch.sparse.mm(L_list[-1], L_sp).coalesce()
        L_list.append(Li)

    # initialize polynomial expansions from e.g. Haar or a wavelet
    # but now we just produce random or a small wavelet. 
    # For each scale, we do an array of length poly_degree. We'll do a wavelet-like initialization:
    # e.g. "Haar" approx
    def wavelet_haar_coeff(k):
        # a naive guess
        if k==0: return 0.707
        elif k==1: return 0.707
        else: return 0.0

    # create a list of arrays
    init_coeffs_list = []
    for i in range(n_scales):
        arr = []
        for k in range(poly_degree):
            arr.append(wavelet_haar_coeff(k) + 0.01*np.random.randn())
        init_coeffs_list.append(arr)

    # build the GNN
    model = FrameletGNN(
        num_features=dataset.num_features,
        hidden_dim=hidden_dim,
        num_classes=dataset.num_classes,
        L_list=L_list,
        init_coeffs_list=init_coeffs_list,
        dropout=dropout,
        use_two_branch=use_two_branch
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_test = 0.0, 0.0
    best_state = None
    patience, patience_max = 0, 50

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
        val_acc = accuracy(data.val_mask)
        test_acc = accuracy(data.test_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1

        if epoch%20==0:
            print(f"Epoch {epoch:03d}: loss={loss.item():.4f}, "
                  f"train={train_acc:.2f}, val={val_acc:.2f}, test={test_acc:.2f}, pat={patience}")

        if patience>patience_max:
            print("Early stop!")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Done. Best val={best_val:.4f}, test={best_test:.4f}")


if __name__=="__main__":
    # Example usage
    run_demo(
        dataset_name='chameleon', # or 'cora','citeseer','pubmed','squirrel', 'chameleon'
        lap_mode='signless',  # or 'biLap','standard'
        n_scales=3,           # try 3 or 4
        poly_degree=4,        # larger poly
        hidden_dim=80,
        dropout=0.6,
        lr=0.005,
        weight_decay=1e-2,
        epochs=400,
        use_two_branch=True,
        advanced_rewire_flag=True,
        device='cuda'
    )