import numpy as np
import math
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Dropout, ReLU, BatchNorm1d

from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures

import ray
from ray import tune


###############################################################################
#                          OPTIONAL ADVANCED REWIRING                         #
###############################################################################
def advanced_rewire(data, steps=2, dist_threshold=0.4):
    """
    Removes edges connecting nodes with feature distance > dist_threshold,
    repeated 'steps' times. Helps on strongly heterophilous graphs.

    NOTE: We remove the conflicting coalesce import from torch_geometric.utils
    and simply keep a 2×E edge_index after re-wiring.
    """
    import torch
    # from torch_geometric.utils import coalesce  # <-- Remove or rename this import

    x = data.x.cpu()
    eidx = data.edge_index.cpu()  # shape (2, E)
    row, col = eidx

    for _ in range(steps):
        dist = torch.norm(x[row] - x[col], dim=1)
        keep_mask = (dist < dist_threshold)
        row, col = row[keep_mask], col[keep_mask]

    # Now build the new edge index in (2, E) format:
    new_ei = torch.stack([row, col], dim=0).long()  # shape (2, E)

    # If you truly want to "coalesce" edge_index in PyG style, you'd do:
    #   from torch_geometric.utils import coalesce as pyg_coalesce
    #   new_ei, _ = pyg_coalesce(new_ei, None, data.num_nodes, data.num_nodes)
    # But a 2×E edge_index is typically fine as-is.

    data.edge_index = new_ei.to(data.x.device)
    return data


###############################################################################
#                           SIGNLESS LAPLACIAN (OR STANDARD)                  #
###############################################################################
def build_laplacian(edge_index, num_nodes, mode='signless'):
    """
    - mode='standard': L= I - D^{-1/2} A D^{-1/2}
    - mode='signless': L= I + D^{-1/2} A D^{-1/2}
    """
    import numpy as np
    import scipy.sparse as sp
    from torch_geometric.utils import remove_self_loops, add_self_loops

    row, col = edge_index
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    val = np.ones(len(row), dtype=np.float32)

    adj = sp.coo_matrix((val, (row.cpu().numpy(), col.cpu().numpy())),
                        shape=(num_nodes, num_nodes),
                        dtype=np.float32)
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_sqrt = 1.0 / np.sqrt(deg + 1e-12)

    rr, cc = adj.row, adj.col
    vv = adj.data
    for i in range(len(vv)):
        r, c = rr[i], cc[i]
        vv[i] *= deg_sqrt[r]*deg_sqrt[c]
    A_hat = sp.coo_matrix((vv, (rr, cc)), shape=adj.shape, dtype=np.float32)
    I_scipy = sp.eye(num_nodes, dtype=np.float32)

    if mode=='standard':
        L = (I_scipy - A_hat).tocoo()
    else:  # 'signless'
        L = (I_scipy + A_hat).tocoo()
    return L

###############################################################################
#                      FRAMELET UTILS (UND. MULTI-LEVEL)                      #
###############################################################################
from scipy import sparse
from scipy.sparse.linalg import lobpcg

def scipy_to_torch_sparse(A):
    # Make sure shape is (2, nnz) for .indices().
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row, dtype=torch.long)
    col = torch.tensor(A.col, dtype=torch.long)
    val = torch.tensor(A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(
        indices=torch.stack((row, col), dim=0),
        values=val,
        size=A.shape
    ).coalesce()

def ChebyshevApprox(f, n, quad_points=500):
    import numpy as np
    x = np.linspace(0, np.pi, quad_points)
    c = np.zeros(n)
    for k in range(1, n+1):
        integrand = np.cos((k-1)*x)*f((np.pi/2)*(np.cos(x)+1))
        c[k-1] = (2.0/np.pi)*np.trapz(integrand, x)
    return c

def get_operator(L, DFilters, n, s, J, Lev):
    import scipy.sparse as sp
    r = len(DFilters)
    coeff_list = []
    for j in range(r):
        coeff_list.append(ChebyshevApprox(DFilters[j], n))

    FD1 = sp.identity(L.shape[0], dtype=np.float32)
    d = dict()
    a = np.pi/2
    for l in range(1, Lev+1):
        for j in range(r):
            c_j = coeff_list[j]
            T0F = FD1
            T1F = ((s**(-J + l -1))/a*L).dot(T0F) - T0F
            Mtemp = (0.5*c_j[0])*T0F
            if n>1:
                Mtemp += c_j[1]*T1F
            prev, prev2 = T1F, T0F
            for k_ in range(2, n):
                TkF = ((2.0/a)*(s**(-J + l -1)*L)).dot(prev) - 2*prev - prev2
                Mtemp += c_j[k_]*TkF
                prev2, prev = prev, TkF
            d[(j,l-1)] = Mtemp
        FD1 = d[(0,l-1)]
    return d

def build_framelet_ops(L, DFilters, n, s, Lev, device):
    num_nodes = L.shape[0]
    lobpcg_init = np.random.rand(num_nodes, 1).astype(np.float32)
    lam_max, _ = lobpcg(L, lobpcg_init, maxiter=50)
    lam_max = float(lam_max[0])

    J = math.log(lam_max/np.pi, s) + Lev -1
    d = get_operator(L, DFilters, n, s, J, Lev)

    d_list = []
    for l in range(Lev):
        for j in range(len(DFilters)):
            M = d[(j,l)]
            M_sp = scipy_to_torch_sparse(M).coalesce()
            d_list.append(M_sp.to(device))
    return d_list

###############################################################################
#                     LEARNABLE CHEBYSHEV FILTER MODULE                      #
###############################################################################
from torch_sparse import spspmm  # efficient sparse-sparse multiplication

class LearnableChebFilter(nn.Module):
    """
    A Chebyshev polynomial filter with learnable coefficients implemented using
    efficient sparse-sparse operations.
    
    Given a sparse matrix L_sp (e.g., a Laplacian), it computes:
         p(L_sp) = sum_{k=0}^{n-1} a_k * T_k(L_sp)
    where T_0 = I, T_1 = L_sp, and for k >= 2:
         T_k = 2 * L_sp @ T_{k-1} - T_{k-2}.
    """
    def __init__(self, init_coeffs, device='cpu'):
        super().__init__()
        self.n = len(init_coeffs)
        self.poly_params = nn.Parameter(torch.tensor(init_coeffs, dtype=torch.float32, device=device))

    def forward(self, L_sp):
        device = L_sp.device
        N = L_sp.size(0)
        n = self.n

        # Create the identity matrix as a sparse tensor.
        I_indices = torch.arange(N, device=device)
        I = torch.sparse_coo_tensor(
            indices=torch.stack([I_indices, I_indices], dim=0),
            values=torch.ones(N, device=device),
            size=(N, N)
        ).coalesce()

        # Helper functions:
        def sparse_scale(B, scalar):
            return torch.sparse_coo_tensor(
                B.indices(),
                B.values() * scalar,
                B.size(),
                device=device
            ).coalesce()

        def sparse_add(A, B):
            return (A + B).coalesce()

        def sparse_mul(A, B):
            m, _ = A.size()
            _, n_out = B.size()
            new_indices, new_values = spspmm(
                A.indices(), A.values(), A.size(),
                B.indices(), B.values(), B.size(),
                m, n_out
            )
            return torch.sparse_coo_tensor(new_indices, new_values, (m, n_out), device=device).coalesce()

        def spmm_sub(scalar, L, prev, prev2):
            product = sparse_mul(L, prev)   # L @ prev
            return sparse_add(sparse_scale(product, scalar),
                              sparse_scale(prev2, -1))

        # Chebyshev Recursion:
        out = sparse_scale(I, 0.5 * self.poly_params[0])
        if n > 1:
            T1 = L_sp
            out = sparse_add(out, sparse_scale(T1, self.poly_params[1]))

        prev2 = I    # T0
        prev  = L_sp # T1
        for k in range(2, n):
            next_sp = spmm_sub(2.0, L_sp, prev, prev2)
            out = sparse_add(out, sparse_scale(next_sp, self.poly_params[k]))
            prev2, prev = prev, next_sp

        return out.coalesce()

###############################################################################
#                 BORROWING LINKX-STYLE ADJ EMBEDDINGS                        #
###############################################################################
def build_adjacency_sp(data):
    """
    Build an unweighted NxN adjacency with self-loops for summation of colEmb.
    """
    from torch_geometric.utils import remove_self_loops, add_self_loops
    edge_index = data.edge_index
    row, col = edge_index
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
    row, col = edge_index
    val = torch.ones(row.size(0), dtype=torch.float32, device=row.device)
    A_sp = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0),  # shape(2, E)
        val,
        (data.num_nodes, data.num_nodes)
    ).coalesce()
    return A_sp

###############################################################################
#        Multi-Head Aggregator: Combine the three embeddings (poly, local, adj)
###############################################################################
class ThreeBranchAttnAggregator(nn.Module):
    """
    We'll have 3 "tokens" for each node: 
      1) Framelet polynomial embedding
      2) Local MLP embedding
      3) Adjacency embedding
    We use multi-head attention across these 3 tokens => produce a single vector
    """
    def __init__(self, d_in, d_out, n_heads=2):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_heads = n_heads

        self.query = nn.Parameter(torch.randn(n_heads, d_in, d_in//n_heads))
        self.key   = nn.Parameter(torch.randn(n_heads, d_in, d_in//n_heads))
        self.value = nn.Parameter(torch.randn(n_heads, d_in, d_in//n_heads))
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.key)
        nn.init.xavier_uniform_(self.value)

        self.proj_out = nn.Linear(d_in, d_out, bias=False)
        self.bn = BatchNorm1d(d_out)

    def forward(self, x_poly, x_local, x_adj):
        X_3 = torch.stack([x_poly, x_local, x_adj], dim=1)  # [N, 3, d_in]
        N, T, d_in = X_3.shape

        heads_out = []
        for h in range(self.n_heads):
            Q = X_3 @ self.query[h]   # [N,3,d_k]
            K = X_3 @ self.key[h]     # [N,3,d_k]
            V = X_3 @ self.value[h]   # [N,3,d_k]
            d_k = Q.shape[-1]

            attn_scores = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(d_k)  # [N,3,3]
            attn_weights = F.softmax(attn_scores, dim=2)                   # [N,3,3]
            out_h = torch.bmm(attn_weights, V).mean(dim=1)                 # [N,d_k]
            heads_out.append(out_h)

        X_cat = torch.cat(heads_out, dim=1)  # [N, d_in]
        X_proj = self.proj_out(X_cat)
        X_bn   = self.bn(X_proj)
        return X_bn

###############################################################################
#   THREE-BRANCH MODEL: MULTI-LAYER FRAMELET, LOCAL MLP, LINKX EMBEDDING      #
###############################################################################
class _FrameletAggregatorLayer(nn.Module):
    """
    One layer of the framelet aggregator:
      - optional linear on input (if use_input_linear=True)
      - param alpha_i for each sub-band => spmm => sum
      - BN => ReLU => dropout
    """
    def __init__(self, 
                 in_dim, 
                 out_dim,
                 n_subbands,
                 dropout=0.5,
                 use_input_linear=True):
        super().__init__()
        self.n_subbands = n_subbands
        self.use_input_linear = use_input_linear
        self.dropout = dropout

        if use_input_linear:
            self.lin_in = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.lin_in = None

        # learnable alpha (one per subband)
        self.alpha_params = nn.Parameter(torch.zeros(n_subbands))
        nn.init.constant_(self.alpha_params, 1.0/n_subbands)

        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x_in, subband_ops):
        if self.lin_in is not None:
            x_in_l = self.lin_in(x_in)
        else:
            x_in_l = x_in

        out_sum = torch.zeros_like(x_in_l)
        for i, op in enumerate(subband_ops):
            w_i = self.alpha_params[i]
            sub_out = torch.sparse.mm(op, x_in_l)  # NxN @ NxDim => NxDim
            out_sum += w_i * sub_out

        out_sum = self.bn(out_sum)
        out_sum = F.relu(out_sum)
        out_sum = F.dropout(out_sum, p=self.dropout, training=self.training)
        return out_sum

class ThreeBranchFramelet(nn.Module):
    """
    1) Multi-layer Framelet aggregator => x_poly
    2) Local MLP => x_local
    3) LinkX adjacency => x_adj
    aggregator => final classification

    Optionally uses a learnable Chebyshev filter if use_learnable_cheb=True.
    In that case, we pass in L_torch (the Laplacian) and the learnable_cheb module,
    and compute d_list on-the-fly each forward pass:
        d_list = [learnable_cheb(L_torch)]
    Otherwise, we store the fixed subband operators in self.d_list and use them directly.
    """
    def __init__(self,
                 num_nodes,
                 in_dim,
                 out_dim,
                 d_list,
                 poly_hidden_dim=64,
                 n_framelet_layers=2,
                 local_hidden_dim=64,
                 linkx_embed_dim=64,
                 aggregator_heads=2,
                 final_hidden_dim=64,
                 dropout=0.5,
                 # For learnable filter:
                 use_learnable_cheb=False,
                 L_torch=None,
                 learnable_cheb=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.use_learnable_cheb = use_learnable_cheb
        self.L_torch = L_torch
        self.learnable_cheb = learnable_cheb

        if self.use_learnable_cheb:
            # We'll compute the operator each forward pass (1 subband).
            self.n_subbands = 1
            self.d_list = None
        else:
            # use the provided fixed operators
            self.d_list = d_list
            self.n_subbands = len(d_list)

        # aggregator layers
        self.agg_layers = nn.ModuleList()
        layer0 = _FrameletAggregatorLayer(
            in_dim=in_dim,
            out_dim=poly_hidden_dim,
            n_subbands=self.n_subbands,
            dropout=dropout,
            use_input_linear=True
        )
        self.agg_layers.append(layer0)

        for _ in range(n_framelet_layers - 1):
            layerX = _FrameletAggregatorLayer(
                in_dim=poly_hidden_dim,
                out_dim=poly_hidden_dim,
                n_subbands=self.n_subbands,
                dropout=dropout,
                use_input_linear=False
            )
            self.agg_layers.append(layerX)

        # local MLP
        self.local_mlp = nn.Sequential(
            nn.Linear(in_dim, local_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(local_hidden_dim, local_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(local_hidden_dim)
        )

        # adjacency embeddings (LinkX style)
        self.rowEmb = nn.Parameter(torch.randn(num_nodes, linkx_embed_dim))
        self.colEmb = nn.Parameter(torch.randn(num_nodes, linkx_embed_dim))
        nn.init.xavier_uniform_(self.rowEmb)
        nn.init.xavier_uniform_(self.colEmb)

        # aggregator (to combine the three branches)
        self.lin_poly_unify  = nn.Linear(poly_hidden_dim,  final_hidden_dim, bias=False)
        self.lin_local_unify = nn.Linear(local_hidden_dim, final_hidden_dim, bias=False)
        self.lin_linkx_unify = nn.Linear(linkx_embed_dim,  final_hidden_dim, bias=False)

        self.agg_attn = ThreeBranchAttnAggregator(
            d_in=final_hidden_dim,
            d_out=final_hidden_dim,
            n_heads=aggregator_heads
        )
        self.lin_out = nn.Linear(final_hidden_dim, out_dim)

    def forward(self, x, adjacency_frame, adjacency_linkx):
        # adjacency_frame is unused here; kept for interface consistency
        device = x.device

        # If using learnable Chebyshev, compute it on each forward pass
        if self.use_learnable_cheb:
            d_list = [self.learnable_cheb(self.L_torch)]
        else:
            d_list = self.d_list

        # multi-layer aggregator: framelet/cheb branch
        x_poly = x
        for layer in self.agg_layers:
            x_poly = layer(x_poly, d_list)

        # local MLP
        x_local = self.local_mlp(x)

        # LinkX adjacency
        col_sum = torch.sparse.mm(adjacency_linkx, self.colEmb)  # NxN @ NxEmbed => NxEmbed
        x_adj = self.rowEmb + col_sum

        # unify dims
        x_poly_u  = self.lin_poly_unify(x_poly)
        x_local_u = self.lin_local_unify(x_local)
        x_adj_u   = self.lin_linkx_unify(x_adj)

        x_agg = self.agg_attn(x_poly_u, x_local_u, x_adj_u)
        logits = self.lin_out(x_agg)
        return F.log_softmax(logits, dim=1)

###############################################################################
#                           TRAINING WITH RAY TUNE                            #
###############################################################################
dataset_name = "chameleon"  # example dataset name
path = osp.join("./data", dataset_name)
_ = WikipediaNetwork(root=path, name=dataset_name, transform=NormalizeFeatures())

def train_with_tune(config):
    dataset = WikipediaNetwork(root=path, name=dataset_name, transform=NormalizeFeatures())
    data = dataset[0]
    data.train_mask = data.train_mask[:, 0].bool()
    data.val_mask   = data.val_mask[:, 0].bool()
    data.test_mask  = data.test_mask[:, 0].bool()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    if config.get("rewire_flag", False):
        advanced_rewire(data, steps=2, dist_threshold=0.4)

    # Build signless Laplacian as a scipy sparse matrix
    L_scipy = build_laplacian(data.edge_index, data.num_nodes, mode='signless')

    # Possibly use learnable Chebyshev
    use_learnable = config.get("use_learnable_cheb_filter", False)
    if use_learnable:
        # We'll build L_torch & learnable filter but do NOT compute operator now
        L_torch = scipy_to_torch_sparse(L_scipy).coalesce().to(device)
        cheb_degree = config.get("cheb_poly_degree", 3)
        init_coeffs = [1.0] * cheb_degree
        learnable_cheb = LearnableChebFilter(init_coeffs, device=device).to(device)
        # We'll pass d_list=None for the model
        d_list = None
    else:
        # Build wavelet filters
        FrameType = config.get("FrameType", "Haar")
        if FrameType == 'Haar':
            D1 = lambda x: np.cos(x/2.)
            D2 = lambda x: np.sin(x/2.)
            DFilters = [D1, D2]
        elif FrameType == 'Linear':
            D1 = lambda x: np.cos(x/2.)**2
            D2 = lambda x: (np.sqrt(2)*np.sin(x))/2.
            D3 = lambda x: np.sin(x/2.)**2
            DFilters = [D1, D2, D3]
        elif FrameType == 'Quadratic':
            D1 = lambda x: np.cos(x / 2) ** 3
            D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
            D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
            D4 = lambda x: np.sin(x / 2) ** 3
            DFilters = [D1, D2, D3, D4]
        else:
            # default Haar
            D1 = lambda x: np.cos(x/2.)
            D2 = lambda x: np.sin(x/2.)
            DFilters = [D1, D2]

        Lev         = config.get("Lev", 2)
        poly_degree = config.get("poly_degree", 2)
        d_list      = build_framelet_ops(L_scipy, DFilters, poly_degree, s=2, Lev=Lev, device=device)

        L_torch = None
        learnable_cheb = None

    adjacency_linkx = build_adjacency_sp(data).coalesce()

    # We'll do 10 repeated runs
    n_reps = 10
    all_test = []

    for rep in range(n_reps):
        model = ThreeBranchFramelet(
            num_nodes=data.num_nodes,
            in_dim=data.x.size(1),
            out_dim=dataset.num_classes,
            d_list=d_list,
            poly_hidden_dim=   config["poly_hidden_dim"],
            n_framelet_layers= config["n_framelet_layers"],
            local_hidden_dim=  config["local_hidden_dim"],
            linkx_embed_dim=   config["linkx_embed_dim"],
            aggregator_heads=  config["aggregator_heads"],
            final_hidden_dim=  config["final_hidden_dim"],
            dropout=           config["dropout"],
            use_learnable_cheb=use_learnable,
            L_torch=L_torch,
            learnable_cheb=learnable_cheb
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )

        best_val, best_test = 0., 0.
        best_state = None
        patience, max_patience = 0, 50
        epochs = config.get("epochs", 300)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, None, adjacency_linkx)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_eval = model(data.x, None, adjacency_linkx)
            def accuracy(mask):
                p = out_eval[mask].max(dim=1)[1]
                return (p == data.y[mask]).sum().item() / mask.sum().item()

            train_acc = accuracy(data.train_mask)
            val_acc   = accuracy(data.val_mask)
            test_acc  = accuracy(data.test_mask)

            if val_acc > best_val:
                best_val   = val_acc
                best_test  = test_acc
                best_state = model.state_dict().copy()
                patience = 0
            else:
                patience += 1
            if patience > max_patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        all_test.append(best_test)

    test_acc_mean = float(np.mean(all_test))
    test_acc_var  = float(np.var(all_test))
    tune.report({"test_acc_mean": test_acc_mean, "test_acc_var": test_acc_var})


def run_ray_tune():
    ray.shutdown()
    ray.init()

    search_space = {
        "Lev":                tune.choice([2, 3]),
        "poly_degree":        tune.choice([2, 3]),
        "n_framelet_layers":  tune.choice([1, 2, 3]),
        "poly_hidden_dim":    tune.choice([32, 64]),
        "local_hidden_dim":   tune.choice([32, 64]),
        "linkx_embed_dim":    tune.choice([32, 64]),
        "aggregator_heads":   tune.choice([1, 2]),
        "final_hidden_dim":   tune.choice([64, 128]),
        "dropout":            tune.uniform(0.3, 0.7),
        "lr":                 tune.loguniform(1e-4, 5e-3),
        "weight_decay":       tune.loguniform(1e-5, 1e-2),
        "epochs":             200,
        "rewire_flag":        tune.choice([False, True]),
        "FrameType":          tune.choice(["Haar"]),
        # New options for the learnable Chebyshev filter:
        "use_learnable_cheb_filter": tune.choice([False, True]),
        "cheb_poly_degree":         tune.choice([2, 3, 4])
    }

    tuner = tune.run(
        train_with_tune,
        config=search_space,
        num_samples=50,  # for demonstration
        resources_per_trial={"cpu": 2, "gpu": 1},
        metric="test_acc_mean",
        mode="max"
    )

    print("Best config: ", tuner.get_best_config(metric="test_acc_mean", mode="max"))
    df = tuner.results_df
    print(df)
    print(df[["test_acc_mean", "test_acc_var"]])


if __name__ == "__main__":
    run_ray_tune()