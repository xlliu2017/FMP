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
#                         LOAD DATASET ONLY ONCE                              #
###############################################################################
dataset_name = "chameleon"  # example dataset name
path = osp.join("./data", dataset_name)
GLOBAL_DATASET = WikipediaNetwork(root=path, name=dataset_name, transform=NormalizeFeatures())

###############################################################################
#                          OPTIONAL ADVANCED REWIRING                         #
###############################################################################
from torch_geometric.utils import coalesce as pyg_coalesce

def advanced_rewire(data, steps=2, dist_threshold=0.4):
    """
    Removes edges connecting nodes with feature distance > dist_threshold,
    repeated 'steps' times. Helps on strongly heterophilous graphs.
    """
    x = data.x.cpu()
    eidx = data.edge_index.cpu()  # shape (2, E)
    row, col = eidx

    for _ in range(steps):
        dist = torch.norm(x[row] - x[col], dim=1)
        keep_mask = (dist < dist_threshold)
        row, col = row[keep_mask], col[keep_mask]

    # clamp to avoid negative or out-of-bound indices
    row = row.clamp(min=0, max=data.num_nodes-1)
    col = col.clamp(min=0, max=data.num_nodes-1)

    # Build the new edge index in (2, E) format
    new_ei = torch.stack([row, col], dim=0).long()

    # Coalesce ensures no duplicates
    new_ei, _ = pyg_coalesce(new_ei, None, data.num_nodes, data.num_nodes)

    # If all edges are removed, fallback to self-loops
    if new_ei.size(1) == 0:
        idx = torch.arange(data.num_nodes, dtype=torch.long)
        new_ei = torch.stack([idx, idx], dim=0).to(new_ei.device)

    data.edge_index = new_ei.to(data.x.device)
    return data


###############################################################################
#                           SIGNLESS LAPLACIAN (OR STANDARD)                  #
###############################################################################
def build_laplacian(edge_index, num_nodes, mode='signless'):
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
import torch

def scipy_to_torch_sparse(A):
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

###############################################################################
#                      LearnableChebFilter for a single L                     #
###############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch_sparse import spspmm

def sparse_scale(B, scalar):
    return torch.sparse_coo_tensor(
        B.indices(), B.values() * scalar, B.size(), device=device
    ).coalesce()

def sparse_add(A, B):
    return (A + B).coalesce()

def sparse_mul(A, B):
    m, kA = A.size()
    kB, n_out = B.size()
    new_indices, new_values = spspmm(
        A.indices(), A.values(), B.indices(), B.values(), m, kA, n_out, coalesced=True
    )
    return torch.sparse_coo_tensor(
        new_indices, new_values, (m, n_out), device=device
    ).coalesce()

def spmm_sub(scalar, L, prev, prev2):
    product = sparse_mul(L, prev)
    return sparse_add(
        sparse_scale(product, scalar),
        sparse_scale(prev2, -1)
    )

class LearnableChebFilter(nn.Module):
    """
    A Chebyshev polynomial filter with learnable coefficients implemented using
    efficient sparse-sparse operations for a single NxN matrix L_sp.
    """
    def __init__(self, init_coeffs, device='cpu'):
        super().__init__()
        self.n = len(init_coeffs)
        self.poly_params = nn.Parameter(torch.tensor(init_coeffs, dtype=torch.float32, device=device))
    
    def forward(self, L_sp):
        device = L_sp.device
        N = L_sp.size(0)
        n = self.n

        I_idx = torch.arange(N, device=device)
        I_sp = torch.sparse_coo_tensor(
            indices=torch.stack([I_idx, I_idx], dim=0),
            values=torch.ones(N, device=device),
            size=(N, N)
        ).coalesce()

        out = sparse_scale(I_sp, 0.5*self.poly_params[0])
        if n>1:
            out = sparse_add(out, sparse_scale(L_sp, self.poly_params[1]))

        prev2 = I_sp
        prev  = L_sp
        for k in range(2, n):
            next_sp = spmm_sub(2.0, L_sp, prev, prev2)
            out = sparse_add(out, sparse_scale(next_sp, self.poly_params[k]))
            prev2, prev = prev, next_sp

        return out.coalesce()

###############################################################################
#          MultiLevelLearnableFilters: r*Lev distinct LearnableChebFilter     #
###############################################################################
def chebyshev_rescale(L_sp, factor):
    return torch.sparse_coo_tensor(
        L_sp.indices(),
        L_sp.values()*factor,
        L_sp.size(),
        device=L_sp.device
    ).coalesce()

class MultiLevelLearnableFilters(nn.Module):
    """
    We store r*Lev distinct LearnableChebFilter, 
    each scale l=1..Lev, band j=0..(r-1) gets its own filter_j_l
    The forward pass returns a dictionary of NxN outputs d[(j,l-1)] 
    that replaces the static approach in get_operator.
    """
    def __init__(self, L_sp, DFilters, n, s, Lev, lam_max=None, device='cpu'):
        super().__init__()
        self.L_sp = L_sp.coalesce()  # NxN
        self.s = s
        self.Lev = Lev
        self.device = device

        self.r = len(DFilters)
        self.n = n
        self.a = math.pi/2

        if lam_max is None:
            # approximate lam_max by lobpcg
            import numpy as np
            from scipy.sparse.linalg import lobpcg
            import scipy.sparse as sp

            coo = L_sp.coalesce()
            row, col = coo.indices()
            val = coo.values()
            N = L_sp.size(0)
            mat = sp.coo_matrix(
                (val.cpu().numpy(), (row.cpu().numpy(), col.cpu().numpy())),
                shape=(N,N)
            )
            lob_init = np.random.rand(N,1).astype(np.float32)
            lam_max_res, _ = lobpcg(mat, lob_init, maxiter=250)
            self.lam_max = float(lam_max_res[0])
        else:
            self.lam_max = lam_max

        self.J = math.log(self.lam_max / math.pi, s) + Lev - 1

        # Store r*Lev distinct filters
        self.filters = nn.ModuleDict()
        for l_ in range(1, Lev+1):
            for j_ in range(self.r):
                init_coeffs = ChebyshevApprox(DFilters[j_], n)
                key = f"filter_{j_}_{l_}"
                self.filters[key] = LearnableChebFilter(init_coeffs, device=device)

    def forward(self):
        d = {}
        N = self.L_sp.size(0)

        I_idx = torch.arange(N, device=self.device)
        FD1 = torch.sparse_coo_tensor(
            indices=torch.stack([I_idx,I_idx], dim=0),
            values=torch.ones(N, device=self.device),
            size=(N,N)
        ).coalesce()

        for l_ in range(1, self.Lev+1):
            scale_factor = (self.s**(-self.J + l_ -1))/self.a
            L_scaled = chebyshev_rescale(self.L_sp, scale_factor)

            for j_ in range(self.r):
                key = f"filter_{j_}_{l_}"
                # NxN polynomial for scaled L
                poly_sp = self.filters[key](L_scaled)
                # wavelet logic => poly_sp * FD1 => NxN
                out_jl = torch.sparse.mm(poly_sp, FD1).coalesce()
                d[(j_, l_-1)] = out_jl

            FD1 = d[(0, l_-1)]  # iterative approach

        return d


###############################################################################
#                      A 3-Branch GNN that can use multiLL                    #
###############################################################################
class ThreeBranchAttnAggregator(nn.Module):
    def __init__(self, d_in, d_out, n_heads=2):
        super().__init__()
        self.d_in = d_in
        self.d_out= d_out
        self.n_heads= n_heads

        self.query = nn.Parameter(torch.randn(n_heads, d_in, d_in//n_heads))
        self.key   = nn.Parameter(torch.randn(n_heads, d_in, d_in//n_heads))
        self.value = nn.Parameter(torch.randn(n_heads, d_in, d_in//n_heads))
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.key)
        nn.init.xavier_uniform_(self.value)

        self.proj_out = nn.Linear(d_in, d_out, bias=False)
        self.bn = BatchNorm1d(d_out)

    def forward(self, x_poly, x_local, x_adj):
        import math
        N, d_in = x_poly.size(0), x_poly.size(1)
        X_3 = torch.stack([x_poly, x_local, x_adj], dim=1)  # [N,3,d_in]

        heads_out=[]
        for h in range(self.n_heads):
            Q = X_3 @ self.query[h]
            K = X_3 @ self.key[h]
            V = X_3 @ self.value[h]
            d_k= Q.shape[-1]

            attn_scores= torch.bmm(Q, K.transpose(1,2))/math.sqrt(d_k)
            attn_weights= F.softmax(attn_scores, dim=2)
            out_h = torch.bmm(attn_weights, V).mean(dim=1)
            heads_out.append(out_h)

        X_cat= torch.cat(heads_out, dim=1)
        X_proj= self.proj_out(X_cat)
        X_bn= self.bn(X_proj)
        return X_bn


class _FrameletAggregatorLayer(nn.Module):
    """
    The 'classic' aggregator that sums over sub-bands. We'll keep it for the old approach.
    """
    def __init__(self, in_dim, out_dim, n_subbands, dropout=0.5, use_input_linear=True):
        super().__init__()
        self.n_subbands= n_subbands
        self.use_input_linear= use_input_linear
        self.dropout= dropout

        if use_input_linear:
            self.lin_in= nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.lin_in= None

        self.alpha_params= nn.Parameter(torch.zeros(n_subbands))
        nn.init.constant_(self.alpha_params, 1.0/n_subbands)

        self.bn= nn.BatchNorm1d(out_dim)

    def forward(self, x_in, subband_ops):
        if self.lin_in is not None:
            x_in_l= self.lin_in(x_in)
        else:
            x_in_l= x_in

        out_sum= torch.zeros_like(x_in_l)
        for i, op in enumerate(subband_ops):
            w_i= self.alpha_params[i]
            sub_out= torch.sparse.mm(op, x_in_l)
            out_sum+= w_i*sub_out

        out_sum= self.bn(out_sum)
        out_sum= F.relu(out_sum)
        out_sum= F.dropout(out_sum, p=self.dropout, training=self.training)
        return out_sum


class ThreeBranchFramelet(nn.Module):
    """
    Now we add 'use_multi_level_learnable' to use the MultiLevelLearnableFilters 
    instead of a static d_list from build_framelet_ops.
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
                 use_learnable_cheb=False,
                 L_torch=None,
                 learnable_cheb=None,
                 use_linkx=True,
                 use_multi_level_learnable=False,
                 multiLL=None,
                 r=2,   # number of wavelet bands if use_multi_level_learnable
                 Lev=2  # number of levels
    ):
        super().__init__()
        self.num_nodes= num_nodes
        self.in_dim= in_dim
        self.out_dim= out_dim
        self.dropout= dropout

        self.use_learnable_cheb= use_learnable_cheb
        self.L_torch= L_torch
        self.learnable_cheb= learnable_cheb
        self.use_linkx= use_linkx

        self.use_multi_level_learnable= use_multi_level_learnable
        self.multiLL= multiLL
        self.r= r
        self.Lev= Lev

        # If we do multi-level learnable, we won't use d_list. We'll compute in forward.
        # Else we do the old approach:
        if self.use_multi_level_learnable:
            self.d_list= None
            # We'll have r*Lev sub-bands
            self.n_subbands= r*Lev
        else:
            self.d_list= d_list
            self.n_subbands= len(d_list) if d_list is not None else 1

        # aggregator layers
        self.agg_layers= nn.ModuleList()
        layer0= _FrameletAggregatorLayer(
            in_dim=in_dim,
            out_dim=poly_hidden_dim,
            n_subbands=self.n_subbands,
            dropout=dropout,
            use_input_linear=True
        )
        self.agg_layers.append(layer0)
        for _ in range(n_framelet_layers -1):
            layerX= _FrameletAggregatorLayer(
                in_dim=poly_hidden_dim,
                out_dim=poly_hidden_dim,
                n_subbands=self.n_subbands,
                dropout=dropout,
                use_input_linear=False
            )
            self.agg_layers.append(layerX)

        # local MLP
        self.local_mlp= nn.Sequential(
            nn.Linear(in_dim, local_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(local_hidden_dim, local_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(local_hidden_dim)
        )

        # linkx
        if self.use_linkx:
            self.rowEmb= nn.Parameter(torch.randn(num_nodes, linkx_embed_dim))
            self.colEmb= nn.Parameter(torch.randn(num_nodes, linkx_embed_dim))
            nn.init.xavier_uniform_(self.rowEmb)
            nn.init.xavier_uniform_(self.colEmb)
            self.lin_linkx_unify= nn.Linear(linkx_embed_dim, final_hidden_dim, bias=False)
        else:
            self.rowEmb= None
            self.colEmb= None
            self.lin_linkx_unify= None

        self.lin_poly_unify=  nn.Linear(poly_hidden_dim, final_hidden_dim, bias=False)
        self.lin_local_unify= nn.Linear(local_hidden_dim, final_hidden_dim, bias=False)
        self.agg_attn= ThreeBranchAttnAggregator(
            d_in= final_hidden_dim,
            d_out= final_hidden_dim,
            n_heads= aggregator_heads
        )
        self.lin_out= nn.Linear(final_hidden_dim, out_dim)

    def forward(self, x, adjacency_frame, adjacency_linkx):
        device= x.device

        # 1) Build sub-band ops
        #    If use_multi_level_learnable, call multiLL -> d
        #    Then we gather NxN ops in a list sorted by (j,l-1).
        if self.use_multi_level_learnable and (self.multiLL is not None):
            d= self.multiLL()  # dictionary d[(j,l-1)] -> NxN
            # convert to a list sorted in some stable order
            # e.g. l-1 ascending, j ascending
            subband_ops= []
            for l_ in range(self.Lev):
                for j_ in range(self.r):
                    subband_ops.append(d[(j_, l_)])  # store NxN in a list
        elif self.use_learnable_cheb:
            # single L
            d_list= [self.learnable_cheb(self.L_torch)]
            subband_ops= d_list
        else:
            # old approach: d_list from build_framelet_ops
            subband_ops= self.d_list

        # aggregator
        x_poly= x
        for layer in self.agg_layers:
            x_poly= layer(x_poly, subband_ops)

        # local MLP
        x_local= self.local_mlp(x)

        # linkx
        if self.use_linkx and adjacency_linkx is not None:
            col_sum= torch.sparse.mm(adjacency_linkx, self.colEmb)
            x_adj= self.rowEmb + col_sum
            x_adj_u= self.lin_linkx_unify(x_adj)
        else:
            # fallback to zeros
            x_adj_u= torch.zeros(
                x_poly.size(0), self.lin_poly_unify.out_features, device=device
            )

        x_poly_u=  self.lin_poly_unify(x_poly)
        x_local_u= self.lin_local_unify(x_local)
        x_agg= self.agg_attn(x_poly_u, x_local_u, x_adj_u)
        logits= self.lin_out(x_agg)
        return F.log_softmax(logits, dim=1)


###############################################################################
#                           TRAINING WITH RAY TUNE                            #
###############################################################################
def train_with_tune(config):
    dataset = GLOBAL_DATASET
    data = dataset[0]
    data.train_mask = data.train_mask[:, 0].bool()
    data.val_mask   = data.val_mask[:, 0].bool()
    data.test_mask  = data.test_mask[:, 0].bool()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    if config.get("rewire_flag", False):
        advanced_rewire(data, steps=2, dist_threshold=0.4)

    # Build adjacency linkx
    from torch_geometric.utils import remove_self_loops, add_self_loops
    eidx = data.edge_index
    row, col = eidx
    eidx, _ = remove_self_loops(eidx)
    eidx, _ = add_self_loops(eidx, num_nodes=data.num_nodes)
    row, col = eidx
    val = torch.ones(row.size(0), dtype=torch.float32, device=device)
    adjacency_linkx = torch.sparse_coo_tensor(
        torch.stack([row,col], dim=0),
        val,
        (data.num_nodes, data.num_nodes)
    ).coalesce()

    # Decide if we do multi-level learnable or old approach
    use_multi = config.get("use_multi_level_learnable", False)
    if use_multi:
        FrameType = config.get("FrameType", "Haar")
        if FrameType=='Haar':
            D1 = lambda x: np.cos(x/2.)
            D2 = lambda x: np.sin(x/2.)
            DFilters= [D1,D2]
        else:
            D1 = lambda x: np.cos(x/2.)**2
            D2 = lambda x: (np.sqrt(2)*np.sin(x))/2.
            D3 = lambda x: np.sin(x/2.)**2
            DFilters= [D1,D2,D3]

        # build L_sp
        from scipy.sparse.linalg import lobpcg
        from scipy import sparse as sp

        L_scipy= build_laplacian(data.edge_index, data.num_nodes, mode='signless')
        L_sp= scipy_to_torch_sparse(L_scipy).coalesce().to(device)

        multiLL = MultiLevelLearnableFilters(
            L_sp=L_sp,
            DFilters=DFilters,
            n=config.get("poly_degree",2),
            s=2,
            Lev=config.get("Lev",2),
            lam_max=None,
            device=device
        )
        d_list= None
    else:
        # old approach
        from scipy.sparse.linalg import lobpcg
        L_scipy= build_laplacian(data.edge_index, data.num_nodes, mode='signless')
        # build d_list
        FrameType = config.get("FrameType", "Haar")
        if FrameType=='Haar':
            D1 = lambda x: np.cos(x/2.)
            D2 = lambda x: np.sin(x/2.)
            DFilters= [D1,D2]
        else:
            D1 = lambda x: np.cos(x/2.)**2
            D2 = lambda x: (np.sqrt(2)*np.sin(x))/2.
            D3 = lambda x: np.sin(x/2.)**2
            DFilters= [D1,D2,D3]

        Lev= config.get("Lev",2)
        poly_degree= config.get("poly_degree",2)
        d_list= build_framelet_ops(L_scipy, DFilters, poly_degree, s=2, Lev=Lev, device=device)
        multiLL= None

    # Build the model
    model= ThreeBranchFramelet(
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
        use_learnable_cheb=False,   # old single filter approach
        L_torch=None,
        learnable_cheb=None,
        use_linkx=config.get("use_linkx", True),
        use_multi_level_learnable=use_multi,
        multiLL=multiLL,
        r=len(DFilters) if use_multi else 2,
        Lev=config.get("Lev",2)
    ).to(device)

    optimizer= torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    n_reps= 3  # reduce for speed
    all_test=[]
    import numpy as np

    for rep in range(n_reps):
        best_val, best_test= 0.,0.
        best_state= None
        patience, max_patience=0, 50
        epochs= config.get("epochs",300)

        # re-init model each rep to measure variance
        model.apply(lambda m: reset_parameters(m))
        # or re-create the model from scratch if you prefer

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out= model(data.x, None, adjacency_linkx)
            loss= F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_eval= model(data.x, None, adjacency_linkx)
            def accuracy(mask):
                p= out_eval[mask].max(dim=1)[1]
                return (p==data.y[mask]).sum().item()/mask.sum().item()
            train_acc= accuracy(data.train_mask)
            val_acc=   accuracy(data.val_mask)
            test_acc=  accuracy(data.test_mask)

            if val_acc>best_val:
                best_val= val_acc
                best_test= test_acc
                best_state= {k:v.clone() for k,v in model.state_dict().items()}
                patience=0
            else:
                patience+=1
            if patience>max_patience:
                break

        if best_state:
            model.load_state_dict(best_state)
        all_test.append(best_test)

    test_acc_mean= float(np.mean(all_test))
    test_acc_var= float(np.var(all_test))
    tune.report({"test_acc_mean": test_acc_mean, "test_acc_var": test_acc_var})


def reset_parameters(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def run_ray_tune():
    ray.shutdown()
    ray.init()

    search_space = {
        "Lev":                tune.choice([2, 3]),
        "poly_degree":        tune.choice([2, 3]),
        "n_framelet_layers":  tune.choice([1, 2]),
        "poly_hidden_dim":    tune.choice([32, 64]),
        "local_hidden_dim":   tune.choice([32, 64]),
        "linkx_embed_dim":    tune.choice([32, 64]),
        "aggregator_heads":   tune.choice([1, 2]),
        "final_hidden_dim":   tune.choice([64, 128]),
        "dropout":            tune.uniform(0.3, 0.7),
        "lr":                 tune.loguniform(1e-4, 5e-3),
        "weight_decay":       tune.loguniform(1e-5, 1e-2),
        "epochs":             200,
        "rewire_flag":        tune.choice([False]),
        "FrameType":          tune.choice(["Haar", "Linear"]),
        "use_linkx":          tune.choice([True, False]),
        "use_multi_level_learnable": tune.choice([True, False])
    }

    tuner = tune.run(
        train_with_tune,
        config=search_space,
        num_samples=8,
        resources_per_trial={"cpu": 2, "gpu": 1},
        metric="test_acc_mean",
        mode="max"
    )

    print("Best config: ", tuner.get_best_config(metric="test_acc_mean", mode="max"))
    df = tuner.results_df
    print(df)
    print(df[["test_acc_mean", "test_acc_var"]])

###############################################################################
#             A SIMPLE TEST FUNCTION WITHOUT RAY (SINGLE RUN)                #
###############################################################################
def test_no_ray():
    """
    A single-run function to test everything without Ray, 
    focusing on use_multi_level_learnable = True.
    """
    from torch_geometric.datasets import WikipediaNetwork
    data = GLOBAL_DATASET[0]
    data.train_mask = data.train_mask[:,0].bool()
    data.val_mask   = data.val_mask[:,0].bool()
    data.test_mask  = data.test_mask[:,0].bool()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # optional rewire
    # advanced_rewire(data, steps=2, dist_threshold=0.4)

    # build adjacency linkx
    from torch_geometric.utils import remove_self_loops, add_self_loops
    eidx = data.edge_index
    row, col = eidx
    eidx, _ = remove_self_loops(eidx)
    eidx, _ = add_self_loops(eidx, num_nodes=data.num_nodes)
    row, col = eidx
    val = torch.ones(row.size(0), dtype=torch.float32, device=device)
    adjacency_linkx = torch.sparse_coo_tensor(
        torch.stack([row,col], dim=0),
        val,
        (data.num_nodes, data.num_nodes)
    ).coalesce()

    # We do multi-level approach
    FrameType = "Linear"
    if FrameType=='Haar':
        D1 = lambda x: np.cos(x/2.)
        D2 = lambda x: np.sin(x/2.)
        DFilters= [D1,D2]
    else:
        D1 = lambda x: np.cos(x/2.)**2
        D2 = lambda x: (np.sqrt(2)*np.sin(x))/2.
        D3 = lambda x: np.sin(x/2.)**2
        DFilters= [D1,D2,D3]

    # build L_sp
    from scipy.sparse.linalg import lobpcg
    from scipy import sparse as sp
    L_scipy= build_laplacian(data.edge_index, data.num_nodes, mode='signless')
    L_sp= scipy_to_torch_sparse(L_scipy).coalesce().to(device)

    # multi-level
    from math import log
    multiLL = MultiLevelLearnableFilters(
        L_sp=L_sp,
        DFilters=DFilters,
        n=3,
        s=2,
        Lev=2,
        lam_max=None,
        device=device
    )

    model= ThreeBranchFramelet(
        num_nodes=data.num_nodes,
        in_dim=data.x.size(1),
        out_dim=GLOBAL_DATASET.num_classes,
        d_list=None,                # not used
        poly_hidden_dim=64,
        n_framelet_layers=2,
        local_hidden_dim=64,
        linkx_embed_dim=64,
        aggregator_heads=2,
        final_hidden_dim=64,
        dropout=0.5,
        use_multi_level_learnable=True,
        multiLL=multiLL,
        r=len(DFilters),            # e.g. 2 for Haar
        Lev=2,
        use_linkx=False
    ).to(device)

    optimizer= torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    best_val, best_test= 0.,0.
    best_state= None
    patience, max_patience=0, 50
    epochs= 100

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out= model(data.x, None, adjacency_linkx)
        loss= F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_eval= model(data.x, None, adjacency_linkx)
        def accuracy(mask):
            p= out_eval[mask].max(dim=1)[1]
            return (p==data.y[mask]).sum().item()/mask.sum().item()
        train_acc= accuracy(data.train_mask)
        val_acc=   accuracy(data.val_mask)
        test_acc=  accuracy(data.test_mask)

        if val_acc>best_val:
            best_val= val_acc
            best_test= test_acc
            best_state= model.state_dict().copy()
            patience=0
        else:
            patience+=1
        if patience>max_patience:
            print(f"Early stop at epoch {epoch}")
            break

        if epoch%20==0:
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} "
                  f"train={train_acc:.2f} val={val_acc:.2f} test={test_acc:.2f} pat={patience}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"\n>>> test_no_ray done. Best val={best_val:.4f}, test={best_test:.4f}")

if __name__=="__main__":
    # run single test
    test_no_ray()
    # or run ray tune
    # run_ray_tune()