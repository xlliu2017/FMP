import numpy as np
import math
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Dropout, ReLU, BatchNorm1d

from scipy import sparse
from scipy.sparse.linalg import lobpcg


from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import remove_self_loops, add_self_loops, coalesce


###############################################################################
#                          OPTIONAL ADVANCED REWIRING                         #
###############################################################################
def advanced_rewire(data, steps=2, dist_threshold=0.4):
    """
    Removes edges connecting nodes with feature distance> dist_threshold,
    repeated 'steps' times. Helps on strongly heterophilous graphs.
    """
    x = data.x.cpu()
    eidx = data.edge_index.cpu()
    row, col = eidx
    for _ in range(steps):
        dist = torch.norm(x[row] - x[col], dim=1)
        keep_mask = (dist < dist_threshold)
        row, col = row[keep_mask], col[keep_mask]
    new_ei = torch.stack([row, col], dim=0).long()
    new_ei = coalesce(new_ei)
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
    row, col = edge_index
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    val = np.ones(len(row), dtype=np.float32)

    # Build adjacency in scipy
    adj = sparse.coo_matrix((val, (row.cpu().numpy(), col.cpu().numpy())),
                            shape=(num_nodes, num_nodes),
                            dtype=np.float32)
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_sqrt = 1.0 / np.sqrt(deg + 1e-12)

    # Build normalized adjacency
    rr, cc = adj.row, adj.col
    vv = adj.data
    for i in range(len(vv)):
        r, c = rr[i], cc[i]
        vv[i] *= deg_sqrt[r]*deg_sqrt[c]
    A_hat = sparse.coo_matrix((vv, (rr, cc)), shape=adj.shape, dtype=np.float32)
    I_scipy = sparse.eye(num_nodes, dtype=np.float32)

    if mode=='standard':
        L = (I_scipy - A_hat).tocoo()
    else: # 'signless'
        L = (I_scipy + A_hat).tocoo()
    return L


###############################################################################
#                      FRAMELET UTILS (UND. MULTI-LEVEL)                      #
###############################################################################
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
    c = np.zeros(n)
    x = np.linspace(0, np.pi, quad_points)
    for k in range(1, n+1):
        integrand = np.cos((k-1)*x)*f((np.pi/2)*(np.cos(x)+1))
        c[k-1] = (2.0/np.pi)*np.trapz(integrand, x)
    return c

def get_operator(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    coeff_list = []
    for j in range(r):
        coeff_list.append(ChebyshevApprox(DFilters[j], n))

    FD1 = sparse.identity(L.shape[0], dtype=np.float32)
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
            for k in range(2, n):
                TkF = ((2.0/a)*(s**(-J + l -1)*L)).dot(prev) - 2*prev - prev2
                Mtemp += c_j[k]*TkF
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
#                 BORROWING LINKX-STYLE ADJ EMBEDDINGS                        #
###############################################################################
def build_adjacency_sp(data):
    """
    Build an unweighted NxN adjacency with self-loops for summation of colEmb.
    """
    edge_index = data.edge_index
    row, col = edge_index
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
    row, col = edge_index
    val = torch.ones(row.size(0), dtype=torch.float32, device=row.device)
    A_sp = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0),
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
    We use multi-head attention across these 3 tokens (like a small "self-attn" for each node)
    => produce a single vector per node => final MLP => classification
    Or we just produce a single "combined" embedding. We'll show the basic approach here.
    """
    def __init__(self, d_in, d_out, n_heads=2):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_heads = n_heads

        # We'll treat 3 tokens => we do standard multi-head self-attn across token dimension=3.
        # Q, K, V for each head
        self.query = nn.Parameter(torch.randn(n_heads, d_in, d_in//n_heads))
        self.key   = nn.Parameter(torch.randn(n_heads, d_in, d_in//n_heads))
        self.value = nn.Parameter(torch.randn(n_heads, d_in, d_in//n_heads))
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.key)
        nn.init.xavier_uniform_(self.value)

        self.proj_out = nn.Linear(d_in, d_out, bias=False)
        self.bn = BatchNorm1d(d_out)

    def forward(self, x_poly, x_local, x_adj):
        """
        For each node, we have 3 embeddings: [d_in], [d_in], [d_in].
        We'll stack them => shape [N, 3, d_in] => multi-head attn across 3 dimension => [N, d_in].
        Then project => BN => return [N, d_out].
        """
        device = x_poly.device
        # stack => [N, 3, d_in]
        X_3 = torch.stack([x_poly, x_local, x_adj], dim=1)  # 3 tokens
        N, T, d_in = X_3.shape  # T=3

        # multi-head
        heads_out = []
        for h in range(self.n_heads):
            # Q= X_3 @ query[h], shape => [N,T, d_in//n_heads]
            Q = X_3 @ self.query[h]
            K = X_3 @ self.key[h]
            V = X_3 @ self.value[h]
            d_k = Q.shape[-1]

            # attn scores => Q bmm K^T along T dimension
            # shape => [N, T, T]
            Q_2 = Q.reshape(N, T, d_k)
            K_2 = K.reshape(N, T, d_k)
            attn_scores = torch.bmm(Q_2, K_2.transpose(1,2)) / math.sqrt(d_k)  # [N,T,T]
            attn_weights = F.softmax(attn_scores, dim=2)

            # out => attn_weights bmm V => [N,T,d_k]
            V_2 = V.reshape(N, T, d_k)
            out_h = torch.bmm(attn_weights, V_2)  # [N,T,d_k]
            # optionally pool over T => e.g. mean => [N, d_k]
            out_mean = out_h.mean(dim=1)
            heads_out.append(out_mean)

        # concat => [N, n_heads*d_k] => n_heads*(d_in//n_heads) = d_in
        X_cat = torch.cat(heads_out, dim=1)  # [N, d_in]
        X_proj = self.proj_out(X_cat)
        X_bn = self.bn(X_proj)
        return X_bn  # [N, d_out]


###############################################################################
#        FINAL MODEL: 3-Branch (Framelet Polynomials + Local MLP + LinkX)     #
###############################################################################
class ThreeBranchFramelet(nn.Module):
    """
    1) Multi-layer Framelet aggregator => x_poly (DEEP aggregator)
    2) Local MLP on x => x_local
    3) LinkX adjacency embeddings => x_adj

    Then aggregator => final MLP => log_softmax
    """
    def __init__(self,
                 num_nodes,
                 in_dim,
                 out_dim,
                 # framelet args
                 d_list, 
                 poly_hidden_dim=64,
                 n_framelet_layers=2,  # number of aggregator layers
                 # local MLP
                 local_hidden_dim=64,
                 # adjacency embedding
                 linkx_embed_dim=64,
                 # aggregator / final
                 aggregator_heads=2,
                 final_hidden_dim=64,
                 dropout=0.5):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim= out_dim
        self.d_list = d_list  # list of sub-band operators
        self.n_subbands = len(d_list)

        self.n_framelet_layers = n_framelet_layers
        self.dropout = dropout

        ## ============= 1) Multi-layer Framelet aggregator =============
        # For layer 1, we map in_dim -> poly_hidden_dim
        # Then each layer does:
        #   x_next = sum_{subband}( alpha_{layer,i} * spmm(subband, x_in) )
        #   optional nonlinearity or BN, dropout, etc.
        # We store these aggregator layers in a ModuleList
        # Example structure: 
        #   aggregator_layers[0]: linear_in + param alpha + spmm
        #   aggregator_layers[1..n-1]: param alpha + spmm
        #   We'll do an optional ReLU or BN between layers for more depth.

        # A simple approach:
        #  - aggregator_layers[0] includes a linear to get dimension poly_hidden_dim
        #  - aggregator_layers[1..n-1] stay in poly_hidden_dim
        # Each aggregator layer will store:
        #   param alpha_subband: [n_subbands], 
        # plus (optionally) a small BN or dropout.

        self.agg_layers = nn.ModuleList()

        # The first aggregator layer has to do a linear from in_dim -> poly_hidden_dim
        layer0 = _FrameletAggregatorLayer(
            in_dim=in_dim,
            out_dim=poly_hidden_dim,
            n_subbands=self.n_subbands,
            dropout=dropout,
            use_input_linear=True
        )
        self.agg_layers.append(layer0)

        # The subsequent aggregator layers go from poly_hidden_dim -> poly_hidden_dim
        for _ in range(n_framelet_layers - 1):
            layerX = _FrameletAggregatorLayer(
                in_dim=poly_hidden_dim,
                out_dim=poly_hidden_dim,
                n_subbands=self.n_subbands,
                dropout=dropout,
                use_input_linear=False
            )
            self.agg_layers.append(layerX)

        ## ============= 2) Local MLP =============
        self.local_mlp = nn.Sequential(
            nn.Linear(in_dim, local_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(local_hidden_dim, local_hidden_dim),
            nn.ReLU(),
            BatchNorm1d(local_hidden_dim)
        )

        ## ============= 3) LinkX adjacency embeddings =============
        self.rowEmb = nn.Parameter(torch.randn(num_nodes, linkx_embed_dim))
        self.colEmb = nn.Parameter(torch.randn(num_nodes, linkx_embed_dim))
        nn.init.xavier_uniform_(self.rowEmb)
        nn.init.xavier_uniform_(self.colEmb)

        ## ============= aggregator (Attn) =============
        # unify dimension => aggregator
        self.lin_poly_unify  = nn.Linear(poly_hidden_dim,  final_hidden_dim, bias=False)
        self.lin_local_unify = nn.Linear(local_hidden_dim, final_hidden_dim, bias=False)
        self.lin_linkx_unify = nn.Linear(linkx_embed_dim,  final_hidden_dim, bias=False)

        self.agg_attn = ThreeBranchAttnAggregator(
            d_in=final_hidden_dim, 
            d_out=final_hidden_dim, 
            n_heads=aggregator_heads
        )
        
        # final classification
        self.lin_out = nn.Linear(final_hidden_dim, out_dim)

    def forward(self, x, adjacency_frame, adjacency_linkx):
        """
        x: node features [N, in_dim]
        adjacency_frame: sub-band operators in self.d_list (list of NxN sp_coo)
        adjacency_linkx: NxN unweighted adjacency for LinkX sum

        Steps:
          1) Multi-layer framelet aggregator -> x_poly
          2) local mlp -> x_local
          3) linkX adjacency -> x_adj
          4) aggregator attn -> final
        """
        device = x.device
        N = x.size(0)

        # ============= 1) Multi-layer Framelet aggregator =============
        x_poly = x
        for layer_idx, layer in enumerate(self.agg_layers):
            x_poly = layer(x_poly, self.d_list)  # spmm => sum => BN => dropout => ReLU?

        # ============= 2) local MLP =============
        x_local = self.local_mlp(x) # [N, local_hidden_dim]

        # ============= 3) LinkX adjacency =============
        # sum_j colEmb[j] => spmm(adjacency_linkx, colEmb) + rowEmb[i]
        col_sum = torch.sparse.mm(adjacency_linkx, self.colEmb)  # [N, linkx_embed_dim]
        x_adj = self.rowEmb + col_sum # [N, linkx_embed_dim]

        # unify dims => aggregator
        x_poly_u  = self.lin_poly_unify(x_poly)   # [N, final_hidden_dim]
        x_local_u = self.lin_local_unify(x_local) # [N, final_hidden_dim]
        x_adj_u   = self.lin_linkx_unify(x_adj)   # [N, final_hidden_dim]

        x_agg = self.agg_attn(x_poly_u, x_local_u, x_adj_u) # [N, final_hidden_dim]

        # final
        logits = self.lin_out(x_agg)
        return F.log_softmax(logits, dim=1)


###############################################################################
# A single aggregator layer: param alpha per sub-band, spmm => sum => BN => ReLU
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

        # learnable alpha
        self.alpha_params = nn.Parameter(torch.zeros(n_subbands))
        nn.init.constant_(self.alpha_params, 1.0/n_subbands)

        self.bn = nn.BatchNorm1d(out_dim)
        self.mlp_agg = nn.Sequential(
            ReLU(),
            Dropout(self.dropout),
            nn.Linear(n_subbands*out_dim, out_dim),
            ReLU(),
            BatchNorm1d(out_dim)
        )

    def forward(self, x_in, subband_ops):
        """
        x_in: [N, in_dim or out_dim]
        subband_ops: list of NxN sp_coo, length=n_subbands
        Steps:
          1) if use_input_linear, do x_in => lin_in => x_in_l => [N, out_dim]
          2) sum of alpha_i * spmm(subband_ops[i], x_in_l)
          3) BN => ReLU => dropout
        """
        if self.lin_in is not None:
            x_in_l = self.lin_in(x_in)  # [N, out_dim]
        else:
            x_in_l = x_in  # assume we already have correct dimension

        # spmm sum
        out_sum = torch.zeros_like(x_in_l)
        for i, op in enumerate(subband_ops):
            w_i = self.alpha_params[i]
            sub_out = torch.sparse.mm(op, x_in_l)
            out_sum += w_i*sub_out

        # BN => ReLU => dropout
        out_sum = self.bn(out_sum)
        out_sum = F.relu(out_sum)
        out_sum = F.dropout(out_sum, p=self.dropout, training=self.training)

        return out_sum


###############################################################################
# A simple aggregator that does multi-head self-attn across the 3 embeddings
###############################################################################
class ThreeBranchAttnAggregator(nn.Module):
    """
    Node i has 3 embeddings: poly, local, adj => we do multi-head self-attn over these 3 'tokens'
    producing one final embedding [N, d_out].
    """
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
        # stack => [N, 3, d_in]
        X_3 = torch.stack([x_poly, x_local, x_adj], dim=1)  # [N, 3, d_in]
        N, T, d_in = X_3.shape

        heads_out=[]
        for h in range(self.n_heads):
            Q = X_3 @ self.query[h]  # [N, 3, d_in//n_heads]
            K = X_3 @ self.key[h]
            V = X_3 @ self.value[h]
            d_k= Q.shape[-1]

            attn_scores= torch.bmm(Q, K.transpose(1,2))/math.sqrt(d_k)  # [N,3,3]
            attn_weights= F.softmax(attn_scores, dim=2)                  # [N,3,3]
            out_h = torch.bmm(attn_weights, V).mean(dim=1)               # [N,d_k]
            heads_out.append(out_h)

        # concat => [N, n_heads*d_k] = [N, d_in]
        X_cat = torch.cat(heads_out, dim=1)
        X_proj= self.proj_out(X_cat)
        X_bn  = self.bn(X_proj)
        return X_bn


###############################################################################
#                             DEMO TRAINING SCRIPT                            #
###############################################################################
def run_demo_3branch_framelet(dataset_name='chameleon',
                              lap_mode='signless',
                              FrameType='Haar',
                              Lev=2,
                              poly_degree=2,
                              # dims
                              poly_hidden_dim=64,
                              local_hidden_dim=64,
                              linkx_embed_dim=64,
                              final_hidden_dim=64,
                              aggregator_heads=2,
                              dropout=0.5,
                              lr=0.005,
                              weight_decay=1e-3,
                              epochs=300,
                              advanced_rewire_flag=False,
                              device='cuda'):
    """
    Merges the ideas:
      - Framelet polynomials (multi-level operators) => 1st branch
      - Local MLP => 2nd branch
      - LinkX adjacency embeddings => 3rd branch
      - Then aggregator with multi-head attn across the 3 embeddings => final
      - signless Lap => advanced rewire => same environment for chameleon
    """
    import torch
    from torch_geometric.transforms import NormalizeFeatures

    ds_lower = dataset_name.lower()
    if ds_lower in ['cora','citeseer','pubmed']:
        path = osp.join('./data', ds_lower)
        dataset = Planetoid(root=path, name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0]
    elif ds_lower in ['chameleon','squirrel']:
        path = osp.join('./data', ds_lower)
        dataset = WikipediaNetwork(root=path, name=ds_lower, transform=NormalizeFeatures())
        data = dataset[0]
        data.train_mask = data.train_mask[:,0].bool()
        data.val_mask   = data.val_mask[:,0].bool()
        data.test_mask  = data.test_mask[:,0].bool()
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # rewiring
    if advanced_rewire_flag:
        data = advanced_rewire(data, steps=2, dist_threshold=0.4)

    # 1) Build Laplacian for framelets
    L_scipy = build_laplacian(data.edge_index, data.num_nodes, mode=lap_mode)

    # 2) choose wavelet filter
    if FrameType=='Haar':
        D1 = lambda x: np.cos(x/2.)
        D2 = lambda x: np.sin(x/2.)
        DFilters = [D1, D2]
    elif FrameType=='Linear':
        D1 = lambda x: np.cos(x/2.)**2
        D2 = lambda x: (np.sqrt(2)*np.sin(x))/2.
        D3 = lambda x: np.sin(x/2.)**2
        DFilters = [D1, D2, D3]
    elif FrameType=='Quadratic':
        D1 = lambda x: np.cos(x/2.)**3
        D2 = lambda x: np.sqrt(3)*np.cos(x/2.)**2*np.sin(x/2.)
        D3 = lambda x: np.sqrt(3)*np.cos(x/2.)*(np.sin(x/2.)**2)
        D4 = lambda x: np.sin(x/2.)**3
        DFilters = [D1,D2,D3,D4]
    else:
        raise ValueError("FrameType must be Haar, Linear, Quadratic")

    d_list = build_framelet_ops(L_scipy, DFilters, poly_degree, s=2, Lev=Lev, device=device)

    # 3) adjacency for LinkX branch
    adjacency_linkx = build_adjacency_sp(data).coalesce()

    # 4) build the 3-branch framelet model
    model = ThreeBranchFramelet(
        num_nodes=data.num_nodes,
        in_dim=data.x.size(1),
        out_dim=dataset.num_classes,
        d_list=d_list,
        poly_hidden_dim=poly_hidden_dim,
        local_hidden_dim=local_hidden_dim,
        linkx_embed_dim=linkx_embed_dim,
        final_hidden_dim=final_hidden_dim,
        aggregator_heads=aggregator_heads,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_test = 0.,0.
    best_state = None
    patience, max_patience = 0, 50

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, d_list, adjacency_linkx)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(data.x, d_list, adjacency_linkx)
        def accuracy(mask):
            p = out_eval[mask].max(dim=1)[1]
            return (p==data.y[mask]).sum().item()/mask.sum().item()

        train_acc = accuracy(data.train_mask)
        val_acc   = accuracy(data.val_mask)
        test_acc  = accuracy(data.test_mask)

        if val_acc>best_val:
            best_val = val_acc
            best_test= test_acc
            best_state = model.state_dict().copy()
            patience=0
        else:
            patience+=1

        if epoch%20==0:
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} "
                  f"train={train_acc:.2f} val={val_acc:.2f} test={test_acc:.2f} pat={patience}")

        if patience>max_patience:
            print("Early stop!")
            break

    if best_state:
        model.load_state_dict(best_state)
    print(f"\n>>> 3-branch framelet final best val={best_val:.4f}, test={best_test:.4f}\n")


if __name__=="__main__":
    # Example usage on Chameleon
    run_demo_3branch_framelet(
        dataset_name='chameleon',
        lap_mode='signless',   # or 'standard'
        FrameType='Quadratic',      # or 'Linear','Quadratic'
        Lev=3,
        poly_degree=3,
        poly_hidden_dim=64,
        local_hidden_dim=64,
        linkx_embed_dim=64,
        final_hidden_dim=64,
        aggregator_heads=2,
        dropout=0.5,
        lr=0.005,
        weight_decay=1e-3,
        epochs=300,
        advanced_rewire_flag=True,
        device='cuda'
    )