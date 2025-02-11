import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import get_laplacian
from torch_geometric.nn import GATConv
import math, functools
import argparse
import os.path as osp

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU, GELU, ELU, Dropout


# function for pre-processing
@torch.no_grad()
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row)
    col = torch.tensor(A.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)

    return torch.sparse_coo_tensor(index, value, A.shape)


# function for pre-processing
def ChebyshevApprox(f, n):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)

    return c


# function for pre-processing
def get_operator(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]

    return d


def framelets(dataset):
    num_nodes = dataset.num_nodes
    L = get_laplacian(dataset.edge_index, num_nodes=num_nodes, normalization='sym')
    L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

    lobpcg_init = np.random.rand(num_nodes, 1)
    lambda_max, _ = lobpcg(L, lobpcg_init, maxiter=50)
    lambda_max = lambda_max[0]

  
    # lambda_max = 4000

    ## FrameType = 'Haar'
    FrameType = 'Haar'
    if FrameType == 'Haar':
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
        RFilters = [D1, D2]
    elif FrameType == 'Linear':
        D1 = lambda x: np.square(np.cos(x / 2))
        D2 = lambda x: np.sin(x) / np.sqrt(2)
        D3 = lambda x: np.square(np.sin(x / 2))
        DFilters = [D1, D2, D3]
        RFilters = [D1, D2, D3]
    elif FrameType == 'Quadratic':  # not accurate so far
        D1 = lambda x: np.cos(x / 2) ** 3
        D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
        D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
        D4 = lambda x: np.sin(x / 2) ** 3
        DFilters = [D1, D2, D3, D4]
        RFilters = [D1, D2, D3, D4]
    else:
        raise Exception('Invalid FrameType')

    Lev = 2  # level of transform
    s = 2  # dilation scale
    n = 2  # n - 1 = Degree of Chebyshev Polynomial Approximation
    J = np.log(lambda_max / np.pi) / np.log(s) + Lev - 1  # dilation level to start the decomposition
    r = len(DFilters)

    # get matrix operators
    d = get_operator(L, DFilters, n, s, J, Lev)
    # enhance sparseness of the matrix operators (optional)
    # d[np.abs(d) < 0.001] = 0.0
    # store the matrix operators (torch sparse format) into a list: row-by-row
    d_list = list()
    for l in range(Lev):
        for i in range(r):
            d_list.append(scipy_to_torch_sparse(d[i, l]))
    return d_list




class UFGLevel(MessagePassing):
    def __init__(self, in_channels, out_channels, init_scale=None, dropout_prob=0.5, atten=False, if_filter=True, channel_mix=False):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        # self.linear = torch.nn.Linear(in_channels, out_channels)
        
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
         
        self.conv = GATConv(in_channels, out_channels, heads=1, dropout=dropout_prob)
        
        
        
    def forward(self, x, edge_index, edge_attr, edge_index_o=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [2, E], i.e. the ajacent matrix at one single level
        # Step 1: Linearly transform node feature matrix.
        if self.channel_mix:
            x = self.linear(x)
        # if self.atten:
        #     x = self.conv(self.propagate(edge_index, x=x, edge_attr=edge_attr), edge_index_o)
        #     return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # else:
        if self.init_scale:
            return self.propagate(edge_index, x=x, edge_attr=edge_attr) * self.filter
        else:
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        

    def message(self, x_j, edge_attr):
        # x_j has shape [E, out_channels]

        # Calculate the framelets coeff.     
        return edge_attr.view(-1, 1) * x_j
    
    
    

class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, d_list, dropout_prob=0.5, shortcut=True, levelMixer='sum', alpha_init=0.4):
        super(Net, self).__init__()
         
        self.shortcut = shortcut
        self.levelMixer = levelMixer
        self.alpha_train = nn.Parameter(torch.tensor(alpha_init))
        
        self.edge_index_list, self.edge_attr_list = [], []
        for i in range(1, len(d_list)):
            self.register_buffer('edge_index_'+str(i), d_list[i].coalesce().indices())
            self.register_buffer('edge_attr_'+str(i), d_list[i].coalesce().values())
            self.edge_index_list.append(getattr(self, 'edge_index_' + str(i))) 
            self.edge_attr_list.append(getattr(self, 'edge_attr_' + str(i))) 
            
        # if levelMixer=='sum':
        # self.ufg_list1 = nn.ModuleList([UFGLevel(nhid, nhid, init_scale=0.7*i, atten=False, dropout_prob=dropout_prob, if_filter=True) for i in range(1,len(d_list))])
        self.ufg_list1 = nn.ModuleList([UFGLevel(nhid, nhid, init_scale=1, atten=False, dropout_prob=dropout_prob, if_filter=False) for i in range(1,len(d_list))])
        self.ufg_list2 = nn.ModuleList([UFGLevel(nhid, nhid, init_scale=1, atten=False, dropout_prob=dropout_prob, if_filter=False) for i in range(1,len(d_list))])
        self.ufg_list3 = nn.ModuleList([UFGLevel(nhid, nhid, init_scale=1, atten=False, dropout_prob=dropout_prob, if_filter=False) for i in range(1,len(d_list))])

        self.mlp1 = Seq(
                       ReLU(),
                       Dropout(dropout_prob),
                       Linear(nhid, num_classes)
                        )
        self.mlp2 = Seq(
                       ReLU(),
                       Dropout(dropout_prob),
                       Linear(3*nhid, nhid)
                        )
        self.mlp3 = Seq(
                       GELU(),
                       Dropout(dropout_prob),
                       Linear(3*nhid, nhid)
                        )
        self.mlp = Seq(
                        ELU(),
                        Dropout(dropout_prob),
                        )
        self.linear1 = Linear(num_features, nhid)
        self.linear = Linear(3*num_classes, num_classes)
        self.linear2 = Linear(nhid, num_classes)

    def forward(self, x, edge_index_o=None):
        # x has shape [num_nodes, num_input_features]                      
        x = self.linear1(x)  
        if self.levelMixer=='sum':
            x += torch.sigmoid(self.alpha_train) * functools.reduce(lambda x, y: x + y, [ufg(x, edge_index, edge_attr, edge_index_o) for 
                       edge_index, edge_attr, ufg in zip(self.edge_index_list, self.edge_attr_list, self.ufg_list1)])
            # x = self.mlp(x)
            # x = functools.reduce(lambda x, y: x + y, [ufg(x, edge_index, edge_attr, edge_index_o) for 
            #            edge_index, edge_attr, ufg in zip(self.edge_index_list, self.edge_attr_list, self.ufg_list2)])
            # x = self.mlp(x)
            x += torch.sigmoid(self.alpha_train) * functools.reduce(lambda x, y: x + y, [ufg(x, edge_index, edge_attr, edge_index_o) for 
                       edge_index, edge_attr, ufg in zip(self.edge_index_list, self.edge_attr_list, self.ufg_list2)])
            x += torch.sigmoid(self.alpha_train) * functools.reduce(lambda x, y: x + y, [ufg(x, edge_index, edge_attr, edge_index_o) for 
                       edge_index, edge_attr, ufg in zip(self.edge_index_list, self.edge_attr_list, self.ufg_list3)])
           
      
            x = self.mlp1(x)
                          
        elif self.levelMixer=='mlp':                          
            x = torch.cat([ufg(x, edge_index, edge_attr) for 
                           edge_index, edge_attr, ufg in zip(self.edge_index_list, self.edge_attr_list, self.ufg_list2)], dim=1)
            # x = torch.cat([ufg(x, edge_index, edge_attr) for edge_index, edge_attr, ufg in self.combo1], dim=1)
            # shortcut = x
            # x = F.relu(shortcut + self.mlp(x))
            x = self.mlp2(x)  #+ self.mlp2(shortcut)

            if self.shortcut:
                shortcut = x

            # # x = self.linear1(self.mlp(x))
            x = torch.cat([ufg(x, edge_index, edge_attr) for 
                           edge_index, edge_attr, ufg in zip(self.edge_index_list, self.edge_attr_list, self.ufg_list2)], dim=1)
            x = self.mlp3(x) + shortcut    
            # x = torch.cat([ufg(x, edge_index, edge_attr) for 
            #                edge_index, edge_attr, ufg in zip(self.edge_index_list, self.edge_attr_list, self.ufg_list3)], dim=1)

            x = self.mlp1(x)
        else: raise Exception('Invalid Level Mixer')
                          
        return F.log_softmax(x, dim=1)
    
    
    
def objective(learning_rate = 0.01, weight_decay = 0.01, nhid = 32, dropout_prob=0.5, levelMixer='mlp', NormalizeFeatures=False, alpha_init=0.4, dataname='Cora'):
    
    torch.manual_seed(0)

    # Training on CPU/GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # load dataset
 
    rootname = osp.join('/home/liux0t/Xinliang/GFA', 'data', dataname)
    if NormalizeFeatures:
        dataset = Planetoid(root=rootname, name=dataname, transform=T.NormalizeFeatures()) #
    else:
        dataset = Planetoid(root=rootname, name=dataname)

    num_nodes = dataset[0].x.shape[0]
    L = get_laplacian(dataset[0].edge_index, num_nodes=num_nodes, normalization='sym')
    L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

    lobpcg_init = np.random.rand(num_nodes, 1)
    lambda_max, _ = lobpcg(L, lobpcg_init, maxiter=50)
    lambda_max = lambda_max[0]

    ## FrameType = 'Haar'
    FrameType = 'Haar'
    if FrameType == 'Haar':
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
        RFilters = [D1, D2]
    elif FrameType == 'Linear':
        D1 = lambda x: np.square(np.cos(x / 2))
        D2 = lambda x: np.sin(x) / np.sqrt(2)
        D3 = lambda x: np.square(np.sin(x / 2))
        DFilters = [D1, D2, D3]
        RFilters = [D1, D2, D3]
    elif FrameType == 'Quadratic':  # not accurate so far
        D1 = lambda x: np.cos(x / 2) ** 3
        D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
        D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
        D4 = lambda x: np.sin(x / 2) ** 3
        DFilters = [D1, D2, D3, D4]
        RFilters = [D1, D2, D3, D4]
    else:
        raise Exception('Invalid FrameType')

    Lev = 2  # level of transform
    s = 2  # dilation scale
    n = 2  # n - 1 = Degree of Chebyshev Polynomial Approximation
    J = np.log(lambda_max / np.pi) / np.log(s) + Lev - 1  # dilation level to start the decomposition
    r = len(DFilters)

    # get matrix operators
    d = get_operator(L, DFilters, n, s, J, Lev)
    # enhance sparseness of the matrix operators (optional)
    # d[np.abs(d) < 0.001] = 0.0
    # store the matrix operators (torch sparse format) into a list: row-by-row
    d_list = list()
    for l in range(Lev):
        for i in range(r):
            d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))

    # extract the data
    data = dataset[0].to(device)
    data.x = data.x.float()
    # create result matrices
    num_epochs = 500
    num_reps = 5
    epoch_loss = dict()
    epoch_acc = dict()
    epoch_loss['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['test_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['test_mask'] = np.zeros((num_reps, num_epochs))
    saved_model_val_acc = np.zeros(num_reps)
    saved_model_test_acc = np.zeros(num_reps)



    # Hyper-parameter Settings
    

    # initialize the learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # training
    for rep in range(num_reps):
        # print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0

        # initialize the model
        model = Net(dataset.num_features, nhid, dataset.num_classes, d_list, dropout_prob=dropout_prob, levelMixer=levelMixer, alpha_init=alpha_init).to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # evaluation mode
            model.eval()
            out = model(data.x, data.edge_index)
            for i, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = out[mask].max(dim=1)[1]
                correct = float(pred.eq(data.y[mask]).sum().item())
                e_acc = correct / mask.sum().item()
                epoch_acc[i][rep, epoch] = e_acc

                e_loss = F.nll_loss(out[mask], data.y[mask])
                epoch_loss[i][rep, epoch] = e_loss

                # scheduler.step(epoch_loss['val_mask'][rep, epoch])

                # print out results
            # print('Epoch: {:3d}'.format(epoch + 1),
            #   'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
            #   'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
            #   'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
            #   'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
            #   'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
            #   'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))

            if epoch_acc['val_mask'][rep, epoch] > max_acc:
                # torch.save(model.state_dict(), args.filename + '.pth')
                # print('=== Model saved at epoch: {:3d}'.format(epoch + 1))
                max_acc = epoch_acc['val_mask'][rep, epoch]
                record_test_acc = epoch_acc['test_mask'][rep, epoch]

        saved_model_val_acc[rep] = max_acc
        saved_model_test_acc[rep] = record_test_acc
        # print('#### Rep {0:2d} Finished! val acc: {1:.4f}, test acc: {2:.4f} ####\n'.format(rep + 1, max_acc, record_test_acc))
#    print('***************************************************************************************************************************')
    print('Average test accuracy over {0:2d} reps: {1:.4f} with stdev {2:.4f}'.format(num_reps, np.mean(saved_model_test_acc), np.std(saved_model_test_acc)))
    
    return np.mean(saved_model_test_acc)




def ray_train():
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler



    class ExperimentTerminationReporter(CLIReporter):
        def should_report(self, trials, done=True):
            """Reports only on experiment termination."""
            return done

    def training_function(config):
        # Hyperparameters
        # modes, width, learning_rate = config["modes"], config["width"], config["learning_rate"]
    #     for step in range(10):
        # Iterative training function - can be any arbitrary training procedure.
        # intermediate_score = fourier_2d_hopt_train.objective(modes, width, learning_rate)
        # Feed the score back back to Tune.
        acc = objective(**config)
        tune.report(acc=acc)

    ray.shutdown()
    ray.init(num_cpus=8, num_gpus=4) 

    asha_scheduler = ASHAScheduler(
        # time_attr='training_iteration',
        metric='loss',
        mode='min',
        max_t=500,
        grace_period=100,
        reduction_factor=2,
        brackets=1)

    
    analysis = tune.run(
        training_function,
        config={
                "dataname": tune.grid_search(['CiteSeer', 'Cora', 'PubMed']), 
                "levelMixer": tune.grid_search(['sum']),
                "learning_rate": tune.grid_search([0.01, 0.02, 0.05, 0.005]),
                "weight_decay": tune.grid_search([5e-3, 1e-2, 2e-2]),

                "alpha_init": tune.grid_search([0.4, 0.3, 0.6,]),
                "nhid" : tune.grid_search( [32, 48, 24, 64]),
                "dropout_prob": tune.grid_search([0.5, 0.6, 0.4, 0.7]), 
                # "tqdm_disable": tune.grid_search(True),

                },
        progress_reporter=ExperimentTerminationReporter(),
        resources_per_trial={'gpu': 1, 'cpu': 2},
        # scheduler=asha_scheduler
        )

    print("Best config: ", analysis.get_best_config(
        metric="acc", mode="max"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    
    
if __name__ == '__main__':
    ray_train()
    



        

        