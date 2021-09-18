from numpy.lib.arraysetops import ediff1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv,GATConv
import networkx as nx
import numpy as np
import scipy.sparse as sp
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler



class DGNN(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout, nhist=7):
        super(DGNN, self).__init__()

        self.window = window
        self.n_nodes = n_nodes
        self.n_state = 50
        self.nhid = nhid
        self.nfeat = nfeat
        self.nhist = nhist
        self.conv1 = GATConv(nfeat, nhid, concat=False)
        self.conv2 = GATConv(nhid, nhid, concat=False)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.rnn1 = nn.GRU(2*nhid, nhid, 1)
        self.rnn2 = nn.GRU(nhid, nhid, 1)
        self.fc1 = nn.Linear(2*nhid+window*nhist, nhid)
        self.fc2 = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.w1 = Parameter(torch.Tensor(nhid, 1))
        self.w2 = Parameter(torch.Tensor(nhid, 1))
        self.m = nn.Softmax(dim=0)

    def forward(self, adj, x):
        lst = list()
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
        skip = x.view(-1,self.window,self.n_nodes,self.nfeat)  
        skip = skip[:,:,:self.n_state,:]
        skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.nfeat)
        skip = skip[:,:,:self.nhist]

        x, (edge,att) = self.conv1(x, adj, return_attention_weights=True)
        x = self.dropout(self.bn1(self.relu(x)))
        x2 = x.reshape(-1,self.n_nodes,self.nhid)
        x2 = x2[:,:self.n_state,:]
        x2 = x2.reshape(-1,self.nhid)
        lst.append(x2)
        x = self.dropout(self.bn2(self.relu(self.conv2(x, adj))))
        x2 = x.reshape(-1,self.n_nodes,self.nhid)
        x2 = x2[:,:self.n_state,:]
        x2 = x2.reshape(-1,self.nhid)
        lst.append(x2)

        x = torch.cat(lst, dim=1)
        x = x.view(-1, self.window, self.n_state, x.size(1))
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))
        out1, hn1 = self.rnn1(x)   
        out2, hn2 = self.rnn2(out1)
        x = torch.cat([hn1[0,:,:],hn2[0,:,:]], dim=1)

        skip = skip.reshape(skip.size(0),-1)
        x = torch.cat([x,skip], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)
        return x


class DGNN_2(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout, nhist=7):
        super(DGNN_2, self).__init__()

        self.window = window
        self.n_nodes = n_nodes
        self.n_state = 50
        self.nhid = nhid
        self.nfeat = nfeat
        self.nhist = nhist
        self.conv1 = GATConv(nfeat, nhid, concat=False)
        self.conv2 = GATConv(nhid, nhid, concat=False)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.rnn1 = nn.GRU(2*nhid, nhid, 1)
        self.rnn2 = nn.GRU(nhid, nhid, 1)
        self.fc1 = nn.Linear(2*nhid+window*nhist, nhid)
        self.fc2 = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.w1 = Parameter(torch.Tensor(nhid, 1))
        self.w2 = Parameter(torch.Tensor(nhid, 1))
        self.m = nn.Softmax(dim=0)

    def forward(self, adj, x, idx):
        lst = list()
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
        skip = x[idx]
        skip = skip.view(-1,self.window,self.n_state,self.nfeat) 
        skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.nfeat)
        skip = skip[:,:,:self.nhist]

        x, (edge,att) = self.conv1(x, adj, return_attention_weights=True)
        x = self.dropout(self.bn1(self.relu(x)))
        lst.append(x[idx])
        The second layer of GAT. (optional)
        x = self.dropout(self.bn2(self.relu(self.conv2(x, adj))))
        lst.append(x[idx])

        x = torch.cat(lst, dim=1)
        x = x.view(-1, self.window, self.n_state, x.size(1))
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))
        out1, hn1 = self.rnn1(x)   
        out2, hn2 = self.rnn2(out1)
        x = torch.cat([hn1[0,:,:],hn2[0,:,:]], dim=1)

        skip = skip.reshape(skip.size(0),-1)
        x = torch.cat([x,skip], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)
        return x, (edge,att)
    


# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)