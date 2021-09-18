import torch
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pandas as pd
from math import ceil
import glob
import unidecode 
from datetime import date, timedelta
import sys
from sklearn import preprocessing
import random
import os
import json
from tqdm import tqdm
from os.path import isfile, join, exists
import csv
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
plt.switch_backend('agg')

    
random.seed(2011)
    
def read_meta_datasets(window,task,edge="",ratio=1, edge_edge=False, model="GAT"):
    os.chdir("../data")
    meta_labs = []
    meta_graphs = []
    meta_features = []
    meta_y = []


    print("processing US covid statistics...")
    os.chdir("US")
    filename = "us_labels.csv" if task != "death" else "us_labels_death.csv"
    labels = pd.read_csv(filename) 

    sdate = date(2020, 5, 15)
    edate = date(2021, 4, 8)
    
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    labels = labels.loc[:,dates]   
    
    Gs = generate_graphs_tmp(dates, "US", edge, ratio, edge_edge)
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    labels = labels.loc[list(range(50)),:]  
    
    meta_labs.append(labels)
    if "GAT" not in model:
        meta_graphs.append(gs_adj) 
    else:
        meta_graphs.append(Gs) 
    features = generate_US_features(Gs, labels, dates, edge, window)
    meta_features.append(features)
    text_emb = []
    
    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            if node >= 50: continue
            y[i].append(labels.loc[node,dates[i]])
    # print(y)  # 50*329 list of list
    meta_y.append(y)


    

def generate_graphs_tmp(dates,country,edge="",ratio=1, edge_edge=False):
    Gs = []

    state = json.load(open("edges/nodes-states.json", 'r'))
    ent_nodes = set()
    extra_edge = 0

    print("...build graphs...")
    for i in tqdm(range(len(dates))):

        date = dates[i]
        d = pd.read_csv("graphs/"+country+"_"+date+".csv",header=None)
        G = nx.DiGraph()
        date_node = set()

        # Add state nodes.
        nodes = list(range(50))  # set(d[0].unique()).union(set(d[1].unique()))
        G.add_nodes_from(nodes)

        # Add mobility edges.
        for row in d.iterrows():
            if country == "US":
                G.add_edge(state[row[1][0]], state[row[1][1]], weight=row[1][2])
            else:
                G.add_edge(row[1][0], row[1][1], weight=row[1][2])

        if "hashtag" in edge or "entity" in edge:

            filefolder= "edges/hashtag-edges/" if edge == "hashtag" else "edges/entity-edges/"
            ent_edges = json.load(open(join(filefolder, date+"-{}-edges.txt".format(edge)), 'r'))
            
            for sid in ent_edges:
                edge_small = ent_edges[sid][:int(ratio)]
                for e1, e2 in edge_small:
                    ent_nodes.add(e1); ent_nodes.add(e2)
                    if e1 >= 50: date_node.add(e1)
                    if e2 >= 50: date_node.add(e2)
                    G.add_edge(int(e2), int(e1), weight=1)  # from entity to state
                    G.add_edge(int(e1), int(e2), weight=1)

            if edge_edge == True:
                filefolder= "edges/entity-edges/" 
                ent_edges = json.load(open(join(filefolder, date+"-entity-entity-edges.txt"), 'r'))
                for e1, e2 in ent_edges:
                    if e1 in date_node and e2 in date_node:
                        G.add_edge(int(e2), int(e1), weight=1)  # from entity to entity
                        G.add_edge(int(e1), int(e2), weight=1)
                        extra_edge += 1

        
        Gs.append(G)
    print("on average, each graph has {} entity-entity edges.".format(round(extra_edge/329,2)))
    print("on average, each graph has {} entity nodes.".format(round(len(ent_nodes)/329,2)))
    return Gs


def plot_graph(G, date, ent_nodes):
    pos = nx.spring_layout(G)
    print("node_num: {}, edge_num: {}".format(G.number_of_nodes(),G.number_of_edges()))
    plt.figure(1,figsize=(15,15))
    node_color = ["red"]*50 + ["blue"]*(len(ent_nodes)-50)
    node_size = [400]*50 + [100]*(len(ent_nodes)-50)
    nx.draw(G, pos, node_size=node_size, font_size=20, edge_color='grey', node_color=node_color, with_labels=True, width=0.4)
    plt.savefig("figure-{}.png".format(date))
    plt.close()



def generate_US_features(Gs, labels, dates, edge, window=7, scaled=False):
    """
    Generate node features
    Features[1] contains the features corresponding to y[1]
    e.g. if window = 7, features[7]= day0:day6, y[7] = day7
    if the window reaches before 0, everything is 0, so features[3] = [0,0,0,0,day0,day1,day2], y[3] = day3
    """
    features = list()
    
    labs = labels.copy()
    nodes = Gs[0].nodes()
    if edge:
        if "state" in edge: emb_file = open("edges/hashtag-edges/all-states-emb.txt", 'r')
        elif "entity" in edge: emb_file = open("edges/entity-edges/all-nodes-emb.txt", 'r')
        emb = {}
        emb_reader = csv.reader(emb_file, delimiter=',')
        for row in emb_reader: emb[int(row[0])] = row[1:]
        text_feat_dim = 768  
    else:  
        text_feat_dim = 0
    
    state = json.load(open("edges/nodes-states.json", 'r'))
    state_inv = {y:x for x,y in state.items()} 

    #print(n_departments)
    print("...generate features...")
    for idx in tqdm(range(len(Gs))):
        G = Gs[idx-1]
        #  Features = population, coordinates, d past cases, one hot region
        
        H = np.zeros([G.number_of_nodes(),window+text_feat_dim]) #+3+n_departments])#])#])
        me = labs.loc[:, dates[:(idx)]].mean(1)
        sd = labs.loc[:, dates[:(idx)]].std(1)+1

        ### enumarate because H[i] and labs[node] are not aligned
        for i,node in enumerate(G.nodes()):
        
            if node < 50: 
                #---- Past cases      
                if(idx < window):
                    if(scaled):
                        H[i,(window-idx):(window)] = (labs.loc[node, dates[0:(idx)]] - me[node])/ sd[node]
                    else:
                        H[i,(window-idx):(window)] = labs.loc[node, dates[0:(idx)]]

                elif idx >= window:
                    if(scaled):
                        H[i,0:(window)] =  (labs.loc[node, dates[(idx-window):(idx)]] - me[node])/ sd[node]
                    else:
                        H[i,0:(window)] = labs.loc[node, dates[(idx-window):(idx)]]
            
            else:
                H[i,0:window] = 0
            if edge:
                H[i,window:] = emb[node]
        features.append(H)
    return features



def generate_new_batches(Gs, features, y, idx, graph_window, shift, batch_size, device, test_sample):
    """
    Generate batches for graphs for MPNN
    """

    # print("...create batches...")
    N = len(idx)
    n_nodes = Gs[0].number_of_nodes()
    n_state = 50 
  
    adj_lst = list()
    features_lst = list()
    y_lst = list()
    node_lst = list()

    batch_data = []
    for i in range(0, N, batch_size):
        if i+batch_size >= N:
            batch_data += [[idx[x] for x in range(i, N)]]
        else:
            batch_data += [[idx[x] for x in range(i, i+batch_size)]]

    for batch in batch_data:
        adj_tmp = list()
        features_tmp = list()
        y_tmp = list()
        num_tmp = list()
        line_idx = 0
        for val in batch:
            for k in range(val-graph_window+1,val+1):
                adj_tmp.append(nx.adjacency_matrix(Gs[k-1]).toarray()) 
                for feat in features[k]:
                    features_tmp.append(feat)
                num_tmp += list(range(line_idx, line_idx+50))
                line_idx += len(features[k])
            y_tmp.append(y[val+shift])

        adj_tmp = sparse_mx_to_torch_sparse_tensor(sp.block_diag(adj_tmp))
        adj_lst.append(adj_tmp.to(device))
        features_tmp = torch.FloatTensor(features_tmp)
        features_lst.append(features_tmp.to(device))
        y_tmp = torch.FloatTensor(y_tmp).reshape(-1)
        y_lst.append(y_tmp.to(device))
        node_lst.append(num_tmp)


    return adj_lst, features_lst, y_lst, node_lst



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

