#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import networkx as nx
import numpy as np
import scipy.sparse as sp


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from datetime import date, timedelta

from os.path import isfile, join, exists
from math import ceil

import itertools
import pandas as pd
import json

from utils import generate_new_features, AverageMeter, read_meta_datasets, generate_new_batches
from utils import 
from utils import sparse_mx_to_torch_sparse_tensor as to_sparse_tensor
from models import DGNN
        
import sys

import random
random.seed(2021)
torch.manual_seed(2021)

    
def train(epoch, adj, features, y, idx):
    optimizer.zero_grad()
    output,_ = model(adj, features, idx)
    loss_train = F.mse_loss(output, y)
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train


def test(adj, features, y, idx):    
    output,(edge,att) = model(adj, features, idx)
    loss_test = F.mse_loss(output, y)
    return output, loss_test, (edge,att) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--hiddens', type=str, default="64",
                        help='Numbers of hidden units.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Size of batch.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--window', type=int, default=7,
                        help='Size of window for features.')
    parser.add_argument('--graph-window', type=int, default=7,
                        help='Size of window for graphs in MPNN LSTM.')
    parser.add_argument('--recur',  default=False,
                        help='True or False.')
    parser.add_argument('--early-stop', type=int, default=100,
                        help='How many epochs to wait before stopping.')
    parser.add_argument('--start-exp', type=int, default=15,
                        help='The first day to start the predictions.')
    parser.add_argument('--ahead', type=int, default=14,
                        help='The number of days ahead of the train set the predictions should reach.')
    parser.add_argument('--sep', type=int, default=10,
                        help='Seperator for validation and train set.')
    parser.add_argument('--shift', type=str, default="1",
                        help='The next xth day to predict.')
    parser.add_argument('--model', type=str, default="DGNN",
                        help='The model to learn.')
    parser.add_argument('--edge', type=str, default="",
                        help='The edge data to use.')
    parser.add_argument('--death', action="store_true",
                        help='Use death dataset to predict.')
    parser.add_argument('--ratio', type=int, default=0,
                        help='Ratio of entities to keep.')
    parser.add_argument('--start-test', type=int, default=0,
                        help='The start of test instance.')
    parser.add_argument('--end-test', type=int, default=329,
                        help='The end of test instance.')
    parser.add_argument('--entity-entity', action="store_true",
                        help='Parse the relations between entity and entity.')

    to_write = True
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    countries = args.country.split(",")
    task = "case" if not args.death else "death"
    meta_labs, meta_graphs, meta_features, meta_y, text_emb = read_meta_datasets(args.window, \
                                                                                task, \
                                                                                args.edge, \
                                                                                args.ratio, \
                                                                                args.entity_entity)


    sdate, edate = date(2020, 5, 15), date(2021, 4, 8)
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    dates = {i:d for i,d in enumerate(dates)}

        
    labels = meta_labs[0]
    gs_adj = meta_graphs[0]
    features = meta_features[0]
    y = meta_y[idx]
    n_samples= len(gs_adj)
    nfeat = meta_features[0][0].shape[1] 

    n_nodes = 50 
    n_state = 50 
    if not os.path.exists('../results/grid/'):
        os.makedirs('../results/grid/')
    if not os.path.exists('../models/'):
        os.makedirs('../models/')
    
    if to_write:
        fw = open("../results/grid/results_{}_{}.csv".format(task,args.model),"a")
        fw.write("shift,edge,topk,window,hidden,MAE,STD,sMAPE"+"\n")
    
    begin=time.time()

	#---- predict days ahead , 0-> next day etc.
    for shift in list([int(x)-1 for x in args.shift.split(",")]): #range(0,args.ahead), [int(args.shift)-1]
        for hidden in list([int(x) for x in args.hiddens.split(",")]):

            if to_write:
                fw = open("../results/grid/results_{}_{}.csv".format(args.model,country),"a")
                if "hashtag" in args.edge or "entity" in args.edge:
                    fw2 = open("../results/grid/results_{}_{}_{}_top{}_window{}.csv".format(task,args.model,args.edge,args.ratio,args.window),"a")
                    att_file = "../results/grid/atts_{}_{}_shift{}_{}_top{}_window{}".format(task,args.model,shift+1,args.edge,args.ratio,args.window)
                    if not os.path.exists(att_file): os.makedirs(att_file) 
                elif "state" in args.edge:
                    fw2 = open("../results/grid/results_{}_{}_{}.csv".format(task,args.model,args.edge),"a")
                    att_file = "../results/grid/atts_{}_{}_shift{}_{}".format(task,args.model,shift-1,args.edge)
                    if not os.path.exists(att_file): os.makedirs(att_file) 
                elif "GAT" in args.model or "LSTM" in args.model or "MPNN" in args.model:
                    fw2 = open("../results/grid/results_{}_{}.csv".format(task,args.model),"a")
                                       

            result,result_rel = [],[]
            exp = 0
            
            for test_sample in range(args.start_exp,n_samples-shift): 
                
                if test_sample < int(args.start_test) or test_sample > int(args.end_test): continue
                exp+=1
                print("test sample {}".format(test_sample))
                test_begin=time.time()

                if test_sample>=108:
                    idx_train = random.sample(list(range(args.window, test_sample-args.sep-10)), 80)
                    idx_train = idx_train + list(range(test_sample-args.sep-10, test_sample-args.sep))
                else: 
                    idx_train = list(range(args.window-1, test_sample-args.sep))                        

                idx_val = list(range(test_sample-args.sep,test_sample,2)) 
                idx_train = idx_train+list(range(test_sample-args.sep+1,test_sample,2))
                #print(idx_train) #[6, 8, 10, 12, 14]<-range(7-1,15-10)+range(15-9,15,2) 
                #print(idx_val) #[5, 7, 9, 11, 13]<-range(15-10,15,2)
                #print(idx_train, idx_val)


                if("DGNN" in args.model):
                    adj_train, features_train, y_train, lidx_train = generate_new_batches(gs_adj, features, y, idx_train, args.graph_window, shift, args.batch_size,device,test_sample)
                    adj_val, features_val, y_val, lidx_val = generate_new_batches(gs_adj, features, y, idx_val, args.graph_window,  shift,args.batch_size, device,test_sample)
                    adj_test, features_test, y_test, lidx_test = generate_new_batches(gs_adj, features, y, [test_sample], args.graph_window,shift, args.batch_size, device,test_sample) #state_indexes
                
                else:
                    sys.exit()

                n_train_batches = ceil(len(idx_train)/args.batch_size)
                n_val_batches = 1
                n_test_batches = 1

                #-------------------- Training
                # Model and optimizer
                stop = False#
                while(not stop):#
                    sign=exp%2
                    if "hashtag" in args.edge or "entity" in args.edge:
                        model_save_name = "../models/model_{}_{}_{}_{}_{}_{}.pth.tar".format(args.model,task,shift+1,args.edge,args.ratio,sign)
                    else:
                        model_save_name = "../models/model_{}_{}_{}_{}_{}.pth.tar".format(args.model,task,shift+1,args.edge,sign)
                    if os.path.exists(model_save_name) and exp not in [1,2]:
                        checkpoint = torch.load(model_save_name)
                        model.load_state_dict(checkpoint['state_dict'])

                    else:

                        if(args.model=="DGNN"):
                            model = DGNN(nfeat=nfeat, nhid=hidden, nout=1, n_nodes=n_nodes, window=args.graph_window, dropout=args.dropout).to(device)
                        elif(args.model=="DGNN2"):
                            model = DGNN2(nfeat=nfeat, nhid=hidden, nout=1, n_nodes=n_nodes, window=args.graph_window, dropout=args.dropout).to(device)


                    optimizer = optim.Adam(model.parameters(), lr=args.lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

                    #------------------- Train
                    best_val_acc= 1e8
                    val_among_epochs = []
                    train_among_epochs = []
                    stop = False

                    start = time.time()
                    for epoch in range(args.epochs):    

                        model.train()
                        train_loss = AverageMeter()

                        for batch in range(n_train_batches):
                            output, loss = train(epoch, adj_train[batch], \
                                                        features_train[batch], \
                                                        y_train[batch],
                                                        lidx_train[batch])
                            train_loss.update(loss.data.item(), output.size(0))


                        # Evaluate on validation set
                        model.eval()

                        output, val_loss, _ = test(adj_val[0], features_val[0], y_val[0], lidx_val[0])
                        val_loss = int(val_loss.detach().cpu().numpy())
                        if country=="US": val_loss /= (args.graph_window*n_state*5)  # only for US


                        if(epoch%50==0):
                            minute=round((time.time() - start)/60,2)
                            print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.2f}".format(train_loss.avg),"val_loss=", "{:.2f}".format(val_loss), "time=", "{:.2f}".format(minute))

                        train_among_epochs.append(train_loss.avg)
                        val_among_epochs.append(val_loss)

                        if(epoch<30 and epoch>10):
                            if(len(set([round(val_e) for val_e in val_among_epochs[-20:]])) == 1 ):
                                break

                        if(epoch>args.early_stop):
                            eps = 0.05 if task != "death" else 0.005
                            if abs(val_loss - np.mean(val_among_epochs[-50:])) <= eps:

                                if val_loss < best_val_acc:
                                    best_val_acc = val_loss
                                    print(model_save_name, "#########")
                                    torch.save({
                                        'state_dict': model.state_dict(),
                                        'optimizer' : optimizer.state_dict(),
                                    }, model_save_name)
                                break

                        if val_loss < best_val_acc:
                            best_val_acc = val_loss
                            torch.save({
                                'state_dict': model.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                            }, model_save_name)
                        stop = True
                        scheduler.step(val_loss)

                #---------------- Testing
                test_loss = AverageMeter()
                checkpoint = torch.load(model_save_name)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                model.eval()

                output, loss, (edge,att) = test(adj_test[0], features_test[0], y_test[0], lidx_test[0])
                o = output.cpu().detach().numpy()
                l = y_test[0].cpu().numpy()

                error = np.mean(abs(o-abs(l)))
                error_rel = np.mean(abs(o-abs(l))/(o+l+0.00001))

                # Print results
                result.append(error)
                result_rel.append(error_rel)
                print("test error={:.2f},{:.2f}, ".format(error, error_rel)+"mean test error={:.2f},{:.2f}".format(np.mean(result), np.mean(result_rel)))
                if to_write:
                    fw2.write("{},{},{},{:.5f},{:.5f},{:.5f},{:.5f}".format(shift+1,hidden,test_sample,error,error_rel,np.mean(result),np.mean(result_rel))+'\n')

            if to_write:
                fw.write(str(shift+1)+","+str(args.edge)+","+str(args.ratio)+","+str(args.window)+","+str(hidden)+",{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+",{:.5f}".format(np.mean(result_rel))+"\n")
                fw.close()
            print("Total time cost: {} minutes.".format(round((time.time()-begin)/60, 2)))





