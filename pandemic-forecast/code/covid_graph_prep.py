import ast
import csv
import json
import numpy as np
import pandas as pd
from multiprocessing import Process
from os import listdir, makedirs
from os.path import isfile, join, exists
from tqdm import tqdm


def build_adj(pid, infiles, outfiles, statemap, statedict):
    
    statedict_inv = {int(y): "_".join(x.lower().split(" ")) for x,y in statedict.items()}
    for (infile, outfile) in zip(infiles, outfiles):
        if "npy" not in infile: continue
        graph_file = open(outfile, 'w')
        graph_writer = csv.writer(graph_file, delimiter=',', quotechar='"')
        data = np.load(infile)
        assert len(data) == 50 and len(data[0]) == 50
        for i,row in enumerate(data):
            for j,col in enumerate(row):
                graph_writer.writerow([statedict_inv[i], statedict_inv[j], col])


def load_state(statemap_file, statedict_file):

    state2index = open(statedict_file, 'r').readline()
    statedict = json.loads(state2index.replace("\'", "\""))

    statemap = {}
    with open(statemap_file) as csvfile:
        csvfile = csv.reader(csvfile, delimiter='\t',)    
        for i,row in enumerate(csvfile):
            if i == 0: continue
            statemap[row[2]] = row[8].replace(" State", "")
    
    return statemap, statedict


def main():

    statemap_file = "./data/safegraph/states.csv"
    statedict_file = "./data/covid-mobility/state_dict.txt"
    inputfolder = "./data/covid-mobility/"
    outputfolder = "./data/graphs/"

    statemap, statedict = load_state(statemap_file, statedict_file)
    filenames_in, filenames_out = [], []
    for fi in [f for f in listdir(inputfolder)]:
        if "npy" not in infile: continue
        filenames_in += [join(inputfolder, fi)]
        filenames_out += [join(outputfolder, "US_"+fi[:10])+".csv"]
    

    n, length = len(filenames_in), len(filenames_in)
    chunks = [range(i,i+n) for i in range(0,len(filenames_in),n)]
    processes = []
    for i,ch in enumerate(chunks):
        fin, fout = [], []
        for j in ch:
            if j >= length: continue
            fin += [filenames_in[j]]
            fout += [filenames_out[j]]
        p = Process(target=build_adj, args=(i, fin, fout, statemap, statedict, ))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()