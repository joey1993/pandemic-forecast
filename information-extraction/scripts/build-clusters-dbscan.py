import sys
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
import hdbscan
import csv
import json
import os
import pandas as pd
from collections import defaultdict,Counter
from tqdm import trange
import umap.umap_ as umap
from os import listdir
from os.path import isfile, join, exists


class Clustering(object):
    def __init__(self, embeddings, entities, output_folder, entity_file, emb_file, counts):
        
        self.emb = embeddings
        self.entities = entities
        self.output_folder = output_folder
        self.entity_file = entity_file
        self.emb_file = emb_file
        self.counts = counts

        self.clusters = defaultdict(dict)
        self.entity_low2high = defaultdict(dict)
        self.centroid = defaultdict(set)


    def dimension_deduction(self, dim=100):

        print("Dimension Deduction...")
        new_file = self.emb_file.replace(".txt", "{}.txt".format(dim))
        if exists(new_file):
            self.emb_low = open(new_file, 'r').readlines()
            self.emb_low = [[float(y) for y in x.split(",")] for x in self.emb_low]
        else:
            self.emb_low = umap.UMAP(n_neighbors=30, 
                            n_components=dim,  
                            metric='cosine').fit_transform(self.emb)        
            f = open(new_file, 'w')
            for emb in self.emb_low: f.write(",".join([str(x) for x in emb])+'\n')
        assert len(self.emb) == len(self.emb_low)


    def clustering(self, mode="all", hyperp=2):

        cluster_file = self.entity_file.replace(".json", "-clusters-{}-{}.json".format(mode, hyperp))
        l2h_file = self.entity_file.replace(".json", "-l2h-{}-{}.json".format(mode, hyperp))
        # if exists(cluster_file) and exists(l2h_file): return

        states = self.entities["STATES"]
        hashtags = self.entities["HASHTAG"]
        entities = self.entities["ENTITIES"]

        if mode == "hashtag": 
            emb = self.emb_low[len(states):len(states)+len(hashtags)]
            nodes = hashtags
        elif mode == "entity":
            emb = self.emb_low[len(states)+len(hashtags):]
            nodes = entities
        else:
            emb = self.emb_low[len(states):]
            nodes = hashtags + entities

        print("Run dbscan clustering algorithm on mode {} and hyperparameter {}...".format(mode, hyperp))
        db = hdbscan.HDBSCAN(min_cluster_size=hyperp,
                    metric='euclidean', core_dist_n_jobs=50,                     
                    cluster_selection_method='eom').fit(emb)

        clusters = defaultdict(list)
        # clusters_id = defaultdict(list)
        self.centroid[mode] = set()
        self.clusters[mode] = {}
        self.entity_low2high[mode] = {} 
        
        cluster_id = 0
        for idx, cid in enumerate(db.labels_):
            if cid == -1: 
                cluster_id -= 1
                clusters[cluster_id] += [nodes[idx]]
            else:
                clusters[cid] += [nodes[idx]]
            # clusters_id[cid] += [idx]

        for cid in clusters: 
            cands = clusters[cid]
            eid = max(cands, key=lambda x: self.counts[x])
            self.clusters[mode][eid] = clusters[cid]
            self.centroid[mode].add(eid)
            for lid in cands:
                self.entity_low2high[mode][lid] = eid

        self.clusters[mode] = sorted(self.clusters[mode].items(), key=lambda x: len(x[1]), reverse=True)

        json.dump(self.clusters[mode], open(cluster_file,'w'))
        json.dump(self.entity_low2high[mode], open(l2h_file,'w'))
        print("--Num of clusters for {}: {} and distribution: {}.".format(mode, max(db.labels_), \
            Counter(sorted(Counter(db.labels_).values(), reverse=True))))


    def build_nodes_features(self, mode="mix", hyperp=2):
        
        print("Build nodes and features for mode {}.".format(mode))
        tags = ["hashtag", "entity"] if mode == "separate" else ["all"]
        entity_low2high = {}  # clusters = {}
        for tag in tags:
            # cluster_file = self.entity_file.replace(".json", "-clusters-{}-{}.json".format(tag, hyperp))
            # clusters.update(json.load(open(cluster_file, 'r')))
            l2h_file = self.entity_file.replace(".json", "-l2h-{}-{}.json".format(tag, hyperp))  
            entity_low2high.update(json.load(open(l2h_file, 'r')))
            
        token_num, token2id = 0, {}
        names = ["STATES", "HASHTAG", "ENTITIES"]
        for name in names:
            for tok in self.entities[name]:
                token2id[tok] = token_num
                token_num += 1

        node2id, id2node, node_num = {}, {}, 0
        for i, name in enumerate(names):
            tag = name.lower()
            for tok in self.entities[name]:
                if name != "STATES":
                    tok = entity_low2high[tok]
                if tok in node2id: continue
                node2id[tok], id2node[node_num] = node_num, tok 
                node_num += 1

            print("--After processing mode {}, there are {} nodes.".format(name, node_num))
            node_file_name = self.entity_file.replace(".json", "-nodes-{}-{}-{}.json".format(mode, tag, hyperp))
            emb_file_name = self.entity_file.replace(".json", "-embs-{}-{}-{}.txt".format(mode, tag, hyperp))
            json.dump(node2id, open(node_file_name, 'w'))
            with open(emb_file_name, 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='"')
                for j in range(node_num):
                    emb_ind = token2id[id2node[j]]
                    writer.writerow([j]+self.emb[emb_ind])
        

if __name__ == "__main__":

    inputembfile = sys.argv[1]
    inputentityfile = sys.argv[2]
    countfile = sys.argv[3]
    outputfolder = sys.argv[4]

    # Load entities.
    print("Load entities.")
    entity_hastag = json.load(open(inputentityfile, 'r'))

    # Load Embeddings.
    print("Load embeddings.")
    embeddings = open(inputembfile,'r').readlines()
    embeddings = [[float(y) for y in x.split(",")[1:]] for x in embeddings]

    # Load counts of entities.
    print("Load counts of entities.")
    counts = json.load(open(countfile, 'r'))
    counts = counts["HASHTAG"] + counts["ENTITIES"]
    counts = {x:int(y) for x,y in counts}

    # Initialize a CL object.
    CL = Clustering(embeddings, entity_hastag, outputfolder, \
                    inputentityfile, inputembfile, counts)
    CL.dimension_deduction(dim=100)
    for hyperp in [2]:
        CL.clustering(mode="hashtag",hyperp=hyperp)
        CL.clustering(mode="entity",hyperp=hyperp)
        CL.clustering(mode="all",hyperp=hyperp)
    
    CL.build_nodes_features(mode="separate",hyperp=2)
    CL.build_nodes_features(mode="mix",hyperp=2)

