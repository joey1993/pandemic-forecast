'''
    python3 node_edge_filter.py 0.05
    0.05: similarity threshold, 3: num of dates threshold
    --Cluster and merge nodes, filter out low-freq nodes & edges.
    --Generate merge.json which records the merging process.
    --Create a new mapping for node ids to save memory because many nodes have been removed. (important)
    --Generate new edge lists (A).
    --Generate node feature embeddings (B).
    Note: A and B are inputs to time series prediction graph models.
'''

import csv
import json
import os
import sys
from tqdm import trange
from collections import defaultdict,Counter
import umap
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
import hnswlib
import matplotlib.pyplot as plt
import pandas as pd
import hdbscan
# from sklearn.cluster import KMeans
import logging
logging.basicConfig(
    filename='node_edge_filter_cluster.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


class Filter(object):
    def __init__(self):

        # To record the old node ids and edges.
        self.nodes = {}
        self.nodes_inv = {}
        self.state_nodes = {}
        self.edges = {}

        # To build a new and small set of mappings.
        self.new_nodes = {}
        self.new_nodes_inv = {}
        self.new_nodes_num = 0

        # To visualize the merging process.
        self.merges = defaultdict(set)

        self.entity_low2high = defaultdict(dict)  # Low freq -> high freq.
        self.entity_location = {}  # Record all the nodes for each states. 
        self.emb_dict = {}

        self.entity_per_type = {"HASHTAG":{}, "ENTITY":{}}
        self.p = {}


    def load_embeddings(self, emb_file):
        with open(emb_file, 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            self.emb_dict = {int(x[0]):x[1:] for x in data}
            print("##Load embeddings of {} nodes.".format(len(self.emb_dict)))
            logging.info("##Load embeddings of {} nodes.".format(len(self.emb_dict)))

    def load_nodes(self, node_file, state_file):
        self.nodes = json.load(open(node_file, 'r'))["node2id"]
        self.nodes_inv = json.load(open(node_file, 'r'))["id2node"]
        self.nodes_inv = {int(x):y for x,y in self.nodes_inv.items()}
        self.state_nodes = json.load(open(state_file, 'r'))["id2node"]
        self.state_nodes = {int(x):y for x,y in self.state_nodes.items()}
        self.state_nodes_inv = json.load(open(state_file, 'r'))["node2id"]
        self.state_nodes_inv = {x:int(y) for x,y in self.state_nodes_inv.items()}
        print("##Load {} nodes, including {} states.".format(len(self.nodes), len(self.state_nodes)))
        logging.info("##Load {} nodes, including {} states.".format(len(self.nodes), len(self.state_nodes)))

    def load_edges(self, edge_file):
        csvfile = open(edge_file, 'r')
        self.edges = csv.reader(csvfile, delimiter=',')

    def load_entity_freq(self, entity_file, entity_per_type_file):
        # Collect all entites of each type.
        if os.path.exists(entity_per_type_file):
            print("--Entity_per_type_file found, loading...")
            logging.info("--Entity_per_type_file found, loading...")
            entity_per_type = json.load(open(entity_per_type_file, 'r'))
            for tag in entity_per_type:
                for k,v in entity_per_type[tag].items():
                    count,dates = int(v["count"]), int(v["dates"])
                    self.entity_per_type[tag][int(k)] = {"count":count, "dates":dates}
        else:
            print("--Entity_per_type_file not found, building...")
            logging.info("--Entity_per_type_file not found, building...")
            entity_per_type = {}
            all_date_entity = json.load(open(entity_file,'r'))
            for date in all_date_entity:
                nodes = all_date_entity[date]["all"]
                # Do not find similar entity for locations. 
                # Group entities except for hashtags together for convenience. 
                for tag in nodes:
                    if tag in ["LOC", "GPE"]: continue
                    entities = nodes[tag]
                    if tag != "HASHTAG": tag = "ENTITY"
                    if tag not in entity_per_type:
                        entity_per_type[tag] = defaultdict(lambda: {"count":0, "dates":set()})
                    for ent,count in entities:
                        if ent not in self.nodes: continue
                        eid = int(self.nodes[ent])
                        entity_per_type[tag][eid]["count"] += count
                        entity_per_type[tag][eid]["dates"].add(date)

            # Convert the date set to counts of dates.
            for tag in entity_per_type:
                for k,v in entity_per_type[tag].items():
                    count,dates = v["count"], len(list(v["dates"]))
                    self.entity_per_type[tag][k] = {"count":count, "dates":dates}
            json.dump(self.entity_per_type, open(entity_per_type_file,'w'))
        print("##Load entity frequency file, inlcuding {} hashtags and {} entities.".format(len(self.entity_per_type["HASHTAG"]), len(self.entity_per_type["ENTITY"])))
        logging.info("##Load entity frequency file, inlcuding {} hashtags and {} entities.".format(len(self.entity_per_type["HASHTAG"]), len(self.entity_per_type["ENTITY"])))

    def cluster_emb(self, index_path, merge_file, sim_threshold=0.01):

        # Cluster each type's entities.
        for tag in self.entity_per_type:
            
            sorted_entities = sorted(self.entity_per_type[tag].items(), key=lambda x: x[1]["count"], reverse=True)
            # Collect the embeddings of all entities in each tag.
            entity_emb, entity_ids = [], []
            for eid,_ in sorted_entities:
                eid = int(eid)
                if eid not in self.emb_dict: continue
                entity_ids += [eid]
                entity_emb += [self.emb_dict[eid]]
            print("--Num of embeddings in {}: {}.".format(tag, len(entity_emb)))
            
            # Use small world graph to look up the closest entiites and merge them. 
            if tag not in self.p:
                num_dim = len(entity_emb[0])
                # if not os.path.exists(os.path.join(index_path, "swn-"+tag)):
                print("--Building small world index for {}.".format(tag))
                p = hnswlib.Index(space='cosine', dim=num_dim)
                p.init_index(max_elements=len(entity_emb), ef_construction=200, M=16, random_seed = 100)
                p.set_num_threads(16)
                p.add_items(entity_emb, entity_ids)
                p.save_index(os.path.join(index_path, "swn-"+tag))
                # else:
                #     print("--Loading existing small world index for {}.".format(tag))
                #     p = hnswlib.Index(space='cosine', dim=num_dim)
                #     p.load_index(index_path, max_elements=len(entity_emb))
                self.p[tag] = p
            else:
                p = self.p[tag]
            
            print("--Computing similarities in type {}.".format(tag))
            for ind in trange(len(sorted_entities)):
                eid,prop = sorted_entities[ind]
                eid = int(eid)
                if eid in self.entity_low2high[tag]: continue
                if eid not in self.emb_dict: continue
                token_emb = self.emb_dict[eid]
                most_similar_word_ids, distances = p.knn_query([token_emb], k=30)

                for lid,d in zip(most_similar_word_ids[0],distances[0]):
                    
                    lid,d = int(lid), float(d)
                    e1, e2 = self.nodes_inv[eid], self.nodes_inv[lid]
                    
                    # Skip e1 itself.
                    if e2 == e1 or lid == eid: continue
                    # Skip the node if it has been clustered as the low-freq item.
                    if lid not in self.entity_per_type[tag]: continue
                    # Skip the node if it has been clustered as the high-freq item.
                    if e2 in self.merges: continue
                    # Skip the node if distance is smaller than threshold.
                    if d > sim_threshold: break
                    

                    self.merges[e1].add(e2)
                    self.entity_low2high[tag][lid] = eid
                    self.entity_per_type[tag][eid]["count"] += self.entity_per_type[tag][lid]["count"]
                    self.entity_per_type[tag][eid]["dates"] |= self.entity_per_type[tag][lid]["dates"]

                    del self.entity_per_type[tag][lid]

        print("--Num of clusters: {} and cluster distribution: {}".format(len(self.merges), sorted([len(list(v))+1 for k,v in self.merges.items()], reverse=True)))        
        print("--Num of hashtags and entities left are {} and {}, respectively.".format(len(self.entity_per_type["HASHTAG"]), \
                                                                                        len(self.entity_per_type["ENTITY"])))
        print("--Preserve the low->high mappings for {} nodes.".format(len(self.entity_low2high["HASHTAG"])+len(self.entity_low2high["ENTITY"])))

        set_default = lambda obj: list(obj) if isinstance(obj, set) else obj
        json.dump(self.merges, open(merge_file.format(sim_threshold),'w'), default=set_default)

    def cluster_emb_2(self, index_path, merge_file, sim_threshold=0.01):

        # Cluster each type's entities.
        for tag in self.entity_per_type:            
            sorted_entities = sorted(self.entity_per_type[tag].items(), key=lambda x: x[1]["count"], reverse=True)
            # Collect the embeddings of all entities in each tag.
            entity_emb, entity_ids = [], []
            for eid,_ in sorted_entities:
                eid = int(eid)
                if eid not in self.emb_dict: continue
                entity_ids += [eid]
                entity_emb += [self.emb_dict[eid]]
            print("--Num of embeddings in {}: {}.".format(tag, len(entity_emb)))
            
            # Use small world graph to look up the closest entiites and merge them. 
            if tag not in self.p:
                num_dim = len(entity_emb[0])
                print("--Building small world index for {}.".format(tag))
                p = hnswlib.Index(space='cosine', dim=num_dim)
                p.init_index(max_elements=len(entity_emb), ef_construction=200, M=16, random_seed = 100)
                p.set_num_threads(16)
                p.add_items(entity_emb, entity_ids)
                p.save_index(os.path.join(index_path, "swn-"+tag))
                self.p[tag] = p
            else:
                p = self.p[tag]
            
            print("--Computing similarities in type {}.".format(tag))
            graph = defaultdict(set)
            orders = [x[0] for x in sorted_entities]
            for i, (eid,_) in enumerate(sorted_entities):
                eid = int(eid)
                token_emb = self.emb_dict[eid]
                most_similar_word_ids, distances = p.knn_query([token_emb], k=30)

                for lid,dist in zip(most_similar_word_ids[0],distances[0]):
                    
                    lid,dist = int(lid), float(dist)                    
                    if lid == eid: continue
                    # Skip the node if distance is smaller than threshold.
                    if dist > sim_threshold: break
                    graph[eid].add(lid)
                    graph[lid].add(eid)

            queue = [orders[0]]
            orders = orders[1:]
            seen = set()
            clusters, num_cluster = defaultdict(list), 0
            while queue or orders:
                if queue == []:
                    queue = [orders[0]]
                    orders = orders[1:]
                    num_cluster += 1
            
                eid = queue.pop()
                if eid in seen: continue
                seen.add(eid)
                clusters[num_cluster].append(eid)
                for lid in list(graph[eid]):
                    queue += [lid]
            
            for _,nodes in clusters.items():
                eid = nodes[0]
                for lid in nodes[1:]:
                    e1, e2 = self.nodes_inv[eid], self.nodes_inv[lid]
                    if e1 == e2: continue
                    
                    self.merges[e1].add(e2)
                    self.entity_low2high[tag][lid] = eid
                    self.entity_per_type[tag][eid]["count"] += self.entity_per_type[tag][lid]["count"]
                    self.entity_per_type[tag][eid]["dates"] |= self.entity_per_type[tag][lid]["dates"]

                    del self.entity_per_type[tag][lid]

        print("--Num of clusters: {} and distribution: {}.".format(num_cluster+1, sorted([len(v) for k,v in clusters.items()], reverse=True)))
        print("--Num of hashtags and entities left are {} and {}, respectively.".format(len(self.entity_per_type["HASHTAG"]), \
                                                                                        len(self.entity_per_type["ENTITY"])))
        print("--Preserve the low->high mappings for {} nodes.".format(len(self.entity_low2high["HASHTAG"])+len(self.entity_low2high["ENTITY"])))

        set_default = lambda obj: list(obj) if isinstance(obj, set) else obj
        json.dump(self.merges, open(merge_file.format(sim_threshold),'w'), default=set_default)


    def entity_emb_dimension_deduction(self):

        print("--Running dimension deduction...")
        self.umap_entity_emb = {}
        self.umap_data = {}
        self.entity_ids = defaultdict(list)
        for tag in self.entity_per_type:
            
            sorted_entities = sorted(self.entity_per_type[tag].items(), key=lambda x: x[1]["count"], reverse=True)
            # Collect the embeddings of all entities in each tag.
            entity_emb = []
            for eid,_ in sorted_entities:
                eid = int(eid)
                if eid not in self.emb_dict: continue  ###
                self.entity_ids[tag] += [eid]
                entity_emb += [self.emb_dict[eid]]
            print("--Num of embeddings in {}: {}.".format(tag, len(entity_emb)))
            logging.info("--Num of embeddings in {}: {}.".format(tag, len(entity_emb)))

            self.umap_entity_emb[tag] = umap.UMAP(n_neighbors=30, 
                                        n_components=100, 
                                        metric='cosine').fit_transform(entity_emb)

            self.umap_data[tag] = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(entity_emb)
            

    def cluster_emb_algo(self, index_path, merge_file, hyperp, algo="hdbscan"):
        
        self.clusters = {}
        # Reset all new node mapping.
        self.new_nodes_num = 0
        self.entity_low2high = defaultdict(dict)
        self.centroid = set()
        for tag in self.entity_per_type: 

            if algo == "hdbscan":
                print("--Running dbscan clustering algorithm...")
                cluster = hdbscan.HDBSCAN(min_cluster_size=hyperp,
                            metric='euclidean', core_dist_n_jobs=32,                     
                            cluster_selection_method='eom').fit(self.umap_entity_emb[tag])

            elif algo == "kmeans":
                print("--Running kmeans clustering algorithm...")
                cluster = KMeans(n_clusters=hyperp, random_state=0, verbose=2, n_jobs=16).fit(self.umap_entity_emb[tag])

            result = pd.DataFrame(self.umap_data[tag], columns=['x', 'y'])
            result['labels'] = cluster.labels_

            # Visualize clusters
            fig, ax = plt.subplots(figsize=(40, 20))
            outliers = result.loc[result.labels == -1, :]
            clustered = result.loc[result.labels != -1, :]
            plt.scatter(outliers.x, outliers.y, color='#e8e8e8', s=0.1)
            plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.20, cmap='hsv_r')
            plt.colorbar()
            plt.savefig(index_path+"clusters_{}_{}_{}".format(algo, tag, hyperp))
            plt.close()

            print("--Num of clusters for {}: {} and distribution: {}.".format(tag, max(cluster.labels_), \
                Counter(sorted(Counter(cluster.labels_).values(), reverse=True))))
            logging.info("--Num of clusters for {}: {} and distribution: {}.".format(tag, max(cluster.labels_), \
                Counter(sorted(Counter(cluster.labels_).values(), reverse=True))))

            clusters = defaultdict(list)
            clusters_id = defaultdict(list)
            for idx, cid in enumerate(cluster.labels_):
                if cid == -1: continue
                clusters[cid] += [self.nodes_inv[self.entity_ids[tag][idx]]]
                clusters_id[cid] += [self.entity_ids[tag][idx]]

            for cid in clusters_id:                
                cands = clusters_id[cid]
                eid = max(cands, key=lambda x: self.entity_per_type[tag][x]["count"])
                self.clusters[self.nodes_inv[eid]] = clusters[cid]
                self.centroid.add(eid)
                for lid in cands:
                    self.entity_low2high[tag][lid] = eid
                    if lid == eid: continue
                    self.entity_per_type[tag][eid]["count"] += self.entity_per_type[tag][lid]["count"]
                    self.entity_per_type[tag][eid]["dates"] |= self.entity_per_type[tag][lid]["dates"]
                    del self.entity_per_type[tag][lid]

        self.clusters = sorted(self.clusters.items(), key=lambda x: len(x[1]), reverse=True)
        json.dump(self.clusters, open(merge_file.format(hyperp),'w'))
        # print("--Num of hashtags and entities left are {} and {}, respectively.".format(len(self.entity_per_type["HASHTAG"]), \
        #                                                                                 len(self.entity_per_type["ENTITY"])))
        print("--Preserve the low->high mappings for {} nodes.".format(len(self.entity_low2high["HASHTAG"])+len(self.entity_low2high["ENTITY"])))

        # logging.info("--Num of hashtags and entities left are {} and {}, respectively.".format(len(self.entity_per_type["HASHTAG"]), \
        #                                                                                 len(self.entity_per_type["ENTITY"])))
        logging.info("--Preserve the low->high mappings for {} nodes.".format(len(self.entity_low2high["HASHTAG"])+len(self.entity_low2high["ENTITY"])))


    def select_edges(self, edge_file, threshold=2):

        print("##Select edges from the large edge set from {}.".format(edge_file.split("/")[-1]))
        logging.info("##Select edges from the large edge set from {}.".format(edge_file.split("/")[-1]))
        self.load_edges(edge_file)
        self.pruned_node = set()
        new_edges, count = set(), 0
        for sr, tg, w, t in self.edges:
            count += 1
            sr, tg, w, t = int(sr), int(tg), int(w), int(t)
            new_ids = []
            for eid in [sr, tg]:
                if eid in self.state_nodes:  # Green light to every state node. 
                    new_ids += [eid]
                else:
                    ent = self.nodes_inv[eid]
                    if ent[0] == "#" and eid in self.entity_low2high["HASHTAG"]: 
                        eid = self.entity_low2high["HASHTAG"][eid]
                    if ent[0] != "#" and eid in self.entity_low2high["ENTITY"]: 
                        eid = self.entity_low2high["ENTITY"][eid]
                    if eid in self.centroid:
                        new_ids += [eid]
                
            if len(new_ids) == 2 and new_ids[0] != new_ids[1]:
                self.pruned_node.update({new_ids[0], new_ids[1]})
                new_edges.add((new_ids[0], new_ids[1], w, t))

        # Create new node id set.
        new_edges = self.id_convert(new_edges)

        # Set the wrting paths.
        new_edge_file = os.path.basename(edge_file).replace(".csv", "-{}.csv".format(threshold))
        new_edge_file = os.path.join(filter_folder, new_edge_file)
        new_node_file = new_edge_file.replace(".csv", "-node.json")

        # Write mappings betweet nodes and ids.
        json.dump({"node2id":self.new_nodes_inv, "id2node":self.new_nodes}, open(new_node_file, 'w'))

        # Write converted and filtered new edges.
        with open(new_edge_file, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"')
            new_edges = sorted(list(new_edges), key=lambda x: (x[-1], x[0], x[1]))
            for edge in new_edges: writer.writerow(edge)

        print("--Reduce num of edges from {} to {}.".format(count, len(new_edges)))
        logging.info("--Reduce num of edges from {} to {}.".format(count, len(new_edges)))



    def id_convert(self, edges):

        # self.new_nodes_inv: node->id
        # self.new_nodes    : id->node
        # self.nodes        : node->id
        # self.nodes_inv    : id->node

        new_edges = []
        if self.new_nodes_num == 0:
            self.new_nodes = {x:y for x,y in self.state_nodes.items()}
            self.new_nodes_num = len(self.state_nodes)
            self.new_nodes_inv = {x:y for x,y in self.state_nodes_inv.items()}
        
        for edge in sorted(list(edges), key=lambda x: (x[-1], x[0], x[1])):
            new_edge = []
            for n in edge[:2]:
                # n is the id in old mappings.
                ent = self.nodes_inv[n]
                if ent not in self.new_nodes_inv:
                    self.new_nodes[self.new_nodes_num] = ent
                    self.new_nodes_inv[ent] = self.new_nodes_num
                    self.new_nodes_num += 1
                new_edge += [self.new_nodes_inv[self.nodes_inv[n]]]
            new_edge += edge[2:]
            new_edges += [new_edge]
        print("--Max new node id is {}".format(self.new_nodes_num))
        logging.info("--Max new node id is {}".format(self.new_nodes_num))
        print("--Num of new edges is {}".format(len(new_edges)))
        logging.info("--Num of new edges is {}".format(len(new_edges)))

        return new_edges

    
    def generate_node_features(self, edge_file, filter_folder, v1):

        # self.pruned_node = sorted(list(self.pruned_node))
        # Set the writing path. 
        feature_file = os.path.basename(edge_file).replace(".csv", "-{}-node-features.csv".format(v1))
        feature_file = os.path.join(filter_folder, feature_file)

        self.new_nodes = sorted(self.new_nodes.items(), key=lambda x: x[0])
        count = 0
        with open(feature_file, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for node_new_id, node in self.new_nodes:
                node_old_id = self.nodes[node]
                if node_old_id not in self.emb_dict: continue  ###
                count += 1
                writer.writerow([node_new_id]+self.emb_dict[node_old_id])
        self.new_nodes = dict(self.new_nodes)
        print("##Write node features for {} nodes.".format(count))
        logging.info("##Write node features for {} nodes.".format(count))


if __name__ == "__main__":

    # To read.
    entity_file = sys.argv[1]   # "/local/yz/nsf-covid/CORD-NER/tweets-event-loc-small-v2.json"
    emb_file = sys.argv[2]      # "/local/yz/nsf-covid/CORD-NER/MLM-model/embeddings/emb.txt"

    initial_folder = sys.argv[3]
    node_file = os.path.join(initial_folder, "nodes.json")
    state_file = os.path.join(initial_folder, "state-nodes.json")

    edge_file_1 = os.path.join(initial_folder, "edge-adjacency-mobility-hashtag.csv")
    edge_file_2 = os.path.join(initial_folder, "edge-adjacency-mobility-hashtag-entity.csv")

    # To write.
    filter_folder = sys.argv[4]
    cluster_path = sys.argv[5]
    entity_per_type_file = os.path.join(filter_folder, "entity-per-type-file.json")
    merge_file = os.path.join(filter_folder, "clusters-{}.json")

    NEF = Filter()
    NEF.load_embeddings(emb_file)
    NEF.load_nodes(node_file, state_file)
    NEF.load_entity_freq(entity_file, entity_per_type_file)
    NEF.entity_emb_dimension_deduction()    

    for v1 in [2,3,4,5,6]:  # 5
        print("New Start! Dbscan hyperparameter: {}".format(v1))
        NEF.load_entity_freq(entity_file, entity_per_type_file)
        NEF.cluster_emb_algo(cluster_path, merge_file, hyperp=v1, algo="hdbscan")
        for edge_file in [edge_file_1, edge_file_2]:
            NEF.select_edges(edge_file, filter_folder, v1)
            NEF.generate_node_features(edge_file, filter_folder, v1)
