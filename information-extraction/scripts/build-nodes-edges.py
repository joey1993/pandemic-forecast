"""
    `python build_entity_edges.py`
Build Edges from Named Entity Recognition and Relation Classiification.
    1. Build edges with map.
    2. Check the NER results for connecting locations, hashtags and entities.
    3. Read relations from eveyday's relation predictions.
    4*. Filter out insignificant entities based on the NER statistics.

Problems:
    1. Haven't set a large threshold to remove insignificant edges or nodes.
    2. Did not consider transition insider states, since numbers are too large. Hard to normalize.
"""


from collections import Counter, defaultdict
import csv
import json
from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
import sys
from tqdm import tqdm


class Build_Edge(object):

    def __init__(self, uniform_weight, nodes, states, hashtagl2h, entityl2h):
        
        self.uniform_weight = uniform_weight
        self.nodes = nodes
        self.state_nodes = states
        self.node_num = len(nodes)
        self.hashtag_l2h = hashtagl2h
        self.entity_l2h = entityl2h
        self.edges = {}  # {time: (node1, node2): weight}
        self.state_edges = {}
        self.location_mapping = {}
        self.dates = {}
    
    def build_location_nodes(self, map_file, state_adjacency_file):

        # Load maps.
        map_csv = open(map_file, 'r').readlines()
        mapping = {}
        for line in map_csv[1:]:
            locations = line.rstrip().replace('\n', '').split('|')
            if len(locations) < 5: continue
            # Data format: "City|State short|State full|County|City alias".
            # Build mapping from every level to State full.
            mapping[locations[1].lower()] = locations[2].lower()

        # Build edges according to adjacency map to initiate each date's counter.
        adj_csv = open(state_adjacency_file, 'r').readlines()
        for line in adj_csv[1:]:
            states = line.rstrip().replace('\n', '').split(',')
            s1 = self.nodes[mapping[states[0].lower()]]
            s2 = self.nodes[mapping[states[1].lower()]]
            self.state_edges[(s1, s2)] = 0
            self.state_edges[(s2, s1)] = 0
            

    def build_adjacency_edges(self, re_pred_folder):

        print("****Build Edges for Adjacency.****")      
        pred_files = [f for f in listdir(re_pred_folder) if isfile(join(re_pred_folder, f))]
        pred_files.sort()
        for pfile in pred_files:
            date = pfile[6:-4].split("-")
            date = "-".join([date[-1], date[0], date[1]])    
            if date not in self.edges:
                self.edges[date] = Counter()
                self.edges[date].update(self.state_edges)


    def build_mobility_edges(self, mobility_data_folder, re_pred_folder):
        def create_file_name(date):
            return date+".npy"

        print("****Build Edges for Mobility.****")        
        pred_files = [f for f in listdir(re_pred_folder) if isfile(join(re_pred_folder, f))]
        pred_files.sort()

        for pfile in tqdm(pred_files):
            date = pfile[6:-4].split("-")
            date = "-".join([date[-1], date[0], date[1]]) 

            mob_file_name = create_file_name(date)
            mob_file = join(mobility_data_folder, mob_file_name)
            if exists(mob_file):
                adj_matrix = np.load(mob_file)
                for i in range(len(adj_matrix)):
                    adj_matrix[i][i] = 0
                    sum_row = sum(adj_matrix[i])
                    for j in range(len(adj_matrix[0])):
                        # Normalize the transition number. 
                        norm_value = int(adj_matrix[i][j]/float(sum_row)*10)
                        if norm_value:
                            # Use negative value to differentiate the mobility edges from others.
                            self.edges[date][(i, j)] = -norm_value


    def build_hashtag_edges(self, ner_tweet_folder, ner_pred_folder):

        print("****Build Edges for Hashtags and Locations.****")
        
        tweet_files = [f for f in listdir(ner_tweet_folder) if isfile(join(ner_tweet_folder, f))]
        pred_files = [f for f in listdir(ner_pred_folder) if isfile(join(ner_pred_folder, f))]
        tweet_files.sort()

        error = 0
        errors = set([b'\xe2\x81\xa6\xe2\x81\xa9', b'\xe2\x80\xaf', b'\xef\xb8\x8f', b'\xc2\xa0', b'\xe2\x81\xa6', \
                b'\xe2\x81\xa9', b'\xe2\x80\x89', b'\xe2\x80\x8b', b'\xc2\xa0\xc2\xa0', b'\xe2\x81\xa6\xe2\x81\xa6',
                b'\xe2\x81\xa6\xe2\x81\xa6\xe2\x81\xa6'])
        tags = ["LOC", "GPE", "DATE", "PRODUCT"]
        self.entity_loc = defaultdict(lambda: defaultdict(dict))

        for tfile in tqdm(tweet_files):            
            # 'covid_01-01-2021.txt' --> '2021-01-01'
            date = tfile[6:-4].split("-")
            date = "-".join([date[-1], date[0], date[1]])  
            
            tname, pname = join(ner_tweet_folder, tfile), join(ner_pred_folder, tfile)
            tf, pf = open(tname, "r"), open(pname, "r")
            tline, pline = tf.readline(), pf.readline()

            while tline != "" and pline != "":
                # Tweet tokens. 
                tweet_dict = json.loads(tline)
                tokens = list(filter(lambda x: x!="" and x.encode('utf-8') not in errors, tweet_dict["tokens"]))
                
                # Location.
                location = tweet_dict["location"].lower()
                lid = self.nodes[location]
                if location == "unknown": tline, pline = tf.readline(), pf.readline()

                # Hashtags.
                hashtags = list(set(list(filter(lambda x: len(x) > 3 and x[0] == "#" and x in self.hashtag_l2h, tokens))))
                hashtags = list(set([self.hashtag_l2h[x] for x in hashtags]))

                # Entities.
                pline = pline.rstrip().replace('\n', '').split()
                if len(pline) == len(tokens): 
                    res = self.match_token_tag(pline, tokens, tags)
                    self.entity_loc[date][location].update(res)

                # Build edges among all hashtags.
                for i in range(len(hashtags)):
                    for j in range(i+1, len(hashtags)):
                        n1, n2 = hashtags[i], hashtags[j]
                        n1_index, n2_index = self.nodes[n1], self.nodes[n2]
                        if n1_index != n2_index:
                            self.edges[date][min(n1_index, n2_index), max(n1_index, n2_index)] += 1
                        self.edges[date][(n1_index, lid)] += 1
                        self.edges[date][(n2_index, lid)] += 1
 
                # Process the next line.        
                tline, pline = tf.readline(), pf.readline()

    def match_token_tag(self, pline, tline, tags):

        tmp, pre_tag = "", ""
        res, output = Counter(), Counter()
        for i, tag in enumerate(pline):
            if tag == "O": continue
            elif tag[:2] == "B_":
                if tmp != "":
                    if tmp[0] != "#" and pre_tag not in tags: 
                        res[tmp] += 1
                    tmp, pre_tag = "", ""
                tmp, pre_tag = tline[i], tag[2:]  
            elif tag[:2] == "I_" and tmp != "":
                tmp = tmp+" "+tline[i] 
        if tmp != "" and tmp[0] != "#" and pre_tag not in tags: 
            res[tmp] += 1
        for k,v in res.items():
            if k in self.entity_l2h:
                output[self.entity_l2h[k]] += v
        return output


    def build_entity_edges(self, re_dateset_folder, re_pred_folder):

        print("****Build Edges for Entities.****")

        # step1: edge between entity and location.
        for date, val in tqdm(self.entity_loc.items()):
            for loc, ent in val.items():
                print("{} contains {} entities.".format(loc, len(ent)))
                lid = self.nodes[loc]
                for e in ent:
                    eid = self.nodes[e]
                    self.edges[date][(lid, eid)] += 1

        # step2: edge between entities.
        tweet_files = [f for f in listdir(re_dateset_folder) if isfile(join(re_dateset_folder, f))]
        pred_files = [f for f in listdir(re_pred_folder) if isfile(join(re_pred_folder, f))]
        pred_files.sort()
        for pfile in tqdm(pred_files):
            
            # 'covid_01-01-2021.txt' --> '2021-01-01'
            date = pfile[6:-4].split("-")
            date = "-".join([date[-1], date[0], date[1]])
                
            pname = join(re_pred_folder, pfile)
            tname = join(re_dateset_folder, pfile)
            pf, tf = open(pname, "r"), open(tname, "r")

            # Read the title line of prediction file first.
            pline = pf.readline()  # Skip the first line.
            pline, tline = pf.readline(), tf.readline()
            while pline != "" and tline != "":
                pred = pline.strip().split('\t')
                if pred == "0":
                    pline, tline = pf.readline(), tf.readline()
                    continue
                tline = json.loads(tline)["text"].split(" ")
                l,r = 0,0
                pair = []
                # Two pointers to locate two entities in one RE input sentence.
                while l < len(tline):
                    if tline[l] == "[S]":
                        r = l + 2
                        while tline[r] != "[E]":
                            r += 1
                        entity = " ".join(tline[l+1:r])
                        if entity in self.entity_l2h:
                            entity = self.entity_l2h[entity]
                            pair += [self.nodes[entity]]
                    l += 1
                if len(pair) == 2 and pair[0] != pair[1]: 
                    self.edges[date][(min(pair), max(pair))] += 1
                pline, tline = pf.readline(), tf.readline() 


    def write_nodes_edges(self, edge_file, date_file, threshold=2):
        
        print("Writing edges...")
        num_date, edge_count = 0, 0
        g = csv.writer(open(edge_file, 'w'), delimiter=',', quotechar='"')
        for date, pairs in sorted(self.edges.items()):
            self.dates[date] = num_date
            pairs = sorted(pairs.items(), key=lambda x: x[0])
            for pair, score in pairs:
                # Remove edges which counts fewer than 2.
                # Negative weights denote mobility edges, keep.
                # Uniform all the entity, location, hashtag edges to 1. 
                if 0 < score < threshold: continue
                if self.uniform_weight:
                    if score < 0:
                        weight = -score
                    else:
                        weight = 1
                else:
                    weight = abs(score)
                g.writerow([pair[0], pair[1], weight, num_date])
                edge_count += 1
            num_date += 1
        print("--Write {} edges.".format(edge_count))
        if not exists(date_file):
            json.dump(self.dates, open(date_file, 'w'))


if __name__ == "__main__":

    re_dateset_folder = sys.argv[1]
    re_pred_folder = sys.argv[2]
    mobility_data_folder = sys.argv[3]
    map_folder = sys.argv[4]
    output_folder = sys.argv[5]
    ner_tweet_folder = sys.argv[6]
    ner_pred_folder = sys.argv[7]

    state_adjacency_file = join(map_folder, "neighbors-states.csv")
    nodes = json.load(open(join(output_folder, "nodes-states-hashtags-entities.json"), 'r'))
    states = json.load(open(join(output_folder, "nodes-states.json"), 'r'))
    map_file = join(map_folder, "us_cities_states_counties.csv")
    date_file = join(output_folder, "dates.json")

    entityl2h = json.load(open(join(output_folder, "l2h-entity.json"), 'r'))
    hashtagl2h = json.load(open(join(output_folder, "l2h-hashtag.json"), 'r'))

    # Write adjacency edges.
    BEO = Build_Edge(uniform_weight=True, nodes=nodes, states=states, entityl2h=entityl2h, hashtagl2h=hashtagl2h)
    BEO.build_location_nodes(map_file, state_adjacency_file)
    BEO.build_adjacency_edges(ner_pred_folder)
    edge_file = join(output_folder, "edges-adjacency.csv")
    #BEO.write_nodes_edges(edge_file, date_file)  

    # Write adjacency+mobility edges.
    # BEO = Build_Edge(uniform_weight=True, nodes=nodes)
    BEO.build_mobility_edges(mobility_data_folder, re_pred_folder)
    edge_file = join(output_folder, "edges-adjacency-mobility.csv")
    #BEO.write_nodes_edges(edge_file, date_file)

    # Write adjacency+mobility+hashtag edges.
    BEO.build_hashtag_edges(ner_tweet_folder, ner_pred_folder)
    edge_file = join(output_folder, "edges-adjacency-mobility-hashtag.csv")
    #BEO.write_nodes_edges(edge_file, date_file)  

    # Write adjacency+mobility+hashtag+entity edges.
    BEO.build_entity_edges(re_dateset_folder, re_pred_folder)
    edge_file = join(output_folder, "edges-adjacency-mobility-hashtag-entity.csv")
    BEO.write_nodes_edges(edge_file, date_file)  
    print("State+hashtag+entity node num: {}.".format(BEO.node_num))
    
