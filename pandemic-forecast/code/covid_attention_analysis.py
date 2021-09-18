import torch
import networkx as nx
import numpy as np
import sys
from os import listdir
from os.path import isfile, join, exists
from collections import defaultdict, OrderedDict, Counter
from datetime import date, timedelta
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import json



def covid_attention_analysis(folder,task,id2state,id2node,shift,entity2type,topk):
    # Compute each state's each activity's count and average att score, maxmimum att score for both low and high case numbers.

    result = defaultdict(dict)
    result_all = defaultdict(dict)
    intereted_types = {"SOCIAL_INDIVIDUAL_BEHAVIOR", "LABORATORY_PROCEDURE", "DISEASE_OR_SYNDROME", \
             "EVENT", "THERAPEUTIC_OR_PREVENTIVE_PROCEDURE", "HASHTAG", "GOVERNMENTAL_OR_REGULATORY_ACTIVITY", \
             "SIGN_OR_SYMPTOM", "DIAGNOSTIC_PROCEDURE", "INJURY_OR_POISONING", \
             "PERSON","ORG", "DAILY_OR_RECREATIONAL_ACTIVITY", "EDUCATIONAL_ACTIVITY"}
    
    case_num, scores = load_case_number(task, folder, shift)
    splitted_cases = split_case(case_num, topk)
    
    print("what amazing happens...")
    for sid in tqdm(range(50)):
        state = id2state[sid]
        high_dates, low_dates = splitted_cases[sid]['high'], splitted_cases[sid]['low']
        for text,dates in [("low",low_dates), ("high",high_dates)]:
            tmp_res = defaultdict(list)
            for d,_ in dates:
                atts, edges = scores[d]['att'], scores[d]['edge']
                indexes = edges[:,1]==sid
                edges_small, atts_small = edges[indexes], atts[indexes]
                combine = [[e1,att[0]] for [e1,_],att in zip(edges_small, atts_small) if e1 >= 50 and att[0]>0]
                combine = sorted(combine, key=lambda x: x[-1], reverse=True)[:10]
                for eid,att in combine:
                    att=np.float64(att)
                    e = id2node[eid]
                    if tmp_res[e] != []:
                        [count, avg, maxm] = tmp_res[e]
                    else:
                        count, avg, maxm = 0, 0, 0
                    avg = round((count*avg+att)/(count+1), 3)
                    count += 1
                    maxm = round(max(att,maxm), 3)
                    tmp_res[e] = [count, avg, maxm]
            
            tmp_res = sorted(tmp_res.items(), key=lambda x:(x[1][2],x[1][1]), reverse=True)[:50]
            tmp_res_2 = OrderedDict()
            for x,y in tmp_res: tmp_res_2[x] = y
            result[state][text] = list(tmp_res_2.keys())

            for key in tmp_res_2:
                if key in result_all[text]:
                    [count_1, avg_1, maxm_1] = result_all[text][key]
                else:
                    count_1, avg_1, maxm_1 = 0, 0, 0
                count_2, avg_2, maxm_2 = tmp_res_2[key]
                avg_ = round((count_1*avg_1+count_2*avg_2)/(count_1+count_2), 3)
                count_ = count_1 + count_2
                maxm_ = max(maxm_1,maxm_2)
                result_all[text][key] = [count_, avg_, maxm_]
        
    result = sorted(result.items(), key=lambda x: x[0])
    outputfile = open(join(folder, "{}_rankings_state.json".format(task)), 'w')
    json.dump(result, outputfile)
    outputfile = open(join(folder, "{}_rankings_all.json".format(task)), 'w')
    json.dump(result_all, outputfile)

    new_result_all = defaultdict(lambda: defaultdict(list))
    for text in result_all:
        for key in result_all[text]:
            keytype = entity2type[key]
            if keytype in intereted_types:
                new_result_all[text][keytype] += [key]
    outputfile = open(join(folder, "{}_rankings_all_type.json".format(task)), 'w')
    json.dump(new_result_all, outputfile)


def load_dict(statefile,nodefile):

    state2id = json.load(open(statefile,'r'))
    node2id = json.load(open(nodefile,'r'))
    id2state = {y:x for x,y in state2id.items()}
    id2node = {y:x for x,y in node2id.items()}

    return id2state, id2node


def load_case_number(task, folder, shift):

    if task == "case":
        filename = "../data/US/us_labels.csv"
    else:
        filename = "../data/US/us_labels_death.csv"

    print("load case numbers...")
    labels = pd.read_csv(filename)
    sdate = date(2020, 5, 30)
    edate = date(2021, 4, 8)    
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(d) for d in dates][shift-1:]
    labels = labels.loc[:,dates]  

    print("load attention scores...")
    scores = {}
    for d in tqdm(dates):
        edges, atts = torch.load(join(folder, d+".pt"))
        atts, edges = atts.detach().cpu().numpy(), edges.detach().cpu().numpy()
        scores[d] = {"att":atts, "edge":edges}

    return labels, scores

def split_case(case_num, x):

    labels = case_num.loc[list(range(50)),:]
    result = {} 
    for i in range(50):
        dates=list(labels.loc[i].index)
        value=list(labels.loc[i])
        tmp = [[x,y] for x,y in zip(dates,value)]
        tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
        high, low = tmp[:x], tmp[-x:]
        result[i] = {"high":high, "low":low}

    return result


def load_entities(folder,outfolder):

    event_daily_files = [f for f in listdir(folder) if isfile(join(folder, f)) and "-all.json" in f]  
    event_daily_files = sorted(event_daily_files) 
    event2type = defaultdict(lambda: Counter())
    for f in tqdm(event_daily_files):
        openf = json.load(open(join(folder,f), 'r'))
        for key in openf:
            for e,c in openf[key].items():
                event2type[e][key] += c
    new_e2type = {}
    for e in event2type:
        max_type = max(event2type[e].items(), key=lambda x:x[1])
        new_e2type[e] = max_type[0]
    json.dump(new_e2type, open(join(outfolder, 'entity_type_dict.json'), 'w'))
    return new_e2type


def rnn_attention():
    atts = torch.load("attention_rnn.pt")
    for i,split in enumerate(atts):
        print(split[0])


if __name__ == "__main__":

    task = sys.argv[1]

    folder = "./tweets-entity-location-0"
    outfolder = "../data/US/edges/"
    if not exists(join(outfolder,'entity_type_dict.json')):
        entity2type = load_entities(folder,outfolder)
    else:
        entity2type = json.load(open(join(outfolder,'entity_type_dict.json'), 'r'))

    folder = "./attention-scores/"
    state_file="./nodes-states.json"
    node_file="./all-nodes.txt"
    shift = int(folder.split("shift")[1].split("_")[0])
    id2state, id2node = load_dict(state_file, node_file)
    covid_attention_analysis(folder,task,id2state,id2node,shift,entity2type,topk=20)

