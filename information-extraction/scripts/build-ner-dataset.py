import json
import re
import sys
from os import listdir, makedirs
from os.path import isfile, join, exists
from multiprocessing import Process
import random
random.seed(2021)

def fun(files, folder, new_folder, pid):
    for i,fi in enumerate(files):
        input_name = join(folder, fi)
        output_name = join(new_folder, fi.replace(".txt", ".json"))
        if exists(output_name):
            continue
        f = open(input_name, 'r')
        g = open(output_name, 'w')
        h = open(output_name.replace(".json",".error"), 'w')
        line = f.readline()
        instance = {"tokens": [], "ner_tags": []}
        errors = []
        count = 0
        while line != "": 
            if count % 1000000 == 0: print("process {} finished {} lines in the {}th file.".format(pid, count, i))  
            try:   
                line = json.loads(line)
                instance["tokens"] = [x.lower() for x in line["tokens"]]
                # assert len(instance["tokens"]) >= 5
                instance["ner_tags"] = ["O"]*len(instance["tokens"])
                # instance["location"] = line["location"]
                json.dump(instance, g)
                g.write('\n')
                count += 1
            except:
                errors += [count]
                pass
            line = f.readline()
        json.dump(errors, h)
        f.close(); g.close(); h.close()

def sample(files, folder, new_folder, pid, threshold, mapping, states):

    for i,fi in enumerate(files):
        input_name = join(folder, fi)
        output_name = join(new_folder, fi)
        if exists(output_name):
            continue
        f = open(input_name, 'r')
        g = open(output_name, 'w')
        line = f.readline()
        count = 0
        while line != "": 
            if count % 1000000 == 0: print("process {} finished {} lines in the {}th file.".format(pid, count, i))  
            try:   
                line = json.loads(line)
                line["tokens"] = [x.lower() for x in line["tokens"]]
                assert len(line["tokens"]) >= 5
                # Only use US and UNKNOWN locations.
                if line["location"] != "UNKNOWN": 
                    locations = line["location"].replace(",", ", ").split(", ")
                    locations = [x.lower() for x in locations]
                    potential_state = None
                    for loc in locations:
                        if loc in states:
                            line["location"] = loc
                            potential_state = loc
                            break
                        elif loc in mapping:
                            potential_state = mapping[loc]
                        if potential_state:
                            line["location"] = potential_state
                    if potential_state:
                        json.dump(line, g)
                        g.write('\n')
                    else:
                        line["location"] = "UNKNOWN"

                value = random.random()
                if value <= threshold and line["location"] == "UNKNOWN":
                    json.dump(line, g)
                    g.write('\n')
            except:
                pass
            count += 1
            line = f.readline()
        f.close(); g.close()


def load_map(map_file, map_dict):
    mapping = {}
    map_csv = open(map_file, 'r').readlines()
    map_dic = json.load(open(map_dict, 'r'))
    map_dic = {k.lower() for k,v in map_dic.items()}
    for line in map_csv[1:]:
        locations = line.rstrip().replace('\n', '').split('|')
        if len(locations) < 5: continue
        # Data format: "City|State short|State full|County|City alias".
        # Build mapping from every level to State full.
        if locations[2].lower() in map_dic:
            for loc in locations[1:3]:
                mapping[loc.lower()] = locations[2].lower()
    return mapping, map_dic


def main(argv):
    folder = argv[1]  # input->tweets.
    new_folder = argv[2]  # output->NER inference datasets.
    threshold = 0
    new_folder += str(threshold)
    if not exists(new_folder): makedirs(new_folder)  # create the folder for outputs. 
    print(new_folder)
    files = [f for f in listdir(folder) if isfile(join(folder, f)) and ".txt" in f]  # load all the input file names.
    files = list(filter(lambda fi: not exists(join(new_folder, fi)), files))
    print(files)
    map_file = argv[3]
    map_dict = argv[4]
    mapping, states = load_map(map_file, map_dict)
    num_files = len(files)
    processes = []
    chunk_files = []
    num_worker = 100
    chunk_size = len(files)//100
    for i in range(0, len(files), chunk_size+1):
        chunk = files[i:i + chunk_size+1]
        chunk_files.append(chunk)
    for i,ch in enumerate(chunk_files):
        print("process {} works on files: {}".format(i, ch))
        p = Process(target=sample, args=(ch,folder,new_folder,i,threshold,mapping,states, ))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()



if __name__ == "__main__":
    main(sys.argv)