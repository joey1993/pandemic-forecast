import json
from os import listdir
from os.path import isfile, join, exists
import sys
from tqdm import tqdm
from collections import Counter

def main(argv):
    input_folder = argv[1]
    output_file = argv[2]
    mode = argv[3]
    entities, hashtags = set(), set()
    states = list(json.load(open("results/nodes-edges-0/nodes-state.json",'r'))["node2id"].keys())

    if mode == "significant":
        files = [f for f in listdir(input_folder) if isfile(join(input_folder, f)) and "-tfidf.json" in f]
        for f in tqdm(files):
            json_dict = json.load(open(join(input_folder, f), 'r'))
            for state, ents in json_dict.items():
                for key, val in ents.items():
                    if val[-1] == "HASHTAG": hashtags.add(key)
                    else: entities.add(key)
        json.dump({"STATES": states, "HASHTAG": sorted(list(hashtags)), "ENTITIES": sorted(list(entities))}, open(output_file, "w"))
        print("Num of hashtags and entities: {} and {}.".format(len(hashtags), len(entities)))


    elif mode == "all":
        entities, hashtags = Counter(), Counter()
        files = [f for f in listdir(input_folder) if isfile(join(input_folder, f)) and "-all.json" in f]
        for f in tqdm(files):
            json_dict = json.load(open(join(input_folder, f), 'r'))
            for tag, ents in json_dict.items():
                for key, count in ents.items():
                    if len(key) < 3: continue
                    if tag == "HASHTAG": 
                        hashtags[key] += count
                    else: 
                        entities[key] += count

        json.dump({"HASHTAG": sorted(hashtags.items(), key=lambda x: x[0]), \
                    "ENTITIES": sorted(entities.items(), key=lambda x: x[0])}, \
                    open(output_file.replace(".json", "-count.json"), "w"))

        json.dump({"STATES": states, "HASHTAG": sorted(hashtags.keys()), "ENTITIES": sorted(entities.keys())}, open(output_file, "w"))
        print("Num of hashtags and entities: {} and {}.".format(len(hashtags), len(entities)))        


if __name__ == "__main__":
    main(sys.argv)