"""
Event extraction:
    1. Match tokens with their types (predicted).
    2. Extract frequent events for each day.
    3. Location wise frequent event extraction for each day.
"""

from collections import Counter, defaultdict, OrderedDict
import json
from os import listdir
from os.path import isfile, join, exists
import sys

def main(argv):

    # Load two files each time to match the tokens with predicted types.
    tweet_folder = argv[1]
    pred_folder = argv[2]
    map_file = argv[3]
    map_dict = argv[4]
    out_file = argv[5]

    tweet_files = [f for f in listdir(tweet_folder) if isfile(join(tweet_folder, f))]
    pred_files = [f for f in listdir(pred_folder) if isfile(join(pred_folder, f))]
    pred_files = sorted(pred_files)

    # Load maps.
    map_csv = open(map_file, 'r').readlines()
    map_dic = json.load(open(map_dict, 'r'))
    mapping = defaultdict(set)
    for line in map_csv[1:]:
        locations = line.rstrip().replace('\n', '').split('|')
        if len(locations) < 5: continue
        # Data format: "City|State short|State full|County|City alias".
        # Build mapping from every level to State full.
        mapping[locations[2].lower()].add(locations[2].lower())
        for loc in locations[:2]+locations[3:4]:
            mapping[loc.lower()].add(locations[2].lower())

    # Map day -> location -> type -> event -> count.
    output = {}
    error = 0
    #pred_files = ["covid_01-01-2021.txt"]
    errors = set([b'\xe2\x81\xa6\xe2\x81\xa9', b'\xe2\x80\xaf', b'\xef\xb8\x8f', b'\xc2\xa0', b'\xe2\x81\xa6', \
              b'\xe2\x81\xa9', b'\xe2\x80\x89', b'\xe2\x80\x8b', b'\xc2\xa0\xc2\xa0', b'\xe2\x81\xa6\xe2\x81\xa6',
              b'\xe2\x81\xa6\xe2\x81\xa6\xe2\x81\xa6'])
    tags = ["LOC", "GPE", "DATE", "PRODUCT"]
    for pfile in pred_files:
        if pfile == "test_results.txt": continue
        print(pfile)
        date = pfile[6:-4]
        # if exists(join(out_file, date+"-loc.json")): continue
        output[date] = output.get(date, {"all":{}, "loc":{}})
        pname = join(pred_folder, pfile)
        tname = join(tweet_folder, pfile)
        pf, tf = open(pname, "r"), open(tname, "r")
        pline, tline = pf.readline(), tf.readline()
        while pline != "" and tline != "":
            pline = pline.rstrip().replace('\n', '').split()
            # Potential problems in data preprocessing: (1) unicode; (2) stemming; (3) hashtag error; (3) special punctuations. 
            tweet_dict = json.loads(tline)
            tline = list(filter(lambda x: x!="" and x.encode('utf-8') not in errors, tweet_dict["tokens"]))
            try:
                assert len(pline) == len(tline)
            except:
                # print(pline, tline)
                error += 1
                pline, tline = pf.readline(), tf.readline()

            # Find the type -> event -> count mappings in current tweet.
            tmp, pre_tag = "", ""
            res = defaultdict(Counter)
            for i, tag in enumerate(pline):
                if len(tline[i])>1 and tline[i][0] == "#":
                    res["HASHTAG"][tline[i]] += 1
                if tag == "O": continue
                elif tag[:2] == "B_":
                    if tmp != "":
                        if tmp[0] != "#": res[pre_tag][tmp] += 1
                        tmp, pre_tag = "", ""
                    tmp, pre_tag = tline[i], tag[2:]  
                elif tag[:2] == "I_" and tmp != "":
                    tmp = tmp+" "+tline[i] 
            if tmp != "" and tmp[0] != "#": res[pre_tag][tmp] += 1
            
            # Remove the tags of no interest.
            for key in tags:
                if key in res: del res[key]

            # Add events to the non location wise hashmap.
            for key in res:
                output[date]["all"][key] = output[date]["all"].get(key, Counter())
                output[date]["all"][key].update(res[key])

            # Add events to the location wise hashmap.
            # Potential problems: 1. Location tagging error propagated. e.g. CA not necessary to be california. 
            #                    2. County-level, city-level name duplicated. e.g. stata clara exists in many states.
            #                    3. Data shortage. Tweets rarely mention some peopleless states. 

            locations = []
            if tweet_dict["location"] != "UNKNOWN":
                locations = [tweet_dict["location"]] 
            # locations = res["LOC"] + res["GPE"]
            for l in locations: 
                if l not in mapping: continue
                for state in list(mapping[l]):
                    output[date]["loc"][state] = output[date]["loc"].get(state, {})
                    for key in res:
                        output[date]["loc"][state][key] = output[date]["loc"][state].get(key, Counter())
                        output[date]["loc"][state][key].update(res[key])

            pline, tline = pf.readline(), tf.readline()

        # Sort and Filter "loc".
        for state in output[date]["loc"]:
            for key,vals in output[date]["loc"][state].items():
                new_vals = {}
                for v,c in vals.items(): 
                    if c>=2: new_vals[v]=c
                tmp1 = sorted(new_vals.items(), key=lambda x: x[1], reverse=True)
                tmp2 = OrderedDict()
                for x,y in tmp1: tmp2[x]=y
                output[date]["loc"][state][key] = tmp2
        
        # json.dump(output[date]["loc"], open(join(out_file, date+"-loc.json"),'w'))

        # Sort and Filter "all".
        for tag in output[date]["all"]:
            tmp1 = output[date]["all"][tag]
            tmp1 = sorted(output[date]["all"][tag].items(), key=lambda x:x[1], reverse=True)
            tmp2 = OrderedDict()
            for x,y in tmp1: 
                if y>=2: tmp2[x]=y
            output[date]["all"][tag] = tmp2 #[:(len(tmp)//5+1)]

        json.dump(output[date]["all"], open(join(out_file, date+"-all.json"),'w'))

    # output = dict(sorted(output.items(), key=lambda x: (int(x[0].split('-')[0]), int(x[0].split('-')[1]))))
    # json.dump(output, open(out_file, 'w'))


if __name__ == "__main__":
    main(sys.argv)