"""
Build tweet relation extraction dataset:
    1. Match tokens with their types (predicted).
    2. Create an instance between each pair of entities. 
"""

from collections import Counter, defaultdict
import json
from os import listdir
from os.path import isfile, join, exists
import sys

def main(argv):

    # Load two files each time to match the tokens with predicted types.
    tweet_folder = argv[1]
    pred_folder = argv[2]
    out_folder = argv[3]

    # tweet_files = [f for f in listdir(tweet_folder) if isfile(join(tweet_folder, f))]
    pred_files = [f for f in listdir(pred_folder) if isfile(join(pred_folder, f))]
    fun = lambda x: [x.split("-")[i] for i in [2, 0, 1]]
    pred_files = list(sorted(pred_files, key=fun))
    entity_num_distribution = Counter()
    # months = ["03","06","07","12"]

    for pfile in pred_files:
        # if not pfile.replace("covid_","").split("-")[0] in months: 
        #     continue 
        count = 0
        pname = join(pred_folder, pfile)
        tname = join(tweet_folder, pfile)
        oname = join(out_folder, pfile)
        if pfile == "test_results.txt":
            continue

        out_io = open(oname, 'w')
        pf, tf = open(pname, "r"), open(tname, "r")
        pline, tline = pf.readline(), tf.readline()
        outlier = set([b'\xe2\x81\xa6\xe2\x81\xa9', b'\xe2\x80\xaf', b'\xef\xb8\x8f', b'\xc2\xa0', b'\xe2\x81\xa6', \
                       b'\xe2\x81\xa9', b'\xe2\x80\x89', b'\xe2\x80\x8b', b'\xc2\xa0\xc2\xa0', b'\xe2\x81\xa6\xe2\x81\xa6',
                       b'\xe2\x81\xa6\xe2\x81\xa6\xe2\x81\xa6'])
        while pline != "" and tline != "":
            pline = pline.rstrip().replace('\n', '').split()
            # Potential problems in data preprocessing: (1) unicode; (2) stemming; (3) hashtag error; (3) special punctuations. 
            tline = json.loads(tline)["tokens"]
            
            tline = list(filter(lambda x: x!="" and x.encode('utf-8') not in outlier, tline))
            try:
                assert len(pline) == len(tline)
            except:
                #print(len(pline),len(tline))
                #print(pline, tline)
                count += 1
                pline, tline = pf.readline(), tf.readline()

            # Parse the relation pairs.
            res = []
            s,e = -1,-1
            for i, tag in enumerate(pline):
                if tag == "O": continue
                elif tag[:2] == "B_":
                    if s != -1 and e != -1:
                        res.append([s, e])
                        s, e = -1, -1
                    s, e = i, i
                elif tag[:2] == "I_" and s != -1:
                    e = i
            if s != -1 and e != -1:
                res.append([s, e])
        
            entity_num_distribution[len(res)] += 1
            
            if len(res) > 1:
                # print(res)
                # print(tline, pline)
                for i in range(len(res)):
                    for j in range(i+1,len(res)):
                        # Create a new sequence marked with [E] and [S].
                        tokens = []
                        
                        ss, se, os, oe = res[i][0], res[i][1], res[j][0], res[j][1]
                        # print(ss, se, os, oe)
                        for j,tok in enumerate(tline):
                            if j == ss or j == os:
                                tokens += ["[S]"]
                            tokens += [tok]
                            if j == se or j == oe:
                                tokens += ["[E]"]
                        if se == len(tline) or oe == len(tline):
                            tokens += ["[E]"]
                        # Write the seq to file.
                        # print(tokens)
                        instance = {"label": "0", "text": " ".join(tokens)}
                        json.dump(instance, out_io)
                        out_io.write('\n')
                # sys.exit()

            pline, tline = pf.readline(), tf.readline()
        print(count)

    # json.dump(entity_num_distribution, open(join(out_folder, "num_entity_dist.json"), 'w'))

if __name__ == "__main__":
    main(sys.argv)