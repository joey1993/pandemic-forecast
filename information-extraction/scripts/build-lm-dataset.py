import json
from os import listdir
from os.path import isfile, join, exists
import sys
from tqdm import tqdm
import random
random.seed(2021)

def main(argv):
    input_folder = argv[1]
    output_folder = argv[2]
    dev = open(join(output_folder, "tweets-LM-valid.txt"), 'w')
    train = open(join(output_folder, "tweet-LM-train.txt"), 'w')
    files = [f for f in listdir(input_folder) if isfile(join(input_folder, f)) and ".txt" in f]
    for file in tqdm(files):
        f = open(join(input_folder, file), "r")
        line = f.readline()
        while line != "":
            line = json.loads(line)
            tweet = line["tweet"]
            value = random.random()
            if value < 0.005: dev.write(tweet+"\n")
            else: train.write(tweet+"\n")
            line = f.readline() 

if __name__ == "__main__":
    main(sys.argv)

