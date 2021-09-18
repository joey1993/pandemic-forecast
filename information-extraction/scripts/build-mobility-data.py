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
    
    for (infile, outfile) in zip(infiles, outfiles):
        print("Process {} working on {}".format(pid, infile))

        # Extract only the origin and destination CBG columns of the original dataset
        main_df = pd.read_csv(infile, compression='gzip', usecols=['origin_census_block_group', 'destination_cbgs'])

        # create a list of datasets each of 1000 rows (except the last one) from the extracted dataset 
        n = 1000 #chunk row size
        chunks_list = [main_df[i:i+n] for i in range(0,main_df.shape[0],n)]

        # Extract each individual destination CBG and its number of visits from the destination_cbgs column
        # Loop over each dataset of 1000 rows and concatenate the resulting datasets together
        parsed_df = []
        for sample in tqdm(chunks_list):
            long_sample = (pd.DataFrame(sample['destination_cbgs'].apply(ast.literal_eval).values.tolist(), index=sample['origin_census_block_group'])
            .stack()
            .reset_index()
            .rename(columns={'level_1':'destination_cbg', 0:'number of visits'})
            )
            long_sample['number of visits'] = long_sample['number of visits'].astype(int)
            parsed_df.append(long_sample)
        parsed_df = pd.concat(parsed_df)    

        # change the CBG column data types to string 
        parsed_df['origin_census_block_group'] = parsed_df['origin_census_block_group'].apply(str)
        parsed_df['destination_cbg'] = parsed_df['destination_cbg'].apply(str)
        #Add leading zeros to both CBG columns
        parsed_df['origin_census_block_group'] = parsed_df['origin_census_block_group'].str.zfill(12)
        parsed_df['destination_cbg'] = parsed_df['destination_cbg'].str.zfill(12)

        # Truncate values in both CBG columns to 2 digits
        parsed_df['origin_state_FIPS'] = parsed_df['origin_census_block_group'].str[:2]
        parsed_df['destination_state_FIPS'] = parsed_df['destination_cbg'].str[:2]
        parsed_df = parsed_df.drop(['origin_census_block_group', 'destination_cbg'], axis=1)

        # Aggregate rows with the same origin and destination state FIPS
        agg_state = parsed_df.groupby(['origin_state_FIPS', 'destination_state_FIPS']).agg({'number of visits': 'sum'})
        agg_state.columns = ['agg_visits']
        agg_state = agg_state.reset_index()

        num_states = len(statedict)
        result = np.zeros(shape=(num_states,num_states))
        for i in range(len(agg_state)):
            fromid = agg_state["origin_state_FIPS"][i]
            toid = agg_state["destination_state_FIPS"][i]
            if fromid in statemap and toid in statemap:
                fromname, toname = statemap[fromid], statemap[toid]
                if fromname in statedict and toname in statedict:
                    result[statedict[fromname]][statedict[toname]] = agg_state["agg_visits"][i]
        np.save(outfile, result)


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

    months = ["01", "02", "03", "04"]
    inputfolder = "data/safegraph/"
    statemap_file = "data/safegraph/states.csv"
    statedict_file = "data/covid-mobility/state_dict.txt"
    outputfolder = "data/covid-mobility-test/"

    statemap, statedict = load_state(statemap_file, statedict_file)

    filenames_in, filenames_out = [], []
    for month in months:
        path1 = join(inputfolder, month)
        for day in [f for f in listdir(path1)]:
            path2 = join(path1, day)
            for filename in [f for f in listdir(path2) if "csv.gz" in f]:
                filenames_in += [join(path2, filename)]
                filenames_out += [join(outputfolder, filename[:10])+".npy"]
    
    # filenames_in = filenames_in[:1]
    n, length = 4, len(filenames_in)
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