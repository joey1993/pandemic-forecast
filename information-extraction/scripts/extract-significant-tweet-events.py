"""
python3 extract-significant-tweet-events.py entityfolder outputfolder days locations
"""

from collections import Counter, defaultdict, OrderedDict
import json
import math
from os import listdir, makedirs
from os.path import isfile, join, exists
import sys
from tqdm import tqdm

def date_tf_idf(event_folder, output_folder, days_occur=12, locations_occur=2):
    '''Extract important entities for each date.
    '''

    event_daily_files = [f for f in listdir(event_folder) if isfile(join(event_folder, f)) and "loc" in f]  
    event_daily_files = sorted(event_daily_files) 

    event_num_dates = defaultdict(Counter)  # Number of dates mentioning each event at the same location.
    event_num_locations = defaultdict(Counter)  # Number of locations mentioning each entity at the same time.
    num_dates = 0
    locations = set()
    all_events_date_loc = {} 
    all_events_total = defaultdict(Counter)

    folder = join(output_folder, "{}days-{}locations".format(days_occur,locations_occur))
    if not exists(folder): 
        makedirs(folder)
    else:
        return
    
    print("Load event files.")
    for f in tqdm(event_daily_files):
        date = f.split("-loc")[0]
        num_dates += 1

        daily_location_events = json.load(open(join(event_folder,f), "r"))
        all_events_date_loc[date] = daily_location_events

        for loc,vals in daily_location_events.items():
            locations.add(loc)
            for key,val in vals.items():
                for e,c in val.items():
                    event_num_dates[loc][e] += 1
                    event_num_locations[date][e] += 1
                    all_events_total[date][loc] += c

    
    print("Compute TF-IDF scores.")
    event_values = defaultdict(lambda: defaultdict(dict))
    num_locations = len(locations)
    for date,locs in tqdm(all_events_date_loc.items()):
        for loc,vals in locs.items():
            for key,val in vals.items():
                for e,c in val.items():
                    if event_num_dates[loc][e] < days_occur or event_num_locations[date][e] < locations_occur \
                        or len(e) <= 2:
                        continue
                    tf = round(c/float(all_events_total[date][loc]), 6)
                    idf1 = round(math.log(float(num_dates)/event_num_dates[loc][e]), 6)
                    idf2 = round(math.log(float(num_locations)/event_num_locations[date][e]), 6)
                    if idf1 < 0 and idf2 < 0: 
                        value = round(-tf * idf1 * idf2, 6)
                    else:
                        value = round(tf * idf1 * idf2, 6)
                    event_values[date][loc][e] = [value, tf, idf1, idf2, key]

    print("Sort and dump the events based on TFIDF.")
    for date,locs in tqdm(event_values.items()):
        for loc,vals in event_values[date].items():
            tmp1 = sorted(vals.items(), key=lambda x:(x[1][0], x[1][1]), reverse=True)
            tmp2 = OrderedDict()
            for key,val in tmp1: tmp2[key] = val
            event_values[date][loc] = tmp2
        json.dump(event_values[date], open(join(folder, date+"-tfidf.json"), "w"))
 
    print("Sort and dump the events based on TF.")
    for date,locs in tqdm(event_values.items()):
        for loc,vals in event_values[date].items():
            tmp1 = sorted(vals.items(), key=lambda x:(x[1][1], x[1][0]), reverse=True)
            tmp2 = OrderedDict()
            for key,val in tmp1: tmp2[key] = val
            event_values[date][loc] = tmp2
        json.dump(event_values[date], open(join(folder, date+"-tf.json"), "w"))

    print("Sort and dump the Hashtags.")
    for date,locs in tqdm(event_values.items()):
        for loc,vals in event_values[date].items():
            tmp1 = sorted(vals.items(), key=lambda x:(x[1][0], x[1][1]), reverse=True)
            tmp1 = list(filter(lambda x: x[1][-1] == "HASHTAG", tmp1))
            tmp2 = OrderedDict()
            for key,val in tmp1: tmp2[key] = val[:-1]
            event_values[date][loc] = tmp2
        json.dump(event_values[date], open(join(folder, date+"-tfidf-hashtag.json"), "w"))


if __name__ == "__main__":
    raw_event_folder = sys.argv[1]
    output_folder = sys.argv[2]
    days_occur = int(sys.argv[3])
    locations_occur = int(sys.argv[4])
    date_tf_idf(raw_event_folder, output_folder, days_occur, locations_occur)