'''
Build the time series prediction dataset.
    para: start date and end date,  covid case and death files.
    return: increasing/decreasing numbers of new cases and deaths compared to the same day one week ago. 
        
'''
import csv
import json
from bisect import bisect_left
from collections import defaultdict, Counter, OrderedDict

from os import listdir, makedirs
from os.path import isfile, join, exists

class Build_TSP_dataset(object):

    def __init__(self):
        self.cases = defaultdict(Counter)
        self.deaths = defaultdict(Counter)
        self.cases_delta = defaultdict(Counter)
        self.deaths_delta = defaultdict(Counter)


    def load_state(self, statemap_file, statedict_file):

        state2index = open(statedict_file, 'r').readline()
        self.statedict = json.loads(state2index.replace("\'", "\""))
        self.statemap = {}
        with open(statemap_file) as csvfile:
            csvfile = csv.reader(csvfile, delimiter='\t',)    
            for i,row in enumerate(csvfile):
                if i == 0: continue
                self.statemap[row[2]] = row[8].replace(" State", "")


    def covid_case_death(self, case_death_file):

        # Find out all the dates needed. 
        start_date = "2020-05-14"  # The first day is used for computing delta. 
        end_date = "2021-04-08"

        # Load death and case statistics.
        self.dates = {"id2date":OrderedDict(),"date2id":OrderedDict()}
        num_dates = 0
        with open(case_death_file) as csvfile:
            csvfile = csv.reader(csvfile, delimiter=',')    
            for i,row in enumerate(csvfile):
                if i == 0 or not start_date <= row[0] <= end_date or \
                    row[2] not in self.statemap: continue
                if row[0] not in self.dates["date2id"]:
                    self.dates["date2id"][row[0]] = num_dates-1
                    self.dates["id2date"][num_dates-1] = row[0]
                    num_dates += 1
                state = self.statemap[row[2]]
                if state not in self.statedict: continue
                stateid = self.statedict[state]
                dateid = self.dates["date2id"][row[0]]
                self.cases[stateid][dateid] = int(row[3])
                self.deaths[stateid][dateid] = int(row[4])

        # Compute the daily change. It should be easier to predict the delta instead of absolute value.
        for stateid in self.cases:
            for i in range(num_dates-2, -1, -1):
                cur, prev = i, i-1
                self.cases_delta[stateid][cur] = self.cases[stateid][cur] - self.cases[stateid][prev]
                self.deaths_delta[stateid][cur] = self.deaths[stateid][cur] - self.deaths[stateid][prev]


    def write_file(self, output_folder):


        death_truth_cls = open(join(output_folder, "us_labels_death.csv"), 'w')
        case_truth_cls = open(join(output_folder, "us_labels.csv"), 'w')
        death_writer_cls = csv.writer(death_truth_cls, delimiter=',', quotechar='"')
        case_writer_cls = csv.writer(case_truth_cls, delimiter=',', quotechar='"')

        row_num = 0
        for (state, sid) in sorted(self.statedict.items(), key=lambda x: x[1]):
            dates = []
            cases, deaths = [], []
            state = "_".join(state.lower().split(" "))
            for (date, did) in self.dates["date2id"].items():
                if did == -1: continue
                dates += [date]
                case_num, death_num = max(0, self.cases_delta[sid][did]), max(0, self.deaths_delta[sid][did])
                cases += [case_num]
                deaths += [death_num]
            if row_num == 0:
                case_writer_cls.writerow(["name"]+dates)
                death_writer_cls.writerow(["name"]+dates)

            case_writer_cls.writerow([state]+cases)
            death_writer_cls.writerow([state]+deaths)
            row_num += 1
        


if __name__ == "__main__":

    statemap_file = "./data/safegraph/states.csv"
    statedict_file = "./data/covid-mobility/state_dict.txt"
    case_death_file = "./data/covid-cases-deaths/us-states.csv"

    output_folder = "./data/"

    BTD = Build_TSP_dataset()
    BTD.load_state(statemap_file, statedict_file)
    BTD.covid_case_death(case_death_file)
    BTD.write_file(output_folder)



