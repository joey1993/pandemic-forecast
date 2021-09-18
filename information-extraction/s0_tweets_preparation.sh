#!/bin/bash
  
# step 0: retrieve from $1 and build NER inference data set.

tokenizedfolder=results/ner-inputs/   
nerdataset=results/ner-inputs/
mapfile=data/maps/us_cities_states_counties.csv
mapdict=data/maps/state_dict.json

# mkdir -p $nerdataset
python3 scripts/build-ner-dataset.py $tokenizedfolder $nerdataset $mapfile $mapdict
