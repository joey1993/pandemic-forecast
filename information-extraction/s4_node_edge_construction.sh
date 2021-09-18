#!/bin/bash

# Step 4.1: 
if [[ "$1" == "mobility" ]];
then
    # Update the mobility data.
    python3 scripts/build_mobility_data.py
    # Update the case/death statistics.
    python3 scripts/build_tsp_dataset.py
fi

ner_tweet_folder=results/ner-inputs/
ner_pred_folder=results/ner-outputs/
re_dataset_folder=results/re-inputs/
re_pred_folder=results/re-outputs/
mobility_data_folder=data/covid-mobility/
map_folder=data/maps/
out_folder=results/nodes-edges/
mkdir -p $out_folder

# Step 4.2:
# Build the graph based on RE predictions and significant entities/events.
# "/local/yz/nsf-covid/CORD-RE/src/build_entity_edges.py"
if [[ "$1" == "edge" ]];
then
    python3 scripts/build-nodes-edges.py \
        $re_dataset_folder \
        $re_pred_folder \
        $mobility_data_folder \
        $map_folder \
        $out_folder \
        $ner_tweet_folder \
        $ner_pred_folder;
fi