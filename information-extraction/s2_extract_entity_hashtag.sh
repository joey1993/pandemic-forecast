#!/bin/bash

# Extract significant entities/events (TF-IDF).
# Extract tweet events from NER results: `/local/yz/nsf-covid/CORD-PIPELINE/scripts/extract-tweet-events.py`
# Extract significant events: `/local/yz/nsf-covid/CORD-PIPELINE/scripts/extract-significant-tweet-events.py`


# Step 2.1:
# Extract entities and hashtags from named entity recognition results.
key="$1"

if [[ $key == "all" ]];
then
    tweetfolder=results/ner-inputs/
    nerfolder=results/ner-outputs/
    mapfile=data/maps/us_cities_states_counties.csv
    mapdict=data/maps/state_dict.json

    entityfolder=results/entity/
    mkdir -p $entityfolder
    outputfolder=results/entity/tweets-entity-location
    mkdir -p $outputfolder
    python3 scripts/extract-tweet-events.py $tweetfolder $nerfolder $mapfile $mapdict $outputfolder
fi

# Step 2.2:
# Extract significant entities and hashtags for display based on TF-IDF scores. 
if [[ $key == "significant" ]];
then 
    outputfolder=results/entity/tweets-entity-location-significant/
    entityfolder=results/entity/tweets-entity-location/
    mkdir -p $outputfolder

    for days in {3,};
    do 
        for locations in {2,};
        do  
            echo "--$days--$locations--"
            python3 scripts/extract-significant-tweet-events.py $entityfolder $outputfolder $days $locations;
        done
    done
fi


# Step 2.3:
# Build a LM dataset and train a LM model
if [[ $key == "build_lm" ]];
then 
    inputfolder=results/ner-inputs/
    outputfolder=data/lm-model-training/
    mkdir -p $outputfolder
    python3 scripts/build-lm-dataset.py $inputfolder $outputfolder
fi

if [[ $key == "train_lm" ]];
then 
    datafolder=data/lm-model-training
    modelfolder=models/language-model-new/
    mkdir -p modelfolder
    CUDA_VISIBLE_DEVICES=1,0 python3 scripts/run_mlm.py \
    --model_name_or_path bert-base-uncased   \
    --train_file $datafolder/tweets-LM-train.txt   \
    --validation_file $datafolder/tweets-LM-valid.txt  \
    --do_train \
    --do_eval \
    --output_dir $modelfolder \
    --line_by_line \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 16  \
    --logging_steps 10000   \
    --save_steps 10000 \
    --max_seq_length 128 
fi


# Step 2.4:
# Build a json file to include all the events and hashtags.
if [[ $key == "json" ]];
then 

    inputfolder=results/entity/tweets-entity-location/
    outputfile=results/entity/hashtag-entity-0.json
    python3 scripts/prepare-entity-emb.py $inputfolder $outputfile all  # significant
fi

# Step 2.5:
# Use Bert to generate the embeddings.
if [[ $key == "infer_emb" ]];
then 
    modelfolder=models/language-model/checkpoint-1680000/
    outputfolder=edges/   
    CUDA_VISIBLE_DEVICES=1,5 python3 scripts/run-emb.py \
        --model_name_or_path $modelfolder \
        --do_predict \
        --output_dir $outputfolder
fi

# Step 2.6:
# Cluster the nodes for filtering.
if [[ $key == "cluster" ]];
then 
    python3 scripts/build-clusters-dbscan.py $inputembfile $inputnodefile $counfile $outputfolder
fi
