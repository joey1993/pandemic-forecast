#!/bin/bash

# step 3.1:
# Build dataset for training a relation extraction model. 
# python3 scripts/build_binary_dataset.py balance


# Step 3.2:
# Train a relation classification model with existing datasets.


# Step 3.3:
# Build the relation extraction inference dataset based on tweets NER results.

key="$1"

if [[ $key == "parse" ]];
then 
    nerinfolder=results/ner-inputs/
    neroutfolder=results/ner-outputs/
    reinfolder=results/re-inputs/
    mkdir -p $reinfolder
    python3 scripts/build-re-dataset.py $nerinfolder $neroutfolder $reinfolder
fi

# Step 3.4:
# Run RE model to do inference on each day's data.
# "/local/yz/nsf-covid/CORD-NER/transformers/examples/text-classification/run_predict_tweets.sh"

if [[ $key == "re" ]];
then
    modelfolder=models/re-model/
    folder=data/re-model-training/
    reoutfolder=results/re-outputs/
    reinfolder=results/re-inputs/
    mkdir -p $reoutfolder

    for file in $(ls $reinfolder); 
    do 
        if [[ $file == covid_* ]];
        then
            echo $file; 
            if [[ ! -f $reoutfolder/$file ]];
            then
                CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/run-re.py \
                    --model_name_or_path bert-base-uncased \
                    --train_file $folder/infer-sample.json \
                    --validation_file $folder/infer-sample.json \
                    --test_file $reinfolder/$file  \
                    --output_dir $reoutfolder \
                    --do_predict \
                    --model_name_or_path $modelfolder \
                    --max_seq_length 128; 
            fi
        fi
    done
fi