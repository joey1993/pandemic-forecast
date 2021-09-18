#!/bin/bash

# step 1: Run NER model to do inference on each day's tweets.
infolder=results/ner-inputs/
outfolder=results/ner-outputs/
folder=data/ner-model-training/
modelfolder=models/ner-model/
mkdir -p $outfolder

# Use four docker containers to simultaneously process different months' tweets. 
for file in $(ls $infolder);
do
  if [[ $file == covid_* ]] ;
  then
    echo $file;
    if [[ ! -f $outfolder/$file ]];
    then
      CUDA_VISIBLE_DEVICES=0 python3 scripts/run-ner-infer.py \
        --model_name_or_path bert-base-uncased \
        --test_file $infolder/$file  \
        --output_dir $outfolder \
        --do_predict \
        --model_name_or_path $modelfolder;
    fi
  fi
done