#!/bin/bash

# Step 5.1:
# Train a language model with MLM.
CUDA_VISIBLE_DEVICES=6,7 python3 run_mlm.py 
    --model_name_or_path bert-base-uncased   
    --train_file ../../../tweets-LM-train.txt   
    --validation_file ../../../tweets-LM-valid.txt  
    --do_train 
    --do_eval 
    --output_dir ../../../MLM-model-large/ 
    --line_by_line 
    --per_device_eval_batch_size 64 
    --per_device_train_batch_size 64  
    --logging_steps 3000   
    --save_steps 3000 
    --max_seq_length 128

# Step 5.2:
# Create node initial embeddings with language model.
emb_file=results/embeddings/emb.txt
if [ -f "$emb_file" ]; then
    echo "$emb_file exists."
else 
    CUDA_VISIBLE_DEVICES=4 python3 scripts/run-emb.py \
        --output_dir results/embeddings/ \
        --do_predict \
        --model_name_or_path models/language-model/;
fi

# Step 5.3:
# Filter the nodes and edges based on clustering.
entity_file=data/entity/tweets-entity-location.json
initial_node_folder=results/node-edge-full/
filter_node_folder=result/node-edge-filter/
mkdir -p $filter_node_folder
cluster_path= result/node-edge-filter/cluster/
mkdir -p $cluster_path

python3 scripts/filter-entity-edges.py 
    $entity_file \
    $emb_file \
    $initial_node_folder \
    $filter_node_folder \
    $cluster_path;