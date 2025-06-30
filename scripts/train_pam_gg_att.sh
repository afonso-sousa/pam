#!/bin/bash

model_name="bert-base-uncased"
batch_size=16
pos_neg_ratio=4  # batch_size must be divisible by pos_neg_ratio
max_seq_length=128
epochs=1
evaluation_steps=1000
weight_decay=0.01
warmup_steps=0
learning_rate=1e-5
optimizer="AdamW" # RMSprop or AdamW
gnn_size=256
add_graph=True
model_save_path="output/pam_gg_att"
wikipedia_dataset_path="data/wiki_train_data.json"

if [ "$add_graph" = True ]; then
    add_graph_flag="--add_graph"
else
    add_graph_flag=""
fi

CUDA_VISIBLE_DEVICES=1 python ct_pretrain.py \
    --arch pam_gg_att \
    --model_name $model_name \
    --batch_size $batch_size \
    --pos_neg_ratio $pos_neg_ratio \
    --max_seq_length $max_seq_length \
    --epochs $epochs \
    --evaluation_steps $evaluation_steps \
    --weight_decay $weight_decay \
    --warmup_steps $warmup_steps \
    --learning_rate $learning_rate \
    --optimizer $optimizer \
    --gnn_size $gnn_size \
    $add_graph_flag \
    --model_save_path $model_save_path \
    --wikipedia_dataset_path $wikipedia_dataset_path
