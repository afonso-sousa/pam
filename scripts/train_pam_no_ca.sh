#!/bin/bash

model_name="bert-base-uncased" # "FacebookAI/roberta-base"
batch_size=16
pos_neg_ratio=4  # batch_size must be divisible by pos_neg_ratio
max_seq_length=128
epochs=1
evaluation_steps=1000
weight_decay=0.01
warmup_steps=0
learning_rate=1e-5
optimizer="AdamW" # RMSprop or AdamW
gnn_size=128
num_gnn_layers=2
add_graph=True
arch="pam_no_ca" # pam, pam_no_ca
model_save_path="output/${arch}_${gnn_size}_$num_gnn_layers"
wikipedia_dataset_path="data/wiki_train_data.json"

if [ "$add_graph" = True ]; then
    add_graph_flag="--add_graph"
else
    add_graph_flag=""
fi

CUDA_VISIBLE_DEVICES=0 python ct_pretrain.py \
    --arch $arch \
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
    --num_gnn_layers $num_gnn_layers \
    $add_graph_flag \
    --model_save_path $model_save_path \
    --wikipedia_dataset_path $wikipedia_dataset_path
