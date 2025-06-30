#!/bin/bash

batch_size=32
max_seq_length=128
epochs=1
evaluation_steps=1000
weight_decay=0.01
warmup_steps=0
learning_rate=2e-5
optimizer="AdamW" # RMSprop or AdamW
model_name="pam"
model_path="output/$model_name"
data_path="data/qqp/processed_qqp.pt"
output_path="output/qqp-finetune-$model_name"

CUDA_VISIBLE_DEVICES=0 python paraphrase_finetune.py \
    --model_path $model_path \
    --dataset_path $data_path \
    --batch_size $batch_size \
    --max_seq_length $max_seq_length \
    --epochs $epochs \
    --evaluation_steps $evaluation_steps \
    --weight_decay $weight_decay \
    --warmup_steps $warmup_steps \
    --learning_rate $learning_rate \
    --optimizer $optimizer \
    --output_path $output_path
