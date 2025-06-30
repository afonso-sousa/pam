#!/bin/bash

dataset_path="data/sick/main/test.json"
gnn_size=128
num_gnn_layers=2
arch="pam_roberta"
model_path="output/${arch}_${gnn_size}_$num_gnn_layers"

CUDA_VISIBLE_DEVICES=0 python test.py \
    --dataset_path $dataset_path \
    --model_path $model_path