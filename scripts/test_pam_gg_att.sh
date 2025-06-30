#!/bin/bash

dataset_path="data/stsb/main/test.json"
model_path="output/pam_gg_att"

CUDA_VISIBLE_DEVICES=0 python test.py \
    --dataset_path $dataset_path \
    --model_path $model_path