#!/bin/bash

dataset_path="data/sick/refra/test.json"
model_path="output/amrsim"

CUDA_VISIBLE_DEVICES=0 python test.py \
    --dataset_path $dataset_path \
    --model_path $model_path