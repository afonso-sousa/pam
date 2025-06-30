#!/bin/bash

dataset_name="sick" # "sick" or "stsb"
split="test" # "dev" or "test"
type="main" # "main" "reify" "syn"
metric="alignscore"

dataset_path="data/$dataset_name/$type/$split.json"
CUDA_VISIBLE_DEVICES=0 python evaluate_metrics.py \
    --dataset_path $dataset_path \
    --metric $metric