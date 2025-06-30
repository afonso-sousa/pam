#!/bin/bash

dataset_name="stsb" # "sick" or "stsb"
split="test" # "dev" or "test"
type="main"
metric="smatch"

src_file_path="data/$dataset_name/$type/raw/src.$split.amr"
tgt_file_path="data/$dataset_name/$type/raw/src.$split.amr"
CUDA_VISIBLE_DEVICES=0 python test_amr_sim_metrics.py \
    --src_file_path $src_file_path \
    --tgt_file_path $tgt_file_path \
    --metric $metric