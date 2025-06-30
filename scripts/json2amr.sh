#!/bin/bash

dataset="stsb" # "sick" or "stsb"
split="test" # "dev" or "test"
type="refra"

input_file="data/$dataset/$type/$split.json"
out_src_file="data/$dataset/$type/raw/src.$split.amr"
out_tgt_file="data/$dataset/$type/raw/tgt.$split.amr"
python preprocess/json2amr.py \
    --input_file $input_file \
    --out_src_file $out_src_file \
    --out_tgt_file $out_tgt_file