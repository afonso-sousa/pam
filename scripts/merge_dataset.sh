#!/bin/bash

dataset="stsb" # "sick" or "stsb"
split="test" # "dev" or "test"
type="syn" # "reify" "syn"

src_file="data/$dataset/$type/raw/src.$split.amr"
tgt_file="data/$dataset/$type/raw/tgt.$split.amr"
score_file="data/$dataset/$split.y"
output_file="data/$dataset/$type/$split.json"
python preprocess/amr2json.py \
    -src $src_file \
    -tgt $tgt_file \
    -score $score_file \
    -output $output_file