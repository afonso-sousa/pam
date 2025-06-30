dataset_path="data/sick/main/test.json"
output_path="data/sick/refra/test.json"
python create_reframed_dataset.py \
    --dataset_path $dataset_path \
    --output_path $output_path