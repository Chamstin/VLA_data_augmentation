python IDM_dump/split_video_instruction.py \
    --source_dir "" \
    --output_dir "IDM_dump/data/gr1_data"

python IDM_dump/preprocess_video.py \
    --src_dir "IDM_dump/data/gr1_data" \
    --dst_dir "IDM_dump/data/gr1_data_split" \
    --dataset gr1 


python IDM_dump/raw_to_lerobot.py \
    --input_dir "IDM_dump/data/gr1_data_split" \
    --output_dir "IDM_dump/data/gr1_unified.data" \
    --cosmos_predict2 


python IDM_dump/dump_idm_actions.py \
    --checkpoint "seonghyeonye/IDM_gr1" \
    --dataset "IDM_dump/data/gr1_unified.data" \
    --output_dir "IDM_dump/data/gr1_unified.data_idm" \
    --num_gpus 8 \
    --video_indices "0 8" 



