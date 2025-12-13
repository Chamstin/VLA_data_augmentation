python IDM_dump/split_video_instruction.py \
    --source_dir "" \
    --output_dir "IDM_dump/data/robocasa"

python IDM_dump/preprocess_video.py \
    --src_dir "IDM_dump/data/robocasa" \
    --dst_dir "IDM_dump/data/robocasa_split" \
    --dataset robocasa \
    --original_width 512 \
    --original_height 512



python IDM_dump/raw_to_lerobot.py \
    --input_dir "IDM_dump/data/robocasa_split" \
    --output_dir "IDM_dump/data/robocasa_panda_omron.data" \
    --cosmos_predict2 


python IDM_dump/dump_idm_actions.py \
    --checkpoint "seonghyeonye/IDM_robocasa" \
    --dataset "IDM_dump/data/robocasa_panda_omron.data" \
    --output_dir "IDM_dump/data/robocasa_panda_omron.data_idm" \
    --num_gpus 8 \
    --video_indices "0 8" \



