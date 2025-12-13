"""
This scripts convert the directory structure from step 2 for step 3
in step 2, the directory structure is like this:

results/dream_gen_benchmark/
└── cosmos_predict2_14b_gr1_object/
    ├── 0_Use_the_right_hand_to_pick_up_the_spatula_and_perform_a_serving_motion_from_the_bowl_onto_the_metal_plate.mp4
    ├── 1_Use_the_right_hand_to_close_lunch_box.mp4
    └── ...

For step 3, the directory structure is like this:

results/dream_gen_benchmark/
└── cosmos_predict2_14b_gr1_object_step3/
    ├── Use_the_right_hand_to_pick_up_the_spatula_and_perform_a_serving_motion_from_the_bowl_onto_the_metal_plate/0.mp4
    ├── Use_the_right_hand_to_close_lunch_box/0.mp4
    └── ...
"""

import os
import shutil
import argparse
import glob

def convert_directory(input_dir, output_dir):
    mp4_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    print(f"Found {len(mp4_files)} mp4 files")
    for mp4_file in mp4_files:
        new_path = os.path.join(output_dir, os.path.basename(mp4_file.replace(".mp4", "")).split("_", 1)[1], "0.mp4")
        print(f"Copying {mp4_file} to {new_path}")
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.copy(mp4_file, new_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="results/dream_gen_benchmark/cosmos_predict2_14b_gr1_object")
    parser.add_argument("--output_dir", type=str, default="results/dream_gen_benchmark/cosmos_predict2_14b_gr1_object_step3")
    args = parser.parse_args()
    assert os.path.exists(args.input_dir), f"Input directory {args.input_dir} does not exist"
    assert not os.path.exists(args.output_dir), f"Output directory {args.output_dir} already exists"
    convert_directory(args.input_dir, args.output_dir)