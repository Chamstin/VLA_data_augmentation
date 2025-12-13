#!/usr/bin/env python3
import os
import glob
from pathlib import Path
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import argparse
import numpy as np
import cv2

def set_seed(seed=42):
    """
    Set all random seeds to a fixed value for reproducibility.
    """
    import random
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    cv2.setRNGSeed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed}")


def load_model(device):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="auto",
        device_map=device,
        trust_remote_code=True
    )
    return model


def load_processor():
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True
    )
    return processor


def evaluate(video_dir: str, output_csv: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    processor = load_processor()

    results = []
    video_paths = glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)

    for vid_path in tqdm(video_paths, desc="Evaluating videos"):
        # derive prompt from filename (remove underscores)
        prompt = Path(vid_path).stem.replace("_", " ")

        user_text = (
            f"The video shows a robot arm completing a specific task. "
            f"Does the video show good physics dynamics and showcase a good alignment with the physical world? Please be a strict judge. If it breaks the laws of physics, please answer 0. "
            f"Answer 0 for No or 1 for Yes. Reply only 0 or 1."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": vid_path},
                    {"type": "text",  "text": user_text},
                ],
            }
        ]

        # prepare inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(device)

        # generate
        generated_ids = model.generate(**inputs, max_new_tokens=4)
        # trim prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        # decode
        output = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        raw_reply = output[0].strip()
        # parse to int prediction (0 or 1)
        pred = 0
        if raw_reply and raw_reply[0] == '1':
            pred = 1
        # record
        results.append((vid_path, prompt, pred))

    total = len(results)
    sum_preds = sum(r[2] for r in results)
    avg_score = sum_preds / total if total > 0 else 0.0

    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write('video_path,prompt,prediction\n')
        for vid, pr, pred in results:
            # escape commas
            pr_esc = pr.replace(',', ' ')
            f.write(f"{vid},{pr_esc},{pred}\n")

    print(f"Evaluation done. Results saved to {output_csv}")
    print(f"Average prediction (1=Yes rate): {avg_score:.4f} ({sum_preds}/{total})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate whether videos follow given instruction prompts using Qwen2.5-VL."
    )
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Root folder containing videos (.mp4)')
    parser.add_argument('--output_csv', type=str, default='eval_results.csv',
                        help='CSV file to write results')
    parser.add_argument('--device', type=str, default=None,
                        help='Torch device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    set_seed(args.seed)
    evaluate(args.video_dir, args.output_csv, args.device)
