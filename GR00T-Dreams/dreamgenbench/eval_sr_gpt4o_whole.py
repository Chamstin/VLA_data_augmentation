#!/usr/bin/env python3
import os
from tqdm import tqdm
import argparse
import glob
from pathlib import Path
import torch
import time
import openai
from openai import OpenAI, OpenAIError
from io import BytesIO
from PIL import Image
import numpy as np
import base64
from dreamgenbench.utils import sample_video_frames, set_seed

def encode_frame_to_data_uri(frame: np.ndarray) -> str:
    """
    Convert an RGB numpy frame into a data URI PNG.
    """
    pil_img = Image.fromarray(frame)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"

def evaluate(video_dir: str, output_csv: str, api_key: str, seed: int = 42, max_retries: int = 3, zeroshot: bool = False):
    openai.api_key = api_key
    client = OpenAI(api_key=api_key)
    set_seed(seed)
    results = []
    video_paths = glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)
    for vid_path in tqdm(video_paths, desc="Evaluating videos"):
        if "cogvideo" in vid_path:
            if "zeroshot" in vid_path:
                prompt = vid_path.split("/", 8)[-1].split("_", 1)[1].replace(".mp4", "")
            else:
                prompt = vid_path.split("/")[-3].split("_", 1)[1]
                prompt = prompt.replace("_", " ")
        elif "hunyuan" in vid_path:
            prompt = vid_path.split("/")[-1].split("_", 1)[1]
            prompt = prompt.split("_", 4)[-1]
            prompt = prompt.replace(".mp4", "")
        elif "wan" in vid_path or "cosmos" in vid_path:
            prompt = Path(vid_path).stem.split("_", 1)[1]
        prompt = prompt.replace("_", " ")
        # make sure the video path is valid
        if not os.path.exists(vid_path):
            print(f"Video path does not exist: {vid_path}")
        if zeroshot:
            user_text = (
                f"You are evaluating if a robot arm correctly follows this instruction: '{prompt}'\n\n"
                f"CRITICAL EVALUATION PROCESS:\n"
                f"1. FIRST CHECK: If you see HUMAN HANDS instead of robot arms, IMMEDIATELY ANSWER 0.\n"
                f"2. SECOND CHECK: Only if robot arms confirmed, verify if the instruction is followed exactly.\n\n"
                f"3. For videos with multiview clip (4 grids), verify if the instruction is followed exactly in each view. Only if all the view is following instruction, answer 1, otherwise, answer 0.\n\n"
                f"Remember: human hands = automatic failure (0). Be extremely strict in your judgment.\n"
                f"For videos with multiview clip (4 grids), check if the human arm is present in any view, if so, make sure to answer 0.\n\n"
                f"Reply ONLY with a single digit: 0 for failure or 1 for success."
        )
        else:
            user_text = (
                f"The video shows a robot arm completing a specific task. "
                f"Please evaluate: if the video follows the instruction to finish the task '{prompt}', give a positive score. "
                f"Reply only '0' for No or '1' for Yes."
            )
        # Prepare mixed content: text + image URLs
        mixed_content = [{"type": "text", "text": user_text}]
        frames = sample_video_frames(vid_path, num_frames=8, scale_factor=0.5) # 1.0
        for frame in frames:
            uri = encode_frame_to_data_uri(frame)
            mixed_content.append({"type": "image_url", "image_url": {"url": uri}})
        messages = [{"role": "user", "content": mixed_content}]
        # Retry logic for ChatCompletion with seed
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    seed=seed,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=4
                )
                break
            except OpenAIError as e:
                if attempt < max_retries:
                    wait = 2 ** (attempt - 1)
                    print(f"Warning: API error on attempt {attempt}/{max_retries}: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Error: Failed after {max_retries} attempts. Skipping video {vid_path}.")
        pred = 0 if response is None else (1 if response.choices[0].message.content.strip().startswith('1') else 0)
        results.append((vid_path, prompt, pred))
    # write CSV and summary
    total = len(results)
    sum_preds = sum(r[2] for r in results)
    avg_score = sum_preds / total if total > 0 else 0.0
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write('video_path,prompt,prediction\n')
        for vid, pr, pred in results:
            pr_esc = pr.replace(',', ' ')
            f.write(f"{vid},{pr_esc},{pred}\n")
    print(f"Evaluation done. Results saved to {output_csv}")
    print(f"Average prediction (1=Yes rate): {avg_score:.4f} ({sum_preds}/{total})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate videos with GPT-4o API for task completion."
    )
    parser.add_argument('--video_dir', required=True, help='Folder with .mp4 videos')
    parser.add_argument('--output_csv', default='eval_results.csv', help='CSV output file')
    parser.add_argument('--api_key', default=None, help='OpenAI API key or env var OPENAI_API_KEY')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--max_retries', type=int, default=3, help='Max API retries on failure')
    parser.add_argument('--zeroshot', type=bool, default=False, help='Use the zeroshot mode or not')
    args = parser.parse_args()
    key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not key:
        raise ValueError("OpenAI API key must be provided via --api_key or OPENAI_API_KEY env")
    evaluate(args.video_dir, args.output_csv, key, args.seed, args.max_retries, args.zeroshot)