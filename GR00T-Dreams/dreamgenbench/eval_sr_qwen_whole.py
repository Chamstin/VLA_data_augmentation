from tqdm import tqdm
import argparse
import glob
from pathlib import Path
import torch
import os
from dreamgenbench.utils import sample_video_frames, set_seed, load_model, load_processor

def evaluate(video_dir: str, output_csv: str, device: str = None, zeroshot: bool = False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    processor = load_processor()

    results = []  # list of tuples (vid_path, prompt, prediction_int)
    video_paths = glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)

    for vid_path in tqdm(video_paths, desc="Evaluating videos"):
        # derive prompt from filename (remove underscores)
        if "cogvideo" in vid_path:
            # split the path to get the prompt
            if "zeroshot" in vid_path:
                prompt = vid_path.split("/", 8)[-1].split("_", 1)[1].replace(".mp4", "")
            else:
                prompt = vid_path.split("/")[-3].split("_", 1)[1]
                prompt = prompt.replace("_", " ")

        elif "hunyuan" in vid_path:
            prompt = vid_path.split("/")[-1].split("_", 1)[1]
            prompt = prompt.split("_", 4)[-1]
            prompt = prompt.replace(".mp4", "")
        elif "wan" in vid_path:
            prompt = Path(vid_path).stem.split("_", 1)[1]
        else: # other models should be merged with WAN later
            prompt = Path(vid_path).stem.split("_", 1)[1]
        
        # make sure the video path is valid
        if not os.path.exists(vid_path):
            print(f"Video path does not exist: {vid_path}")
            continue

        prompt = prompt.replace("_", " ")
        if zeroshot:
            user_text = (
                f"You are evaluating if a robot arm correctly follows this instruction: '{prompt}'\n\n"
                f"CRITICAL EVALUATION PROCESS:\n"
                f"1. FIRST CHECK: If you see HUMAN HANDS instead of robot arms, IMMEDIATELY ANSWER 0.\n"
                f"2. SECOND CHECK: Only if robot arms confirmed, verify if the instruction is followed exactly.\n\n"
                f"Remember: human hands = automatic failure (0). Be extremely strict in your judgment.\n"
                f"Reply ONLY with a single digit: 0 for failure or 1 for success."
        )
        else:
            user_text = (
                f"The video shows a robot arm completing a specific task. "
                f"Does the video follow the instruction to finish the task: '{prompt}'? If it fails to follow the instruction (e.g. miss the object, action or do some other actions), please answer 0. "
                f"Answer 0 for No or 1 for Yes. Reply only 0 or 1."
            )
        
        # Sample 16 frames from the video and resize to half the original size
        frame_images = sample_video_frames(vid_path, num_frames=49, scale_factor=0.3)
        
        # Create a message with multiple images (frames) instead of a video
        message_content = [{"type": "text", "text": user_text}]
        
        # Add each frame as an image
        for frame in frame_images:
            message_content.append({"type": "image", "image": frame})
        
        messages = [
            {
                "role": "user",
                "content": message_content
            }
        ]

        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process all frames as images
        image_inputs = frame_images
        
        inputs = processor(
            text=[text],
            images=image_inputs,
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

    # write to CSV and compute average
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
    parser.add_argument('--zeroshot', type=bool, default=False, help='Use the zeroshot mode or not')

    args = parser.parse_args()
    
    set_seed(args.seed)
    evaluate(args.video_dir, args.output_csv, args.device)