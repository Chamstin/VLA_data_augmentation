#!/usr/bin/env python3
import os
import torch
import cv2
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from decord import VideoReader, cpu

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
    # Load Qwen2.5-VL model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="auto",
        device_map=device,
        trust_remote_code=True
    )
    return model


def load_processor():
    # Load processor
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True
    )
    return processor

def sample_video_frames(video_path, num_frames=16, scale_factor=0.5):
    """
    Sample num_frames evenly from the video and resize them to a scaled size using decord.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to sample (default: 16)
        scale_factor: Scale factor for resizing (default: 0.5)
        
    Returns:
        List of sampled frames as numpy arrays in RGB format
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        raise RuntimeError(f"Failed to open video with decord: {video_path}\n{e}")
    
    frame_count = len(vr)
    orig_height, orig_width = vr[0].shape[0:2]
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)

    # Compute frame indices to sample
    if frame_count < num_frames:
        indices = np.arange(frame_count)
        pad_count = num_frames - frame_count
    else:
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        pad_count = 0
    # Load and process frames
    frames = []
    for idx in indices:
        frame = vr[idx].asnumpy()  # RGB format
        frame = cv2.resize(frame, (new_width, new_height)) # , interpolation=cv2.INTER_AREA)
        frames.append(frame)

    # Pad with last frame if needed
    if pad_count > 0:
        last_frame = frames[-1] if frames else np.zeros((new_height, new_width, 3), dtype=np.uint8)
        frames.extend([last_frame.copy() for _ in range(pad_count)])

    return frames
