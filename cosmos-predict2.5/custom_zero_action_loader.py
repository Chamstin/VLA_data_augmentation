# Custom action loader that generates zero actions for testing
import numpy as np
import mediapy


def load_zero_action_fn():
    """
    Custom action loading function that generates zero actions.
    Useful for testing the model without real robot actions.
    """
    
    def load_fn(json_data: dict, video_path: str, args) -> dict:
        """
        Load zero actions for testing.
        
        Args:
            json_data: JSON data (not used in this case)
            video_path: Path to the initial image
            args: Inference arguments
            
        Returns:
            Dictionary containing zero actions and initial frame
        """
        # Calculate number of actions needed
        # For 15 seconds at 20 fps = 300 frames
        # With chunk_size=12, we need at least 25 chunks * 12 = 300 actions
        fps = args.save_fps  # 20 fps
        duration_seconds = 15
        total_frames = fps * duration_seconds  # 300 frames
        num_actions = total_frames
        
        # Create zero actions (7D: x, y, z, roll, pitch, yaw, gripper)
        # Shape: (num_actions, 7)
        actions = np.zeros((num_actions, 7), dtype=np.float32)
        
        # Load initial frame
        # Assuming video_path is actually an image path for the initial frame
        try:
            img_array = mediapy.read_image(video_path)
        except:
            # If it's a video, extract first frame
            video_array = mediapy.read_video(video_path)
            img_array = video_array[0]
        
        # Resize to specified resolution
        if args.resolution != "none":
            h, w = map(int, args.resolution.split(","))
            img_array = mediapy.resize_image(img_array, (h, w))
        
        print(f"Generated {num_actions} zero actions for {duration_seconds} seconds at {fps} fps")
        print(f"Initial frame shape: {img_array.shape}")
        
        return {
            "actions": actions,
            "initial_frame": img_array,
            "video_array": np.array([img_array]),  # Single frame as video
            "video_path": video_path,
        }
    
    return load_fn
