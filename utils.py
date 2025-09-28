import cv2
import os
import numpy as np
from colored_logger import logger

def video_to_fixed_frames(video_path, output_dir, num_frames, prefix="frame"):
    """
    Extract exactly `num_frames` frames evenly spaced across the video.

    Parameters
    ----------
    video_path : str
        Path to input video file.
    output_dir : str
        Directory where frames will be saved.
    num_frames : int
        Number of frames to extract from the video.
    prefix : str, optional
        Prefix for saved frame filenames.
    """
    logger.info(f"Processing video: {video_path}, extracting {num_frames} frames.")
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError("Video has no frames.")

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    count = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_path = os.path.join(output_dir, f"{prefix}_{count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1

    cap.release()
    logger.info(f"Extracted {count} frames to {output_dir}")