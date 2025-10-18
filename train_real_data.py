import os
import io
import cv2
import glob
import math
import boto3
import torch
import shutil
import tempfile
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from colored_logger import logger
import config
from datasets.frame_dataset import FrameDataset
from model.cnn_lstm_model import CNNLSTM
import pandas as pd

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def read_video_index():
    df = pd.read_csv("video_index.csv")
    return df

def list_s3_videos(bucket, prefix):
    s3 = boto3.client(
            "s3",
            endpoint_url="https://s3min2.e-science.pl",  
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name="us-east-1"  
    )
    token = None
    video_index = read_video_index()
    while True:
        kwargs = dict(Bucket=bucket, Prefix=prefix)
        if token: kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            logger.debug(f"Found S3 object: {obj['Key']}")
            if obj["Key"].lower().endswith(".mov") and obj["Key"].startswith("L") and obj["Key"].split(".")[0] in video_index["Video name"].values:
                yield obj["Key"]
        token = resp.get("NextContinuationToken")
        if not token: break

def download_s3(bucket, key, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    s3 = boto3.client(
        "s3",
        endpoint_url="https://s3min2.e-science.pl",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )

    with open(dest_path, "wb") as f:
        obj = s3.get_object(Bucket=bucket, Key=key)
        f.write(obj["Body"].read())

def sample_frame_indices(total, num_frames, start=0):
    if total <= num_frames:
        return list(range(start, total))
    step = (total - start) // num_frames
    return [start + i * step for i in range(num_frames)]

def detect_first_motion_frame(cap, roi, max_check=100, diff_thresh=7, max_diff=100):
    logger.info(f"Detecting first motion frame in ROI: {roi} with max_check={max_check}, diff_thresh={diff_thresh}, max_diff={max_diff}")
    max_val_found = 0
    x, y, w, h = roi
    prev_gray = None
    for i in range(max_check):
        ok, frame = cap.read()
        if not ok:
            logger.warning(f"Failed to read frame {i} during motion detection")
            break
        gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        logger.debug(f"Motion detection frame {i} with mean pixel value {gray.mean()}")
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = diff.mean()
            logger.debug(f"Frame {i} mean diff: {mean_diff}, diff_thresh: {diff_thresh}")
            if max_val_found < mean_diff:
                max_val_found = mean_diff

            if mean_diff > diff_thresh and mean_diff < max_diff and gray.mean() < 10:
                logger.info(f"Detected motion at frame {i} with mean diff {mean_diff}")
                return i 
        prev_gray = gray
    logger.info(f"No significant motion detected within {max_check} frames. Max diff found: {max_val_found}")
    return 0

def detect_first_motion_frame_v2(cap, roi, max_check=500, low_thresh=3, high_thresh=25, sustain_frames=5):
    """
    Detect first frame where motion inside ROI exceeds low_thresh for
    several consecutive frames, but ignore very strong spikes (hands).
    """
    logger.info(f"Detecting first motion frame (v2) in ROI: {roi} with max_check={max_check}, low_thresh={low_thresh}, high_thresh={high_thresh}, sustain_frames={sustain_frames}")
    x, y, w, h = roi
    prev_gray = None
    motion_streak = 0

    for i in range(max_check):
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = diff.mean()

            if mean_diff > high_thresh:
                motion_streak = 0
                continue

            if mean_diff > low_thresh:
                motion_streak += 1
                if motion_streak >= sustain_frames:
                    logger.info(f"Detected sustained motion starting at frame {i - sustain_frames + 1}")
                    return max(0, i - sustain_frames)
            else:
                motion_streak = 0

        prev_gray = gray

    return 0

def detect_first_larva_frame(cap, roi, max_check=500, diff_thresh=15, 
                             min_larva_pixels=500, max_larva_pixels=3000, 
                             sustain_frames=3):
    """
    Detects the first frame where a larva is present by analyzing the area of motion.

    It ignores very large motion (hands) and very small motion (noise).

    Args:
        cap: The video capture object.
        roi: A tuple (x, y, w, h) defining the region of interest.
        max_check: Maximum number of frames to check.
        diff_thresh: Threshold for pixel difference to be considered motion (0-255).
        min_larva_pixels: The minimum number of changed pixels to be considered larva motion.
        max_larva_pixels: The maximum number of changed pixels. Above this is a hand.
        sustain_frames: How many consecutive frames motion must be sustained.
    """
    logger.info(f"Detecting first larva frame with pixel counting...")
    
    x, y, w, h = roi
    prev_gray = None
    motion_streak = 0
    hand_detected_recently = False 

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(max_check):
        ok, frame = cap.read()
        if not ok:
            logger.warning("Could not read frame from video.")
            break

        roi_frame = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is not None:
            frame_delta = cv2.absdiff(prev_gray, gray)

            thresh = cv2.threshold(frame_delta, diff_thresh, 255, cv2.THRESH_BINARY)[1]

            pixel_change_count = cv2.countNonZero(thresh)

            if pixel_change_count > max_larva_pixels:
                motion_streak = 0
                hand_detected_recently = True
                logger.info(f"Frame {i}: Large motion detected ({pixel_change_count} pixels). Assuming hand, resetting.")
            
            elif hand_detected_recently and pixel_change_count < min_larva_pixels:
                hand_detected_recently = False
                logger.info(f"Frame {i}: Scene stabilizing after hand.")

            elif not hand_detected_recently and pixel_change_count >= min_larva_pixels:
                motion_streak += 1
                logger.info(f"Frame {i}: Potential larva motion detected ({pixel_change_count} pixels). Streak: {motion_streak}/{sustain_frames}")
                if motion_streak >= sustain_frames:
                    start_frame = max(0, i - sustain_frames + 1)
                    logger.info(f"Sustained larva motion detected! Start frame is ~{start_frame}")
                    return start_frame
            else:
                motion_streak = 0

        prev_gray = gray

    logger.warning("No sustained larva motion found within the first max_check frames.")
    return 0 

def extract_8_dishes_to_frame_folders(video_path, out_root, num_frames, roi_boxes, dish_to_class):
    """
    Writes frames into: out_root/<ClassName>/frames_<video-stem>_dishK/frame_XXXX.png
    Returns total sequences written.
    """
    os.makedirs(out_root, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video {video_path} has {total} frames.")

    stem = os.path.splitext(os.path.basename(video_path))[0]
    lower_name = os.path.splitext(os.path.basename(video_path))[0].lower()
    logger.info(f"Inferring dish classes from video name: {lower_name}")

    if "etoh" in lower_name:  # this is to get ones that only hve ethanol
        name_data = lower_name.split("_")
        strength = float(name_data[3]) if len(name_data) > 1 and name_data[3].replace('.', '', 1).isdigit() else 50.0
        logger.info(f"Detected 'etoh' in filename, assigning all dishes to Ethanol {strength}%")
        cname = "Ethanol"
        cname_full = f"{cname} {strength}%"
        dish_to_class = {k: cname_full for k in dish_to_class}

    fps = cap.get(cv2.CAP_PROP_FPS)
    offset_seconds = [170.0,155.0,95.0, 60.0, 22.5, 0.0, 0.0, 0.0]
    start_offsets = [int(s * fps) for s in offset_seconds]
    
    # start_offsets = []
    # for roi in roi_boxes:
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #     start_offsets.append(detect_first_larva_frame(cap, roi, max_check=4300, diff_thresh=15, sustain_frames=3))
    
    logger.info(f"Detected start offsets per dish: {start_offsets}")


    idxs_per_dish = [
        sample_frame_indices(total - start_offsets[k], num_frames, start=start_offsets[k])
        for k in range(len(roi_boxes))
    ]
    targets = []
    logger.info(f"Writing frames to {out_root} ...")

    for k, roi in enumerate(roi_boxes):
        cls = dish_to_class[k]
        seq_dir = os.path.join(out_root, cls, f"frames_{stem}_dish{k}")
        os.makedirs(seq_dir, exist_ok=True)
        targets.append(seq_dir)

    for k, (x, y, w, h) in enumerate(roi_boxes):
        for j, fi in enumerate(idxs_per_dish[k]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                logger.warning(f"Failed to read frame {fi} for dish {k}")
                continue
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                logger.warning(f"Empty crop for frame {fi} dish {k}")
                continue
            out_path = os.path.join(targets[k], f"frame_{j:04d}.png")
            logger.debug(f"Writing {out_path}")
            cv2.imwrite(out_path, crop)
    cap.release()
    logger.info("Done writing frames.")

    return len(targets)

def train_one_video(model, optimizer, data_dir, device, num_frames, batch_size, epochs):
    """
    Minimal train loop (per video) that reuses your dataset and settings.
    """
    dataset = FrameDataset(data_dir, num_frames=num_frames, transform=transform)
    if len(dataset) == 0:
        return 0, 0.0

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    model.train()
    total, correct, running = 0, 0, 0.0
    for _ in range(epochs):
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            logits = model(frames)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = (100.0 * correct / total) if total else 0.0
    return total, acc

def main():
    device = config.DEVICE
    model = CNNLSTM(num_classes=config.NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    if os.path.exists(config.CHECKPOINT_PATH):
        ckpt = torch.load(config.CHECKPOINT_PATH, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            if "opt_state" in ckpt:
                optimizer.load_state_dict(ckpt["opt_state"])
        else:
            model.load_state_dict(ckpt)

    for key in list_s3_videos(config.S3_BUCKET, config.S3_PREFIX):
        with tempfile.TemporaryDirectory() as tmp:
            local_mp4 = os.path.join(tmp, os.path.basename(key))
            tmp_data = os.path.join(tmp, "data")   

            logger.info(f"Downloading s3://{config.S3_BUCKET}/{key}")
            download_s3(config.S3_BUCKET, key, local_mp4)

            logger.info("Extracting 8 dishes to frame folders")
            extract_8_dishes_to_frame_folders(
                video_path=local_mp4,
                out_root=tmp_data,
                num_frames=config.NUM_FRAMES,
                roi_boxes=config.ROI_BOXES,
                dish_to_class=config.DISH_TO_CLASS
            )

            logger.info("Training on this videos generated sequences")
            nseq, acc = train_one_video(
                model, optimizer,
                data_dir=tmp_data,
                device=device,
                num_frames=config.NUM_FRAMES,
                batch_size=config.BATCH_SIZE,
                epochs=config.EPOCHS_PER_VIDEO
            )
            logger.info(f"Trained on {nseq} sequences | approx acc: {acc:.1f}%")

            torch.save({"model_state": model.state_dict(),
                        "opt_state": optimizer.state_dict()},
                       config.CHECKPOINT_PATH)


        if device.type == "cuda":
            torch.cuda.empty_cache()

    torch.save(model.state_dict(), config.SAVE_PATH)
    logger.info(f"Saved final model to {config.SAVE_PATH}")

if __name__ == "__main__":
    main()
