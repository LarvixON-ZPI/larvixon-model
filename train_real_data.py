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

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def list_s3_videos(bucket, prefix):
    s3 = boto3.client(
            "s3",
            endpoint_url="https://s3min2.e-science.pl",  
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name="us-east-1"  
    )
    token = None
    while True:
        kwargs = dict(Bucket=bucket, Prefix=prefix)
        if token: kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            if obj["Key"].lower().endswith(".mp4"):
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

def detect_first_motion_frame(cap, roi, max_check=50, diff_thresh=15):
    x, y, w, h = roi
    prev_gray = None
    for i in range(max_check):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = diff.mean()
            if mean_diff > diff_thresh:
                return i 
        prev_gray = gray
    return 0


def extract_6_dishes_to_frame_folders(video_path, out_root, num_frames, roi_boxes, dish_to_class):
    """
    Writes frames into: out_root/<ClassName>/frames_<video-stem>_dishK/frame_XXXX.png
    Returns total sequences written.
    """
    os.makedirs(out_root, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    stem = os.path.splitext(os.path.basename(video_path))[0]
    stem = os.path.splitext(os.path.basename(video_path))[0]
    lower_name = stem.lower()

    for key, cname in dish_to_class.items():
        if cname.lower()[:3] in lower_name:  # this is to get ones that only hve ethanol
            dish_to_class = {k: cname for k in dish_to_class}
            break

    start_offsets = []
    for roi in roi_boxes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        start_offsets.append(detect_first_motion_frame(cap, roi))


    idxs_per_dish = [
        sample_frame_indices(total - start_offsets[k], num_frames, start=start_offsets[k])
        for k in range(len(roi_boxes))
    ]
    targets = []
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
                continue
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            out_path = os.path.join(targets[k], f"frame_{j:04d}.png")
            cv2.imwrite(out_path, crop)
    cap.release()

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

            logger.info("Extracting 6 dishes to frame folders")
            extract_6_dishes_to_frame_folders(
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
