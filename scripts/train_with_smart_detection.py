import os
import sys
sys.path.append(".")
import tempfile
import torch
import boto3
from torch.utils.data import DataLoader
import torch.nn as nn
from src.utils.logger import logger
import src.config as config
from src.datasets.frame_dataset import FrameDataset
from src.models.cnn_lstm_model import CNNLSTM
from src.utils.smart_extraction import SmartVideoExtractor
from torchvision import transforms
import pandas as pd

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def read_video_index():
    """Read the video index CSV file."""
    df = pd.read_csv("video_index.csv")
    return df

def list_s3_videos(bucket, prefix):
    """List videos from S3 bucket that match the video index."""
    s3 = boto3.client(
        "s3",
        endpoint_url="https://s3min2.e-science.pl",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )
    token = None
    video_index = read_video_index()
 
    while True:
        kwargs = dict(Bucket=bucket, Prefix=prefix)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)

        for obj in resp.get("Contents", []):
            logger.debug(f"Found S3 object: {obj['Key']}")
            if (obj["Key"].lower().endswith(".mov") and 
                obj["Key"].startswith("L") and 
                obj["Key"].split(".")[0] in video_index["Video name"].values):
                yield obj["Key"]

        token = resp.get("NextContinuationToken")
        if not token:
            break

def download_s3(bucket, key, dest_path):
    """Download file from S3."""
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

def train_one_video(model, optimizer, data_dir, device=config.DEVICE,
                   num_frames=config.NUM_FRAMES, 
                   batch_size=config.BATCH_SIZE, 
                   epochs=config.EPOCHS_PER_VIDEO):
    """
    Train model on sequences from one video.

    Parameters:
    - model: The neural network model
    - optimizer: The optimizer
    - data_dir: Directory containing the training sequences
    - device: Device to train on
    - num_frames: Number of frames per sequence
    - batch_size: Batch size for training
    - epochs: Number of epochs to train

    Returns:
    - total_sequences: Number of sequences processed
    - accuracy: Training accuracy
    """
    dataset = FrameDataset(data_dir, num_frames=num_frames, 
                          transform=transform)
    if len(dataset) == 0:
        logger.warning(f"No sequences found in {data_dir}, skipping training.")
        return 0, 0.0

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Training on {len(dataset)} sequences for {epochs} epochs.")
    model.train()
    total, correct, running_loss = 0, 0, 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)


            logits = model(frames)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            epoch_loss += loss.item()

            pred = logits.argmax(1)
            correct_batch = (pred == labels).sum().item()
            batch_size_actual = labels.size(0)

            correct += correct_batch
            total += batch_size_actual
            epoch_correct += correct_batch
            epoch_total += batch_size_actual

        epoch_acc = (100.0 * epoch_correct / epoch_total) if epoch_total else 0.0
        epoch_avg_loss = epoch_loss / len(loader) if len(loader) else 0.0
        logger.info(f"Epoch {epoch+1}/{epochs}: Loss={epoch_avg_loss:.4f}, "
                   f"Acc={epoch_acc:.1f}%")

    acc = (100.0 * correct / total) if total else 0.0
    avg_loss = running_loss / (len(loader) * epochs) if len(loader) else 0.0

    logger.info(f"Training completed: {total} total samples, "
               f"Avg Loss={avg_loss:.4f}, Acc={acc:.1f}%")

    return total, acc

def analyze_video_before_training(video_path: str) -> bool:
    """
    Analyze video quality and determine if it's suitable for training.

    Parameters:
    - video_path: Path to the video file

    Returns:
    - True if video is suitable for training
    """
    try:
        extractor = SmartVideoExtractor(use_detected_rois=True)
        analysis = extractor.analyze_video_quality(video_path)
 
        if "error" in analysis:
            logger.error(f"Video analysis failed: {analysis['error']}")
            return False

        video_info = analysis["video_info"]
        quality_metrics = analysis["quality_metrics"]
        detection_results = analysis["detection_results"]

        logger.info(f"Video Analysis:")
        logger.info(f"  Duration: {video_info['duration_seconds']:.1f}s")
        logger.info(f"  Resolution: {video_info['resolution']}")
        logger.info(f"  FPS: {video_info['fps']:.1f}")
        logger.info(f"  Avg Brightness: {quality_metrics['avg_brightness']:.1f}")
        logger.info(f"  Avg Contrast: {quality_metrics['avg_contrast']:.1f}")
        logger.info(f"  Detected Dishes: {detection_results['num_dishes_detected']}")

        for rec in analysis["recommendations"]:
            logger.info(f"  Recommendation: {rec}")

        if detection_results["num_dishes_detected"] == 0:
            logger.warning("No dishes detected - video may not be suitable")
            return False

        if quality_metrics["avg_brightness"] < 20:
            logger.warning("Video very dark - may affect training quality")

        if quality_metrics["avg_contrast"] < 10:
            logger.warning("Very low contrast - may affect training quality")

        return True

    except Exception as e:
        logger.error(f"Video analysis failed with exception: {e}")
        return False

def main():
    """Main training function with smart detection."""
    logger.info("Starting training with smart petri dish detection:")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Model: {CNNLSTM.__name__}")
    logger.info(f"Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"Batch Size: {config.BATCH_SIZE}")
    logger.info(f"Num Frames: {config.NUM_FRAMES}")
    logger.info(f"Epochs per Video: {config.EPOCHS_PER_VIDEO}")
    logger.info(f"S3 Bucket: {config.S3_BUCKET}")
    logger.info(f"Class Names: {config.CLASS_NAMES}")

    device = config.DEVICE
    model = CNNLSTM(num_classes=config.NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    if os.path.exists(config.CHECKPOINT_PATH):
        logger.info(f"Loading checkpoint from {config.CHECKPOINT_PATH}")
        ckpt = torch.load(config.CHECKPOINT_PATH, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            if "opt_state" in ckpt:
                optimizer.load_state_dict(ckpt["opt_state"])
                logger.info("Loaded optimizer state")
        else:
            model.load_state_dict(ckpt)
        logger.info("Checkpoint loaded successfully")

    extractor = SmartVideoExtractor(use_detected_rois=True)

    total_videos_processed = 0
    total_sequences_created = 0

    for key in list_s3_videos(config.S3_BUCKET, config.S3_PREFIX):
        logger.info(f"Processing video: {key}")

        with tempfile.TemporaryDirectory() as tmp:
            local_video_path = os.path.join(tmp, os.path.basename(key))
            tmp_data_dir = os.path.join(tmp, "data")

            logger.info(f"Downloading s3://{config.S3_BUCKET}/{key}")
            download_s3(config.S3_BUCKET, key, local_video_path)
            
            logger.info("Analyzing video quality...")
            if not analyze_video_before_training(local_video_path):
                logger.warning(f"Skipping video {key} due to quality issues")
                continue

            logger.info("Extracting frames with smart detection...")
            sequences_created = extractor.extract_video_intelligently(
                video_path=local_video_path,
                output_dir=tmp_data_dir,
                num_frames=config.NUM_FRAMES,
                dish_to_class=config.DISH_TO_CLASS
            )

            if sequences_created == 0:
                logger.warning(f"No sequences created from {key}, skipping")
                continue

            total_sequences_created += sequences_created
            logger.info(f"Created {sequences_created} sequences from video")

            logger.info("Training on extracted sequences...")
            num_samples, accuracy = train_one_video(
                model,
                optimizer,
                data_dir=tmp_data_dir,
                device=device,
                num_frames=config.NUM_FRAMES,
                batch_size=config.BATCH_SIZE,
                epochs=config.EPOCHS_PER_VIDEO,
            )

            if num_samples > 0:
                logger.info(f"Trained on {num_samples} samples | "
                           f"Accuracy: {accuracy:.1f}%")
                total_videos_processed += 1
            else:
                logger.warning(f"No training samples from {key}")
                continue

            checkpoint = {
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "videos_processed": total_videos_processed,
                "sequences_created": total_sequences_created,
            }
            torch.save(checkpoint, config.CHECKPOINT_PATH)
            logger.info(f"Checkpoint saved after processing video {total_videos_processed}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    torch.save(model.state_dict(), config.SAVE_PATH)
    logger.info(f"Training completed!")
    logger.info(f"Processed {total_videos_processed} videos")
    logger.info(f"Created {total_sequences_created} total sequences")
    logger.info(f"Final model saved to {config.SAVE_PATH}")


if __name__ == "__main__":
    main()
