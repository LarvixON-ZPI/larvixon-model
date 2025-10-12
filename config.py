import os
import torch
from dotenv import load_dotenv

load_dotenv()

FRAME_DIR = os.getenv("FRAME_DIR", "inference_frames/seq1")
DATA_DIR = os.getenv("DATA_DIR", "data/")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))  # Adjust based on your GPU memory
NUM_FRAMES = int(os.getenv("NUM_FRAMES", "150"))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "5"))
MODEL_PATH = os.getenv("MODEL_PATH", "cnn_lstm.pt")
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "25"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

_default_classes = "ethanol,cocaine,ketamine,morphine,tetrodotoxin"
CLASS_NAMES = os.getenv("CLASS_NAMES", _default_classes).split(",")

S3_BUCKET = "your-bucket"
S3_PREFIX = "videos/"


DISH_TO_CLASS = {
    0: "Cocaine",
    1: "Ethanol",
    2: "Ketamine",
    3: "Morphine",
    4: "Tetrodotoxin",
    5: "Ethanol",       
}

ROI_BOXES = [
    (x1, y1, w1, h1),  # dish 0
    (x2, y2, w2, h2),  # dish 1
    (x3, y3, w3, h3),  # dish 2
    (x4, y4, w4, h4),  # dish 3
    (x5, y5, w5, h5),  # dish 4
    (x6, y6, w6, h6),  # dish 5
]

if len(CLASS_NAMES) != NUM_CLASSES:
    raise ValueError(
        f"Number of class names ({len(CLASS_NAMES)}) doesn't match NUM_CLASSES ({NUM_CLASSES})"
    )
