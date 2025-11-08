import os
import sys

from src import config
sys.path.append(".")
import tempfile
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision import transforms
from src.models.cnn_lstm_model import CNNLSTM
from src.utils.logger import logger
from src.utils.video_utils import video_to_fixed_frames
from src.config import NUM_FRAMES, NUM_CLASSES, MODEL_PATH, DEVICE, CLASS_NAMES, LEARNING_RATE, CHECKPOINT_PATH, SAVE_PATH


transform = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

model = CNNLSTM(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

app = FastAPI(title="Larvae Injection Classification API")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "video.mp4")
        with open(video_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        test_dir = "data/"
        frames_dir = os.path.join(test_dir, "frames")
        video_to_fixed_frames(video_path, frames_dir, NUM_FRAMES)

        frames = []
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith((".jpg", ".png"))])

        for fname in frame_files[:NUM_FRAMES]:
            img_path = os.path.join(frames_dir, fname)
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            frames.append(img)

        while len(frames) < NUM_FRAMES:
            frames.append(torch.zeros_like(frames[0]) if frames else torch.zeros(3, 112, 112))

        input_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)
        logger.info(f"Processed {len(frames)} frames for prediction.")
        logger.info(f"Input tensor shape: {input_tensor.shape}")

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.sigmoid(output,)
            probs = probs.squeeze().cpu().numpy()
            logger.info(f"Prediction probabilities: {probs}")

        results = { cls: float(p * 100) for cls, p in zip(CLASS_NAMES, probs)}

        return {"predictions": results}


@app.post("/train")
async def train(file: UploadFile = File(...), class_name: str = "Nothing", epochs: int = 2):
    if class_name not in CLASS_NAMES:
        return {"error": f"Invalid class name. Choose from {CLASS_NAMES}"}

    device = DEVICE
    model = CNNLSTM(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            if "opt_state" in ckpt:
                optimizer.load_state_dict(ckpt["opt_state"])
        else:
            model.load_state_dict(ckpt)

    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "video.mp4")
        from scripts.train_real_data import train_one_video
        train_one_video(model, optimizer=optimizer, data_dir=video_path, device=device, num_frames=NUM_FRAMES, batch_size=NUM_CLASSES, epochs=epochs)

    torch.save(model.state_dict(), SAVE_PATH)
    logger.info(f"Saved final model to {SAVE_PATH}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"status": f"Training initiated for class '{class_name}' for {epochs} epochs."}

