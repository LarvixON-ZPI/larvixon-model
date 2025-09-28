import io
import os
import tempfile
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision import transforms
from model.cnn_lstm_model import CNNLSTM
from utils import video_to_fixed_frames

NUM_CLASSES = 5
NUM_FRAMES = 150
MODEL_PATH = "cnn_lstm.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["ethanol", "cocaine", "ketamine", "morphine", "tetrodotoxin"]

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

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
        
        frames_dir = os.path.join(temp_dir, "frames")
        video_to_fixed_frames(video_path, frames_dir, NUM_FRAMES)

        frames = []
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
        
        for fname in frame_files[:NUM_FRAMES]:
            img_path = os.path.join(frames_dir, fname)
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            frames.append(img)
        
        while len(frames) < NUM_FRAMES:
            frames.append(torch.zeros_like(frames[0]) if frames else torch.zeros(3, 112, 112))
        
        input_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            probs = probs.squeeze().cpu().numpy()
        
        result = {cls: float(p * 100) for cls, p in zip(CLASS_NAMES, probs)}
        
        return {"predictions": result}
