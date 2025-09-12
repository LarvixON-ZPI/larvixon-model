import io
import zipfile
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision import transforms
from model.cnn_lstm_model import CNNLSTM

NUM_CLASSES = 5
NUM_FRAMES = 16
MODEL_PATH = "cnn_lstm.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["ethanol", "meth", "cocaine", "substance_d", "substance_e"]

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
    contents = await file.read()
    zip_bytes = io.BytesIO(contents)
    with zipfile.ZipFile(zip_bytes, "r") as zip_ref:
        frame_files = sorted(zip_ref.namelist())
        frames = []

        for fname in frame_files[:NUM_FRAMES]:
            with zip_ref.open(fname) as f:
                img = Image.open(f).convert("RGB")
                img = transform(img)
                frames.append(img)

    while len(frames) < NUM_FRAMES:
        frames.append(torch.zeros_like(frames[0]))

    input_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)  

    with torch.no_grad():
        output = model(input_tensor)         
        probs = F.softmax(output, dim=1)     
        probs = probs.squeeze().cpu().numpy()

    result = {cls: float(p * 100) for cls, p in zip(CLASS_NAMES, probs)}

    return {"predictions": result}
