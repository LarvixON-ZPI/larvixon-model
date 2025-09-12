import os
import glob
import torch
from PIL import Image
from torchvision import transforms
from model.cnn_lstm_model import CNNLSTM

FRAME_DIR = "inference_frames/seq1"   
NUM_FRAMES = 16
NUM_CLASSES = 5
MODEL_PATH = "cnn_lstm.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Cocaine', 'Ethanol', 'Ketamine', 'Morphine', 'Tetrodotoxin']

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = CNNLSTM(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

frame_paths = sorted(glob.glob(os.path.join(FRAME_DIR, "*.jpg")))[:NUM_FRAMES]

frames = []
for fp in frame_paths:
    img = Image.open(fp).convert("RGB")
    img = transform(img)
    frames.append(img)

while len(frames) < NUM_FRAMES:
    frames.append(torch.zeros_like(frames[0]))

input_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)  

with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)    
    probs = probs.squeeze().cpu().numpy()   
    top = sorted(zip(class_names, probs), key=lambda x: -x[1])
    for cls, p in top:
        print(f"{cls}: {p*100:.2f}%")


