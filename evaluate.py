import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.frame_dataset import FrameDataset
from model.cnn_lstm_model import CNNLSTM
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import config


transform = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

dataset = FrameDataset(
    config.FRAME_DIR, num_frames=config.NUM_FRAMES, transform=transform
)
loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

model = CNNLSTM(num_classes=config.NUM_CLASSES).to(config.DEVICE)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for frames, labels in loader:
        frames = frames.to(config.DEVICE)
        outputs = model(frames)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

print(classification_report(all_labels, all_preds, digits=4))

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
