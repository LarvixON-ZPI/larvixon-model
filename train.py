import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.frame_dataset import FrameDataset
from model.cnn_lstm_model import CNNLSTM
from tqdm import tqdm

DATA_DIR = "data/"
NUM_CLASSES = 5  
NUM_EPOCHS = 20
BATCH_SIZE = 2
NUM_FRAMES = 150
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
from torch.utils.data import random_split

dataset = FrameDataset(DATA_DIR, num_frames=NUM_FRAMES, transform=transform)

total_size = len(dataset)
val_size = int(0.2 * total_size)
train_size = total_size - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CNNLSTM(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0

    for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        frames, labels = frames.to(DEVICE), labels.to(DEVICE)

        outputs = model(frames)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

model.eval()
val_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for frames, labels in val_loader:
        frames, labels = frames.to(DEVICE), labels.to(DEVICE)
        outputs = model(frames)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

val_acc = 100.0 * correct / total
val_loss /= len(val_loader)
print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

torch.save(model.state_dict(), "cnn_lstm.pt")
