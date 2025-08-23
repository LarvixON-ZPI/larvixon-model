import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class FrameDataset(Dataset):
    def __init__(self, root_dir, num_frames=16,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.samples = []

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                frame_paths = sorted(glob.glob(os.path.join(class_dir, "*.jpg")))
                for i in range(0, len(frame_paths) - num_frames + 1, num_frames):
                    self.samples.append(frame_paths[i:i + num_frames])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_path, label = self.samples[idx]
        frame_paths = sorted(glob.glob(os.path.join(seq_path, "*.jpg")))[:self.num_frames]

        frames = []
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        while len(frames) < self.num_frames:
            frames.append(torch.zeros_like(frames[0]))

        return torch.stack(frames[:self.num_frames]), label

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
