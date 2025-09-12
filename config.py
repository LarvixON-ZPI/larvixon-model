import torch

FRAME_DIR = "inference_frames/seq1"   
DATA_DIR = "data/"
BATCH_SIZE = 2 # Adjust based on your GPU memory, dont go above 2 for <=4GB GPU (trust)
NUM_FRAMES = 150
NUM_CLASSES = 5
MODEL_PATH = "cnn_lstm.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Cocaine', 'Ethanol', 'Ketamine', 'Morphine', 'Tetrodotoxin']