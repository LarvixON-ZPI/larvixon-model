# Classification Model Larvixon

This model performs classification of larvae behavior based on sequences of image frames extracted from video. It predicts the substance that the larvae were injected with, based on visible behavioral patterns.

The architecture combines:

- **CNN (ResNet18)** – extracts spatial features from each frame  
- **LSTM** – captures temporal dynamics across the frame sequence  
- **Fully Connected Layer** – outputs the predicted class

Input: sequence of `N` frames, shape `[N, 3, 112, 112]`  
Output: class label (`substance_a`, `substance_b`, ...)

The model is trained using PyTorch on frame folders grouped by class.

## Manual Use

- Input frames into inference_frames/seq1 directory, in .png format
- Set config variables in evaluate.py
- run python evaluate.py

## ⚙️ Configuration

🔹 Variable Explanations

- **FRAME_DIR** – path to a folder with frames for inference (prediction).

- **DATA_DIR** – root folder containing the dataset used for training/validation, organized by class subfolders.

- **BATCH_SIZE** – number of sequences processed in parallel during training/evaluation. Lower it if GPU memory is small (≤4 GB → use 1–2).

- **NUM_FRAMES** – how many frames per sequence the model uses. Extra frames are cut, fewer frames are padded.

- **NUM_CLASSES** – total number of classes (substances) to predict. Must match your dataset.

- **MODEL_PATH** – file path where the trained model weights are saved/loaded.

- **DEVICE** – automatically selects "cuda" if a GPU is available, otherwise "cpu".

- **CLASS_NAMES** – list of human-readable class labels in the order they are mapped during training.
