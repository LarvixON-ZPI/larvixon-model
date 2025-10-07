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

For easy configuration copy the contents of `.env.example` into your own `.env` file and modify the variables as needed.

🔹 Variable Explanations

- **FRAME_DIR** – path to a folder with frames for inference (prediction).

- **DATA_DIR** – root folder containing the dataset used for training/validation, organized by class subfolders.

- **BATCH_SIZE** – number of sequences processed in parallel during training/evaluation. Lower it if GPU memory is small (≤4 GB → use 1–2).

- **NUM_FRAMES** – how many frames per sequence the model uses. Extra frames are cut, fewer frames are padded.

- **NUM_CLASSES** – total number of classes (substances) to predict. Must match your dataset.

- **MODEL_PATH** – file path where the trained model weights are saved/loaded.

- **DEVICE** – automatically selects "cuda" if a GPU is available, otherwise "cpu".

- **CLASS_NAMES** – list of human-readable class labels in the order they are mapped during training.


## 🐳 Docker Setup
To build the docker image run one of the following commands
```bash
docker buildx build -t larvixon-model . 
```
```bash
docker build -t larvixon-model .
```

To run the docker container with environment variables from a file:
```bash
docker run --env-file .env -p 8000:8000 larvixon-model
```
Alternatively, you can pass environment variables directly in the command:
```bash
docker run -e API_PORT=8000 -e MODEL_PATH=cnn_lstm.pt -p 8000:8000 larvixon-model
```