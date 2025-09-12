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
