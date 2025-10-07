FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libgl1 && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV NUM_CLASSES=5
ENV NUM_FRAMES=150
ENV MODEL_PATH=cnn_lstm.pt
ENV IMG_SIZE=112

EXPOSE $API_PORT
CMD uvicorn app:app --host $API_HOST --port $API_PORT