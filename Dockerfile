# Multi-stage Dockerfile for mammography risk prediction pipeline
# ================================================================
# Stage 1: Build environment with all dependencies
# Stage 2: Slim runtime image for inference

# Build stage
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS builder

LABEL maintainer="Manuel"
LABEL description="Deep Learning Pipeline for Breast Cancer Risk Prediction"

# System dependencies for OpenCV and pydicom
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

# Runtime stage
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /app /app

# Create directories for data and output
RUN mkdir -p /data /output /checkpoints

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Default entrypoint: training
ENTRYPOINT ["python"]
CMD ["scripts/train.py", "--config", "configs/default.yaml"]

# Example usage:
# docker build -t mammo-risk .
#
# Training:
# docker run --gpus all \
#     -v /path/to/data:/data \
#     -v /path/to/output:/output \
#     mammo-risk scripts/train.py \
#         --config configs/default.yaml \
#         --data-dir /data \
#         --output-dir /output
#
# Evaluation:
# docker run --gpus all \
#     -v /path/to/data:/data \
#     -v /path/to/checkpoints:/checkpoints \
#     -v /path/to/output:/output \
#     mammo-risk scripts/evaluate.py \
#         --checkpoint /checkpoints/best_model.pt \
#         --data-dir /data \
#         --labels-csv /data/labels_test.csv \
#         --output-dir /output
#
# Report generation:
# docker run --gpus all \
#     -v /path/to/patient:/dicom \
#     -v /path/to/checkpoints:/checkpoints \
#     -v /path/to/output:/output \
#     mammo-risk scripts/generate_report.py \
#         --checkpoint /checkpoints/best_model.pt \
#         --dicom-dir /dicom \
#         --output /output/report.pdf
