# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements-tf2.20.txt requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-tf2.20.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy the model and necessary files
COPY public/models/combined_saved_model /app/public/models/combined_saved_model
COPY public/assets/data/agriculture_datasets.json /app/public/assets/data/agriculture_datasets.json
COPY server.py predict_combined.py ./

# Create necessary directories
RUN mkdir -p /app/public/assets/data

# Expose port
EXPOSE 8001

# Run with proper production settings
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1", "--limit-concurrency", "20"]
