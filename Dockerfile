FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY *.py .
COPY *.json .
COPY .env .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV MASTER_ADDR=localhost
ENV MASTER_PORT=29500

# Command to run training
CMD ["deepspeed", "--num_gpus=1", "train.py", "--deepspeed", "deepspeed_config.json"]
