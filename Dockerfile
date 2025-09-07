FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Install system dependencies including common lib* packages
RUN apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3-pip git \
    build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip

WORKDIR /app

# Copy requirements.txt first for cache optimization
COPY requirements.txt ./

# Install dependencies from requirements.txt
RUN python3.11 -m pip install -r requirements.txt

# Copy entire 'sam' folder containing 'sam2' package inside the image at /app/sam
COPY sam /app/sam

# Install SAM2 package in editable mode from /app/sam/sam2
RUN python3.11 -m pip install -e /app/sam/sam2

# Set working directory to the actual Python package dir to avoid import shadowing
WORKDIR /app/sam/sam2




version: "3.8"

services:
  sam2:
    image: sam2-image
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - C:/Users/gopic/OneDrive/Desktop/gitcheck/sam:/app/sam
    working_dir: /app/sam/sam2
    command: bash -c "python3 your_script.py"






