# Use official CUDA-enabled Ubuntu base image for GPU support
#FROM nvidia/cuda:12.6.0-runtime-ubuntu20.04
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04
# Set working directory inside container
WORKDIR /app

# Install git, Python 3.11, python3-pip, and clean up cache to reduce image size
RUN apt-get update && \
    apt-get install -y git python3.11 python3.11-dev python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN python3.11 -m pip install --upgrade pip

RUN python3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#copy requirement.txt
COPY requirement.txt ./

RUN python3.11 -m pip install -r requirement.txt

# Clone SAM2 repository and install it editable mode
RUN git clone https://github.com/facebookresearch/sam2.git && \
    python3.11 -m pip install -e sam2


