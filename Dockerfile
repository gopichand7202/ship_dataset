# enable qemu emulation
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

# create and use a buildx builder (container driver ensures local export works)
docker buildx create --name mybuilder --driver docker-container --use
docker buildx inspect --bootstrap

# build for ARM64 and write a local tar you can copy to the Jetson
docker buildx build --platform linux/arm64 -t myapp:arm64 --output type=tar,dest=./myapp-arm64.tar .

# on the Jetson (after copying the tar) load the image
# scp myapp-arm64.tar jetson:/path/ && ssh jetson
docker load -i ./myapp-arm64.tar
```# enable qemu emulation
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

# create and use a buildx builder (container driver ensures local export works)
docker buildx create --name mybuilder --driver docker-container --use
docker buildx inspect --bootstrap

# build for ARM64 and write a local tar you can copy to the Jetson
docker buildx build --platform linux/arm64 -f Dockerfile -t myapp:arm64 --output type=tar,dest=./myapp-arm64.tar .

# on the Jetson (after copying the tar) load the image
# scp myapp-arm64.tar jetson:/path/ && ssh jetson
docker load -i ./myapp-arm64.tar











FROM python:3.12

# Install system packages for git, Python, build tools
# RUN apt-get update && \
#     apt-get install -y python3.12 python3.12-dev python3-pip \
#     libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 curl git && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 curl git build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Install CUDA-enabled PyTorch (change cu118 to your CUDA version if needed)
RUN python3.12 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126 --trusted-host download.pytorch.org --trusted-host pypi.org --trusted-host files.pythonhosted.org

COPY requirement.txt /opt/app_code/requirement.txt

RUN python3.12 -m pip install --no-cache-dir -r /opt/app_code/requirement.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org

COPY sam/ /opt/sam/
RUN python3.12 -m pip install -e /opt/sam/ --trusted-host pypi.org --trusted-host files.pythonhosted.org

