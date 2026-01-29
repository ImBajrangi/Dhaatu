FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.10 and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Create user for HF Spaces (before copying files)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /home/user/app

# Copy requirements first for caching
COPY --chown=user:user requirements.txt .

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY --chown=user:user . .

# Switch to user
USER user

# Set environment
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    SPCONV_ALGO=native \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
