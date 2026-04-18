# ============================================================
# FreeGrasp Demo - Dockerfile
# Base: CUDA 12.4 + cuDNN 9 + Ubuntu 22.04
# Compatible with: RTX 3090, RTX 4080, RTX A4000 (18GB+ VRAM)
# ============================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9"   
# 8.6 = RTX 3090/A4000, 8.9 = RTX 4080/4090

# ---------- System deps ----------
RUN apt-get update && apt-get install -y \
    git wget curl build-essential ninja-build \
    python3.10 python3.10-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    pip install --upgrade pip

WORKDIR /app

# ---------- PyTorch (CUDA 12.4) ----------
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ---------- GraspNet requirements ----------
COPY models/FGC_graspnet/requirements.txt /tmp/graspnet_req.txt
RUN pip install -r /tmp/graspnet_req.txt

# ---------- LangSAM requirements ----------
RUN pip install supervision requests transformers openai einops accelerate \
    open3d-plus \
    git+https://github.com/IDEA-Research/GroundingDINO.git \
    git+https://github.com/facebookresearch/segment-anything.git

# Note: 'ray[all]' and 'tensorflow' skipped — not used in demo path

# ---------- GraspNetAPI ----------
RUN git clone https://github.com/graspnet/graspnetAPI.git /tmp/graspnetAPI && \
    pip install /tmp/graspnetAPI && \
    rm -rf /tmp/graspnetAPI

# ---------- Copy project ----------
COPY . /app

# ---------- Build GraspNet C++ extensions ----------
RUN cd /app/models/FGC_graspnet/pointnet2 && python setup.py install && \
    cd /app/models/FGC_graspnet/knn && python setup.py install

# ---------- Other Python deps ----------
RUN pip install \
    gradio \
    opencv-python-headless \
    trimesh \
    matplotlib \
    pandas \
    numpy==1.24.4 \
    scipy \
    Pillow \
    tqdm \
    bitsandbytes

# ---------- Expose Gradio port ----------
EXPOSE 7860

# ---------- Wait for pre-downloaded models or download on first run ----------
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

CMD ["python", "demo.py"]
