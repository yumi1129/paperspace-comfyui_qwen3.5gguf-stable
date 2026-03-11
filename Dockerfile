FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    git-lfs \
    wget \
    curl \
    unzip \
    aria2 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    python3.10 \
    python3-pip \
    python3-dev \
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /tmp

COPY requirements.txt /tmp/requirements.txt

# まずPyTorchを先に固定
RUN pip install \
    torch \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu124

# 土台ライブラリ
RUN pip install -r /tmp/requirements.txt

# xformers / triton は必要性が高いので残す
RUN pip install xformers triton

# Jupyter系
RUN pip install jupyterlab jupyter-server-proxy

# llama-cpp-python を、対応wheelが取れればビルド回避
RUN CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86" \
    pip install "llama-cpp-python==0.3.16"

# llama-server を事前ビルド
RUN git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp && \
    cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/llama.cpp/build -j 2 && \
    ln -sf /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

EXPOSE 8888 8188 6006 8000
WORKDIR /notebooks
