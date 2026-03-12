FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
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
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /tmp

COPY requirements.txt /tmp/requirements.txt
COPY wheels /tmp/wheels

# PyTorch を先に固定
RUN pip install \
    torch \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu124

# 基本ライブラリ
RUN pip install -r /tmp/requirements.txt

# 追加ライブラリ
RUN pip install xformers triton
RUN pip install jupyterlab jupyter-server-proxy
RUN pip install comfyui-manager

# 事前作成済み wheel
RUN pip install /tmp/wheels/llama_cpp_python-0.3.16-cp311-cp311-linux_x86_64.whl

# llama-server を事前ビルド
RUN git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp && \
    cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="80;86" \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/llama.cpp/build -j 2 && \
    ln -sf /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

EXPOSE 8888 8188 6006 8000
WORKDIR /notebooks
