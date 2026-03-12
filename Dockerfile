FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

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

RUN python3.11 -m venv /opt/venv && \
    /opt/venv/bin/python --version && \
    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel

WORKDIR /tmp

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install \
    torch \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install -r /tmp/requirements.txt
RUN python -m pip install xformers triton
RUN python -m pip install jupyterlab jupyter-server-proxy
RUN python -m pip install comfyui-manager

RUN curl -fL -o /tmp/llama_cpp_python.whl \
    https://github.com/yumi1129/paperspace-comfyui_qwen3.5gguf-stable/releases/download/v1/llama_cpp_python-0.3.16-cp311-cp311-linux_x86_64.whl

RUN ls -lh /tmp/llama_cpp_python.whl

RUN python --version && \
    python -m pip --version

RUN python -m pip install packaging

RUN python - <<'PY'
from packaging.utils import parse_wheel_filename
from packaging import tags
wheel = "llama_cpp_python-0.3.16-cp311-cp311-linux_x86_64.whl"
name, version, build, wheel_tags = parse_wheel_filename(wheel)
supported = set(tags.sys_tags())
print("wheel tags:", wheel_tags)
print("supported match:", any(t in supported for t in wheel_tags))
PY

RUN python -m pip install \
    diskcache \
    jinja2 \
    markupsafe \
    typing_extensions \
    numpy

RUN python -m pip install --no-deps --force-reinstall /tmp/llama_cpp_python.whl

RUN rm -f /tmp/llama_cpp_python.whl

RUN git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp && \
    cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="80;86" \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/llama.cpp/build -j 2 && \
    ln -sf /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

EXPOSE 8888 8188 6006 8000
WORKDIR /notebooks
