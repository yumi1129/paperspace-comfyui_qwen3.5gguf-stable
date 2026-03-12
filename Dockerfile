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
    https://github.com/yumi1129/paperspace-comfyui_qwen3.5gguf-stable/releases/download/v2/llama_cpp_python-0.3.16-cp311-cp311-manylinux_2_35_x86_64.whl

RUN ls -lh /tmp/llama_cpp_python.whl

RUN mkdir -p /tmp/llama_wheel_unpack && \
    python - <<'PY'
import zipfile
wheel = "/tmp/llama_cpp_python.whl"
dest = "/tmp/llama_wheel_unpack"
with zipfile.ZipFile(wheel, "r") as z:
    z.extractall(dest)
print("extracted to", dest)
PY

RUN echo "=== unpacked wheel contents ===" && \
    find /tmp/llama_wheel_unpack -maxdepth 2 -mindepth 1 | sort

RUN python - <<'PY'
import os
import shutil
import sysconfig

src = "/tmp/llama_wheel_unpack"
dst = sysconfig.get_paths()["purelib"]

print("copy from:", src)
print("copy to:", dst)

for name in os.listdir(src):
    s = os.path.join(src, name)
    d = os.path.join(dst, name)
    print("copying:", s, "->", d)
    if os.path.isdir(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)
PY

RUN python - <<'PY'
import os
import sysconfig

site = sysconfig.get_paths()["purelib"]
print("site-packages:", site)

targets = [
    os.path.join(site, "llama_cpp"),
    os.path.join(site, "llama_cpp_python.libs"),
]

for t in targets:
    print(t, "exists =", os.path.exists(t))
PY

RUN find /opt/venv/lib/python3.11/site-packages/llama_cpp -maxdepth 3 -type f | sort

RUN rm -rf /tmp/llama_cpp_python.whl /tmp/llama_wheel_unpack

RUN curl -fL -o /tmp/llama-server.tar.gz \
    https://github.com/yumi1129/paperspace-comfyui_qwen3.5gguf-stable/releases/download/v3/llama-server-cuda12.4-sm80_86.tar.gz && \
    tar -xzf /tmp/llama-server.tar.gz -C /tmp && \
    mv /tmp/llama-server-cuda12.4-sm80_86 /usr/local/bin/llama-server && \
    chmod +x /usr/local/bin/llama-server && \
    rm -f /tmp/llama-server.tar.gz

EXPOSE 8888 8188 6006 8000
WORKDIR /notebooks
