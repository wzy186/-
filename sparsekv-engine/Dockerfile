# SparseKV-Engine Docker 镜像
# 基于 NVIDIA CUDA + PyTorch，支持一键编译 CUDA 扩展并运行 Benchmark

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# 安装 PyTorch (CUDA 12.1 版本)
RUN pip3 install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 复制项目代码
WORKDIR /workspace/sparsekv-engine
COPY . /workspace/sparsekv-engine/

# 安装 Python 依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 编译 CUDA 扩展
RUN cd cuda && \
    python3 setup.py build_ext --inplace && \
    python3 setup.py install && \
    cd ..

# 运行测试验证安装
RUN python3 tests/test_sparse_attention.py

# 默认入口：运行 benchmark
CMD ["python3", "benchmark/benchmark_longbench.py", "--seq-len", "32768", "--num-runs", "10"]
