#!/bin/bash
# SparseKV-Engine 一键运行脚本
# 支持：本地(CPU)、Docker(GPU)、云服务器 三种运行模式

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="sparsekv-engine"
BENCHMARK_SEQ_LEN=${BENCHMARK_SEQ_LEN:-32768}
BENCHMARK_RUNS=${BENCHMARK_RUNS:-10}
SPARSE_RATIO=${SPARSE_RATIO:-0.3}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检测运行环境
detect_env() {
    log_info "检测运行环境..."
    
    if command -v nvidia-smi &> /dev/null; then
        log_info "检测到 NVIDIA GPU:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        ENV="docker_gpu"
    elif [ -f "/.dockerenv" ]; then
        log_warn "当前在 Docker 容器内，但未检测到 nvidia-smi"
        ENV="docker_cpu"
    else
        log_warn "未检测到 NVIDIA GPU，将使用 CPU 模式运行（仅测试 Python 逻辑，无法编译 CUDA Kernel）"
        ENV="local_cpu"
    fi
    
    echo ""
}

# 本地 CPU 模式运行
run_local_cpu() {
    log_info "模式：本地 CPU 测试"
    log_warn "注意：macOS / 无 GPU 环境只能跑 PyTorch fallback，无法编译 CUDA 扩展"
    
    cd "$SCRIPT_DIR"
    
    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        log_error "未找到 python3，请先安装 Python 3.10+"
        exit 1
    fi
    
    # 创建虚拟环境（如果不存在）
    if [ ! -d ".venv" ]; then
        log_info "创建 Python 虚拟环境..."
        python3 -m venv .venv
    fi
    
    source .venv/bin/activate
    
    # 安装 CPU 版 PyTorch
    log_info "安装 PyTorch (CPU 版)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    
    # 运行 Python 逻辑测试（跳过 CUDA）
    log_info "运行 CPU 模式基准测试（小序列验证逻辑正确性）..."
    python benchmark/benchmark_longbench.py \
        --seq-len 1024 \
        --num-heads 8 \
        --head-dim 64 \
        --sparse-ratio "$SPARSE_RATIO" \
        --num-runs 3
    
    log_info "CPU 测试完成。要跑完整 GPU Benchmark，请使用 Docker 或在 Linux GPU 服务器上运行。"
}

# Docker GPU 模式运行
run_docker_gpu() {
    log_info "模式：Docker GPU"
    
    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        log_error "未找到 docker，请先安装 Docker"
        exit 1
    fi
    
    # 检查 nvidia-docker
    if ! docker info | grep -q "nvidia"; then
        log_warn "Docker 未配置 nvidia runtime，尝试使用 --gpus all 参数"
    fi
    
    cd "$SCRIPT_DIR"
    
    log_info "构建 Docker 镜像..."
    docker build -t "$PROJECT_NAME:latest" .
    
    log_info "运行 Benchmark（seq_len=$BENCHMARK_SEQ_LEN, runs=$BENCHMARK_RUNS）..."
    mkdir -p benchmark_results
    docker run --rm --gpus all \
        -e BENCHMARK_SEQ_LEN="$BENCHMARK_SEQ_LEN" \
        -e BENCHMARK_RUNS="$BENCHMARK_RUNS" \
        -e SPARSE_RATIO="$SPARSE_RATIO" \
        -v "$(pwd)/benchmark_results:/workspace/sparsekv-engine/benchmark_results" \
        "$PROJECT_NAME:latest" \
        python3 benchmark/benchmark_longbench.py \
            --seq-len "$BENCHMARK_SEQ_LEN" \
            --num-runs "$BENCHMARK_RUNS" \
            --sparse-ratio "$SPARSE_RATIO"
    
    log_info "Benchmark 完成，结果保存在 ./benchmark_results/"
}

# Docker Compose 模式
run_compose() {
    log_info "模式：Docker Compose"
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "未找到 docker-compose"
        exit 1
    fi
    
    cd "$SCRIPT_DIR"
    docker-compose up --build
}

# 交互式进入容器
run_shell() {
    log_info "启动交互式容器..."
    docker run --rm --gpus all -it \
        -v "$(pwd):/workspace/sparsekv-engine" \
        "$PROJECT_NAME:latest" \
        /bin/bash
}

# 运行单元测试
run_test() {
    log_info "运行单元测试..."
    cd "$SCRIPT_DIR"
    
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    if [ "$ENV" = "docker_gpu" ]; then
        docker run --rm --gpus all "$PROJECT_NAME:latest" \
            python3 -m pytest tests/ -v
    else
        python3 -m pytest tests/ -v || python3 tests/test_sparse_attention.py
    fi
}

# 帮助信息
show_help() {
    cat << EOF
SparseKV-Engine 一键运行脚本

用法: ./run.sh [命令] [选项]

命令:
    auto        自动检测环境并运行（默认）
    docker      使用 Docker GPU 模式构建并运行 Benchmark
    compose     使用 docker-compose 运行
    local       强制使用本地 CPU 模式（macOS / 无 GPU）
    test        运行单元测试
    shell       进入交互式 Docker 容器
    help        显示帮助

环境变量:
    BENCHMARK_SEQ_LEN   Benchmark 序列长度 (默认: 32768)
    BENCHMARK_RUNS      Benchmark 运行次数 (默认: 10)
    SPARSE_RATIO        稀疏比例 (默认: 0.3)

示例:
    # 一键自动检测并运行
    ./run.sh

    # Docker GPU 模式，指定 64K 序列长度
    BENCHMARK_SEQ_LEN=65536 ./run.sh docker

    # 本地 macOS 测试
    ./run.sh local

    # 运行测试
    ./run.sh test

运行环境说明:
    • Linux + NVIDIA GPU : 推荐 Docker 模式，可编译 CUDA Kernel，跑完整 Benchmark
    • macOS / 无 GPU     : 只能 local 模式，跑 PyTorch CPU fallback，验证逻辑正确性
    • 云服务器(A100/V100): Docker 模式，完整功能
EOF
}

# 主逻辑
main() {
    COMMAND=${1:-auto}
    
    case "$COMMAND" in
        auto)
            detect_env
            if [ "$ENV" = "docker_gpu" ]; then
                run_docker_gpu
            else
                run_local_cpu
            fi
            ;;
        docker)
            detect_env
            run_docker_gpu
            ;;
        compose)
            run_compose
            ;;
        local)
            run_local_cpu
            ;;
        test)
            detect_env
            run_test
            ;;
        shell)
            run_shell
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
