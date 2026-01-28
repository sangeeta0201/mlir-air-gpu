# Build MLIR-AIR for GPU Target

This guide provides instructions for building MLIR-AIR for GPU targets without AIE dependencies. Tested on MI300X.

## Prerequisites

- ROCm installation (tested with ROCm 6.x)
- CMake 3.20+
- Ninja build system
- Python 3.8+

### Cluster Access (Optional)

If using the OCI MI300X cluster:

```bash
salloc -p amd-arad -N 1 --gres=gpu:2 -t 0-1
srun --pty $SHELL
bash
```

## Quick Build (Recommended)

This is the fastest way to build MLIR-AIR for GPU targets using pre-built LLVM wheels.

```bash
# Clone the repository
git clone https://github.com/Xilinx/mlir-air.git
cd mlir-air

# Setup Python environment
source utils/setup_python_packages.sh

# Build LLVM
./utils/clone-llvm.sh
./utils/build-llvm-local.sh llvm

# Build MLIR-AIR for GPU (without AIE)
./utils/build-mlir-air-gpu.sh llvm

# Setup environment
source sandbox/bin/activate
source utils/env_setup.sh build llvm/install
```

## Build Script Details

The `build-mlir-air-gpu.sh` script builds MLIR-AIR with:
- `-DAIR_ENABLE_AIE=OFF` - Disables AIE backend dependency
- GPU/ROCDL passes enabled
- AIR core dialect and transformations

## Manual Build

For more control over the build process:

```bash
# Clone and setup
git clone https://github.com/Xilinx/mlir-air.git
cd mlir-air
source utils/setup_python_packages.sh

# Build LLVM
./utils/clone-llvm.sh
./utils/build-llvm-local.sh llvm

# Configure MLIR-AIR for GPU
mkdir -p build && cd build
cmake .. \
    -GNinja \
    -DMLIR_DIR=$(pwd)/../llvm/install/lib/cmake/mlir \
    -DLLVM_DIR=$(pwd)/../llvm/install/lib/cmake/llvm \
    -DAIR_ENABLE_AIE=OFF \
    -DCMAKE_BUILD_TYPE=Release

# Build
ninja
```

## Available Passes

With GPU-only build, the following passes are available:

| Pass | Description |
|------|-------------|
| `air-to-rocdl` | Lower AIR dialect to ROCDL dialect |
| `air-to-async` | Lower AIR dialect to async dialect |
| `air-gpu-outlining` | Outline GPU kernels |
| `convert-to-air` | Convert operations to AIR dialect |

AIE-specific passes (e.g., `air-to-aie`) are registered but will emit an error if invoked, indicating that AIE support is required.

## GPU Compilation with aircc-gpu.sh

The `aircc-gpu.sh` script provides an easy way to compile AIR MLIR to GPU targets.

### Quick Start

```bash
# Compile the 4k x 4k matrix multiplication example for MI300X
./utils/aircc-gpu.sh --gpu-arch=gfx942 -o output.mlir test/gpu/4k_4k_mul/air_sync.mlir

# With verbose output to see compilation steps
./utils/aircc-gpu.sh -v --gpu-arch=gfx942 -o output.mlir test/gpu/4k_4k_mul/air_sync.mlir

# Keep intermediate files for debugging
./utils/aircc-gpu.sh -v --gpu-arch=gfx942 --tmpdir=/tmp/mytest -o output.mlir test/gpu/4k_4k_mul/air_sync.mlir
```

### aircc-gpu.sh Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o <file>` | stdout | Output file |
| `--gpu-arch <arch>` | `gfx942` | GPU architecture |
| `--gpu-runtime <rt>` | `HIP` | GPU runtime: `HIP` or `OpenCL` |
| `--tmpdir <dir>` | auto | Directory for intermediate files |
| `-v, --verbose` | off | Show compilation steps |

### Supported GPU Architectures

| Architecture | GPU |
|--------------|-----|
| `gfx942` | MI300X, MI300A |
| `gfx90a` | MI200 series |
| `gfx908` | MI100 |
| `gfx1100` | RDNA3 (RX 7900) |

### Compilation Pipeline

The script runs the following passes:

1. **AIR to ROCDL** (`air-opt -air-to-rocdl`)
   - Converts `air.launch`, `air.segment`, `air.herd` → `gpu.launch`
   - Converts `air.dma_memcpy_nd` → memory operations

2. **GPU Kernel Outlining** (`mlir-opt gpu-kernel-outlining`)
   - Outlines GPU kernels into `gpu.module`

3. **ROCDL Binary Generation** (`mlir-opt convert-gpu-to-rocdl, gpu-module-to-binary`)
   - Converts GPU dialect to ROCDL
   - Generates embedded GPU binary

4. **Final Lowering** (`mlir-opt gpu-to-llvm, convert-to-llvm`)
   - Lowers to LLVM dialect for execution

## GPU Test Examples

### 4k x 4k Matrix Multiplication

The `test/gpu/4k_4k_mul/` directory contains a matrix multiplication example.

```bash
# Compile the example
./utils/aircc-gpu.sh -v --gpu-arch=gfx942 \
    -o /tmp/matmul_output.mlir \
    test/gpu/4k_4k_mul/air_sync.mlir

# View the generated LLVM IR
head -100 /tmp/matmul_output.mlir
```

### Running on GPU

To run the compiled output on an AMD GPU:

```bash
# Compile to executable
mlir-cpu-runner /tmp/matmul_output.mlir \
    --entry-point-result=void \
    --shared-libs=$PWD/llvm/install/lib/libmlir_rocm_runtime.so \
    --shared-libs=$PWD/llvm/install/lib/libmlir_runner_utils.so
```

## Manual Compilation Steps

For more control, you can run the passes manually:

### Step 1: AIR to ROCDL

```bash
air-opt test/gpu/4k_4k_mul/air_sync.mlir \
    -air-to-rocdl -canonicalize -cse \
    -o step1_rocdl.mlir
```

### Step 2: GPU Kernel Outlining

```bash
mlir-opt step1_rocdl.mlir \
    --pass-pipeline="builtin.module(func.func(lower-affine, convert-scf-to-cf), gpu-kernel-outlining)" \
    -o step2_outlined.mlir
```

### Step 3: ROCDL Binary Generation

```bash
mlir-opt step2_outlined.mlir \
    --pass-pipeline="builtin.module(rocdl-attach-target{chip=gfx942 O=3}, gpu.module(convert-gpu-to-rocdl{chipset=gfx942 runtime=HIP}, reconcile-unrealized-casts), gpu-module-to-binary)" \
    -o step3_binary.mlir
```

### Step 4: Final LLVM Lowering

```bash
mlir-opt step3_binary.mlir \
    --pass-pipeline="builtin.module(func.func(gpu-async-region), gpu-to-llvm, convert-to-llvm, reconcile-unrealized-casts)" \
    -o step4_llvm.mlir
```

## Python AIRCC (Alternative)

The Python `aircc` compiler also supports GPU targets (requires Python bindings):

```bash
# Compile AIR MLIR to GPU (MI300X)
aircc.py --target=gpu --gpu-arch=gfx942 input.mlir -o output.mlir

# Auto-detect GPU from device name
aircc.py --device=gfx942 input.mlir -o output.mlir
```

### AIRCC Python Options

| Option | Default | Description |
|--------|---------|-------------|
| `--target` | `aie` | Target backend: `aie` or `gpu` |
| `--gpu-arch` | `gfx942` | GPU architecture |
| `--gpu-runtime` | `HIP` | GPU runtime: `HIP` or `OpenCL` |

## Environment Setup

To reactivate the environment from a new terminal:

```bash
cd mlir-air
source sandbox/bin/activate
source utils/env_setup.sh build llvm/install
```

## Troubleshooting

### Missing OpenSSL

If you encounter OpenSSL errors during CMake or LLVM build:

```bash
# Install OpenSSL locally
wget https://github.com/openssl/openssl/releases/download/openssl-3.5.0/openssl-3.5.0.tar.gz
tar zxvf openssl-3.5.0.tar.gz
cd openssl-3.5.0
./config --prefix=$HOME/openssl --openssldir=$HOME/openssl no-ssl2
make && make install

# Add to environment
export PATH=$HOME/openssl/bin:$PATH
export LD_LIBRARY_PATH=$HOME/openssl/lib:$LD_LIBRARY_PATH
export OPENSSL_ROOT_DIR=$HOME/openssl
```

### AIE Pass Errors

If you see errors like:
```
error: AIRToAIE pass requires AIE support. Rebuild with -DAIR_ENABLE_AIE=ON
```

This is expected behavior. The GPU-only build does not include AIE backend support. Use `air-to-rocdl` instead of `air-to-aie` for GPU targets.

### ROCm Runtime Not Found

Ensure ROCm is installed and the runtime library path is correct:
```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

## Building with AIE Support

If you need both GPU and AIE backends, build with AIE enabled:

```bash
# Follow the full build instructions in the main README
./utils/clone-mlir-aie.sh
./utils/build-mlir-aie-local.sh llvm mlir-aie/cmake/modulesXilinx aienginev2/install mlir-aie
./utils/build-mlir-air.sh llvm mlir-aie/cmake/modulesXilinx mlir-aie aienginev2/install
```
