# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is **Cute-Learning**, a collection of standalone CUDA kernel examples and benchmarks using NVIDIA CUTLASS CuTe. It is **not** a web application or service — each subdirectory is an independent CUDA program compiled with `nvcc`.

### Prerequisites

- **NVIDIA GPU** (Ampere sm_80+ minimum; Hopper sm_90a for `gemm/hopper/*`)
- **CUDA Toolkit** (nvcc, CUDA runtime, cuBLAS) — installed at `/usr/local/cuda-12.6`
- **CUTLASS headers** via git submodule at `flashdecoding/src/cutlass/`
- Add `/usr/local/cuda-12.6/bin` to `PATH` and `/usr/local/cuda-12.6/lib64` to `LD_LIBRARY_PATH`

### Building

Each example has its own `Makefile` in its subdirectory. Build individually with `make` in the relevant directory. The default Makefile targets both compile and run the binary, so to **compile only**, invoke `nvcc` directly (see the Makefile for flags).

### Known issues

- **Hardcoded paths**: Several Makefiles (`gemm/gemm_v2`, `gemm/gemm_v3`, `gemm/gemm_v4`, `gemm/hopper/*`) reference the original developer's home directory for CUTLASS includes. Symlinks at `/home/zhichen/dayou/cutlass` and `/home/dudayou/dayou/repo/cutlass` pointing to `flashdecoding/src/cutlass` are created by the environment setup to work around this.
- **`others/dequant/Makefile`** has an incorrect relative path (`-I../flashdecoding/...` should be `-I../../flashdecoding/...`). Compile manually with corrected include path.
- **`gemm/gemm_stream-k`** has a pre-existing code error (`d_workspace` and `gemm_host` undefined in `main.cu`); it does not compile.
- **`sm90_decode/src/cutlass`** submodule is tracked in git but missing from `.gitmodules`. A symlink to `flashdecoding/src/cutlass` is used as a workaround.

### Cloud VM limitations

- The Cloud VM does **not** have an NVIDIA GPU. Compilation works, but **execution requires a GPU-equipped machine**.
- The `flashdecoding/` and `sm90_decode/` CMake projects also require PyTorch/LibTorch for linking, which is not installed by default.
