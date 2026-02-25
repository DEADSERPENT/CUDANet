# CUDAnet: A Race-Free, Stateful-Optimizer MLP Engine on NVIDIA GPU

## M.Tech High-Performance Computing — Mini Project

> A from-scratch CUDA implementation of a Multi-Layer Perceptron (MLP) featuring
> correct 2-phase gradient accumulation, full SGD/Momentum/Adam optimizer support
> with Batch Normalization, LR scheduling, Tensor Core inference, and NVML energy
> profiling. Trained and benchmarked on MNIST and Fashion-MNIST.

---

## Key Results (NVIDIA RTX A2000, sm_86 — 20 epochs, 2×128, batch 2048)

| Configuration | Test Acc | Time/Epoch | Binary |
|---|---|---|---|
| SGD (lr=0.1) | 74.88% | ~0.12 s | `cuda` |
| Momentum (lr=0.1) | 74.65% | ~0.17 s | `cuda_momentum` |
| Adam (lr=0.01) | 80.80% | ~0.17 s | `cuda_adam` |
| Adam + LR Schedule | 82.16% | ~0.17 s | `cuda_adam_lr` |
| BN + SGD (lr=0.1) | 60.68% | ~0.21 s | `cuda_batchnorm` |
| **BN + Adam + LR** | **85.64%** | **~0.15 s** | `cuda_full` |

All BN+Adam results reflect the corrected optimizer integration (see §Implementation Details).

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Prerequisites & Installation](#prerequisites--installation)
4. [Building](#building)
5. [Running](#running)
6. [Inference](#inference)
7. [Visualisation](#visualisation)
8. [Implementation Details](#implementation-details)
9. [Benchmarks](#benchmarks)
10. [Future Work](#future-work)
11. [References](#references)

---

## Features

### Implemented

| Feature | Status | Notes |
|---|---|---|
| Softmax + Cross-Entropy output | ✅ | default |
| Sigmoid + MSE output | ✅ | `output_type=0` |
| Shared-memory tiled forward pass | ✅ | TILE_SIZE=32, ~10% speedup |
| SGD optimizer | ✅ | |
| Momentum optimizer | ✅ | 2-phase gradient accumulation |
| Adam optimizer | ✅ | 2-phase, bias-corrected |
| Batch Normalisation | ✅ | All three optimizers supported |
| LR scheduling (exponential decay) | ✅ | `lr × 0.9753^epoch` |
| Xavier/He weight initialisation | ✅ | `σ = √(2/fan_in)` |
| Validation split (80/20) | ✅ | prevents overfitting monitoring |
| Save / load model checkpoint | ✅ | `model_checkpoint.bin` |
| Standalone GPU inference binary | ✅ | `infer` |
| WMMA Tensor Core kernels | ✅ | fallback for non-16-divisible layers |
| NVML energy profiling | ✅ | Joules per epoch |
| Fashion-MNIST support | ✅ | compile-time flag |
| Training visualisation | ✅ | reads `*_output.txt` → 3 PNG plots |

---

## Project Structure

```
CUDA-based-MLP/
├── mnist_nn_cuda.cu          Main CUDA source (~2,280 lines)
│                             All training configs via compile-time #defines
├── infer.cu                  Standalone CUDA inference binary
├── mnist_nn_serial_c.c       Serial CPU baseline
├── load_data.h               MNIST/Fashion-MNIST binary reader
├── Makefile                  11+ build targets
│
├── getdata.sh                Download MNIST/Fashion-MNIST datasets
├── generate_test_input.py    Pipe a test image to ./infer
├── visualize_training.py     Generate 3 publication-quality PNG plots
│
├── run.md                    ← COMPLETE RUN GUIDE (start here)
├── README.md                 This file
├── vision.md                 Conference-level design document
├── main.tex                  LaTeX paper source
│
├── data/                     MNIST dataset (git-ignored, download via getdata.sh)
├── *_output.txt              Training results per configuration
└── *.png                     Generated visualisation plots
```

---

## Prerequisites & Installation

```bash
# CUDA toolkit (nvcc)
sudo apt install nvidia-cuda-toolkit

# Host compiler (g++-12 required by Makefile)
sudo apt install g++-12

# Python visualisation dependencies
pip3 install matplotlib numpy

# Download MNIST dataset
bash getdata.sh
```

**GPU requirement:** NVIDIA GPU with Compute Capability ≥ 5.0.
Tested on RTX A2000 12 GB (sm_86). Edit `ARCH` on line 5 of `Makefile` to
match your GPU.

---

## Building

```bash
# Core binaries
make everything           # serial serial_omp cuda cuda_adam_lr
                          # cuda_batchnorm cuda_full cuda_wmma infer

# Additional optimizer binaries
make cuda_momentum cuda_adam cuda_lr_schedule cuda_lr_bn cuda_adam_bn

# Clean
make clean
make re                   # clean + everything
```

---

## Running

**Argument order — same for all binaries:**
```
./binary  <nl>  <nh>  <ne>  <nb>  <alpha>  <output_type>  <log_interval>  <save_flag>
```

> `log_interval` **must be ≥ 1** — passing `0` causes SIGFPE (integer division by zero).

### Recommended runs

```bash
# Best overall: BN + Adam + LR scheduling  →  85.64%
./cuda_full 2 128 20 2048 0.01 1 5 0

# Adam baseline  →  80.80%
./cuda_adam 2 128 20 2048 0.01 1 5 0

# SGD baseline  →  74.88%
./cuda 2 128 20 2048 0.1 1 5 0

# SGD + LR schedule  →  71.80%  (LR decay hurts SGD at this scale)
./cuda_lr_schedule 2 128 20 2048 0.1 1 1 0
```

See `run.md` for the complete guide including all configurations.

---

## Inference

```bash
# 1. Train and save a model
./cuda_full 2 128 20 2048 0.01 1 5 1       # save_flag=1

# 2. Predict a single digit
python3 generate_test_input.py 0 | ./infer          # → 7

# 3. Verbose mode (shows per-class probabilities)
python3 generate_test_input.py 0 | ./infer -v
```

---

## Visualisation

```bash
python3 visualize_training.py
```

Reads `*_output.txt` files and writes:

| File | Content |
|---|---|
| `training_accuracy_comparison.png` | Train/test accuracy curves for all configs |
| `training_time_comparison.png` | Per-epoch timing bar chart |
| `optimizer_convergence_analysis.png` | Early convergence + full run side-by-side |

---

## Implementation Details

### Architecture

```
Input (784) → [Linear → BN → ReLU] × nl → Linear → Softmax → Output (10)
```

Default: `nl=2`, `nh=128`, `nb=2048`, giving a 784→128→128→10 network.

### 2-Phase Gradient Accumulation (Correctness Argument)

A critical design principle for Momentum and Adam: optimiser state must be
updated **once per batch**, not once per sample.

**Phase 1** — `one_learning_cycle` kernel (non-BN) / `grad_accum_layer_kernel` (BN):
each of the `nb` GPU threads computes its per-sample gradient and writes to a
shared gradient buffer via `atomicAdd`. No optimiser state is touched.

**Phase 2** — `apply_optimizer_update` kernel: a single kernel call per batch.
Divides the accumulated gradient by `nb` (averaging), then applies the
Momentum/Adam update formula and resets the gradient buffer to zero.

This guarantees race-free, correct Momentum and Adam updates at any batch size.

### BN + Optimizer Integration Bug (Fixed)

Prior to this fix, the BN training path (`#if USE_BATCH_NORM`) always called
`sgd_layer_update_kernel` regardless of `OPTIMIZER_TYPE`. With Adam (`lr=0.01`)
this produced an effective weight step of `0.01 / 2048 ≈ 4.9 × 10⁻⁶` —
essentially no learning.

**Fix:** The BN path now dispatches via `#if OPTIMIZER_TYPE`:
- `== 0` (SGD): direct atomic update via `sgd_layer_update_kernel`
- `== 1/2` (Momentum/Adam): 2-phase via `grad_accum_layer_kernel` →
  `apply_optimizer_update`

The BN learnable parameters `γ` and `β` also received the same fix:
`bn_params_update_kernel` now implements SGD/Momentum/Adam with full
bias-corrected Adam formulae.

**Impact:** BN+Adam+LR accuracy went from **17.91% → 85.64%**.

### Shared-Memory Tiling

Forward and backward kernels use 32×32 tiled shared memory
(`USE_SHARED_MEMORY=1`, TILE_SIZE=32) to reduce global memory traffic:

```cuda
__shared__ float sh_w[TILE_SIZE][TILE_SIZE];
__shared__ float sh_a[TILE_SIZE];
```

### Learning Rate Schedule

Exponential decay applied at the start of every epoch:
```
lr_epoch = lr_0 × exp(-0.025 × epoch)   ≈   lr_0 × 0.9753^epoch
```

### WMMA Tensor Cores

Layers whose `n_in` and `n_out` are both multiples of 16 use warp-level
`wmma::mma_sync` on 16×16×16 FP16 tiles. Non-divisible layers fall back to
scalar kernels automatically.

---

## Benchmarks

### Optimizer Comparison (2×128, 20 epochs, batch 2048)

| Optimizer | Test@Ep5 | Test@Ep20 | Relative to SGD |
|---|---|---|---|
| SGD | 40.1% | 74.88% | baseline |
| Momentum | 61.4% | 74.65% | 1.53× faster early |
| Adam | 78.7% | 80.80% | 1.96× faster early, +6% final |
| Adam+LR | 79.4% | 82.16% | best stable convergence |
| **BN+Adam+LR** | **57.5%** | **85.64%** | **best final accuracy** |

### Serial vs CUDA

| Implementation | Time/Epoch | Speedup |
|---|---|---|
| Serial CPU | ~90 s | 1× |
| CUDA SGD | ~0.12 s | **750×** |
| CUDA BN+Adam+LR | ~0.15 s | **600×** |

---

## Future Work

1. **Kernel Fusion** — fuse Linear+BN+ReLU into a single kernel to reduce
   memory round-trips and halve BN overhead
2. **Roofline Analysis** — profile arithmetic intensity vs memory bandwidth
   to identify bound kernels
3. **PyTorch/cuDNN Comparison** — quantitative benchmark against framework
   implementations on identical architectures
4. **Scale to 4L×512** — expected 95–97% MNIST accuracy; validates that the
   current implementation is not the accuracy bottleneck
5. **INT8 Inference Quantisation** — 4× throughput, minimal accuracy loss
6. **Multi-GPU Training** — gradient synchronisation with NCCL
7. **Fashion-MNIST Ablation** — systematic BN/dropout/LR study on harder dataset

---

## References

1. LeCun, Y. et al. "Gradient-Based Learning Applied to Document Recognition."
   *Proc. IEEE* 86(11), 1998. — MNIST dataset.
2. Kingma, D. P. & Ba, J. "Adam: A Method for Stochastic Optimization."
   *ICLR 2015.*
3. Ioffe, S. & Szegedy, C. "Batch Normalization: Accelerating Deep Network
   Training by Reducing Internal Covariate Shift." *ICML 2015.*
4. NVIDIA Corporation. *CUDA C++ Programming Guide*, 2024.
5. Glorot, X. & Bengio, Y. "Understanding the Difficulty of Training Deep
   Feedforward Neural Networks." *AISTATS 2010.* — Xavier initialisation.

---

## Quick Start

```bash
bash getdata.sh                                   # 1. download MNIST
make everything cuda_momentum cuda_adam           # 2. build
./cuda_full 2 128 20 2048 0.01 1 5 0             # 3. best accuracy run
python3 visualize_training.py                     # 4. generate plots
python3 generate_test_input.py 0 | ./infer -v    # 5. inference demo
```

See **`run.md`** for the complete step-by-step guide.

---

*Hardware: NVIDIA RTX A2000 12 GB (sm\_86) · CUDA 12.2 · g++-12*
*February 2026 — M.Tech HPC Systems*
