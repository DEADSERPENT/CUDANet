# CUDAnet — Complete Run Guide

End-to-end instructions: from a fresh clone to training all configurations,
running inference, and generating publication-quality visualisation plots.

---

## 0. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| NVIDIA GPU | Compute Capability ≥ 5.0 | Tested on RTX A2000 (sm_86) |
| CUDA Toolkit | 11.0 – 12.x | `/usr/lib/nvidia-cuda-toolkit/bin/nvcc` |
| g++ | 12 | `sudo apt install g++-12` |
| GNU Make | any | `sudo apt install make` |
| Python 3 | ≥ 3.8 | for visualisation + inference helper |
| matplotlib, numpy | latest | `pip3 install matplotlib numpy` |

Check your GPU compute capability:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

---

## 1. Clone the Repository

```bash
git clone <repository-url>
cd CUDA-based-MLP
```

---

## 2. Download the MNIST Dataset

```bash
bash getdata.sh
```

Verify the four files are present:
```bash
ls data/
# t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte
# train-images-idx3-ubyte  train-labels-idx1-ubyte
```

---

## 3. Configure the Makefile for Your GPU

Open `Makefile` and check line 5:
```makefile
ARCH = sm_86    # change to your compute capability
```

Common values:
| GPU | ARCH |
|---|---|
| GTX 1080 / 1070 | `sm_61` |
| RTX 2080 / 2070 | `sm_75` |
| RTX 3090 / 3080 / A2000 | `sm_86` |
| RTX 4090 / 4080 | `sm_89` |

---

## 4. Build All Binaries

```bash
make everything        # builds: serial serial_omp cuda cuda_adam_lr
                       #         cuda_batchnorm cuda_full cuda_wmma infer

make cuda_momentum     # Momentum optimizer
make cuda_adam         # Adam optimizer
make cuda_lr_schedule  # SGD with LR scheduling
make cuda_lr_bn        # BN + SGD + LR
make cuda_adam_bn      # BN + Adam + LR  (alias with same flags as cuda_full minus LR)
```

Alternatively, build every target at once:
```bash
make everything cuda_momentum cuda_adam cuda_lr_schedule cuda_lr_bn cuda_adam_bn
```

Expected: only warnings (no errors).

---

## 5. Argument Reference

**All CUDA and serial binaries share the same positional argument order:**

```
./binary  <nl>  <nh>  <ne>  <nb>  <alpha>  <output_type>  <log_interval>  <save_flag>
```

| Arg | Meaning | Typical Value |
|---|---|---|
| `nl` | Hidden layers (not counting input/output) | `2` |
| `nh` | Neurons per hidden layer | `128` |
| `ne` | Training epochs | `20` |
| `nb` | Batch size | `2048` |
| `alpha` | Learning rate | `0.1` (SGD) / `0.01` (Adam) |
| `output_type` | `0`=Sigmoid+MSE  `1`=Softmax+CrossEntropy | `1` |
| `log_interval` | Print every N epochs (**must be ≥ 1**) | `5` |
| `save_flag` | `0`=none  `1`=save  `2`=load+train | `0` |

> **Warning:** `log_interval=0` causes a division-by-zero crash (SIGFPE).
> Always pass a value ≥ 1.

---

## 6. Run All Training Configurations

### 6.1 Baseline — SGD  (74.88%)
```bash
./cuda 2 128 20 2048 0.1 1 5 0 | tee sgd_output.txt
```

### 6.2 Momentum  (74.65%)
```bash
./cuda_momentum 2 128 20 2048 0.1 1 5 0 | tee momentum_output.txt
```

### 6.3 Adam  (80.80%)
```bash
./cuda_adam 2 128 20 2048 0.01 1 5 0 | tee adam_output.txt
```

### 6.4 Adam + LR Scheduling  (82.16%)
```bash
./cuda_adam_lr 2 128 20 2048 0.01 1 5 0 | tee adam_lr_output.txt
```

### 6.5 SGD + LR Scheduling  (71.80%)
```bash
./cuda_lr_schedule 2 128 20 2048 0.1 1 1 0 | tee lr_schedule_output.txt
```

### 6.6 Batch Normalisation + SGD  (60.68%)
```bash
./cuda_batchnorm 2 128 20 2048 0.1 1 5 0 | tee batchnorm_output.txt
```

### 6.7 BN + Adam + LR Scheduling — Best Config  (85.64%)
```bash
./cuda_full 2 128 20 2048 0.01 1 5 0 | tee full_output.txt
```

### 6.8 Serial CPU Baseline
```bash
./serial 2 128 20 2048 0.1 1 5 0 | tee serial_output.txt
```

### Run All in Parallel (GPU contention — sequential is more accurate for timing)
```bash
./cuda           2 128 20 2048 0.1  1 5 0 > sgd_output.txt     2>&1 &
./cuda_momentum  2 128 20 2048 0.1  1 5 0 > momentum_output.txt 2>&1 &
./cuda_adam      2 128 20 2048 0.01 1 5 0 > adam_output.txt     2>&1 &
wait
./cuda_adam_lr   2 128 20 2048 0.01 1 5 0 > adam_lr_output.txt  2>&1
./cuda_full      2 128 20 2048 0.01 1 5 0 > full_output.txt     2>&1
./cuda_batchnorm 2 128 20 2048 0.1  1 5 0 > batchnorm_output.txt 2>&1
```

---

## 7. Generate Visualisation Plots

Requires the `*_output.txt` files from step 6.

```bash
python3 visualize_training.py
```

Produces three PNG files:
```
training_accuracy_comparison.png    — accuracy curves for all optimizers
training_time_comparison.png        — per-epoch timing bar chart
optimizer_convergence_analysis.png  — early vs full-run convergence
```

Expected summary printed to stdout:
```
Optimizer       Final Test Acc   Avg Time/Epoch
SGD              74.88%           0.118s
Momentum         74.65%           0.168s
Adam             80.80%           0.171s
Adam+LR          82.16%           0.169s
BN+SGD           60.68%           0.206s
BN+Adam+LR       85.64%           0.150s
```

---

## 8. Train and Save a Model, Then Run Inference

### Train and save model checkpoint
```bash
./cuda_full 2 128 20 2048 0.01 1 5 1
# save_flag=1 writes model_checkpoint.bin
```

### Run inference on a single test image
```bash
python3 generate_test_input.py 0 | ./infer
# Output: 7   (predicted digit)
```

### Verbose inference (shows probabilities)
```bash
python3 generate_test_input.py 0 | ./infer -v
```

### Loop over first 10 test images
```bash
for i in $(seq 0 9); do
    echo -n "Image $i → "; python3 generate_test_input.py $i | ./infer
done
```

---

## 9. Scale Up for Higher Accuracy

For conference-paper quality results (targeting 97%+), run with a larger network:

```bash
./cuda_full 4 512 100 2048 0.001 1 10 0 | tee full_4x512_output.txt
```

Expected: ~95–97% test accuracy on MNIST.

---

## 10. Fashion-MNIST Variant

```bash
./cuda_fashion     2 128 20 2048 0.1  1 5 0   # SGD on Fashion-MNIST
./cuda_fashion_full 2 128 20 2048 0.01 1 5 0  # BN+Adam+LR on Fashion-MNIST
```

---

## 11. Makefile Quick Reference

```bash
make everything           # Build main binaries
make cuda_momentum        # Momentum optimizer
make cuda_adam            # Adam optimizer
make cuda_adam_bn         # BN + Adam + LR
make cuda_lr_schedule     # SGD + LR schedule
make cuda_lr_bn           # BN + SGD + LR
make cuda_wmma            # Tensor Core (WMMA) variant
make cuda_nvml            # NVML energy profiling
make clean                # Remove all binaries
make re                   # clean + everything
```

---

## 12. Troubleshooting

### SIGFPE crash at startup
```
Floating point exception (core dumped)
```
**Cause:** `log_interval` argument is `0`.
**Fix:** Pass `5` (or any value ≥ 1) as the 7th argument.

### CUDA out of memory
```
cudaMalloc failed
```
**Fix:** Reduce batch size (`nb`): try `1024` or `512`.

### Wrong compute capability
```
nvcc fatal: Unsupported gpu architecture 'compute_86'
```
**Fix:** Edit `ARCH` on line 5 of `Makefile` to match your GPU.

### Data files not found
```
Error opening file: data/train-images-idx3-ubyte
```
**Fix:** Run `bash getdata.sh` to download the MNIST dataset.

### Model checkpoint not found (inference)
```
Error opening model file: ./model_checkpoint.bin
```
**Fix:** Run training with `save_flag=1`:
```bash
./cuda_full 2 128 20 2048 0.01 1 5 1
```

---

*Generated: February 2026 | Hardware: NVIDIA RTX A2000 12GB (sm\_86)*
