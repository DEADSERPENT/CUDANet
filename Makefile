COMPILER1  = gcc
CFLAGS1    = -O2 -lm
CFLAGS1_G  = -g $(CFLAGS1)

COMPILER2  = /usr/lib/nvidia-cuda-toolkit/bin/nvcc
ARCH       = sm_86
# Use GCC 12 as host compiler (CUDA 12.0 supports GCC up to 12)
CFLAGS2    = -arch $(ARCH) -O2 -ccbin g++-12
CFLAGS2_G  = -arch $(ARCH) -g -G -ccbin g++-12

# ─────────────────────────────────────────────────────────────────────────────
# Default targets
# ─────────────────────────────────────────────────────────────────────────────

all: serial cuda

# ─────────────────────────────────────────────────────────────────────────────
# CPU baseline targets
# ─────────────────────────────────────────────────────────────────────────────

serial: mnist_nn_serial_c.c
	$(COMPILER1) mnist_nn_serial_c.c -o serial $(CFLAGS1_G)

# Multi-threaded CPU baseline using OpenMP (replaces the "strawman" single-thread)
# Makes speedup claims credible – compare against this, not the serial target.
serial_omp: mnist_nn_serial_c.c
	$(COMPILER1) mnist_nn_serial_c.c -o serial_omp $(CFLAGS1_G) -fopenmp
	@echo "Built OpenMP baseline with $$(nproc) threads available"

# ─────────────────────────────────────────────────────────────────────────────
# Core CUDA targets
# ─────────────────────────────────────────────────────────────────────────────

cuda: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda $(CFLAGS2)

# Momentum optimizer
cuda_momentum: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_momentum $(CFLAGS2) -DOPTIMIZER_TYPE=1

# Adam optimizer
cuda_adam: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_adam $(CFLAGS2) -DOPTIMIZER_TYPE=2

# Adam + cosine LR scheduling (matches paper description)
cuda_adam_lr: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_adam_lr $(CFLAGS2) \
	    -DOPTIMIZER_TYPE=2 -DUSE_LR_SCHEDULE=1 -DLR_SCHEDULE_TYPE=2

# SGD + LR scheduling
cuda_lr_schedule: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_lr_schedule $(CFLAGS2) -DUSE_LR_SCHEDULE=1

# ─────────────────────────────────────────────────────────────────────────────
# Batch Normalisation (FIXED – correct multi-kernel implementation)
# ─────────────────────────────────────────────────────────────────────────────

# Fixed BN with Adam + LR schedule (achieves ~97% test accuracy)
cuda_batchnorm: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_batchnorm $(CFLAGS2) \
	    -DUSE_BATCH_NORM=1

cuda_lr_bn: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_lr_bn $(CFLAGS2) \
	    -DUSE_LR_SCHEDULE=1 -DUSE_BATCH_NORM=1

cuda_adam_bn: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_adam_bn $(CFLAGS2) \
	    -DOPTIMIZER_TYPE=2 -DUSE_LR_SCHEDULE=1 -DUSE_BATCH_NORM=1

# Full feature stack: BN + Adam + cosine LR (per-batch decay, fixes over-decay)
cuda_full: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_full $(CFLAGS2) \
	    -DOPTIMIZER_TYPE=2 -DUSE_LR_SCHEDULE=1 -DUSE_BATCH_NORM=1 -DLR_SCHEDULE_TYPE=2

# ─────────────────────────────────────────────────────────────────────────────
# WMMA Tensor Core target (requires layer dims ≥ 16, compile-time multiples)
# ─────────────────────────────────────────────────────────────────────────────

cuda_wmma: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_wmma $(CFLAGS2) \
	    -DUSE_TENSOR_CORES=1 -DOPTIMIZER_TYPE=2 -DUSE_LR_SCHEDULE=1

# ─────────────────────────────────────────────────────────────────────────────
# NVML energy profiling target (links against nvidia-ml)
# ─────────────────────────────────────────────────────────────────────────────

cuda_nvml: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_nvml $(CFLAGS2) \
	    -DUSE_NVML -DOPTIMIZER_TYPE=2 -DUSE_LR_SCHEDULE=1 \
	    -lnvidia-ml

# Full features + NVML profiling (conference-grade build)
cuda_nvml_full: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_nvml_full $(CFLAGS2) \
	    -DUSE_NVML -DOPTIMIZER_TYPE=2 -DUSE_LR_SCHEDULE=1 \
	    -DUSE_BATCH_NORM=1 -lnvidia-ml

# ─────────────────────────────────────────────────────────────────────────────
# Fashion-MNIST targets
# ─────────────────────────────────────────────────────────────────────────────

cuda_fashion: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_fashion $(CFLAGS2) \
	    -DFASHION_MNIST=1 -DOPTIMIZER_TYPE=2 -DUSE_LR_SCHEDULE=1

cuda_fashion_full: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_fashion_full $(CFLAGS2) \
	    -DFASHION_MNIST=1 -DOPTIMIZER_TYPE=2 -DUSE_LR_SCHEDULE=1 \
	    -DUSE_BATCH_NORM=1

# ─────────────────────────────────────────────────────────────────────────────
# Inference binary
# ─────────────────────────────────────────────────────────────────────────────

infer: infer.cu
	$(COMPILER2) infer.cu -o infer $(CFLAGS2)

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark / profiling targets
# ─────────────────────────────────────────────────────────────────────────────

# Full ablation study (builds each configuration and measures speedup/accuracy)
benchmark: cuda serial_omp
	bash run_ablation.sh 30 2048 128 2

# Nsight Compute profiling (requires ncu in PATH)
profiling: cuda
	ncu --csv --target-processes all \
	    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes.sum,dram__bytes.sum,flop_count_sp \
	    ./cuda 2 128 5 2048 0.01 1 > ncu_output.csv
	python3 roofline_analysis.py --ncu ncu_output.csv
	@echo "Roofline plot saved as roofline_plot.pdf"

# Theoretical roofline only (no hardware needed)
roofline:
	python3 roofline_analysis.py --theoretical
	@echo "Roofline plot saved as roofline_plot.pdf"

# PyTorch comparison (requires: pip install torch torchvision)
pytorch_baseline:
	python3 benchmark_pytorch.py --layers 2 --hidden 128 --epochs 20 --batch 2048 --lr 0.1

pytorch_full:
	python3 benchmark_pytorch.py --layers 4 --hidden 512 --epochs 50 --batch 256 \
	    --lr 0.001 --optimizer adam

# ─────────────────────────────────────────────────────────────────────────────
# Data download
# ─────────────────────────────────────────────────────────────────────────────

data_mnist:
	bash getdata.sh mnist

data_fashion:
	bash getdata.sh fashion

# ─────────────────────────────────────────────────────────────────────────────
# Build everything
# ─────────────────────────────────────────────────────────────────────────────

everything: serial serial_omp cuda cuda_adam_lr cuda_batchnorm cuda_full \
            cuda_fashion cuda_wmma infer

# ─────────────────────────────────────────────────────────────────────────────

clean:
	rm -f serial serial_omp cuda cuda_momentum cuda_adam cuda_batchnorm \
	      cuda_lr_schedule cuda_adam_lr cuda_lr_bn cuda_full cuda_adam_bn \
	      cuda_wmma cuda_nvml cuda_nvml_full \
	      cuda_fashion cuda_fashion_full infer \
	      ablation_results.csv ncu_output.csv roofline_plot.pdf

re: clean all

.PHONY: all serial serial_omp cuda cuda_momentum cuda_adam cuda_adam_lr \
        cuda_lr_schedule cuda_batchnorm cuda_lr_bn cuda_adam_bn cuda_full \
        cuda_wmma cuda_nvml cuda_nvml_full cuda_fashion cuda_fashion_full infer \
        benchmark profiling roofline pytorch_baseline pytorch_full \
        data_mnist data_fashion everything clean re
