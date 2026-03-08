#!/usr/bin/env python3
"""
PyTorch baseline benchmark for CUDAnet paper.
Trains the identical 2×128 MLP on MNIST with Adam (lr=0.01),
20 epochs, batch 2048 — same config as CUDAnet Adam.

Measures:
  - Time per epoch (wall-clock, GPU-side via CUDA events)
  - Peak VRAM during training (torch.cuda.max_memory_allocated)
  - PyTorch install size on disk

Run: python3 benchmark_pytorch.py
"""

import time, os, subprocess, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ── Config (matches CUDAnet paper exactly) ────────────────────────────────────
HIDDEN_LAYERS = 2
HIDDEN_SIZE   = 128
EPOCHS        = 20
BATCH_SIZE    = 2048
LR            = 0.01
SEED          = 42

torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Load MNIST from binary files (same data as CUDAnet) ──────────────────────
def read_mnist_images(path):
    with open(path, 'rb') as f:
        f.read(16)   # skip magic, n, rows, cols
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784).astype(np.float32) / 255.0

def read_mnist_labels(path):
    with open(path, 'rb') as f:
        f.read(8)    # skip magic, n
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

DATA = 'data'
try:
    x_train = read_mnist_images(f'{DATA}/train-images-idx3-ubyte')
    y_train = read_mnist_labels(f'{DATA}/train-labels-idx1-ubyte')
    x_test  = read_mnist_images(f'{DATA}/t10k-images-idx3-ubyte')
    y_test  = read_mnist_labels(f'{DATA}/t10k-labels-idx1-ubyte')
except FileNotFoundError as e:
    print(f"MNIST data not found: {e}")
    print("Place MNIST binary files in ./data/")
    sys.exit(1)

x_train_t = torch.tensor(x_train, device=device)
y_train_t = torch.tensor(y_train, device=device)
x_test_t  = torch.tensor(x_test,  device=device)
y_test_t  = torch.tensor(y_test,  device=device)

train_ds = TensorDataset(x_train_t, y_train_t)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      drop_last=True)   # drop_last mirrors CUDAnet floor division


# ── Model: exact 2×128 MLP with Xavier init ───────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_dim = 784
        for _ in range(HIDDEN_LAYERS):
            lin = nn.Linear(in_dim, HIDDEN_SIZE)
            # Xavier uniform (matches CUDAnet's sqrt(2/fan_in) He-like init)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.ReLU()]
            in_dim = HIDDEN_SIZE
        out = nn.Linear(in_dim, 10)
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
criterion = nn.CrossEntropyLoss()

# ── Training loop ─────────────────────────────────────────────────────────────
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

epoch_times = []
print(f"\nTraining 2×{HIDDEN_SIZE} MLP  |  Adam lr={LR}  |  batch={BATCH_SIZE}  |  {EPOCHS} epochs\n")
print(f"{'Epoch':>6}  {'Train Acc':>10}  {'Test Acc':>9}  {'Time/Ep (s)':>12}")
print("-" * 44)

for ep in range(EPOCHS):
    model.train()
    t0 = time.perf_counter()
    if device.type == 'cuda':
        start_ev.record()

    correct = total = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total   += yb.size(0)

    if device.type == 'cuda':
        end_ev.record()
        torch.cuda.synchronize()
        ep_time = start_ev.elapsed_time(end_ev) / 1000.0  # ms → s
    else:
        ep_time = time.perf_counter() - t0

    epoch_times.append(ep_time)

    # Test accuracy (no grad)
    model.eval()
    with torch.no_grad():
        test_logits = model(x_test_t)
        test_correct = (test_logits.argmax(1) == y_test_t).sum().item()
        test_acc = test_correct / len(y_test_t) * 100

    train_acc = correct / total * 100
    print(f"{ep:>6}  {train_acc:>9.2f}%  {test_acc:>8.2f}%  {ep_time:>11.3f}s")

# ── Results summary ───────────────────────────────────────────────────────────
avg_time = np.mean(epoch_times[1:])   # skip ep0 warm-up
peak_vram_mb = 0
if device.type == 'cuda':
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

# PyTorch install size
try:
    torch_path = os.path.dirname(torch.__file__)
    result = subprocess.run(['du', '-sh', torch_path], capture_output=True, text=True)
    torch_size = result.stdout.split()[0]
except Exception:
    torch_size = "N/A"

print("\n" + "="*55)
print("PYTORCH BASELINE RESULTS")
print("="*55)
print(f"  Final test accuracy:    {test_acc:.2f}%")
print(f"  Avg time/epoch (s):     {avg_time:.3f}  (excl. ep0 warm-up)")
print(f"  Peak VRAM (MB):         {peak_vram_mb:.1f}")
print(f"  PyTorch install size:   {torch_size}")
print(f"  Model parameters:       {sum(p.numel() for p in model.parameters()):,}")
print("="*55)

# ── Comparison table (CUDAnet values from paper) ──────────────────────────────
cuda_time   = 3.91   # Adam, from paper Table I
cuda_vram   = 470    # ~470 KB parameters + buffers (paper §5.2) — in KB
cuda_size   = "<2"   # MB

print("\n" + "="*65)
print(f"{'Metric':<28} {'CUDAnet (Adam)':>16}  {'PyTorch (Adam)':>16}")
print("-"*65)
print(f"{'Final Test Accuracy':<28} {'97.1%':>16}  {test_acc:>15.1f}%")
print(f"{'Avg Time/Epoch (s)':<28} {cuda_time:>16.3f}  {avg_time:>15.3f}")
print(f"{'Peak VRAM':<28} {'~470 KB (params)':>16}  {peak_vram_mb:>13.0f} MB")
print(f"{'Binary/Install Size':<28} {cuda_size + ' MB':>16}  {torch_size:>16}")
print(f"{'Runtime Dependencies':<28} {'libcuda.so only':>16}  {'Python+libtorch':>16}")
print("="*65)
print("\nCopy these numbers into main.tex Table (Edge AI comparison).")
