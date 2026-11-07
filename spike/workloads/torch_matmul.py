#!/usr/bin/env python3
"""PyTorch workload for E1 (find channel groups in a real framework) and an
E3 cross-check. Runs back-to-back matmuls and prints matmuls per second.

Usage: torch_matmul.py --seconds 20 --warmup 4 [--start-at <epoch>] [--label x]
"""
import argparse
import os
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--seconds", type=float, default=20)
p.add_argument("--warmup", type=float, default=4)
p.add_argument("--start-at", type=float, default=0)
p.add_argument("--label", default="torch")
p.add_argument("--n", type=int, default=4096)
a = p.parse_args()

x = torch.randn(a.n, a.n, device="cuda", dtype=torch.float16)
y = torch.randn(a.n, a.n, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
while a.start_at and time.time() < a.start_at:
    time.sleep(0.001)

t0 = time.time()
count = 0
BATCH = 16
while True:
    for _ in range(BATCH):
        z = x @ y
    torch.cuda.synchronize()
    t = time.time()
    if t - t0 >= a.seconds:
        break
    if t - t0 >= a.warmup:
        count += BATCH
window = a.seconds - a.warmup
print(f"label={a.label} pid={os.getpid()} window_s={window:.1f} matmuls={count} kps={count / window:.2f}")
