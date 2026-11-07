#!/usr/bin/env python3
"""PyTorch workload: back-to-back FP16 matmuls, prints matmuls per second.

Week 1 used --sync-every 16. Week 2 adds:
  --sync-every 0   never idle: keep 4 batches queued using CUDA events
  --graph          capture a batch of matmuls as one CUDA graph

Usage: torch_matmul.py --seconds 20 --warmup 4 [--start-at <epoch>]
                       [--sync-every 16] [--batch 16] [--graph] [--label x]
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
p.add_argument("--batch", type=int, default=16)
p.add_argument("--sync-every", type=int, default=16,
               help="matmuls between full waits; 0 = keep the GPU queue full")
p.add_argument("--graph", action="store_true")
a = p.parse_args()

x = torch.randn(a.n, a.n, device="cuda", dtype=torch.float16)
y = torch.randn(a.n, a.n, device="cuda", dtype=torch.float16)
z = torch.empty_like(x)

if a.graph:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            torch.matmul(x, y, out=z)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(a.batch):
            torch.matmul(x, y, out=z)

    def run_batch():
        g.replay()
else:
    def run_batch():
        for _ in range(a.batch):
            torch.matmul(x, y, out=z)

torch.cuda.synchronize()
while a.start_at and time.time() < a.start_at:
    time.sleep(0.001)

t0 = time.time()
count = 0
if a.sync_every > 0:
    per_wait = max(1, a.sync_every // a.batch)
    while True:
        for _ in range(per_wait):
            run_batch()
        torch.cuda.synchronize()
        t = time.time()
        if t - t0 >= a.seconds:
            break
        if t - t0 >= a.warmup:
            count += per_wait * a.batch
else:
    events = []
    while True:
        run_batch()
        e = torch.cuda.Event()
        e.record()
        events.append(e)
        if len(events) > 4:
            events.pop(0).synchronize()
            t = time.time()
            if t - t0 >= a.seconds:
                break
            if t - t0 >= a.warmup:
                count += a.batch
    torch.cuda.synchronize()

window = a.seconds - a.warmup
print(f"label={a.label} pid={os.getpid()} graph={int(a.graph)} sync_every={a.sync_every} "
      f"window_s={window:.1f} matmuls={count} kps={count / window:.2f}")
