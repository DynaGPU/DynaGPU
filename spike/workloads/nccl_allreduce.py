#!/usr/bin/env python3
"""E11 workload: NCCL all-reduce loop across ranks (one GPU per rank).

Run with torchrun:
  python -m torch.distributed.run --nproc-per-node 2 nccl_allreduce.py --seconds 30

Each step fills a tensor with (rank + 1), all-reduces it and checks that every
element equals the expected sum, so a corrupted or half-finished collective is
caught. Prints one "start" line per rank (with its pid, so the agent can find
the rank's shim socket), "step" progress lines, and a final summary line.
"""
import argparse
import datetime
import os
import time

import torch
import torch.distributed as dist


def out(msg):
    """One write per line, so the ranks' lines never interleave."""
    os.write(1, (msg + "\n").encode())


p = argparse.ArgumentParser()
p.add_argument("--seconds", type=float, default=30)
p.add_argument("--mb", type=int, default=64, help="tensor size in MiB")
p.add_argument("--progress-ms", type=int, default=50)
p.add_argument("--timeout-s", type=int, default=120, help="NCCL collective timeout")
a = p.parse_args()

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=a.timeout_s),
                        device_id=torch.device("cuda", local))
n = a.mb * 1024 * 1024 // 4
x = torch.empty(n, device="cuda", dtype=torch.float32)
want = float(sum(range(1, world + 1)))

# warm-up
for _ in range(3):
    x.fill_(rank + 1)
    dist.all_reduce(x)
torch.cuda.synchronize()
out(f"start rank={rank} pid={os.getpid()} t={time.time():.4f}")
dist.barrier()

t0 = time.time()
steps = errors = 0
step_ms = []
last_prog = 0.0
stop = torch.zeros(1, device="cuda")
while True:
    # Ranks agree on when to stop, so none leaves while the other is mid-collective.
    stop.fill_(1.0 if time.time() - t0 >= a.seconds else 0.0)
    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
    if stop.item() > 0:
        break
    s0 = time.time()
    x.fill_(rank + 1)
    dist.all_reduce(x)
    ok = bool((x == want).all().item())  # waits for the collective
    s1 = time.time()
    steps += 1
    errors += not ok
    step_ms.append((s1 - s0) * 1000)
    if (s1 - last_prog) * 1000 >= a.progress_ms:
        out(f"step rank={rank} t={s1:.4f} i={steps}")
        last_prog = s1
step_ms.sort()
q = lambda f: step_ms[min(len(step_ms) - 1, int(len(step_ms) * f))]
out(f"done rank={rank} pid={os.getpid()} steps={steps} errors={errors} "
      f"step_p50_ms={q(0.5):.2f} step_p99_ms={q(0.99):.2f} step_max_ms={step_ms[-1]:.1f} "
      f"gib_per_s={steps * a.mb / 1024 / (time.time() - t0):.2f}")
dist.destroy_process_group()
