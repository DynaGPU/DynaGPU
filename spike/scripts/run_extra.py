#!/usr/bin/env python3
"""Extra checks after the main spike run.

runtime: two W1 apps start with CUDA's default timeslice (no override); after
         their contexts exist, timeslices 7000/3000 us are written to each
         app's DYNAGPU_CTL file. The shim applies them to the running app.
         The measured window starts after the change, so a 70/30 result means
         the runtime path works.
torch:   two PyTorch matmul apps, default timeslices and then 7000/3000 us.

Usage: run_extra.py --w1 BIN --torch-py PY --torch-script SCRIPT --shim SO --out DIR
"""
import argparse
import csv
import os
import statistics
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--w1", required=True)
p.add_argument("--torch-py", required=True)
p.add_argument("--torch-script", required=True)
p.add_argument("--shim", required=True)
p.add_argument("--out", required=True)
p.add_argument("--reps", type=int, default=3)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
logs = os.path.join(a.out, "shim-logs")
os.makedirs(logs, exist_ok=True)


def kps(line):
    return float(dict(kv.split("=", 1) for kv in line.split())["kps"])


def launch(cmd_for, apps, tag, ctl_after=None):
    start = time.time() + 6.0
    procs, ctls = [], []
    for name, ts in apps:
        env = dict(os.environ, LD_PRELOAD=a.shim,
                   DYNAGPU_LOG=os.path.join(logs, f"{tag}-{name}.log".replace(":", "-")))
        ctl = f"/tmp/dynagpu-ctl-{tag}-{name}".replace(":", "-")
        if os.path.exists(ctl):
            os.remove(ctl)
        if ctl_after:
            env["DYNAGPU_CTL"] = ctl
        elif ts:
            env["DYNAGPU_TIMESLICE_US"] = str(ts)
        ctls.append((ctl, ts))
        procs.append(subprocess.Popen(cmd_for(name, start), env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True))
    if ctl_after:
        while time.time() < start - 1.0:  # contexts exist, measurement not started
            time.sleep(0.05)
        for ctl, ts in ctls:
            with open(ctl, "w") as f:
                f.write(str(ts))
    out = [kps(pr.communicate()[0].strip().splitlines()[-1]) for pr in procs]
    return out[0] / (out[0] + out[1]), out


rows, summary = [], []
w1 = lambda n, s: [a.w1, "--seconds", "20", "--warmup", "4", "--start-at", f"{s:.3f}", "--label", n]
tq = lambda n, s: [a.torch_py, a.torch_script, "--seconds", "20", "--warmup", "4",
                   "--start-at", f"{s:.3f}", "--label", n]
cases = [
    ("runtime-w1-7:3", w1, [("a", 7000), ("b", 3000)], True, 0.7),
    ("torch-default", tq, [("a", None), ("b", None)], False, 0.5),
    ("torch-7:3", tq, [("a", 7000), ("b", 3000)], False, 0.7),
]
for name, cmd, apps, runtime, target in cases:
    shares = []
    for rep in range(a.reps):
        s, k = launch(cmd, apps, f"{name}-{rep}", ctl_after=runtime)
        shares.append(s)
        rows.append({"case": name, "rep": rep, "target_a": target, "rate_a": k[0],
                     "rate_b": k[1], "share_a": round(s, 4)})
    m = statistics.mean(shares)
    summary.append(f"{name}: target {target:.3f} measured {m:.3f} "
                   f"(sd {statistics.pstdev(shares):.3f}) error {(m - target) * 100:+.2f} pts")

with open(os.path.join(a.out, "extra.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(a.out, "extra_summary.txt"), "w") as f:
    f.write("\n".join(summary) + "\n")
print("\n".join(summary))
