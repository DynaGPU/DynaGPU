#!/usr/bin/env python3
"""E7 follow-up: total throughput vs turn length, to find the shortest turn
that keeps the switching loss within 5%.

Usage: run_e7_sweep.py --spike DIR --out DIR [--reps 3]
Needs the solo speed in <out>/solo.json (written by run_week2.py).
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--spike", required=True)
p.add_argument("--out", required=True)
p.add_argument("--reps", type=int, default=3)
a = p.parse_args()
W2 = f"{a.spike}/workloads/w2_work"
SHIM = f"{a.spike}/shim/libdynagpu_shim.so"
solo = json.load(open(os.path.join(a.out, "solo.json")))["short"]

rows, lines = [], []
for napps, ts in [(2, 2000), (2, 3000), (2, 5000), (2, 10000), (4, 3000), (4, 5000)]:
    effs = []
    for rep in range(a.reps):
        start = time.time() + 6
        ps = []
        for i in range(napps):
            env = dict(os.environ, LD_PRELOAD=SHIM, DYNAGPU_TIMESLICE_US=str(ts),
                       DYNAGPU_LOG="/dev/null")
            ps.append(subprocess.Popen([W2, "--iters", "4000", "--seconds", "20", "--warmup", "4",
                                        "--start-at", f"{start:.3f}", "--label", f"x{i}"],
                                       env=env, stdout=subprocess.PIPE, text=True))
        total = 0.0
        for pr in ps:
            line = pr.communicate()[0].strip().splitlines()[-1]
            total += float(dict(x.split("=", 1) for x in line.split())["kps"])
        e = total / solo
        effs.append(e)
        rows.append({"apps": napps, "ts_us": ts, "rep": rep, "sum_kps": round(total, 2),
                     "efficiency": round(e, 4)})
    m = statistics.mean(effs)
    lines.append(f"E7 sweep {napps} apps at {ts} us turns: throughput {m * 100:.1f}% of solo "
                 f"(loss {(1 - m) * 100:.1f}%)")
    print(lines[-1], flush=True)

with open(os.path.join(a.out, "e7_sweep.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(a.out, "summary.txt"), "a") as f:
    f.write("\n".join(lines) + "\n")
