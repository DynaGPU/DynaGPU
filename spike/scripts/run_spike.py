#!/usr/bin/env python3
"""Run the week-1 spike tests (E3, E4 and overhead) and write raw CSVs.

Every run starts its processes at a shared wall-clock time, so their
measured windows overlap exactly. Shares are computed from kernels
completed per second (kps); W1 kernels are fixed units of work.

Usage: run_spike.py --bin ../workloads/w1_busy --shim ../shim/libdynagpu_shim.so \
                    --out ../results [--reps 5] [--seconds 20] [--warmup 4]
"""
import argparse
import csv
import os
import statistics
import subprocess
import time

P = argparse.ArgumentParser()
P.add_argument("--bin", required=True)
P.add_argument("--shim", required=True)
P.add_argument("--out", required=True)
P.add_argument("--reps", type=int, default=5)
P.add_argument("--seconds", type=float, default=20)
P.add_argument("--warmup", type=float, default=4)
P.add_argument("--only", default="", help="comma list: solo,default,e3,e4")
A = P.parse_args()
os.makedirs(A.out, exist_ok=True)
LOGDIR = os.path.join(A.out, "shim-logs")
os.makedirs(LOGDIR, exist_ok=True)


def parse(line):
    return dict(kv.split("=", 1) for kv in line.split())


def run_group(apps, tag):
    """apps: list of dicts {label, ts_us or None, extra args}. Returns kps per label."""
    start = time.time() + 4.0
    procs = []
    for a in apps:
        env = dict(os.environ)
        if a.get("ts_us") is not None or a.get("shim"):
            env["LD_PRELOAD"] = A.shim
            env["DYNAGPU_LOG"] = os.path.join(LOGDIR, f"{tag}-{a['label']}.log".replace(":", "-"))
            if a.get("ts_us") is not None:
                env["DYNAGPU_TIMESLICE_US"] = str(a["ts_us"])
        cmd = [A.bin, "--seconds", str(A.seconds), "--warmup", str(A.warmup),
               "--start-at", f"{start:.3f}", "--label", a["label"]] + a.get("args", [])
        procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True))
    out = {}
    for a, p in zip(apps, procs):
        line = p.communicate()[0].strip().splitlines()[-1]
        if p.returncode != 0:
            raise SystemExit(f"{tag} {a['label']} failed: {line}")
        out[a["label"]] = float(parse(line)["kps"])
    return out


def writer(name, fields):
    f = open(os.path.join(A.out, name), "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    return f, w


only = set(A.only.split(",")) if A.only else {"solo", "default", "e3", "e4"}
summary = []

# Solo speed, with and without the shim (also measures shim overhead).
solo = None
if "solo" in only or True:
    f, w = writer("solo.csv", ["rep", "shim", "kps"])
    vals = {False: [], True: []}
    for rep in range(A.reps):
        for shim in (False, True):
            k = run_group([{"label": "solo", "shim": shim}], f"solo-{rep}-{int(shim)}")["solo"]
            vals[shim].append(k)
            w.writerow({"rep": rep, "shim": int(shim), "kps": k})
            f.flush()
    f.close()
    solo = statistics.mean(vals[False])
    over = (statistics.mean(vals[False]) - statistics.mean(vals[True])) / solo * 100
    summary.append(f"solo kps={solo:.1f} (sd {statistics.pstdev(vals[False]):.1f}); "
                   f"with shim {statistics.mean(vals[True]):.1f}; shim overhead {over:+.2f}%")

# Baseline: two apps with the driver's default turns (no shim).
if "default" in only:
    f, w = writer("default_pair.csv", ["rep", "kps_a", "kps_b", "share_a", "efficiency"])
    shares = []
    for rep in range(A.reps):
        r = run_group([{"label": "a"}, {"label": "b"}], f"default-{rep}")
        s = r["a"] / (r["a"] + r["b"])
        shares.append(s)
        w.writerow({"rep": rep, "kps_a": r["a"], "kps_b": r["b"], "share_a": round(s, 4),
                    "efficiency": round((r["a"] + r["b"]) / solo, 4)})
        f.flush()
    f.close()
    summary.append(f"default pair share_a={statistics.mean(shares):.3f} "
                   f"(sd {statistics.pstdev(shares):.3f})")

# E3: weighted shares from timeslice lengths.
E3 = [  # (name, ts_a_us, ts_b_us)
    ("1:1", 5000, 5000),
    ("7:3", 7000, 3000),
    ("9:1", 9000, 1000),
    ("95:5", 19000, 1000),
    ("7:3-short", 1400, 600),
]
if "e3" in only:
    f, w = writer("e3_weighted.csv", ["split", "ts_a_us", "ts_b_us", "target_a", "rep",
                                      "kps_a", "kps_b", "share_a", "efficiency"])
    for name, ta, tb in E3:
        target = ta / (ta + tb)
        shares, effs = [], []
        for rep in range(A.reps):
            r = run_group([{"label": "a", "ts_us": ta}, {"label": "b", "ts_us": tb}],
                          f"e3-{name}-{rep}")
            s = r["a"] / (r["a"] + r["b"])
            e = (r["a"] + r["b"]) / solo
            shares.append(s)
            effs.append(e)
            w.writerow({"split": name, "ts_a_us": ta, "ts_b_us": tb,
                        "target_a": round(target, 4), "rep": rep, "kps_a": r["a"],
                        "kps_b": r["b"], "share_a": round(s, 4), "efficiency": round(e, 4)})
            f.flush()
        m = statistics.mean(shares)
        summary.append(f"E3 {name}: target {target:.3f} measured {m:.3f} "
                       f"(sd {statistics.pstdev(shares):.3f}) error {(m - target) * 100:+.2f} pts; "
                       f"efficiency {statistics.mean(effs):.3f}")
    f.close()

# E4: work conservation. A has the small share; B is mostly idle.
if "e4" in only:
    f, w = writer("e4_work_conserving.csv", ["case", "rep", "kps_a", "kps_b", "a_vs_solo"])
    cases = [
        ("b-bursty-5ms-per-100ms", ["--burst-ms", "5", "--idle-ms", "95"]),
        ("b-idle-1ms-per-1s", ["--burst-ms", "1", "--idle-ms", "1000"]),
    ]
    for case, bargs in cases:
        vals = []
        for rep in range(A.reps):
            r = run_group([{"label": "a", "ts_us": 3000},
                           {"label": "b", "ts_us": 7000, "args": bargs}], f"e4-{case}-{rep}")
            v = r["a"] / solo
            vals.append(v)
            w.writerow({"case": case, "rep": rep, "kps_a": r["a"], "kps_b": r["b"],
                        "a_vs_solo": round(v, 4)})
            f.flush()
        summary.append(f"E4 {case}: busy app got {statistics.mean(vals) * 100:.1f}% of solo "
                       f"(min {min(vals) * 100:.1f}%)")
    f.close()

with open(os.path.join(A.out, "summary.txt"), "a") as s:
    s.write("\n".join(summary) + "\n")
print("\n".join(summary))
