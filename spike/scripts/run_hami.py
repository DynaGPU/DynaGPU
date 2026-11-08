#!/usr/bin/env python3
"""DynaGPU vs HAMi-core on the same GPU, and both loaded together.

Sections (--only):
  calib    solo speed of each workload, without and with HAMi-core loaded (no limits)
  split    the E5 cases at 70/30: DynaGPU turn lengths 7000/3000 us vs HAMi-core
           CUDA_DEVICE_SM_LIMIT 70/30
  cap      one app alone, capped: DynaGPU agent pausing vs HAMi-core SM limit
  idle     one app alone with a 30% share: DynaGPU weight vs HAMi-core limit
  launch   host cost of a kernel launch: nothing, HAMi-core, DynaGPU
  coexist  HAMi-core for memory + DynaGPU for time in the same app
  torch    PyTorch 70/30 with each

HAMi-core setup: built with its own build.sh, loaded with LD_PRELOAD, and
CUDA_REDIRECT pointing at the same library (HAMi normally installs it at
/usr/local/vgpu/libvgpu.so). Each app gets its own shared-cache file, as each
container would.

Usage: run_hami.py --spike DIR --hami PATH/libvgpu.so --py PYTHON --out DIR [--only ...]
"""
import argparse
import csv
import json
import os
import re
import socket
import statistics
import subprocess
import threading
import time

P = argparse.ArgumentParser()
P.add_argument("--spike", required=True)
P.add_argument("--hami", required=True)
P.add_argument("--py", required=True)
P.add_argument("--out", required=True)
P.add_argument("--reps", type=int, default=3)
P.add_argument("--seconds", type=float, default=15)
P.add_argument("--warmup", type=float, default=3)
P.add_argument("--only", default="calib,split,cap,idle,launch,coexist,torch")
A = P.parse_args()
ONLY = set(A.only.split(","))
W2 = f"{A.spike}/workloads/w2_work"
MEM = f"{A.spike}/workloads/mem_probe"
TORCH = f"{A.spike}/workloads/torch_matmul.py"
SHIM = f"{A.spike}/shim/libdynagpu_shim.so"
V = A.hami
os.makedirs(A.out, exist_ok=True)
LOGS = os.path.join(A.out, "logs")
os.makedirs(LOGS, exist_ok=True)

CFG = {"short": ["--iters", "4000"], "graph": ["--iters", "4000", "--graph", "32"],
       "long": ["--iters", "2000000"]}
RUN = 0


def say(line):
    print(line, flush=True)
    with open(os.path.join(A.out, "summary.txt"), "a") as f:
        f.write(line + "\n")


def kv(line):
    return dict(x.split("=", 1) for x in line.split() if "=" in x)


def safe(tag):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def writer(name, fields):
    f = open(os.path.join(A.out, name), "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    return f, w


def env(tag, hami=False, sm=None, mem=None, dyn=False, ts=None, sock=None, lease=None):
    """Environment for one app. hami/dyn choose which libraries are preloaded."""
    global RUN
    RUN += 1
    e = dict(os.environ)
    pre = []
    if hami:
        pre.append(V)
        e["CUDA_REDIRECT"] = V
        e["CUDA_DEVICE_MEMORY_SHARED_CACHE"] = f"/tmp/hami-{RUN}-{safe(tag)}.cache"
        e["LIBCUDA_LOG_LEVEL"] = "1"
        if sm is not None:
            e["CUDA_DEVICE_SM_LIMIT"] = str(sm)
        if mem is not None:
            e["CUDA_DEVICE_MEMORY_LIMIT"] = mem
    if dyn:
        pre.append(SHIM)
        e["DYNAGPU_LOG"] = os.path.join(LOGS, safe(tag) + ".log")
        if ts:
            e["DYNAGPU_TIMESLICE_US"] = str(ts)
        if sock:
            e["DYNAGPU_SOCK"] = sock
        if lease:
            e["DYNAGPU_LEASE_MS"] = str(lease)
    if pre:
        e["LD_PRELOAD"] = ":".join(pre)
    return e


class Agent:
    def __init__(self, name="agent"):
        self.me = f"/tmp/dgh-{name}.{os.getpid()}.{threading.get_ident()}.sock"
        if os.path.exists(self.me):
            os.unlink(self.me)
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.s.bind(self.me)
        self.s.settimeout(2.0)
        self.bad = 0
        self.lat = []

    def send(self, cmd, socks):
        if not socks:
            return []
        t0 = time.perf_counter()
        sent, out = 0, []
        for p in socks:
            try:
                self.s.sendto(cmd.encode(), p)
                sent += 1
            except OSError:
                out.append("refused")
                self.bad += 1
        for _ in range(sent):
            try:
                r = self.s.recv(128).decode()
            except OSError:
                r = "timeout"
            out.append(r)
            self.bad += not r.startswith("ok")
        self.lat.append((time.perf_counter() - t0) * 1e6)
        return out

    def close(self):
        self.s.close()
        os.unlink(self.me)


def clear(*paths):
    for p in paths:
        if os.path.exists(p):
            os.unlink(p)


def wait_ready(socks, timeout=30):
    ag = Agent("ready")
    end = time.time() + timeout
    try:
        while time.time() < end:
            if all(os.path.exists(p) for p in socks):
                if all(x.startswith("ok") for x in ag.send("hb", socks)):
                    return True
            time.sleep(0.2)
        return False
    finally:
        ag.close()


def run_apps(apps, during=None):
    """apps: (label, cmd, env); all start together. Returns {label: rate}."""
    start = time.time() + 6
    ps = []
    for label, cmd, e in apps:
        full = cmd + ["--seconds", str(A.seconds), "--warmup", str(A.warmup),
                      "--start-at", f"{start:.3f}", "--label", label]
        ps.append((label, subprocess.Popen(full, env=e, stdout=subprocess.PIPE,
                                           stderr=subprocess.DEVNULL, text=True)))
    th = None
    if during:
        th = threading.Thread(target=during, args=(start, start + A.seconds))
        th.start()
    out = {}
    for label, p in ps:
        lines = p.communicate()[0].strip().splitlines()
        r = kv(lines[-1]) if lines else {}
        out[label] = float(r.get("kps", "nan"))
    if th:
        th.join()
    return out


def duty_cycle(sock, cap, window=0.1):
    """DynaGPU cap: the agent pauses the app for (1 - cap) of every window."""
    def run(start, end):
        if not wait_ready([sock]):
            return
        ag = Agent("cap")
        t = start - 1.0
        while time.time() < t:
            time.sleep(0.001)
        while t < end - 0.2:
            t += cap * window
            time.sleep(max(0, t - time.time()))
            ag.send("pause", [sock])
            t += (1 - cap) * window
            time.sleep(max(0, t - time.time()))
            ag.send("resume", [sock])
        ag.close()
    return run


# ---------------------------------------------------------------- calibration
SOLO = {}
cal = os.path.join(A.out, "solo.json")
if "calib" in ONLY or not os.path.exists(cal):
    f, w = writer("solo.csv", ["config", "loaded", "rep", "kps"])
    for name, args in CFG.items():
        for loaded in ("none", "hami"):
            if loaded == "hami" and name == "long":
                continue
            vals = []
            for rep in range(A.reps):
                e = env(f"solo-{name}-{loaded}-{rep}", hami=loaded == "hami")
                k = run_apps([("solo", [W2] + args, e)])["solo"]
                vals.append(k)
                w.writerow({"config": name, "loaded": loaded, "rep": rep, "kps": k})
                f.flush()
            SOLO[f"{name}/{loaded}"] = statistics.mean(vals)
            say(f"solo {name} ({'HAMi-core loaded, no limit' if loaded == 'hami' else 'nothing loaded'}): "
                f"{statistics.mean(vals):.2f} kps (sd {statistics.pstdev(vals):.2f})")
        if f"{name}/hami" in SOLO:
            say(f"  HAMi-core overhead with no limit, {name}: "
                f"{(1 - SOLO[name + '/hami'] / SOLO[name + '/none']) * 100:+.2f}%")
    f.close()
    json.dump(SOLO, open(cal, "w"))
else:
    SOLO = json.load(open(cal))
S = lambda cfg: SOLO[f"{cfg}/none"]

# ---------------------------------------------------------------- split (E5 at 70/30)
if "split" in ONLY:
    f, w = writer("split.csv", ["case", "system", "rep", "kps_a", "kps_b", "norm_a", "norm_b",
                                "share_a", "total_vs_solo"])
    cases = [("short vs short", "short", "short"), ("graph vs graph", "graph", "graph"),
             ("long vs long", "long", "long"), ("long vs short", "long", "short"),
             ("short vs long", "short", "long"), ("graph vs short", "graph", "short")]
    for name, ca, cb in cases:
        for system in ("DynaGPU", "HAMi-core"):
            shares, totals = [], []
            for rep in range(A.reps):
                tag = f"split-{name}-{system}-{rep}"
                if system == "DynaGPU":
                    ea, eb = env(tag + "-a", dyn=True, ts=7000), env(tag + "-b", dyn=True, ts=3000)
                else:
                    ea, eb = env(tag + "-a", hami=True, sm=70), env(tag + "-b", hami=True, sm=30)
                r = run_apps([("a", [W2] + CFG[ca], ea), ("b", [W2] + CFG[cb], eb)])
                na, nb = r["a"] / S(ca), r["b"] / S(cb)
                share = na / (na + nb)
                shares.append(share)
                totals.append(na + nb)
                w.writerow({"case": name, "system": system, "rep": rep, "kps_a": r["a"], "kps_b": r["b"],
                            "norm_a": round(na, 4), "norm_b": round(nb, 4), "share_a": round(share, 4),
                            "total_vs_solo": round(na + nb, 4)})
                f.flush()
            m = statistics.mean(shares)
            say(f"split 70/30 {name}, {system}: A share {m * 100:.1f}% (error {(m - 0.7) * 100:+.1f} pts, "
                f"runs {min(shares) * 100:.1f}-{max(shares) * 100:.1f}%), GPU used "
                f"{statistics.mean(totals) * 100:.1f}% of one app alone")
    f.close()

# ---------------------------------------------------------------- cap (one app alone)
if "cap" in ONLY:
    f, w = writer("cap.csv", ["case", "system", "cap", "rep", "kps", "share"])
    for cfg, cap in (("short", 0.2), ("short", 0.4), ("short", 0.8), ("graph", 0.4), ("long", 0.4)):
        for system in ("DynaGPU", "HAMi-core"):
            vals = []
            for rep in range(A.reps):
                tag = f"cap-{cfg}-{cap}-{system}-{rep}"
                if system == "DynaGPU":
                    sock = "/tmp/dgh-cap.sock"
                    clear(sock)
                    r = run_apps([("a", [W2] + CFG[cfg], env(tag, dyn=True, sock=sock))],
                                 during=duty_cycle(sock, cap))
                else:
                    r = run_apps([("a", [W2] + CFG[cfg], env(tag, hami=True, sm=int(cap * 100)))])
                s = r["a"] / S(cfg)
                vals.append(s)
                w.writerow({"case": cfg, "system": system, "cap": cap, "rep": rep, "kps": r["a"],
                            "share": round(s, 4)})
                f.flush()
            m = statistics.mean(vals)
            say(f"cap {cap * 100:.0f}% alone, {cfg} kernels, {system}: {m * 100:.1f}% of solo "
                f"(error {(m - cap) * 100:+.1f} pts, runs {min(vals) * 100:.1f}-{max(vals) * 100:.1f}%)")
    f.close()

# ---------------------------------------------------------------- idle (work conservation)
if "idle" in ONLY:
    f, w = writer("idle.csv", ["system", "setting", "rep", "share"])
    for system, setting in (("DynaGPU", "weight 30% (3000 us turns), no cap"),
                            ("HAMi-core", "CUDA_DEVICE_SM_LIMIT=30")):
        vals = []
        for rep in range(A.reps):
            tag = f"idle-{system}-{rep}"
            e = env(tag, dyn=True, ts=3000) if system == "DynaGPU" else env(tag, hami=True, sm=30)
            r = run_apps([("a", [W2] + CFG["short"], e)])
            vals.append(r["a"] / S("short"))
            w.writerow({"system": system, "setting": setting, "rep": rep, "share": round(vals[-1], 4)})
        say(f"idle GPU, app with a 30% setting, {system}: gets {statistics.mean(vals) * 100:.1f}% of solo")
    f.close()

# ---------------------------------------------------------------- launch cost
if "launch" in ONLY:
    f, w = writer("launch.csv", ["setup", "rep", "launch_ns", "roundtrip_p50_us", "roundtrip_p99_us"])
    setups = [("nothing loaded", lambda t: env(t)),
              ("HAMi-core, no limit", lambda t: env(t, hami=True)),
              ("HAMi-core, SM limit 50", lambda t: env(t, hami=True, sm=50)),
              ("DynaGPU shim", lambda t: env(t, dyn=True)),
              ("both loaded, no limits", lambda t: env(t, hami=True, dyn=True))]
    res = {s: [] for s, _ in setups}
    for rep in range(5):
        for name, mk in setups:
            out = subprocess.run([W2, "--launch-bench", "--label", "lb"], env=mk(f"launch-{rep}"),
                                 capture_output=True, text=True).stdout.strip().splitlines()
            r = kv(out[-1])
            res[name].append(r)
            w.writerow({"setup": name, "rep": rep, "launch_ns": r["launch_ns"],
                        "roundtrip_p50_us": r["roundtrip_p50_us"], "roundtrip_p99_us": r["roundtrip_p99_us"]})
    f.close()
    for name, _ in setups:
        m = lambda k: statistics.mean(float(r[k]) for r in res[name])
        say(f"launch, {name}: {m('launch_ns'):.0f} ns per launch, round trip p50 "
            f"{m('roundtrip_p50_us'):.1f} us, p99 {m('roundtrip_p99_us'):.1f} us")

# ---------------------------------------------------------------- coexist
if "coexist" in ONLY:
    # (a) memory limit
    f, w = writer("coexist_memory.csv", ["setup", "reported_total_mib", "allocated_gib", "stop_reason"])
    for name, e in (("nothing loaded", env("mem-none")),
                    ("DynaGPU only", env("mem-dyn", dyn=True)),
                    ("HAMi-core, 8 GiB limit", env("mem-hami", hami=True, mem="8g")),
                    ("HAMi-core 8 GiB + DynaGPU", env("mem-both", hami=True, mem="8g", dyn=True, ts=7000))):
        out = subprocess.run([MEM], env=e, capture_output=True, text=True).stdout.strip().splitlines()
        r = kv(out[-1]) if out else {}
        w.writerow({"setup": name, "reported_total_mib": r.get("reported_total_mib"),
                    "allocated_gib": r.get("allocated_gib"), "stop_reason": r.get("stop_reason")})
        say(f"memory, {name}: app sees {r.get('reported_total_mib')} MiB, allocated "
            f"{r.get('allocated_gib')} GiB before {r.get('stop_reason')}")
    f.close()
    # (b) 70/30 with both libraries in both apps (HAMi-core memory only, DynaGPU time)
    shares = []
    for rep in range(A.reps):
        r = run_apps([("a", [W2] + CFG["short"], env(f"co-split-{rep}-a", hami=True, mem="8g", dyn=True, ts=7000)),
                      ("b", [W2] + CFG["short"], env(f"co-split-{rep}-b", hami=True, mem="8g", dyn=True, ts=3000))])
        shares.append(r["a"] / (r["a"] + r["b"]))
    say(f"both loaded, DynaGPU 7000/3000: A share {statistics.mean(shares) * 100:.1f}% "
        f"(runs {min(shares) * 100:.1f}-{max(shares) * 100:.1f}%)")
    with open(os.path.join(A.out, "coexist_split.csv"), "w") as g:
        g.write("rep,share_a\n" + "".join(f"{i},{s:.4f}\n" for i, s in enumerate(shares)))
    # (c) 1000 pauses with checksum, both libraries loaded
    ref = kv(subprocess.run([W2, "--iters", "600000", "--kernels", "1200"], capture_output=True,
                            text=True).stdout.strip().splitlines()[-1])
    sock = "/tmp/dgh-co.sock"
    clear(sock)
    app = subprocess.Popen([W2, "--iters", "600000", "--kernels", "1200"],
                           env=env("co-pause", hami=True, mem="8g", dyn=True, sock=sock, lease=1000),
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    wait_ready([sock])
    ag = Agent("co")
    for _ in range(1000):
        time.sleep(0.003)
        ag.send("pause", [sock])
        time.sleep(0.002)
        ag.send("resume", [sock])
    ag.close()
    t = kv(app.communicate(timeout=600)[0].strip().splitlines()[-1])
    say(f"both loaded, 1000 DynaGPU pauses: agent errors {ag.bad}, checksum "
        f"{'MATCH' if t.get('checksum') == ref.get('checksum') else 'MISMATCH'}")
    # (d) HAMi-core's SM limit and DynaGPU turns at the same time (should be avoided)
    r = run_apps([("a", [W2] + CFG["short"], env("co-double-a", hami=True, sm=70, dyn=True, ts=7000)),
                  ("b", [W2] + CFG["short"], env("co-double-b", hami=True, sm=30, dyn=True, ts=3000))])
    na, nb = r["a"] / S("short"), r["b"] / S("short")
    say(f"both limiting at once (HAMi-core 70/30 + DynaGPU 7000/3000): A share {na / (na + nb) * 100:.1f}%, "
        f"GPU used {(na + nb) * 100:.1f}% of one app alone")

# ---------------------------------------------------------------- torch
if "torch" in ONLY:
    f, w = writer("torch.csv", ["system", "rep", "kps_a", "kps_b", "share_a", "total_vs_solo"])
    tcmd = [A.py, TORCH, "--sync-every", "0"]
    solo = statistics.mean(run_apps([("s", tcmd, env(f"torch-solo-{i}"))])["s"] for i in range(2))
    say(f"torch solo: {solo:.1f} matmuls/s")
    for system in ("DynaGPU", "HAMi-core"):
        shares, tots = [], []
        for rep in range(A.reps):
            tag = f"torch-{system}-{rep}"
            if system == "DynaGPU":
                ea, eb = env(tag + "-a", dyn=True, ts=7000), env(tag + "-b", dyn=True, ts=3000)
            else:
                ea, eb = env(tag + "-a", hami=True, sm=70), env(tag + "-b", hami=True, sm=30)
            r = run_apps([("a", tcmd, ea), ("b", tcmd, eb)])
            s = r["a"] / (r["a"] + r["b"])
            shares.append(s)
            tots.append((r["a"] + r["b"]) / solo)
            w.writerow({"system": system, "rep": rep, "kps_a": r["a"], "kps_b": r["b"],
                        "share_a": round(s, 4), "total_vs_solo": round(tots[-1], 4)})
            f.flush()
        say(f"torch 70/30, {system}: A share {statistics.mean(shares) * 100:.1f}% (runs "
            f"{min(shares) * 100:.1f}-{max(shares) * 100:.1f}%), GPU used {statistics.mean(tots) * 100:.1f}% of one app alone")
    f.close()

print("done", flush=True)
