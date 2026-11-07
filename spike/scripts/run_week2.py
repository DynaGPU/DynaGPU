#!/usr/bin/env python3
"""Week-2 spike tests: E5 (graphs, long kernels), PyTorch shortfall,
E6 (priority latency), E7 (cost), E8 (10,000 pause cycles).

Shares are normalised by each app's solo speed in the same configuration, so
apps with different kernel sizes can be compared:
    share_a = (rate_a / solo_a) / (rate_a / solo_a + rate_b / solo_b)

Usage: run_week2.py --spike DIR --py PYTHON --out DIR [--only calib,e5,torch,e6,e7,e8]
"""
import argparse
import csv
import json
import os
import re
import socket
import statistics
import subprocess
import time

P = argparse.ArgumentParser()
P.add_argument("--spike", required=True, help="spike/ directory on the GPU machine")
P.add_argument("--py", required=True, help="python with torch")
P.add_argument("--out", required=True)
P.add_argument("--reps", type=int, default=5)
P.add_argument("--only", default="calib,e5,torch,e6,e7,e8")
A = P.parse_args()
ONLY = set(A.only.split(","))
W2 = f"{A.spike}/workloads/w2_work"
HP = f"{A.spike}/workloads/hp_probe"
TORCH = f"{A.spike}/workloads/torch_matmul.py"
SHIM = f"{A.spike}/shim/libdynagpu_shim.so"
os.makedirs(A.out, exist_ok=True)
LOGS = os.path.join(A.out, "shim-logs")
os.makedirs(LOGS, exist_ok=True)
SUMMARY = []

CFG = {  # name -> w2 args
    "short": ["--iters", "4000"],                     # ~67 us kernels
    "long": ["--iters", "2000000"],                   # ~33 ms kernels (longer than any turn)
    "graph": ["--iters", "4000", "--graph", "32"],    # CUDA graphs of 32 kernels
}


def say(line):
    print(line, flush=True)
    SUMMARY.append(line)
    with open(os.path.join(A.out, "summary.txt"), "a") as f:
        f.write(line + "\n")


def kv(line):
    return dict(x.split("=", 1) for x in line.split() if "=" in x)


def env_for(tag, ts=None, sock=None, il=None, shim=True):
    e = dict(os.environ)
    if shim:
        e["LD_PRELOAD"] = SHIM
        e["DYNAGPU_LOG"] = os.path.join(LOGS, re.sub(r"[^A-Za-z0-9._-]+", "_", tag.replace(":", "-")) + ".log")
    if ts:
        e["DYNAGPU_TIMESLICE_US"] = str(ts)
    if sock:
        e["DYNAGPU_SOCK"] = sock
    if il is not None:
        e["DYNAGPU_INTERLEAVE"] = str(il)
    return e


def run_apps(apps, seconds=20, warmup=4):
    """apps: list of (label, cmd_prefix, env). All start together. Returns {label: kps}."""
    start = time.time() + 6
    ps = []
    for label, cmd, env in apps:
        full = cmd + ["--seconds", str(seconds), "--warmup", str(warmup),
                      "--start-at", f"{start:.3f}", "--label", label]
        ps.append((label, subprocess.Popen(full, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.DEVNULL, text=True)))
    out = {}
    for label, p in ps:
        line = p.communicate()[0].strip().splitlines()[-1]
        out[label] = float(kv(line)["kps"])
    return out


def writer(name, fields):
    f = open(os.path.join(A.out, name), "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    return f, w


def dmesg_xid():
    r = subprocess.run(["sudo", "dmesg"], capture_output=True, text=True).stdout
    return sum(1 for l in r.splitlines() if "xid" in l.lower())


# ---------------------------------------------------------------- calibration
SOLO = {}
cal_path = os.path.join(A.out, "solo.json")
if "calib" in ONLY or not os.path.exists(cal_path):
    f, w = writer("solo.csv", ["config", "rep", "kps"])
    for name, args in CFG.items():
        vals = []
        for rep in range(3):
            k = run_apps([("solo", [W2] + args, env_for(f"solo-{name}-{rep}"))])["solo"]
            vals.append(k)
            w.writerow({"config": name, "rep": rep, "kps": k})
        SOLO[name] = statistics.mean(vals)
        say(f"solo {name}: {SOLO[name]:.2f} kps (sd {statistics.pstdev(vals):.2f})")
    f.close()
    json.dump(SOLO, open(cal_path, "w"))
else:
    SOLO = json.load(open(cal_path))

# ---------------------------------------------------------------- E5
if "e5" in ONLY:
    cases = [  # name, cfg_a, cfg_b, ts_a, ts_b
        ("short-vs-short 7:3 (control)", "short", "short", 7000, 3000),
        ("graph-vs-graph 7:3", "graph", "graph", 7000, 3000),
        ("long-vs-long 7:3", "long", "long", 7000, 3000),
        ("long-vs-short 7:3", "long", "short", 7000, 3000),
        ("short-vs-long 7:3", "short", "long", 7000, 3000),
        ("graph-vs-short 7:3", "graph", "short", 7000, 3000),
        ("long-vs-short default (no DynaGPU)", "long", "short", None, None),
    ]
    f, w = writer("e5_graphs_long.csv", ["case", "rep", "target_a", "kps_a", "kps_b",
                                         "norm_a", "norm_b", "share_a"])
    for name, ca, cb, ta, tb in cases:
        target = ta / (ta + tb) if ta else 0.5
        shares = []
        for rep in range(A.reps):
            shim = ta is not None
            r = run_apps([("a", [W2] + CFG[ca], env_for(f"e5-{name}-{rep}-a", ta, shim=shim)),
                          ("b", [W2] + CFG[cb], env_for(f"e5-{name}-{rep}-b", tb, shim=shim))])
            na, nb = r["a"] / SOLO[ca], r["b"] / SOLO[cb]
            s = na / (na + nb)
            shares.append(s)
            w.writerow({"case": name, "rep": rep, "target_a": round(target, 3), "kps_a": r["a"],
                        "kps_b": r["b"], "norm_a": round(na, 4), "norm_b": round(nb, 4),
                        "share_a": round(s, 4)})
            f.flush()
        m = statistics.mean(shares)
        say(f"E5 {name}: target {target:.3f} measured {m:.3f} (sd {statistics.pstdev(shares):.3f}) "
            f"error {(m - target) * 100:+.2f} pts")
    f.close()

# ---------------------------------------------------------------- PyTorch shortfall
if "torch" in ONLY:
    cases = [
        ("sync every 16 (week 1)", ["--sync-every", "16"]),
        ("sync every 256", ["--sync-every", "256"]),
        ("never idle (events)", ["--sync-every", "0"]),
        ("CUDA graph, never idle", ["--graph", "--sync-every", "0"]),
    ]
    f, w = writer("torch_shortfall.csv", ["case", "rep", "kps_a", "kps_b", "share_a"])
    for name, args in cases:
        shares = []
        for rep in range(3):
            cmd = [A.py, TORCH] + args
            r = run_apps([("a", cmd, env_for(f"torch-{name}-{rep}-a".replace(" ", "_"), 7000)),
                          ("b", cmd, env_for(f"torch-{name}-{rep}-b".replace(" ", "_"), 3000))])
            s = r["a"] / (r["a"] + r["b"])
            shares.append(s)
            w.writerow({"case": name, "rep": rep, "kps_a": r["a"], "kps_b": r["b"],
                        "share_a": round(s, 4)})
            f.flush()
        m = statistics.mean(shares)
        say(f"torch 7:3 {name}: measured {m:.3f} (sd {statistics.pstdev(shares):.3f}) "
            f"error {(m - 0.7) * 100:+.2f} pts")
    f.close()

# ---------------------------------------------------------------- E6
if "e6" in ONLY:
    f, w = writer("e6_latency.csv", ["case", "lp_kernels", "p50_us", "p90_us", "p99_us",
                                     "max_us", "pause_p50_us", "pause_p99_us", "failures"])
    REQ = 1000

    def e6_case(name, lp_cfg, lp_ts=None, hp_il=None, pause=False):
        lps = []
        socks = []
        for i in range(2 if lp_cfg else 0):
            sock = f"/tmp/e6-lp{i}.sock"
            socks.append(sock)
            env = env_for(f"e6-{name}-lp{i}".replace(" ", "_"), lp_ts, sock=sock)
            lps.append(subprocess.Popen([W2] + CFG[lp_cfg] + ["--seconds", "40", "--warmup", "0",
                                        "--label", f"lp{i}"], env=env, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL))
        time.sleep(4)
        cmd = [HP, "--requests", str(REQ), "--period-ms", "10",
               "--csv", os.path.join(A.out, "e6-" + name.replace(", ", "-").replace(" ", "_") + ".csv")]
        if pause:
            cmd += ["--pause", ",".join(socks)]
        env = env_for(f"e6-{name}-hp".replace(" ", "_"), il=hp_il) if hp_il is not None else None
        line = subprocess.run(cmd, env=env, capture_output=True, text=True).stdout.strip()
        for p in lps:
            p.kill()
            p.wait()
        r = kv(line)
        w.writerow({"case": name, "lp_kernels": lp_cfg or "-", "p50_us": r["p50_us"],
                    "p90_us": r["p90_us"], "p99_us": r["p99_us"], "max_us": r["max_us"],
                    "pause_p50_us": r["pause_p50_us"], "pause_p99_us": r["pause_p99_us"],
                    "failures": r["failures"]})
        f.flush()
        say(f"E6 {name}: p50 {float(r['p50_us']):.0f} us, p99 {float(r['p99_us']):.0f} us, "
            f"max {float(r['max_us']):.0f} us, pause step p99 {float(r['pause_p99_us']):.0f} us, "
            f"failures {r['failures']}")
        return float(r["p99_us"])

    base = e6_case("HP alone", None)
    for lp in ("short", "long"):
        e6_case(f"{lp} LP, default turns", lp)
        e6_case(f"{lp} LP, LP turns 1 ms", lp, lp_ts=1000)
        e6_case(f"{lp} LP, HP interleave HIGH", lp, hp_il=2)
        p = e6_case(f"{lp} LP, pause on request", lp, pause=True)
        say(f"E6 {lp} LP: pausing adds {p - base:.0f} us to p99 (pass mark 1000 us)")
    f.close()

# ---------------------------------------------------------------- E7
if "e7" in ONLY:
    f, w = writer("e7_cost.csv", ["case", "apps", "ts_us", "rep", "sum_kps", "efficiency"])
    for napps, ts in [(2, 1000), (4, 1000), (8, 1000), (4, None)]:
        effs = []
        for rep in range(3):
            apps = [(f"x{i}", [W2] + CFG["short"], env_for(f"e7-{napps}-{ts}-{rep}-{i}", ts))
                    for i in range(napps)]
            r = run_apps(apps)
            e = sum(r.values()) / SOLO["short"]
            effs.append(e)
            w.writerow({"case": "throughput", "apps": napps, "ts_us": ts or "default-2048",
                        "rep": rep, "sum_kps": round(sum(r.values()), 2), "efficiency": round(e, 4)})
            f.flush()
        say(f"E7 {napps} apps at {ts or 'default 2048'} us turns: total throughput "
            f"{statistics.mean(effs) * 100:.1f}% of one app alone (loss "
            f"{(1 - statistics.mean(effs)) * 100:.1f}%)")
    f.close()
    f, w = writer("e7_launch.csv", ["shim", "rep", "launch_ns", "roundtrip_p50_us",
                                    "roundtrip_p99_us"])
    res = {0: [], 1: []}
    for rep in range(5):
        for shim in (0, 1):
            env = env_for(f"e7-launch-{rep}", shim=bool(shim))
            line = subprocess.run([W2, "--launch-bench", "--label", "lb"], env=env,
                                  capture_output=True, text=True).stdout.strip()
            r = kv(line)
            res[shim].append(r)
            w.writerow({"shim": shim, "rep": rep, "launch_ns": r["launch_ns"],
                        "roundtrip_p50_us": r["roundtrip_p50_us"],
                        "roundtrip_p99_us": r["roundtrip_p99_us"]})
    f.close()
    for shim in (0, 1):
        m = lambda k: statistics.mean(float(r[k]) for r in res[shim])
        say(f"E7 launch {'with' if shim else 'without'} shim: launch {m('launch_ns'):.0f} ns, "
            f"round trip p50 {m('roundtrip_p50_us'):.1f} us, p99 {m('roundtrip_p99_us'):.1f} us")

# ---------------------------------------------------------------- E8
if "e8" in ONLY:
    def pauser(sock_path, cycles, run_ms, pause_ms, csv_path):
        me = f"/tmp/e8-pauser.{os.getpid()}.sock"
        if os.path.exists(me):
            os.unlink(me)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        s.bind(me)
        s.settimeout(2.0)
        ok = bad = 0
        lat = []
        with open(csv_path, "w") as out:
            out.write("cycle,pause_us,resume_us,pause_reply,resume_reply\n")
            for c in range(cycles):
                time.sleep(run_ms / 1000)
                try:
                    t0 = time.perf_counter()
                    s.sendto(b"pause", sock_path)
                    r1 = s.recv(128).decode()
                    t1 = time.perf_counter()
                    time.sleep(pause_ms / 1000)
                    t2 = time.perf_counter()
                    s.sendto(b"resume", sock_path)
                    r2 = s.recv(128).decode()
                    t3 = time.perf_counter()
                except OSError as e:  # app gone or no reply within 2 s
                    bad += 1
                    out.write(f"{c},-,-,error,{type(e).__name__}\n")
                    if bad > 5:
                        break
                    continue
                good = r1.startswith("ok") and r2.startswith("ok")
                ok += good
                bad += not good
                lat.append(((t1 - t0) * 1e6, (t3 - t2) * 1e6))
                out.write(f"{c},{(t1 - t0) * 1e6:.1f},{(t3 - t2) * 1e6:.1f},"
                          f"{r1.split()[0]},{r2.split()[0]}\n")
        s.close()
        os.unlink(me)
        return ok, bad, lat

    f, w = writer("e8_correctness.csv", ["case", "kernels", "iters", "ref_checksum",
                                         "test_checksum", "match", "cycles_ok", "cycles_failed",
                                         "xid_before", "xid_after", "ref_s", "test_s", "hang"])
    for name, iters, kernels in [("10 ms kernels", "600000", "2500"),
                                 ("33 ms kernels (every pause lands mid-kernel)", "2000000", "900")]:
        ref = subprocess.run([W2, "--iters", iters, "--kernels", kernels, "--label", "ref"],
                             capture_output=True, text=True).stdout.strip()
        ref = kv(ref)
        xb = dmesg_xid()
        sock = "/tmp/e8-app.sock"
        comp = subprocess.Popen([W2] + CFG["short"] + ["--seconds", "400", "--warmup", "0",
                                "--label", "competitor"], env=env_for(f"e8-{name[:2]}-competitor"),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        app = subprocess.Popen([W2, "--iters", iters, "--kernels", kernels, "--label", "test"],
                               env=env_for(f"e8-{name[:2]}-app", sock=sock),
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        time.sleep(3)
        ok, bad, lat = pauser(sock, 10000, 3, 2,
                              os.path.join(A.out, f"e8-cycles-{name[:2].strip()}ms.csv"))
        hang = False
        try:
            out = app.communicate(timeout=900)[0].strip()
        except subprocess.TimeoutExpired:
            app.kill()
            out, hang = "", True
        comp.kill()
        comp.wait()
        xa = dmesg_xid()
        t = kv(out) if out else {}
        match = t.get("checksum") == ref["checksum"]
        w.writerow({"case": name, "kernels": kernels, "iters": iters,
                    "ref_checksum": ref["checksum"], "test_checksum": t.get("checksum", "-"),
                    "match": match, "cycles_ok": ok, "cycles_failed": bad, "xid_before": xb,
                    "xid_after": xa, "ref_s": ref["elapsed_s"], "test_s": t.get("elapsed_s", "-"),
                    "hang": hang})
        f.flush()
        pp = sorted(x[0] for x in lat)
        rp = sorted(x[1] for x in lat)
        say(f"E8 {name}: {ok} pause cycles OK, {bad} failed; checksum "
            f"{'MATCH' if match else 'MISMATCH'} ({t.get('checksum')} vs {ref['checksum']}); "
            f"new Xid {xa - xb}; hang {hang}; pause call p50/p99 {pp[len(pp) // 2]:.0f}/"
            f"{pp[int(len(pp) * 0.99)]:.0f} us, resume p50/p99 {rp[len(rp) // 2]:.0f}/"
            f"{rp[int(len(rp) * 0.99)]:.0f} us")
    f.close()

print("done")
