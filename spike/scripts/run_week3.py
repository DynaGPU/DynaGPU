#!/usr/bin/env python3
"""Week-3 spike tests: E9 (hard caps), E10 (agent dies -> lease), E11 (NCCL),
and "quick", the short re-check used for each driver version in E12.

Usage: run_week3.py --spike DIR --py PYTHON --out DIR [--only calib,e9,e10,e11,quick]

Caps (E9) are enforced by the agent with pause/resume on a fixed window:
an app capped at c% runs c% of each window and is paused for the rest. With
a busy neighbour the windows are exclusive: the neighbour is paused while the
capped app runs. Measured share = the app's kernels/s / its solo kernels/s.
"""
import argparse
import csv
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

P = argparse.ArgumentParser()
P.add_argument("--spike", required=True)
P.add_argument("--py", required=True, help="python with torch")
P.add_argument("--out", required=True)
P.add_argument("--reps", type=int, default=3)
P.add_argument("--only", default="calib,e9,e10,e11")
A = P.parse_args()
ONLY = set(A.only.split(","))
W2 = f"{A.spike}/workloads/w2_work"
SHIM = f"{A.spike}/shim/libdynagpu_shim.so"
NCCL = f"{A.spike}/workloads/nccl_allreduce.py"
os.makedirs(A.out, exist_ok=True)
LOGS = os.path.join(A.out, "shim-logs")
os.makedirs(LOGS, exist_ok=True)

CFG = {"short": ["--iters", "4000"],      # ~67 us kernels
       "long": ["--iters", "2000000"]}    # ~33 ms kernels


def say(line):
    print(line, flush=True)
    with open(os.path.join(A.out, "summary.txt"), "a") as f:
        f.write(line + "\n")


def kv(line):
    return dict(x.split("=", 1) for x in line.split() if "=" in x)


def safe(tag):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def logpath(tag):
    return os.path.join(LOGS, safe(tag) + ".log")


def env_for(tag, ts=None, sock=None, lease=None, shim=True, gpu=None):
    e = dict(os.environ)
    if shim:
        e["LD_PRELOAD"] = SHIM
        e["DYNAGPU_LOG"] = logpath(tag)
    if ts:
        e["DYNAGPU_TIMESLICE_US"] = str(ts)
    if sock:
        e["DYNAGPU_SOCK"] = sock
    if lease:
        e["DYNAGPU_LEASE_MS"] = str(lease)
    if gpu is not None:
        e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return e


def writer(name, fields):
    f = open(os.path.join(A.out, name), "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    return f, w


def dmesg_xid():
    r = subprocess.run(["sudo", "dmesg"], capture_output=True, text=True).stdout
    return sum(1 for l in r.splitlines() if "xid" in l.lower())


class Agent:
    """Talks to shim sockets. send() sends to several apps at once, then waits."""

    def __init__(self, name="agent"):
        self.me = f"/tmp/dg-{name}.{os.getpid()}.{threading.get_ident()}.sock"
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
        sent = 0
        out = []
        for p in socks:
            try:
                self.s.sendto(cmd.encode(), p)
                sent += 1
            except OSError:          # app not up yet, or already gone
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


def clear_socks(*paths):
    """Remove socket files left over from an earlier run."""
    for p in paths:
        if os.path.exists(p):
            os.unlink(p)


def wait_ready(socks, timeout=30):
    """Wait until every shim socket answers with a compute channel group."""
    ag = Agent("ready")
    end = time.time() + timeout
    try:
        while time.time() < end:
            if all(os.path.exists(p) for p in socks):
                r = ag.send("hb", socks)
                if all(x.startswith("ok") for x in r):
                    return True
            time.sleep(0.2)
        return False
    finally:
        ag.close()


def run_apps(apps, seconds=20, warmup=4, during=None):
    """apps: (label, cmd, env). All start together at `start`. during(start, end)
    runs in a thread while they run. Returns ({label: kps}, {label: pid})."""
    start = time.time() + 6
    ps = []
    for label, cmd, env in apps:
        full = cmd + ["--seconds", str(seconds), "--warmup", str(warmup),
                      "--start-at", f"{start:.3f}", "--label", label]
        ps.append((label, subprocess.Popen(full, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.DEVNULL, text=True)))
    th = None
    if during:
        th = threading.Thread(target=during, args=(start, start + seconds))
        th.start()
    out, pids = {}, {}
    for label, p in ps:
        r = kv(p.communicate()[0].strip().splitlines()[-1])
        out[label] = float(r["kps"])
        pids[label] = int(r["pid"])
    if th:
        th.join()
    return out, pids


# ---------------------------------------------------------------- calibration
SOLO = {}
cal_path = os.path.join(A.out, "solo.json")
if "calib" in ONLY or not os.path.exists(cal_path):
    f, w = writer("solo.csv", ["config", "rep", "kps"])
    for name, args in CFG.items():
        vals = []
        for rep in range(3):
            k = run_apps([("solo", [W2] + args, env_for(f"solo-{name}-{rep}"))])[0]["solo"]
            vals.append(k)
            w.writerow({"config": name, "rep": rep, "kps": k})
        SOLO[name] = statistics.mean(vals)
        say(f"solo {name}: {SOLO[name]:.2f} kps (sd {statistics.pstdev(vals):.2f})")
    f.close()
    json.dump(SOLO, open(cal_path, "w"))
else:
    SOLO = json.load(open(cal_path))


# ---------------------------------------------------------------- E9 helpers
def duty_cycle(phases, stats):
    """phases: list of (socks_paused_in_this_phase, ms). Returns a during() fn that
    cycles through the phases from 1 s before the apps start until they end."""

    def run(start, end):
        all_socks = sorted({s for ps, _ in phases for s in ps})
        if not wait_ready(all_socks):
            stats["error"] = "sockets not ready"
            return
        ag = Agent("cap")
        paused = set()
        t = start - 1.0
        while time.time() < t:
            time.sleep(0.001)
        while t < end:
            for socks, ms in phases:
                want = set(socks)
                ag.send("pause", sorted(want - paused))     # pause first, then resume:
                ag.send("resume", sorted(paused - want))    # no moment with both running
                paused = want
                t += ms / 1000
                d = t - time.time()
                if d > 0:
                    time.sleep(d)
        stats["calls"] = len(ag.lat)
        stats["bad"] = ag.bad       # errors while the apps were running
        ag.send("resume", sorted(paused))
        stats["late"] = 0
        lat = sorted(ag.lat)
        stats["call_p99_us"] = lat[int(len(lat) * 0.99)] if lat else 0
        ag.close()

    return run


def pmon_start(seconds):
    return subprocess.Popen(["nvidia-smi", "pmon", "-s", "u", "-d", "1", "-c", str(int(seconds))],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)


def pmon_mean(p, pids):
    """Mean SM% per pid from nvidia-smi pmon output ('-' counts as 0)."""
    out = p.communicate()[0]
    vals = {pid: [] for pid in pids}
    for line in out.splitlines():
        if line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 4 or not f[1].isdigit():
            continue
        pid = int(f[1])
        if pid in vals:
            vals[pid].append(0.0 if f[3] == "-" else float(f[3]))
    return {pid: (statistics.mean(v) if v else float("nan")) for pid, v in vals.items()}


# ---------------------------------------------------------------- E9
if "e9" in ONLY:
    f, w = writer("e9_caps.csv", ["case", "rep", "cap", "window_ms", "kernels", "neighbour",
                                  "share_a", "share_b", "error_pts", "pmon_sm_a", "pmon_sm_b",
                                  "agent_calls", "agent_bad", "call_p99_us"])
    cases = [  # name, cap, window_ms, cfg, neighbour
        ("alone cap 20%", 0.20, 100, "short", False),
        ("alone cap 40%", 0.40, 100, "short", False),
        ("alone cap 60%", 0.60, 100, "short", False),
        ("alone cap 80%", 0.80, 100, "short", False),
        ("alone cap 40%, 50 ms window", 0.40, 50, "short", False),
        ("alone cap 40%, 200 ms window", 0.40, 200, "short", False),
        ("alone cap 40%, 33 ms kernels", 0.40, 100, "long", False),
        ("with busy neighbour, cap 30%", 0.30, 100, "short", True),
        ("with busy neighbour, cap 60%", 0.60, 100, "short", True),
    ]
    for name, cap, win, cfg, nb in cases:
        errs = []
        for rep in range(A.reps):
            sa, sb = "/tmp/e9-a.sock", "/tmp/e9-b.sock"
            clear_socks(sa, sb)
            run_ms, off_ms = cap * win, (1 - cap) * win
            phases = [([sb] if nb else [], run_ms), ([sa], off_ms)]
            st = {}
            apps = [("a", [W2] + CFG[cfg], env_for(f"e9-{name}-{rep}-a", sock=sa))]
            if nb:
                apps.append(("b", [W2] + CFG["short"], env_for(f"e9-{name}-{rep}-b", sock=sb)))
            pm = pmon_start(26)
            r, pids = run_apps(apps, during=duty_cycle(phases, st))
            pm_mean = pmon_mean(pm, list(pids.values()))
            share_a = r["a"] / SOLO[cfg]
            share_b = r["b"] / SOLO["short"] if nb else ""
            errs.append((share_a - cap) * 100)
            w.writerow({"case": name, "rep": rep, "cap": cap, "window_ms": win, "kernels": cfg,
                        "neighbour": nb, "share_a": round(share_a, 4),
                        "share_b": round(share_b, 4) if nb else "",
                        "error_pts": round((share_a - cap) * 100, 2),
                        "pmon_sm_a": round(pm_mean[pids["a"]], 1),
                        "pmon_sm_b": round(pm_mean[pids["b"]], 1) if nb else "",
                        "agent_calls": st.get("calls"), "agent_bad": st.get("bad"),
                        "call_p99_us": round(st.get("call_p99_us", 0))})
            f.flush()
        m = statistics.mean(errs)
        say(f"E9 {name}: measured {cap * 100 + m:.1f}% for a {cap * 100:.0f}% cap, error "
            f"{m:+.2f} pts (worst {max(errs, key=abs):+.2f}), pass mark 3 pts")
    f.close()
    # Accuracy of NVML per-process utilisation as a usage source, at known shares.
    f, w = writer("e9_nvml_accuracy.csv", ["case", "true_a", "true_b", "pmon_sm_a", "pmon_sm_b"])
    for name, ta, tb in [("7:3 weighted", 7000, 3000), ("1:1 weighted", 5000, 5000)]:
        pm = pmon_start(26)
        r, pids = run_apps([("a", [W2] + CFG["short"], env_for(f"e9n-{name}-a", ta)),
                            ("b", [W2] + CFG["short"], env_for(f"e9n-{name}-b", tb))])
        pmm = pmon_mean(pm, list(pids.values()))
        w.writerow({"case": name, "true_a": round(r["a"] / SOLO["short"], 4),
                    "true_b": round(r["b"] / SOLO["short"], 4),
                    "pmon_sm_a": round(pmm[pids["a"]], 1), "pmon_sm_b": round(pmm[pids["b"]], 1)})
        say(f"E9 NVML check {name}: true A {r['a'] / SOLO['short'] * 100:.1f}% B "
            f"{r['b'] / SOLO['short'] * 100:.1f}%; pmon SM A {pmm[pids['a']]:.0f}% B "
            f"{pmm[pids['b']]:.0f}%")
    f.close()


# ---------------------------------------------------------------- E10 helpers
AGENT_SRC = r"""
import socket, sys, time, os
me = f"/tmp/dg-e10agent.{os.getpid()}.sock"
s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM); s.bind(me); s.settimeout(2)
hb_s = float(sys.argv[1]); cmds = sys.argv[2:]
for c in cmds:
    path, cmd = c.split("|", 1)
    s.sendto(cmd.encode(), path); r = s.recv(128).decode()
    print(f"cmd t={time.time():.4f} sock={path} cmd={cmd.replace(' ', '_')} reply={r.split()[0]}", flush=True)
end = time.time() + hb_s
socks = sorted({c.split("|", 1)[0] for c in cmds})
while time.time() < end:
    time.sleep(0.25)
    for p in socks:
        s.sendto(b"hb", p); s.recv(128)
    print(f"hb t={time.time():.4f}", flush=True)
print("idle", flush=True)
time.sleep(3600)
"""


def start_agent(hb_seconds, cmds):
    """A separate agent process we can SIGKILL. Returns (proc, lines)."""
    p = subprocess.Popen([sys.executable, "-c", AGENT_SRC, str(hb_seconds)] + cmds,
                         stdout=subprocess.PIPE, text=True)
    lines = []

    def rd():
        for l in p.stdout:
            lines.append(l.strip())

    threading.Thread(target=rd, daemon=True).start()
    return p, lines


def progress(out):
    """[(t, done)] from W2 --progress-ms lines."""
    res = []
    for l in out.splitlines():
        if l.startswith("prog "):
            r = kv(l)
            res.append((float(r["t"]), int(r["done"])))
    return res


def rate(prog, t0, t1):
    pts = [(t, d) for t, d in prog if t0 <= t <= t1]
    if len(pts) < 2:
        return 0.0
    return (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])


def lease_events(tag):
    ev = []
    try:
        for l in open(logpath(tag)):
            if "LEASE_EXPIRED" in l:
                m = re.search(r"\[dynagpu \d+ ([\d.]+)\]", l)
                ev.append((float(m.group(1)), l.strip()))
    except FileNotFoundError:
        pass
    return ev


# ---------------------------------------------------------------- E10
if "e10" in ONLY:
    LEASE = 1000
    # (a) agent pauses an app, then dies
    f, w = writer("e10_lease.csv", ["case", "rep", "last_cmd_t", "lease_expired_t",
                                    "resumed_t", "resume_after_last_cmd_s", "pause_reply",
                                    "new_agent_reply", "share_before", "share_after"])
    res = []
    for rep in range(5):
        tag = f"e10a-{rep}-a"
        sa = "/tmp/e10-a.sock"
        clear_socks(sa)
        a = subprocess.Popen([W2] + CFG["short"] + ["--seconds", "14", "--warmup", "0",
                             "--progress-ms", "2", "--label", "a"],
                             env=env_for(tag, sock=sa, lease=LEASE), stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True)
        b = subprocess.Popen([W2] + CFG["short"] + ["--seconds", "16", "--warmup", "0"],
                             env=env_for(f"e10a-{rep}-b", shim=False), stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        wait_ready([sa])
        time.sleep(3)
        ag, lines = start_agent(0, [f"{sa}|pause"])
        time.sleep(0.4)
        ag.send_signal(signal.SIGKILL)   # agent crashes while the app is paused
        ag.wait()
        time.sleep(3)
        # A new agent can take over afterwards.
        na = Agent("e10new")
        r2 = na.send("pause", [sa])[0]
        na.send("resume", [sa])
        na.close()
        out = a.communicate()[0]
        b.wait()
        cmd = [l for l in lines if l.startswith("cmd ")]
        t_cmd = float(kv(cmd[0])["t"]) if cmd else float("nan")
        prog = progress(out)
        after = [t for t, _ in prog if t > t_cmd + 0.05]
        t_res = after[0] if after else float("nan")
        ev = lease_events(tag)
        t_exp = ev[0][0] if ev else float("nan")
        d = t_res - t_cmd
        res.append(d)
        w.writerow({"case": "agent dies while app paused", "rep": rep,
                    "last_cmd_t": f"{t_cmd:.4f}", "lease_expired_t": f"{t_exp:.4f}",
                    "resumed_t": f"{t_res:.4f}", "resume_after_last_cmd_s": round(d, 3),
                    "pause_reply": kv(cmd[0]).get("reply") if cmd else "none",
                    "new_agent_reply": r2.split()[0], "share_before": "", "share_after": ""})
        f.flush()
        say(f"E10a rep {rep}: app resumed {d:.3f} s after the agent's last command "
            f"(lease {LEASE} ms; expiry logged {t_exp - t_cmd:.3f} s); new agent pause: {r2.split()[0]}")
    say(f"E10a: resume after {min(res):.3f}-{max(res):.3f} s (pass: 1.0-1.3 s)")

    # (b) agent sets 9:1 turns, then dies -> both apps back to CUDA's default turns
    for rep in range(3):
        sa, sb = "/tmp/e10-a.sock", "/tmp/e10-b.sock"
        clear_socks(sa, sb)
        procs = {}
        for x, s in (("a", sa), ("b", sb)):
            procs[x] = subprocess.Popen([W2] + CFG["short"] + ["--seconds", "18", "--warmup", "0",
                                        "--progress-ms", "20", "--label", x],
                                        env=env_for(f"e10b-{rep}-{x}", sock=s, lease=LEASE),
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        wait_ready([sa, sb])
        time.sleep(1)
        ag, lines = start_agent(6, [f"{sa}|ts 9000", f"{sb}|ts 1000"])
        time.sleep(6.5)                   # heartbeats for 6 s, then goes silent
        t_kill = time.time()
        ag.send_signal(signal.SIGKILL)
        ag.wait()
        outs = {x: p.communicate()[0] for x, p in procs.items()}
        hbs = [float(kv(l)["t"]) for l in lines if l.startswith("hb ")]
        t_last = hbs[-1] if hbs else t_kill
        ev = lease_events(f"e10b-{rep}-a")
        t_exp = ev[0][0] if ev else t_last + 1
        pa, pb = progress(outs["a"]), progress(outs["b"])
        ra0, rb0 = rate(pa, t_last - 4, t_last), rate(pb, t_last - 4, t_last)
        ra1, rb1 = rate(pa, t_exp + 0.5, t_exp + 6), rate(pb, t_exp + 0.5, t_exp + 6)
        s0, s1 = ra0 / (ra0 + rb0), ra1 / (ra1 + rb1)
        w.writerow({"case": "agent sets 9:1 then dies", "rep": rep, "last_cmd_t": f"{t_last:.4f}",
                    "lease_expired_t": f"{t_exp:.4f}", "resumed_t": "",
                    "resume_after_last_cmd_s": round(t_exp - t_last, 3), "pause_reply": "",
                    "new_agent_reply": "", "share_before": round(s0, 4), "share_after": round(s1, 4)})
        f.flush()
        say(f"E10b rep {rep}: A's share {s0 * 100:.1f}% while the agent lived (target 90%), "
            f"{s1 * 100:.1f}% after its lease ran out (CUDA default: 50%); reset "
            f"{t_exp - t_last:.3f} s after the last heartbeat")

    # (c) healthy agent: long pauses kept alive by heartbeats, no false expiry
    tag = "e10c-a"
    sa = "/tmp/e10-a.sock"
    clear_socks(sa)
    a = subprocess.Popen([W2] + CFG["short"] + ["--seconds", "75", "--warmup", "0",
                         "--progress-ms", "5", "--label", "a"],
                         env=env_for(tag, sock=sa, lease=LEASE), stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True)
    wait_ready([sa])
    ag = Agent("e10c")
    pauses = []
    t_end = time.time() + 60
    while time.time() < t_end:
        tp = time.time()
        ag.send("pause", [sa])
        while time.time() < tp + 3:      # 3 s pause, 3x the lease, heartbeat every 250 ms
            time.sleep(0.25)
            ag.send("hb", [sa])
        tr = time.time()
        ag.send("resume", [sa])
        pauses.append((tp, tr))
        time.sleep(1)
    ag.close()
    out = a.communicate()[0]
    prog = progress(out)
    held = 0
    for tp, tr in pauses:
        inside = [t for t, _ in prog if tp + 0.05 < t < tr - 0.01]
        held += not inside
    ev = lease_events(tag)
    f.close()
    say(f"E10c healthy agent: {len(pauses)} pauses of 3 s (3x the lease) with heartbeats; "
        f"{held}/{len(pauses)} held with no progress; false lease expiries: {len(ev)}; "
        f"agent errors {ag.bad}")


# ---------------------------------------------------------------- E11
def gpu_sampler(stop, rows):
    p = subprocess.Popen(["nvidia-smi", "--query-gpu=index,utilization.gpu,power.draw",
                          "--format=csv,noheader,nounits", "-lms", "200"],
                         stdout=subprocess.PIPE, text=True)
    for l in p.stdout:
        if stop.is_set():
            break
        f = [x.strip() for x in l.split(",")]
        if len(f) == 3 and f[0].isdigit():
            rows.append((time.time(), int(f[0]), float(f[1]), float(f[2])))
    p.kill()


def nccl_run(tag, seconds, script, ts=None, competitors=False, comp_socks=None):
    """Start 2 ranks with the shim; script(socks_by_rank, t_start) drives pauses.
    Returns dict with per-rank summaries, stderr text, gpu samples."""
    env = env_for(tag, ts=ts, sock="/tmp/e11-%p.sock")
    env["NCCL_DEBUG"] = "WARN"
    comps = []
    if competitors:
        for g in (0, 1):
            comps.append(subprocess.Popen([W2] + CFG["short"] + ["--seconds", str(seconds + 60),
                                          "--warmup", "0"],
                                          env=env_for(f"{tag}-comp{g}", ts=None if comp_socks else 3000,
                                                      gpu=g, sock=comp_socks[g] if comp_socks else None),
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    p = subprocess.Popen([A.py, "-m", "torch.distributed.run", "--nproc-per-node", "2",
                          NCCL, "--seconds", str(seconds)], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines, errl = [], []
    threading.Thread(target=lambda: [lines.append(l.strip()) for l in p.stdout], daemon=True).start()
    threading.Thread(target=lambda: [errl.append(l.rstrip()) for l in p.stderr], daemon=True).start()
    stop, rows = threading.Event(), []
    th = threading.Thread(target=gpu_sampler, args=(stop, rows), daemon=True)
    th.start()
    end = time.time() + 120
    while time.time() < end and sum(l.startswith("start ") for l in lines) < 2:
        time.sleep(0.1)
    socks = {}
    for l in lines:
        if l.startswith("start "):
            r = kv(l)
            socks[int(r["rank"])] = f"/tmp/e11-{r['pid']}.sock"
    marks = []
    if len(socks) == 2 and wait_ready(list(socks.values())):
        if script:
            marks = script(socks, time.time())
    hang = False
    try:
        p.wait(timeout=seconds + 180)
    except subprocess.TimeoutExpired:
        hang = True
        p.kill()
    for c in comps:
        c.kill()
        c.wait()
    stop.set()
    time.sleep(0.5)
    done = {int(kv(l)["rank"]): kv(l) for l in lines if l.startswith("done ")}
    nccl_err = [l for l in errl if re.search(r"error|timeout|watchdog|abort", l, re.I)]
    return {"done": done, "hang": hang, "rc": p.returncode, "nccl_err": nccl_err,
            "rows": rows, "marks": marks, "socks": socks, "lines": lines}


def window_util(rows, t0, t1):
    res = {}
    for g in (0, 1):
        u = [x[2] for x in rows if x[1] == g and t0 <= x[0] <= t1]
        pw = [x[3] for x in rows if x[1] == g and t0 <= x[0] <= t1]
        res[g] = (statistics.mean(u) if u else float("nan"), statistics.mean(pw) if pw else float("nan"))
    return res


if "e11" in ONLY:
    f, w = writer("e11_nccl.csv", ["case", "steps_r0", "steps_r1", "errors", "step_p50_ms",
                                   "step_p99_ms", "step_max_ms", "gib_per_s", "hang", "rc",
                                   "nccl_error_lines", "agent_bad", "xid_new"])

    def record(name, r, agent_bad, xb):
        d0, d1 = r["done"].get(0, {}), r["done"].get(1, {})
        errors = sum(int(d.get("errors", 0)) for d in (d0, d1)) if d0 and d1 else "missing"
        w.writerow({"case": name, "steps_r0": d0.get("steps", "-"), "steps_r1": d1.get("steps", "-"),
                    "errors": errors, "step_p50_ms": d0.get("step_p50_ms", "-"),
                    "step_p99_ms": d0.get("step_p99_ms", "-"), "step_max_ms": d0.get("step_max_ms", "-"),
                    "gib_per_s": d0.get("gib_per_s", "-"), "hang": r["hang"], "rc": r["rc"],
                    "nccl_error_lines": len(r["nccl_err"]), "agent_bad": agent_bad,
                    "xid_new": dmesg_xid() - xb})
        f.flush()
        say(f"E11 {name}: steps {d0.get('steps')}/{d1.get('steps')}, check errors {errors}, "
            f"step p50 {d0.get('step_p50_ms')} ms p99 {d0.get('step_p99_ms')} ms max "
            f"{d0.get('step_max_ms')} ms, {d0.get('gib_per_s')} GiB/s, hang {r['hang']}, rc "
            f"{r['rc']}, NCCL error lines {len(r['nccl_err'])}, agent errors {agent_bad}")
        for l in r["nccl_err"][:5]:
            say(f"    stderr: {l[:200]}")

    # (a) baseline
    xb = dmesg_xid()
    r = nccl_run("e11-baseline", 30, None)
    record("baseline (no pauses)", r, 0, xb)
    base_steps = int(r["done"].get(0, {}).get("steps", 0) or 0)

    # (b) who spins? pause one rank, then the other, then both
    spin = []

    def spin_script(socks, t0):
        ag = Agent("e11spin")
        marks = []
        for label, ranks in (("pause rank 0 only", [0]), ("pause rank 1 only", [1]),
                             ("pause both ranks", [0, 1]), ("running (no pause)", [])):
            time.sleep(3)
            s = [socks[k] for k in ranks]
            ag.send("pause", s)
            tp = time.time()
            time.sleep(5)
            ag.send("resume", s)
            marks.append((label, tp, time.time()))
        spin.append(ag.bad)
        ag.close()
        return marks

    xb = dmesg_xid()
    r = nccl_run("e11-spin", 40, spin_script)
    record("pause one rank / both ranks, 5 s each", r, spin[0] if spin else "-", xb)
    fs, ws = writer("e11_spin.csv", ["window", "gpu0_util", "gpu0_power_w", "gpu1_util",
                                     "gpu1_power_w"])
    for label, tp, tr in r["marks"]:
        u = window_util(r["rows"], tp + 1, tr - 0.5)
        ws.writerow({"window": label, "gpu0_util": round(u[0][0], 1), "gpu0_power_w": round(u[0][1]),
                     "gpu1_util": round(u[1][0], 1), "gpu1_power_w": round(u[1][1])})
        say(f"E11 {label}: GPU0 util {u[0][0]:.0f}% ({u[0][1]:.0f} W), GPU1 util "
            f"{u[1][0]:.0f}% ({u[1][1]:.0f} W)")
    fs.close()
    # Save the raw samples.
    with open(os.path.join(A.out, "e11_gpu_samples.csv"), "w") as g:
        g.write("t,gpu,util,power_w\n")
        for row in r["rows"]:
            g.write(f"{row[0]:.3f},{row[1]},{row[2]},{row[3]}\n")
        for label, tp, tr in r["marks"]:
            g.write(f"# {label} {tp:.3f} {tr:.3f}\n")

    # (c) stress: many short pauses, both ranks together, then one rank alone
    for name, ranks, cycles in (("300 gang pauses (both ranks, 50 ms)", [0, 1], 300),
                                ("300 one-rank pauses (rank 0, 50 ms)", [0], 300)):
        bad = []

        def stress(socks, t0, ranks=ranks, cycles=cycles):
            ag = Agent("e11stress")
            s = [socks[k] for k in ranks]
            for _ in range(cycles):
                time.sleep(0.1)
                ag.send("pause", s)
                time.sleep(0.05)
                ag.send("resume", s)
            bad.append(ag.bad)
            ag.close()
            return []

        xb = dmesg_xid()
        r = nccl_run("e11-stress-" + safe(name), 60, stress)
        record(name, r, bad[0] if bad else "-", xb)

    # (d) sharing: each GPU also runs a busy app; ranks get 7 ms turns, the apps 3 ms
    xb = dmesg_xid()
    r = nccl_run("e11-shared", 30, None, ts=7000, competitors=True)
    record("sharing each GPU 7:3 with a busy app", r, 0, xb)
    st = int(r["done"].get(0, {}).get("steps", 0) or 0)
    if base_steps:
        say(f"E11 sharing: all-reduce rate {st / base_steps * 100:.1f}% of the GPUs-alone rate")
    f.close()


# ---------------------------------------------------------------- E11 gang scheduling
if "e11gang" in ONLY:
    # Each GPU runs one rank and one busy app. The agent gives the ranks 70 ms
    # and the apps 30 ms of every 100 ms by pausing. "aligned": both GPUs
    # switch together. "offset": GPU1's schedule runs 50 ms behind GPU0's.
    base = None
    f, w = writer("e11_gang.csv", ["case", "steps", "errors", "step_p50_ms", "step_p99_ms",
                                   "rate_vs_alone", "agent_bad"])
    for name, offset in (("GPUs alone (reference)", None), ("aligned 70/30 slots", 0.0),
                         ("offset 70/30 slots (GPU1 50 ms behind)", 0.05)):
        comp = ["/tmp/e11-comp0.sock", "/tmp/e11-comp1.sock"]
        clear_socks(*comp)
        bad = []

        def gang(socks, t0, offset=offset):
            if not wait_ready(comp):
                bad.append("comp not ready")
                return []
            ag = Agent("gang")
            # events: (time within the 100 ms window, gpu, phase)
            ev = sorted([(0.0, 0, "rank"), (0.07, 0, "comp"),
                         ((0.0 + offset) % 0.1, 1, "rank"), ((0.07 + offset) % 0.1, 1, "comp")])
            t, end = time.time(), time.time() + 29
            while t < end:
                for dt, g, ph in ev:
                    d = t + dt - time.time()
                    if d > 0:
                        time.sleep(d)
                    on, off = (socks[g], comp[g]) if ph == "rank" else (comp[g], socks[g])
                    ag.send("pause", [off])
                    ag.send("resume", [on])
                t += 0.1
            ag.send("resume", [socks[0], socks[1]] + comp)
            bad.append(ag.bad)
            ag.close()
            return []

        if offset is None:
            r = nccl_run("e11g-ref", 30, None)
        else:
            r = nccl_run("e11g-" + safe(name), 30, gang, competitors=True, comp_socks=comp)
        d0 = r["done"].get(0, {})
        steps = int(d0.get("steps", 0) or 0)
        base = base or steps
        w.writerow({"case": name, "steps": steps, "errors": d0.get("errors"),
                    "step_p50_ms": d0.get("step_p50_ms"), "step_p99_ms": d0.get("step_p99_ms"),
                    "rate_vs_alone": round(steps / base, 4), "agent_bad": bad[0] if bad else 0})
        f.flush()
        say(f"E11 gang {name}: {steps} steps ({steps / base * 100:.1f}% of GPUs alone), errors "
            f"{d0.get('errors')}, step p50 {d0.get('step_p50_ms')} ms p99 {d0.get('step_p99_ms')} ms, "
            f"agent errors {bad[0] if bad else 0}, hang {r['hang']}")
    f.close()


# ---------------------------------------------------------------- quick (E12)
if "quick" in ONLY:
    # E1: groups found, timeslice set and read back
    tag = "quick-e1"
    subprocess.run([W2] + CFG["short"] + ["--seconds", "3", "--warmup", "0"],
                   env=env_for(tag, ts=7000), capture_output=True)
    log = open(logpath(tag)).read()
    tsg = len(re.findall(r"TSG_ALLOC", log))
    eng = sorted(set(re.findall(r"TSG_ALLOC .*engineType=(\d+)", log)))
    set_ok = "SET_TIMESLICE" in log and " OK" in log
    get = re.findall(r"GET_TIMESLICE .*value_us=(\d+)", log)
    app = re.findall(r"APP_SET_TIMESLICE .*app_value_us=(\d+)", log)
    chans = len(re.findall(r"CHANNEL_ALLOC", log))
    say(f"quick E1: {tsg} channel groups (engines {','.join(eng)}), {chans} channels, SET ok "
        f"{set_ok}, GET {get[-1] if get else '-'} us, CUDA's own value {app[0] if app else '-'} us")
    # E3: 7:3
    r, _ = run_apps([("a", [W2] + CFG["short"], env_for("quick-e3-a", 7000)),
                     ("b", [W2] + CFG["short"], env_for("quick-e3-b", 3000))])
    say(f"quick E3 7:3: A share {r['a'] / (r['a'] + r['b']) * 100:.1f}% (target 70%)")
    # Pause effect + E8-lite: 1000 pause cycles, checksum
    ref = kv(subprocess.run([W2, "--iters", "600000", "--kernels", "1200"], capture_output=True,
                            text=True).stdout.strip())
    sa = "/tmp/quick-a.sock"
    clear_socks(sa)
    a = subprocess.Popen([W2, "--iters", "600000", "--kernels", "1200"],
                         env=env_for("quick-e8", sock=sa, lease=1000), stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True)
    wait_ready([sa])
    ag = Agent("quick")
    first = ag.send("pause", [sa])[0]
    time.sleep(2)
    ag.send("resume", [sa])
    for _ in range(1000):
        time.sleep(0.003)
        ag.send("pause", [sa])
        time.sleep(0.002)
        ag.send("resume", [sa])
    ag.close()
    t = kv(a.communicate(timeout=600)[0].strip())
    lat = sorted(ag.lat)
    say(f"quick E8: first pause reply '{first}'; 1000 cycles, agent errors {ag.bad}; checksum "
        f"{'MATCH' if t.get('checksum') == ref.get('checksum') else 'MISMATCH'}; call p50 "
        f"{lat[len(lat) // 2]:.0f} us p99 {lat[int(len(lat) * 0.99)]:.0f} us; app "
        f"{float(t.get('elapsed_s', 0)):.1f} s vs {float(ref.get('elapsed_s', 0)):.1f} s alone")

print("done")
