#!/usr/bin/env python3
"""E9 follow-up: does a duty-cycled app run at a higher clock than one that
never stops? Samples SM clock and power while W2 runs (a) flat out and (b)
under a 40% cap, and reports kernels per second of *running* time.

Usage: clock_check.py --spike DIR --out DIR
"""
import argparse, os, socket, statistics, subprocess, threading, time
P = argparse.ArgumentParser(); P.add_argument("--spike", required=True); P.add_argument("--out", required=True)
A = P.parse_args()
W2 = f"{A.spike}/workloads/w2_work"; SHIM = f"{A.spike}/shim/libdynagpu_shim.so"

def sample(stop, rows):
    p = subprocess.Popen(["nvidia-smi", "-i", "0", "--query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active",
                          "--format=csv,noheader,nounits", "-lms", "100"], stdout=subprocess.PIPE, text=True)
    for l in p.stdout:
        if stop.is_set(): break
        f = [x.strip() for x in l.split(",")]
        try: rows.append((float(f[0]), float(f[1]), float(f[2]), f[3]))
        except ValueError: pass
    p.kill()

def run(cap):
    sock = "/tmp/clk.sock"
    if os.path.exists(sock): os.unlink(sock)
    env = dict(os.environ, LD_PRELOAD=SHIM, DYNAGPU_SOCK=sock)
    app = subprocess.Popen([W2, "--iters", "4000", "--seconds", "24", "--warmup", "4", "--label", "a"], env=env,
                           stdout=subprocess.PIPE, text=True)
    stop, rows = threading.Event(), []
    th = threading.Thread(target=sample, args=(stop, rows)); th.start()
    time.sleep(1.5)
    run_s = 0.0
    if cap < 1:
        me = "/tmp/clk-agent.sock"
        if os.path.exists(me): os.unlink(me)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM); s.bind(me); s.settimeout(2)
        t = time.time(); end = t + 21   # the app runs ~24 s from now; stop well before
        try:
            while t < end:
                t += cap * 0.1; time.sleep(max(0, t - time.time()))
                s.sendto(b"pause", sock); s.recv(128)
                t += (1 - cap) * 0.1; time.sleep(max(0, t - time.time()))
                s.sendto(b"resume", sock); s.recv(128)
        except OSError as e:
            print(f"agent stopped early: {e!r}", flush=True)
            try:
                s.sendto(b"resume", sock)
            except OSError:
                pass
        s.close(); os.unlink(me)
    out = app.communicate()[0].strip().splitlines()[-1]
    stop.set(); th.join()
    kps = float(dict(x.split("=", 1) for x in out.split() if "=" in x)["kps"])
    rows = rows[40:-10]  # steady part
    busy = [r for r in rows if r[1] > 120] or rows  # samples while the GPU is working
    return kps, statistics.mean(r[0] for r in busy), statistics.mean(r[1] for r in rows), max(r[2] for r in rows), sorted({r[3] for r in rows})

os.makedirs(A.out, exist_ok=True)
with open(os.path.join(A.out, "e9_clocks.csv"), "w") as f:
    f.write("case,kps,share_of_solo,sm_clock_mhz_busy,power_w_mean,temp_max_c,throttle_reasons\n")
    solo = None
    for name, cap in (("flat out", 1.0), ("cap 40%", 0.4), ("flat out (repeat)", 1.0), ("cap 80%", 0.8)):
        kps, clk, pw, temp, why = run(cap)
        solo = solo or kps
        f.write(f"{name},{kps:.1f},{kps / solo:.4f},{clk:.0f},{pw:.0f},{temp:.0f},{' '.join(why)}\n"); f.flush()
        print(f"clock check {name}: {kps:.0f} kps ({kps / solo * 100:.1f}% of flat out), SM clock while busy {clk:.0f} MHz, "
              f"mean power {pw:.0f} W, max temp {temp:.0f} C, throttle reasons {' '.join(why)}", flush=True)
