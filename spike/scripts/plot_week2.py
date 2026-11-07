#!/usr/bin/env python3
"""Draw week-2 charts as SVG (no third-party packages).

Usage: plot_week2.py <results/week2> <out-dir>
Writes spike-week2-e6.svg (priority latency) and spike-week2-e7.svg
(throughput vs turn length).
"""
import csv
import statistics
import sys
from collections import defaultdict
from xml.sax.saxutils import escape

res, outdir = sys.argv[1], sys.argv[2]
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
INK, MUTED, GRID, BLUE, GREEN, GREY, RED = ("#0f172a", "#475569", "#eef2f7", "#3b82f6",
                                           "#16a34a", "#94a3b8", "#dc2626")


def t(x, y, s, size=13, fill=INK, weight="normal", anchor="start"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>')


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">\n'
            f'<rect width="{w}" height="{h}" rx="12" fill="#ffffff"/>\n' + "\n".join(body) +
            "\n</svg>\n")


# ---- E6: p99 latency of the high-priority app, per case (long-kernel neighbours)
rows = list(csv.DictReader(open(f"{res}/e6_latency.csv")))
pick = [("HP alone", "Alone on the GPU", GREY),
        ("long LP, default turns", "2 busy apps, CUDA default turns", GREY),
        ("long LP, HP interleave HIGH", "2 busy apps, HP interleave HIGH", GREY),
        ("long LP, LP turns 1 ms", "2 busy apps, their turns cut to 1 ms", GREY),
        ("long LP, pause on request", "2 busy apps, DynaGPU pauses them", GREEN)]
by = {r["case"]: r for r in rows}
W, left, right = 800, 316, 690
vmax = 5000
X = lambda v: left + min(v, vmax) / vmax * (right - left)
H = 100 + 44 * len(pick) + 46
b = [t(24, 34, "Pausing the neighbours keeps a high-priority request near 1 ms", 18, INK, "bold"),
     t(24, 56, "p99 time for a small high-priority request; neighbours run 33 ms kernels. "
               "NVIDIA L40S, 1000 requests each", 12.5, MUTED)]
for v in range(0, vmax + 1, 1000):
    b.append(f'<line x1="{X(v):.1f}" y1="76" x2="{X(v):.1f}" y2="{H - 40}" stroke="{GRID}"/>')
    b.append(t(X(v), H - 24, f"{v / 1000:.0f} ms", 11.5, MUTED, anchor="middle"))
for i, (case, label, color) in enumerate(pick):
    y = 86 + i * 44
    p99 = float(by[case]["p99_us"])
    b.append(t(24, y + 18, label, 13, INK, "bold" if color == GREEN else "normal"))
    b.append(f'<rect x="{left}" y="{y + 4}" width="{X(p99) - left:.1f}" height="22" rx="3" fill="{color}"/>')
    b.append(t(X(p99) + 8, y + 20, f"{p99 / 1000:.2f} ms", 13, INK, "bold"))
open(f"{outdir}/spike-week2-e6.svg", "w").write(svg(W, H, b))

# ---- E7: throughput vs turn length (2 apps), with the 95% pass line
pts = defaultdict(list)
for r in csv.DictReader(open(f"{res}/e7_cost.csv")):
    if r["ts_us"] != "default-2048":
        pts[(int(r["apps"]), int(r["ts_us"]))].append(float(r["efficiency"]))
for r in csv.DictReader(open(f"{res}/e7_sweep.csv")):
    pts[(int(r["apps"]), int(r["ts_us"]))].append(float(r["efficiency"]))
series = defaultdict(list)
for (n, ts), v in pts.items():
    series[n].append((ts, statistics.mean(v)))
W, H = 800, 400
x0, x1, y0, y1 = 90, 720, 330, 80
xs = [1000, 2000, 3000, 5000, 10000]
import math
XL = lambda ts: x0 + (math.log(ts) - math.log(1000)) / (math.log(10000) - math.log(1000)) * (x1 - x0)
lo, hi = 0.85, 1.0
YL = lambda e: y0 - (e - lo) / (hi - lo) * (y0 - y1)
b = [t(24, 34, "Turns of 5 ms or more keep the switching loss under 5%", 18, INK, "bold"),
     t(24, 56, "Total work of all apps vs one app alone, by turn length (W2, 67 µs kernels, "
               "NVIDIA L40S, 3 runs each)", 12.5, MUTED)]
for e in (0.85, 0.90, 0.95, 1.0):
    b.append(f'<line x1="{x0}" y1="{YL(e):.1f}" x2="{x1}" y2="{YL(e):.1f}" stroke="{GRID}"/>')
    b.append(t(x0 - 10, YL(e) + 4, f"{e * 100:.0f}%", 11.5, MUTED, anchor="end"))
b.append(f'<line x1="{x0}" y1="{YL(0.95):.1f}" x2="{x1}" y2="{YL(0.95):.1f}" stroke="{RED}" '
         f'stroke-dasharray="5,4" stroke-width="1.5"/>')
b.append(t(x1, YL(0.95) - 6, "pass mark: loss ≤ 5%", 11.5, RED, anchor="end"))
for ts in xs:
    b.append(t(XL(ts), y0 + 20, f"{ts / 1000:g} ms", 11.5, MUTED, anchor="middle"))
b.append(t((x0 + x1) / 2, y0 + 42, "turn length per app", 12, MUTED, anchor="middle"))
colors = {2: BLUE, 4: GREEN, 8: "#f59e0b"}
for n in sorted(series):
    s = sorted(series[n])
    c = colors.get(n, MUTED)
    path = " ".join(f"{'M' if i == 0 else 'L'}{XL(ts):.1f},{YL(e):.1f}" for i, (ts, e) in enumerate(s))
    b.append(f'<path d="{path}" fill="none" stroke="{c}" stroke-width="2.5"/>')
    for ts, e in s:
        b.append(f'<circle cx="{XL(ts):.1f}" cy="{YL(e):.1f}" r="4.5" fill="{c}"/>')
    ts, e = s[-1]
    dy = 34 if n == 4 else 4  # keep the 4-app label off the 2-app line
    b.append(t(XL(ts) + 10, YL(e) + dy, f"{n} apps", 12.5, c, "bold"))
open(f"{outdir}/spike-week2-e7.svg", "w").write(svg(W, H, b))
print("wrote charts")
