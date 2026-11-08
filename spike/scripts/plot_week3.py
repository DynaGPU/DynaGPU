#!/usr/bin/env python3
"""Draw week-3 charts as SVG (no third-party packages).

Usage: plot_week3.py <results/week3> <out-dir>
Writes spike-week3-e9.svg (cap error per case) and spike-week3-e11.svg
(all-reduce rate when the GPUs are shared).
"""
import csv
import statistics
import sys
from collections import OrderedDict
from xml.sax.saxutils import escape

res, outdir = sys.argv[1], sys.argv[2]
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
INK, MUTED, GRID, BLUE, GREEN, GREY, RED, BAND = ("#0f172a", "#475569", "#eef2f7", "#3b82f6",
                                                 "#16a34a", "#94a3b8", "#dc2626", "#ecfdf5")


def t(x, y, s, size=13, fill=INK, weight="normal", anchor="start"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>')


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">\n'
            f'<rect width="{w}" height="{h}" rx="12" fill="#ffffff"/>\n' + "\n".join(body) +
            "\n</svg>\n")


# ---- E9: error vs the cap (points), mean and range of 3 runs, with the ±3 pass band
rows = list(csv.DictReader(open(f"{res}/e9_caps.csv")))
cases = OrderedDict()
for r in rows:
    cases.setdefault(r["case"], []).append(float(r["error_pts"]))
labels = {
    "alone cap 20%": "Alone, cap 20%",
    "alone cap 40%": "Alone, cap 40%",
    "alone cap 60%": "Alone, cap 60%",
    "alone cap 80%": "Alone, cap 80%",
    "alone cap 40%, 50 ms window": "Alone, cap 40%, 50 ms window",
    "alone cap 40%, 200 ms window": "Alone, cap 40%, 200 ms window",
    "alone cap 40%, 33 ms kernels": "Alone, cap 40%, 33 ms kernels",
    "with busy neighbour, cap 30%": "Busy neighbour, cap 30%",
    "with busy neighbour, cap 60%": "Busy neighbour, cap 60%",
}
W, left, right = 800, 290, 760
lo, hi = -3, 5
X = lambda v: left + (v - lo) / (hi - lo) * (right - left)
H = 104 + 34 * len(cases) + 70
top, bot = 84, 84 + 34 * len(cases)
b = [t(24, 34, "Caps hold GPU time; alone, the app does a bit more work per ms", 18, INK, "bold"),
     t(24, 56, "Measured share minus cap, in points (dot = mean of 3 runs, line = range). "
               "NVIDIA L40S, 100 ms windows unless noted", 12.5, MUTED),
     f'<rect x="{X(-3):.1f}" y="{top - 6}" width="{X(3) - X(-3):.1f}" height="{bot - top + 6}" fill="{BAND}"/>']
for v in range(lo, hi + 1):
    b.append(f'<line x1="{X(v):.1f}" y1="{top - 6}" x2="{X(v):.1f}" y2="{bot}" '
             f'stroke="{"#cbd5e1" if v == 0 else GRID}"/>')
    b.append(t(X(v), bot + 18, f"{v:+d}" if v else "0", 11.5, MUTED, anchor="middle"))
b.append(f'<line x1="{X(3):.1f}" y1="{top - 6}" x2="{X(3):.1f}" y2="{bot}" stroke="{RED}" '
         f'stroke-dasharray="5,4" stroke-width="1.5"/>')
b.append(t(X(3) + 4, top + 6, "pass mark ±3", 11.5, RED))
for i, (case, errs) in enumerate(cases.items()):
    y = top + 6 + i * 34
    nb = "neighbour" in case
    c = GREEN if nb else BLUE
    m = statistics.mean(errs)
    b.append(t(24, y + 5, labels.get(case, case), 13, INK))
    b.append(f'<line x1="{X(min(errs)):.1f}" y1="{y}" x2="{X(max(errs)):.1f}" y2="{y}" '
             f'stroke="{c}" stroke-width="3" stroke-linecap="round"/>')
    b.append(f'<circle cx="{X(m):.1f}" cy="{y}" r="6" fill="{c}"/>')
    b.append(t(X(max(errs)) + 10, y + 5, f"{m:+.1f}", 12.5, INK, "bold"))
b.append(t(24, bot + 44, "Blue: app alone. The GPU runs 3.5–6.6% faster clocks when it is paused part of "
                         "the time (measured),", 12, MUTED))
b.append(t(24, bot + 60, "so the app does more work in its share of time. Green: a busy neighbour keeps the "
                         "clock steady, and the cap holds.", 12, MUTED))
open(f"{outdir}/spike-week3-e9.svg", "w").write(svg(W, H, b))

# ---- E11: all-reduce rate when both GPUs are shared 70/30
gang = {r["case"]: float(r["rate_vs_alone"]) for r in csv.DictReader(open(f"{res}/e11_gang.csv"))}
nccl = {r["case"]: r for r in csv.DictReader(open(f"{res}/e11_nccl.csv"))}
base = int(nccl["baseline (no pauses)"]["steps_r0"])
shared = int(nccl["sharing each GPU 7:3 with a busy app"]["steps_r0"]) / base
pick = [("Both GPUs to itself", gang["GPUs alone (reference)"], GREY),
        ("Shared 70/30, each GPU's own turns", shared, GREY),
        ("Shared 70/30, agent slots 50 ms apart", gang["offset 70/30 slots (GPU1 50 ms behind)"], GREY),
        ("Shared 70/30, agent slots aligned", gang["aligned 70/30 slots"], GREEN)]
W, left, right = 800, 316, 700
X = lambda v: left + v * (right - left)
H = 100 + 44 * len(pick) + 46
b = [t(24, 34, "A 2-GPU job only runs at its share if both GPUs switch together", 18, INK, "bold"),
     t(24, 56, "NCCL all-reduce rate vs the GPUs alone; each GPU also runs a busy app. "
               "Target 70%. 2 × NVIDIA L40S", 12.5, MUTED)]
for v in (0, 0.25, 0.5, 0.75, 1.0):
    b.append(f'<line x1="{X(v):.1f}" y1="76" x2="{X(v):.1f}" y2="{H - 40}" stroke="{GRID}"/>')
    b.append(t(X(v), H - 24, f"{v * 100:.0f}%", 11.5, MUTED, anchor="middle"))
b.append(f'<line x1="{X(0.7):.1f}" y1="76" x2="{X(0.7):.1f}" y2="{H - 40}" stroke="{RED}" '
         f'stroke-dasharray="5,4" stroke-width="1.5"/>')
b.append(t(X(0.7) + 4, 72, "target 70%", 11.5, RED))
for i, (label, v, color) in enumerate(pick):
    y = 86 + i * 44
    b.append(t(24, y + 18, label, 13, INK, "bold" if color == GREEN else "normal"))
    b.append(f'<rect x="{left}" y="{y + 4}" width="{X(v) - left:.1f}" height="22" rx="3" fill="{color}"/>')
    b.append(t(X(v) + 8, y + 20, f"{v * 100:.1f}%", 13, INK, "bold"))
open(f"{outdir}/spike-week3-e11.svg", "w").write(svg(W, H, b))
print("wrote charts")
