#!/usr/bin/env python3
"""Draw the DynaGPU vs HAMi-core charts as SVG (no third-party packages).

Usage: plot_hami.py <results/hami> <out-dir>
Writes spike-hami-split.svg (70/30 by kernel type) and spike-hami-cap.svg
(40% cap, app alone).
"""
import csv
import statistics
import sys
from collections import OrderedDict
from xml.sax.saxutils import escape

res, outdir = sys.argv[1], sys.argv[2]
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
INK, MUTED, GRID, GREEN, ORANGE, RED = ("#0f172a", "#475569", "#eef2f7", "#16a34a", "#ea580c", "#dc2626")
SYS = (("DynaGPU", GREEN), ("HAMi-core", ORANGE))


def t(x, y, s, size=13, fill=INK, weight="normal", anchor="start"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>')


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">\n'
            f'<rect width="{w}" height="{h}" rx="12" fill="#ffffff"/>\n' + "\n".join(body) +
            "\n</svg>\n")


def chart(rows, title, subtitle, target, target_label, vmax, fname, note):
    """rows: OrderedDict label -> {system: [values]}"""
    W, left, right = 800, 250, 720
    X = lambda v: left + min(v, vmax) / vmax * (right - left)
    rowh = 52
    top = 100
    H = top + rowh * len(rows) + 74
    b = [t(24, 34, title, 18, INK, "bold"), t(24, 56, subtitle, 12.5, MUTED)]
    lx = 24
    for name, c in SYS:  # legend
        b.append(f'<rect x="{lx}" y="68" width="12" height="12" rx="2" fill="{c}"/>')
        b.append(t(lx + 18, 79, name, 12.5, INK))
        lx += 120
    bot = top + rowh * len(rows)
    for v in range(0, int(vmax * 100) + 1, 25):
        b.append(f'<line x1="{X(v / 100):.1f}" y1="{top - 8}" x2="{X(v / 100):.1f}" y2="{bot}" stroke="{GRID}"/>')
        b.append(t(X(v / 100), bot + 18, f"{v}%", 11.5, MUTED, anchor="middle"))
    b.append(f'<line x1="{X(target):.1f}" y1="{top - 8}" x2="{X(target):.1f}" y2="{bot}" stroke="{RED}" '
             f'stroke-dasharray="5,4" stroke-width="1.5"/>')
    b.append(t(X(target) + 4, top - 12, target_label, 11.5, RED))
    for i, (label, by) in enumerate(rows.items()):
        y = top + i * rowh
        b.append(t(24, y + 24, label, 13, INK))
        for j, (name, c) in enumerate(SYS):
            vals = by[name]
            m = statistics.mean(vals)
            yy = y + 4 + j * 20
            b.append(f'<rect x="{left}" y="{yy}" width="{X(m) - left:.1f}" height="15" rx="2" fill="{c}"/>')
            if max(vals) - min(vals) > 0.01:
                b.append(f'<line x1="{X(min(vals)):.1f}" y1="{yy + 7.5}" x2="{X(max(vals)):.1f}" '
                         f'y2="{yy + 7.5}" stroke="{INK}" stroke-width="1.5"/>')
            b.append(t(X(max(vals)) + 6, yy + 12, f"{m * 100:.1f}%", 11.5, INK, "bold"))
    b.append(t(24, bot + 44, note, 12, MUTED))
    open(f"{outdir}/{fname}", "w").write(svg(W, H, b))


rows = OrderedDict()
names = {"short vs short": "Short vs short", "graph vs graph": "Graphs vs graphs",
         "long vs long": "Long vs long", "long vs short": "Long vs short",
         "short vs long": "Short vs long", "graph vs short": "Graphs vs short"}
for r in csv.DictReader(open(f"{res}/split.csv")):
    rows.setdefault(names[r["case"]], {"DynaGPU": [], "HAMi-core": []})[r["system"]].append(float(r["share_a"]))
chart(rows, "A 70/30 split, by kind of GPU work",
      "App A's share when two busy apps are set to 70/30 (bar = mean of 3 runs, line = range). "
      "NVIDIA H100",
      0.70, "target 70%", 1.0, "spike-hami-split.svg",
      "Short = 67 µs kernels, long = 38 ms kernels, graphs = 32 kernels per launch. "
      "50% = no effect (the GPU's default split).")

rows = OrderedDict()
names = {"short": "Short kernels", "graph": "CUDA graphs", "long": "Long kernels (38 ms)"}
for r in csv.DictReader(open(f"{res}/cap.csv")):
    if abs(float(r["cap"]) - 0.4) < 1e-6:
        rows.setdefault(names[r["case"]], {"DynaGPU": [], "HAMi-core": []})[r["system"]].append(float(r["share"]))
chart(rows, "40% cap: launch throttling doesn't hold back graphs or long kernels",
      "Speed of one app alone with a 40% cap, vs no cap (bar = mean of 3 runs, line = range). NVIDIA H100",
      0.40, "cap 40%", 1.0, "spike-hami-cap.svg",
      "DynaGPU pauses the app on the GPU; HAMi-core delays its kernel launches.")
print("wrote charts")
