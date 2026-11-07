#!/usr/bin/env python3
"""Draw the E3 result (target vs measured share of app A) as an SVG.

Usage: plot_e3.py <results-dir> <out.svg>
Reads default_pair.csv and e3_weighted.csv; no third-party packages needed.
"""
import csv
import statistics
import sys
from xml.sax.saxutils import escape

res, out = sys.argv[1], sys.argv[2]
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
BLUE, AMBER, INK, MUTED, LINE = "#3b82f6", "#f59e0b", "#0f172a", "#475569", "#cbd5e1"


def rows(name):
    with open(f"{res}/{name}") as f:
        return list(csv.DictReader(f))


groups = []
d = [float(r["share_a"]) for r in rows("default_pair.csv")]
groups.append(("No DynaGPU", 0.5, d, "CUDA default 2048 µs"))
by = {}
for r in rows("e3_weighted.csv"):
    by.setdefault(r["split"], []).append(r)
for split, rs in by.items():
    groups.append((f"Target {split}", float(rs[0]["target_a"]),
                   [float(r["share_a"]) for r in rs], f"{rs[0]['ts_a_us']} / {rs[0]['ts_b_us']} µs"))


def t(x, y, s, size=13, fill=INK, weight="normal", anchor="start"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>')


W, left, right = 800, 190, 700
rowh = 46
H = 110 + rowh * len(groups) + 50
X = lambda v: left + v * (right - left)
o = [f'<rect width="{W}" height="{H}" rx="12" fill="#ffffff"/>',
     t(24, 34, "Timeslice lengths give the GPU split we ask for", 18, INK, "bold"),
     t(24, 56, "Share of GPU work done by app A, two busy apps on one NVIDIA L40S "
               "(driver 610.57.04), 5 runs each", 12.5, MUTED)]
for v in (0, 0.25, 0.5, 0.75, 1.0):
    o.append(f'<line x1="{X(v):.1f}" y1="80" x2="{X(v):.1f}" y2="{H - 46}" stroke="#eef2f7"/>')
    o.append(t(X(v), H - 28, f"{int(v * 100)}%", 11.5, MUTED, anchor="middle"))
for i, (name, target, vals, ts) in enumerate(groups):
    y = 92 + i * rowh
    m = statistics.mean(vals)
    o.append(t(24, y + 14, name, 13, INK, "bold"))
    o.append(t(24, y + 30, ts, 11.5, MUTED))
    o.append(f'<rect x="{left}" y="{y + 4}" width="{X(m) - left:.1f}" height="22" fill="{BLUE}"/>')
    o.append(f'<rect x="{X(m):.1f}" y="{y + 4}" width="{right - X(m):.1f}" height="22" fill="{AMBER}"/>')
    o.append(f'<line x1="{X(target):.1f}" y1="{y}" x2="{X(target):.1f}" y2="{y + 30}" '
             f'stroke="#dc2626" stroke-width="3"/>')
    o.append(t(right + 10, y + 20, f"{m * 100:.1f}%", 13, INK, "bold"))
    o.append(t(right + 58, y + 20, f"({(m - target) * 100:+.1f})", 11.5, MUTED))
o.append(t(24, H - 8, "Blue = app A, amber = app B, red tick = target. "
                      "Number = measured share of A (error vs target, points).", 11.5, MUTED))
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" font-family="{FONT}">\n' + "\n".join(o) + "\n</svg>\n")
open(out, "w").write(svg)
print("wrote", out)
