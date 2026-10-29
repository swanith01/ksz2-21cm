#!/usr/bin/env python
"""
Digitise Zhou+25 (ApJ 991, 195; arXiv:2503.09462) Fig. 6 EXACTLY from the vector paths of the PDF.

Fig. 6, page 10: D_ell = ell(ell+1) C_ell^{kSZ2 x 21cm2} / 2pi [muK^4] vs z0 (left) and vs xbar_HII (right) for the three
reionization scenarios (Fiducial solid, Late dashed, Short dotted; Zhou+25 Table 1 / Fig. 1) at ell = 400 (blue),
800 (orange), 1600 (green). Top row: HERA noise filter; bottom row: SKA1-Low (filter ~ unity). Optimistic foreground
filter k_par,0 = 0.01, dz = 2 chunks. The axes are calibrated on the TICK MARKS (black 2-point paths); the tick values
come from the tick labels. Nothing is eyeballed.

Usage:  python tools/digitize_zhou25_fig6.py Lidz_ksz2_21cm2.pdf > src/ksz2_21cm/plotting/zhou25_fig6.py
"""
import re
import sys

import numpy as np
import pymupdf as fitz

pdf = sys.argv[1]
pg = fitz.open(pdf)[9]
assert "Figure 6." in pg.get_text(), "page 10 is not the Fig. 6 page"
dr = pg.get_drawings()
words = pg.get_text("words")

# the four axes (white-filled rectangles of ~187 x 139 pt); order: (row, col) = HERA z, HERA x, SKA z, SKA x
axes = sorted([p["rect"] for p in dr if p.get("fill") == (1.0, 1.0, 1.0) and 180 < p["rect"].width < 200 and 130 < p["rect"].height < 150],
              key=lambda r: (round(r.y0 / 50), r.x0))
assert len(axes) == 4, axes
NAMES = [("HERA", "z"), ("HERA", "xhii"), ("SKA", "z"), ("SKA", "xhii")]


def ticks(ax, axis):
    """tick positions (pt) and values for one axis of one panel."""
    pos = []
    for p in dr:
        if p.get("color") == (0.0, 0.0, 0.0) and len(p["items"]) == 1 and p["items"][0][0] == "l":
            a, b = p["items"][0][1], p["items"][0][2]
            if axis == "y" and abs(a.y - b.y) < 0.01 and abs(a.x - b.x) < 5 and abs(a.x - ax.x0) < 6 and ax.y0 - 1 < a.y < ax.y1 + 1:
                pos.append(a.y)
            if axis == "x" and abs(a.x - b.x) < 0.01 and abs(a.y - b.y) < 5 and abs(a.y - ax.y1) < 6 and ax.x0 - 1 < a.x < ax.x1 + 1:
                pos.append(a.x)
    pos = sorted(set(round(v, 2) for v in pos))
    labs = []
    for w in words:
        c = ((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)
        if axis == "y" and ax.x0 - 25 < c[0] < ax.x0 - 1 and ax.y0 - 5 < c[1] < ax.y1 + 5 and re.fullmatch(r"[0-9.]+", w[4]):
            labs.append((c[1], float(w[4])))
        if axis == "x" and ax.y1 + 1 < c[1] < ax.y1 + 20 and ax.x0 - 5 < c[0] < ax.x1 + 5 and re.fullmatch(r"[0-9.]+", w[4]):
            labs.append((c[0], float(w[4])))
    vals = []
    for t in pos:
        j = int(np.argmin([abs(t - l[0]) for l in labs]))
        assert abs(labs[j][0] - t) < 1.5, (axis, t, labs)
        vals.append(labs[j][1])
    vals = np.array(vals)
    if axis == "y":                       # the text layer has no minus sign: below the '0' tick values are negative
        pos_a = np.array(pos)
        z0 = pos_a[np.argmin(np.abs(vals))] if 0.0 in vals else None
        if z0 is not None:
            vals = np.where(pos_a > z0 + 0.5, -vals, vals)
    return np.array(pos), vals


cal = []
for ax in axes:
    px, vx = ticks(ax, "x")
    py, vy = ticks(ax, "y")
    ax_ = np.polyfit(px, vx, 1)
    ay_ = np.polyfit(py, vy, 1)
    assert np.max(np.abs(np.polyval(ax_, px) - vx)) < 0.02 * (np.ptp(vx) + 1e-9), "x calibration not linear"
    assert np.max(np.abs(np.polyval(ay_, py) - vy)) < 0.01 * np.ptp(vy), "y calibration not linear"
    cal.append((ax_, ay_))

COL = {(0.12, 0.47, 0.71): 400, (1.0, 0.5, 0.05): 800, (0.17, 0.63, 0.17): 1600}


def style(dashes):
    if dashes.strip() in ("[] 0", ""):
        return "Fiducial"
    first = float(dashes.strip("[] 0").split()[0])
    return "Late" if first > 2 else "Short"


out = {}
for p in dr:
    col = p.get("color")
    key = tuple(round(c, 2) for c in col) if col else None
    if key not in COL or len(p["items"]) < 10:
        continue
    r = p["rect"]
    k = [i for i, a in enumerate(axes) if a.contains(fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))]
    assert len(k) == 1, r
    pts = [p["items"][0][1]] + [it[2] for it in p["items"] if it[0] == "l"]
    xy = np.array([[q.x, q.y] for q in pts])
    ax_, ay_ = cal[k[0]]
    data = np.column_stack([np.polyval(ax_, xy[:, 0]), np.polyval(ay_, xy[:, 1])])
    out.setdefault(NAMES[k[0]], {}).setdefault(style(p.get("dashes", "[] 0")), {})[COL[key]] = data

print('"""Zhou+25 (ApJ 991, 195) Fig. 6, digitised from the vector paths of the PDF by tools/digitize_zhou25_fig6.py.')
print('D_ell = ell(ell+1) C_ell^{kSZ2 x 21cm2}/2pi [muK^4]. ZHOU_FIG6[instrument][axis][scenario][ell] = array [x, D]')
print('instrument: HERA (top row) / SKA (SKA1-Low, bottom row); axis: "z" (x = z0) or "xhii" (x = paper\'s own xbar_HII);')
print('scenario: Fiducial / Late / Short (Zhou+25 Table 1). Optimistic filter k_par,0 = 0.01, dz = 2."""')
print("import numpy as np\n\nZHOU_FIG6 = {")
for (inst, axn), sc in sorted(out.items()):
    print(f'    ("{inst}", "{axn}"): {{')
    for s in ("Fiducial", "Late", "Short"):
        print(f'        "{s}": {{')
        for ell in (400, 800, 1600):
            a = sc[s][ell]
            body = ", ".join(f"[{x:.4f}, {y:.3f}]" for x, y in a)
            print(f"            {ell}: np.array([{body}]),")
        print("        },")
    print("    },")
print("}")
