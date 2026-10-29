# Two top-of-page figures from figx/data.json (values read by key by extract_x.py). Inline SVG, no compute.
import json, math
from decimal import Decimal, ROUND_HALF_UP
D = json.load(open("/private/tmp/claude-501/-Users-aswinsurya-probe-shift-viewer/b3b0fe45-ac17-4911-908e-445ff32d8b31/scratchpad/status/figx/data.json"))
BLUE, ORANGE, GREY, TEAL, PURPLE = "#2f6db5", "#e08a1e", "#8a8a8a", "#2a9d8f", "#7b5ea7"
M = "−"
def r1(x): return float(Decimal(repr(x)).quantize(Decimal("0.1"), ROUND_HALF_UP))
def f1(x): return f"{r1(x):.1f}".replace("-", M)
def sg(x): v = r1(x); return "0.0" if v == 0 else f"{v:+.1f}".replace("-", M)
def esc(t): return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def T(x, y, s, anchor="start", extra=""): return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="currentColor" {extra}>{esc(s)}</text>'
def svg(W, H, label, body): return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{esc(label)}" style="font-family:inherit;font-size:12px;display:block">{"".join(body)}</svg>')

# ---------- Figure A: error to add ----------
ROWS = [("v6_with_strategy", "Registered headline (v6; 85 with-strategy tasks)", BLUE),
        ("v6_rerun_with_strategy", "Real rerun, run 2 − run 1 (same 85 tasks)", GREY),
        ("v7_pooled", "Expanded probe family (v7; all 90 tasks)", TEAL),
        ("a25_pooled", "Fully simulated: alarm line on simulated honest (all tasks)", ORANGE),
        ("servedA", "Served context, layout A (all 90 tasks)", PURPLE)]
def get(d, key):
    if key == "servedA": return (d.get("served") or {}).get("A")
    return d.get(key)
def panel_x(lab):
    d = D[lab]; W, L, R = 440, 8, 10; X0 = 8; XMAX = 28; pw = W - L - R
    xs = lambda v: L + pw * min(v, XMAX) / XMAX
    top = 44; rh = 46; H = top + rh * len(ROWS) + 40
    b = [T(W / 2, 16, lab, "middle", 'font-weight="600"')]
    b.append(f'<rect x="{xs(0):.1f}" y="{top-8}" width="{xs(1)-xs(0):.1f}" height="{rh*len(ROWS)}" fill="#2a9d8f" fill-opacity="0.13"/>')
    b.append(f'<rect x="{xs(1):.1f}" y="{top-8}" width="{xs(5)-xs(1):.1f}" height="{rh*len(ROWS)}" fill="currentColor" fill-opacity="0.05"/>')
    for v, t in ((1, "1 fine"), (5, "5 sad")):
        b.append(f'<line x1="{xs(v):.1f}" y1="{top-8}" x2="{xs(v):.1f}" y2="{top-8+rh*len(ROWS)}" stroke="currentColor" stroke-opacity="0.45" stroke-dasharray="3 3"/>')
        b.append(T(xs(v) + 3, top - 12, t, "start", 'font-size="11" fill-opacity="0.8"'))
    for i, (key, name, col) in enumerate(ROWS):
        y = top + rh * i; v = get(d, key)
        b.append(T(L, y + 4, name, "start", 'font-size="11" fill-opacity="0.85"'))
        yy = y + 17
        if not v:
            b.append(T(L, yy + 4, "not available (fewer than 2 eligible probes)" if key != "servedA" else "not available", "start", 'font-size="11" fill-opacity="0.6"'))
            continue
        X, U = v["X"], v["U"]
        b.append(f'<line x1="{xs(X):.1f}" y1="{yy}" x2="{xs(U):.1f}" y2="{yy}" stroke="{col}" stroke-width="2"/>')
        b.append(f'<line x1="{xs(U):.1f}" y1="{yy-5}" x2="{xs(U):.1f}" y2="{yy+5}" stroke="{col}" stroke-width="2"/>')
        b.append(f'<rect x="{xs(0):.1f}" y="{yy-6}" width="{max(xs(X)-xs(0),2.5):.1f}" height="12" rx="3" fill="{col}"/>')
        lbl = "estimate " + (f"{f1(X)}" + (" (floored)" if X == 0 else "")) + f", approximate upper limit {f1(U)}"
        if key == "a25_pooled" and D[lab].get("a25_yardstick_U"):
            yu = D[lab]["a25_yardstick_U"]
            b.append(f'<line x1="{xs(yu):.1f}" y1="{yy-8}" x2="{xs(yu):.1f}" y2="{yy+8}" stroke="currentColor" stroke-width="2"/>')
            lbl += f"; real yardstick limit {f1(yu)} (black tick)"
        b.append(T(L, yy + 19, lbl, "start", 'font-size="11"'))
    ya = top + rh * len(ROWS) + 4
    b.append(f'<line x1="{xs(0):.1f}" y1="{ya}" x2="{xs(XMAX):.1f}" y2="{ya}" stroke="currentColor" stroke-opacity="0.4"/>')
    for t in (0, 5, 10, 15, 20, 25): b.append(T(xs(t), ya + 14, str(t), "middle", 'font-size="11" fill-opacity="0.7"'))
    b.append(T(W / 2, ya + 30, "points of catch rate to add to error bars (lower is better)", "middle", 'font-size="11" fill-opacity="0.8"'))
    return svg(W, H, f"Error to add, {lab}: estimate bars with approximate upper limits for five versions", b)

# ---------- Figure B: FPR sweep ----------
LX0, LX1 = math.log10(1e-4), math.log10(1e-1)
def panel_s(nm):
    s = D["sweep"][nm]; pts = s["points"]; W, L, R = 300, 40, 8; pw = W - L - R
    xs = lambda f: L + pw * (math.log10(f) - LX0) / (LX1 - LX0)
    t1, b1 = 34, 150; t2, b2 = 184, 290
    yt = lambda v: b1 - (b1 - t1) * v
    lo, hi = -30, 30
    yd = lambda v: b2 - (b2 - t2) * (max(min(v, hi), lo) - lo) / (hi - lo)
    fmin = 1 / s["n_honest_cal"]
    b = [T(W / 2, 14, nm, "middle", 'font-weight="600"')]
    for (tt, bb) in ((t1, b1), (t2, b2)):
        b.append(f'<rect x="{xs(1e-4):.1f}" y="{tt}" width="{xs(fmin)-xs(1e-4):.1f}" height="{bb-tt}" fill="currentColor" fill-opacity="0.12"/>')
    b.append(T((xs(1e-4) + xs(fmin)) / 2, t1 + 14, "not reachable", "middle", 'font-size="10" fill-opacity="0.75"'))
    b.append(T((xs(1e-4) + xs(fmin)) / 2, t1 + 26, "yet at run", "middle", 'font-size="10" fill-opacity="0.75"'))
    b.append(T((xs(1e-4) + xs(fmin)) / 2, t1 + 38, "level", "middle", 'font-size="10" fill-opacity="0.75"'))
    # top: TPR bands + lines
    for arm, col in (("real", BLUE), ("sim", ORANGE)):
        up = " ".join(f"{xs(p['fpr']):.1f},{yt(p[arm][2]):.1f}" for p in pts)
        dn = " ".join(f"{xs(p['fpr']):.1f},{yt(p[arm][1]):.1f}" for p in reversed(pts))
        b.append(f'<polygon points="{up} {dn}" fill="{col}" fill-opacity="0.2"/>')
        line = " ".join("%.1f,%.1f" % (xs(p["fpr"]), yt(p[arm][0])) for p in pts)
        b.append(f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="2"/>')
    for v in (0, 0.5, 1): b.append(T(L - 4, yt(v) + 4, f"{int(v*100)}%", "end", 'font-size="10" fill-opacity="0.7"'))
    b.append(f'<line x1="{L}" y1="{b1}" x2="{W-R}" y2="{b1}" stroke="currentColor" stroke-opacity="0.35"/>')
    # bottom: delta
    b.append(f'<rect x="{xs(fmin):.1f}" y="{yd(5):.1f}" width="{xs(1e-1)-xs(fmin):.1f}" height="{yd(-5)-yd(5):.1f}" fill="currentColor" fill-opacity="0.07"/>')
    b.append(f'<line x1="{L}" y1="{yd(0):.1f}" x2="{W-R}" y2="{yd(0):.1f}" stroke="currentColor" stroke-opacity="0.5"/>')
    for key, col, op in (("rr", GREY, 0.3), ("d", "#c0392b" if False else "#444", 0.25)):
        ok = [p for p in pts if p[key][1] is not None]
        if not ok: continue
        up = " ".join(f"{xs(p['fpr']):.1f},{yd(100*p[key][2]):.1f}" for p in ok)
        dn = " ".join(f"{xs(p['fpr']):.1f},{yd(100*p[key][1]):.1f}" for p in reversed(ok))
        b.append(f'<polygon points="{up} {dn}" fill="{GREY if key=="rr" else "currentColor"}" fill-opacity="{op if key=="rr" else 0.18}"/>')
    dline = " ".join("%.1f,%.1f" % (xs(p["fpr"]), yd(100 * p["d"][0])) for p in pts)
    b.append(f'<polyline points="{dline}" fill="none" stroke="currentColor" stroke-width="2"/>')
    for v in (-20, 0, 20): b.append(T(L - 4, yd(v) + 4, (f"{v:+d}" if v else "0").replace("-", M), "end", 'font-size="10" fill-opacity="0.7"'))
    for f, t in ((1e-4, "1e-4"), (1e-3, "1e-3"), (1e-2, "1e-2"), (1e-1, "1e-1")):
        b.append(T(xs(f), b2 + 13, t, "middle", 'font-size="10" fill-opacity="0.7"'))
    b.append(T(L + pw / 2, b2 + 26, "run-level false-alarm rate (log)", "middle", 'font-size="10" fill-opacity="0.8"'))
    b.append(T(L + 2, t1 - 4, "catch (attack runs caught)", "start", 'font-size="10" fill-opacity="0.8"'))
    b.append(T(L + 2, t2 - 4, "simulated − real (points)", "start", 'font-size="10" fill-opacity="0.8"'))
    p1 = min(pts, key=lambda p: abs(p["fpr"] - 0.01)); p5 = min(pts, key=lambda p: abs(p["fpr"] - 0.05))
    cap = f"at about 1%: real {100*p1['real'][0]:.0f}%, sim {100*p1['sim'][0]:.0f}%, gap {sg(100*p1['d'][0])} [{sg(100*p1['d'][1])}, {sg(100*p1['d'][2])}]"
    b.append(T(W / 2, b2 + 42, cap, "middle", 'font-size="10"'))
    return svg(W, b2 + 50, f"{nm}: real and simulated catch against the false-alarm rate, and simulated minus real", b)

FLEX = '<div style="display:flex;flex-wrap:wrap;gap:10px">'
CELL = lambda w: f'<div style="flex:1 1 {w}px;min-width:0;max-width:100%">'
CELLA = '<div style="flex:1 1 360px;min-width:0;max-width:560px">'
outA = FLEX + "".join(CELLA + panel_x(l) + "</div>" for l in ("Replies, 1%", "Replies, 5%", "Commands, 1%", "Commands, 5%")) + "</div>"
CELLB = '<div style="flex:1 1 260px;min-width:0;max-width:420px">'
outB = FLEX + "".join(CELLB + panel_s(n) + "</div>" for n in D["sweep"]) + "</div>"
leg = (f'<p class="small"><span style="color:{BLUE}">■</span> real (mean of runs 1 and 2) &nbsp; <span style="color:{ORANGE}">■</span> simulated, with 95% bands (top); '
       f'black line and band: simulated − real with its 95% band; grey band: real run 2 − run 1 (bottom); light band: ±5 points; shaded left zone: below 1 in 386 honest runs.</p>')
open("/private/tmp/claude-501/-Users-aswinsurya-probe-shift-viewer/b3b0fe45-ac17-4911-908e-445ff32d8b31/scratchpad/status/figx/figA.html", "w").write(outA)
open("/private/tmp/claude-501/-Users-aswinsurya-probe-shift-viewer/b3b0fe45-ac17-4911-908e-445ff32d8b31/scratchpad/status/figx/figB.html", "w").write(leg + outB)
print(len(outA), len(outB))
