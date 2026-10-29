# Draft main-question figures from figdata.json (written by extract.py from committed records). No compute.
import json
D = json.load(open("/private/tmp/claude-501/-Users-aswinsurya-probe-shift-viewer/b3b0fe45-ac17-4911-908e-445ff32d8b31/scratchpad/status/draftfig/figdata.json"))
BLUE, ORANGE, GREY = "#2f6db5", "#e08a1e", "#8a8a8a"
M = "−"
def sg(v): return "0.0" if abs(v) < 0.05 else f"{v:+.1f}".replace("-", M)
def nm(v): return f"{v:.1f}".replace("-", M)
def ivs(t): return f"{sg(t[0])} [{sg(t[1])}, {sg(t[2])}]"
def esc(t): return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def svg(W, H, label, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{esc(label)}" '
            f'style="font-family:inherit;font-size:12px;display:block">{"".join(body)}</svg>')
def T(x, y, s, anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="currentColor" {extra}>{esc(s)}</text>'
def whisker(x, y1, y2, color, w, cap=0):
    s = f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{w}" stroke-linecap="round"/>'
    if cap:
        for y in (y1, y2): s += f'<line x1="{x-cap:.1f}" y1="{y:.1f}" x2="{x+cap:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="{w}"/>'
    return s
def hwhisker(y, x1, x2, color, w):
    return f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="{w}" stroke-linecap="round"/>'

def panel1_one(key, title, sub):
    rows = D["panel1"][key]; W, L = 420, 46; pw = W - L - 6; slot = pw / 3
    top, bot = 50, 200; dt, db, lo, hi = 236, 396, -20, 15
    yb = lambda v: bot - (bot - top) * v / 100
    yd = lambda v: db - (db - dt) * (v - lo) / (hi - lo)
    b = [T(W / 2, 16, title, "middle", 'font-weight="600"'), T(W / 2, 32, sub, "middle", 'fill-opacity="0.75"')]
    b.append(f'<line x1="{L}" y1="{bot}" x2="{W-6}" y2="{bot}" stroke="currentColor" stroke-opacity="0.4"/>')
    for t in (0, 50, 100): b.append(T(L - 5, yb(t) + 4, str(t), "end", 'fill-opacity="0.7"'))
    b.append(T(12, (top + bot) / 2, "attack runs caught (%)", "middle", f'transform="rotate(-90 12 {(top+bot)/2})" fill-opacity="0.8"'))
    b.append(f'<rect x="{L}" y="{yd(5):.1f}" width="{pw}" height="{yd(-5)-yd(5):.1f}" fill="currentColor" fill-opacity="0.09"/>')
    b.append(f'<line x1="{L}" y1="{yd(0):.1f}" x2="{W-6}" y2="{yd(0):.1f}" stroke="currentColor" stroke-opacity="0.5"/>')
    for t in (-15, -5, 0, 5, 15): b.append(T(L - 5, yd(t) + 4, sg(t).replace(".0", "") if t else "0", "end", 'fill-opacity="0.7"'))
    b.append(T(12, (dt + db) / 2, "simulated − real (points)", "middle", f'transform="rotate(-90 12 {(dt+db)/2})" fill-opacity="0.8"'))
    for i, rate in enumerate(sorted(rows, key=float)):
        r = rows[rate]; cx = L + slot * i + slot / 2; bw = 32
        for j, (arm, c) in enumerate((("real", BLUE), ("sim", ORANGE))):
            v, l, h = r[arm]; bx = cx - bw - 3 + j * (bw + 6)
            b.append(f'<rect x="{bx:.1f}" y="{yb(v):.1f}" width="{bw}" height="{bot-yb(v):.1f}" rx="3" fill="{c}"/>')
            b.append(whisker(bx + bw / 2, yb(l), yb(h), "currentColor", 1.6, 5))
            b.append(T(bx + bw / 2, yb(h) - 5, nm(v), "middle", 'font-size="11.5"'))
        b.append(T(cx, bot + 15, f"{float(rate)*100:g}% target", "middle", 'font-size="11"'))
        d, rr = r["d"], r["rr"]
        b.append(whisker(cx - 9, yd(max(d[1], lo)), yd(min(d[2], hi)), "currentColor", 4.5, 6))
        b.append(f'<circle cx="{cx-9:.1f}" cy="{yd(d[0]):.1f}" r="5" fill="currentColor"/>')
        b.append(whisker(cx + 9, yd(max(rr[1], lo)), yd(min(rr[2], hi)), GREY, 3, 4))
        b.append(f'<circle cx="{cx+9:.1f}" cy="{yd(rr[0]):.1f}" r="4" fill="{GREY}"/>')
        b.append(T(cx, db + 16, ivs(d), "middle", 'font-size="11.5" font-weight="600"'))
        b.append(f'<text x="{cx:.1f}" y="{db+31:.1f}" text-anchor="middle" fill="{GREY}" font-size="11">{esc(ivs(rr))}</text>')
    ly = db + 52
    b.append(f'<rect x="{L}" y="{ly-9}" width="10" height="10" fill="{BLUE}"/>' + T(L + 14, ly, "real (mean of runs 1, 2)"))
    b.append(f'<rect x="{L+170}" y="{ly-9}" width="10" height="10" fill="{ORANGE}"/>' + T(L + 184, ly, "simulated"))
    b.append(whisker(L + 268, ly - 10, ly + 1, "currentColor", 1.6, 4) + T(L + 278, ly, "95% interval"))
    ly2 = ly + 18
    b.append(whisker(L + 4, ly2 - 11, ly2 + 1, "currentColor", 4.5) + T(L + 14, ly2, "simulated − real, 95%"))
    b.append(whisker(L + 160, ly2 - 11, ly2 + 1, GREY, 3) + T(L + 170, ly2, "run 2 − run 1, 95%"))
    b.append(f'<rect x="{L+296}" y="{ly2-9}" width="14" height="10" fill="currentColor" fill-opacity="0.15"/>' + T(L + 314, ly2, "±5 points"))
    return svg(W, ly2 + 8, f"{title}: real against simulated catch with 95% intervals, and simulated minus real", b)

LAB = {"sol_reply": "GPT-5.6-Sol-trained, averaged reply (B∪C)", "sol_command": "GPT-5.6-Sol-trained, any command (B)",
       "fibs": "Published deception probe, averaged reply (B∪C)", "refusal": "Refusal direction (candidate), averaged reply (B∪C)",
       "harmfulness": "Harmfulness direction, averaged reply (B∪C)", "highstakes": "High-stakes, weak gate, averaged reply (B∪C)",
       "apollo": "Apollo deception, H200 refit, reply (B∪C)",
       "control": "Positive control, last token, vs run 2 (A∪B)"}
def panel2_one(rate):
    W, rowh, top = 420, 50, 34; n = len(D["panel2"]); x0, x1, lo, hi = 8, 262, -15, 40
    xv = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)
    H0 = top + rowh * n
    b = [T(W / 2, 16, f"At the {float(rate)*100:g}% alarm line", "middle", 'font-weight="600"')]
    b.append(f'<rect x="{xv(-5):.1f}" y="{top-6}" width="{xv(5)-xv(-5):.1f}" height="{rowh*n}" fill="currentColor" fill-opacity="0.09"/>')
    b.append(f'<line x1="{xv(0):.1f}" y1="{top-6}" x2="{xv(0):.1f}" y2="{H0-6}" stroke="currentColor" stroke-opacity="0.5"/>')
    for t in (-10, 0, 10, 20, 30, 40): b.append(T(xv(t), H0 + 8, sg(t).replace(".0", "") if t else "0", "middle", 'fill-opacity="0.7" font-size="11"'))
    b.append(T((x0 + x1) / 2, H0 + 24, "simulated − real catch (points)", "middle", 'fill-opacity="0.8" font-size="11"'))
    for i, row in enumerate(D["panel2"]):
        y = top + i * rowh; q = row[rate]; d, rr = q["d"], q["rr"]
        if i: b.append(f'<line x1="{x0}" y1="{y-6}" x2="{W-4}" y2="{y-6}" stroke="currentColor" stroke-opacity="0.12"/>')
        b.append(T(x0, y + 6, LAB[row["key"]], "start", 'font-size="11.5" font-weight="600"'))
        real_word = "run 2" if row["key"] == "control" else "real"
        b.append(T(W - 4, y + 6, f"{real_word} {nm(q['real'])}%, sim {nm(q['sim'])}%", "end", 'font-size="11" fill-opacity="0.75"'))
        yb1, yb2 = y + 20, y + 32
        b.append(hwhisker(yb1, xv(d[1]), xv(d[2]), "currentColor", 4.5) + f'<circle cx="{xv(d[0]):.1f}" cy="{yb1}" r="5" fill="currentColor"/>')
        b.append(hwhisker(yb2, xv(rr[1]), xv(rr[2]), GREY, 3) + f'<circle cx="{xv(rr[0]):.1f}" cy="{yb2}" r="3.5" fill="{GREY}"/>')
        b.append(T(x1 + 10, yb1 + 4, ivs(d), "start", 'font-size="11.5" font-weight="600"'))
        b.append(f'<text x="{x1+10:.1f}" y="{yb2+4}" fill="{GREY}" font-size="11">{esc(ivs(rr))}</text>')
    ly = H0 + 44
    b.append(whisker(x0 + 4, ly - 11, ly + 1, "currentColor", 4.5) + T(x0 + 14, ly, "simulated − real, 95%"))
    b.append(whisker(x0 + 160, ly - 11, ly + 1, GREY, 3) + T(x0 + 170, ly, "run 2 − run 1, 95%"))
    b.append(f'<rect x="{x0+300}" y="{ly-9}" width="14" height="10" fill="currentColor" fill-opacity="0.15"/>' + T(x0 + 318, ly, "±5 points"))
    return svg(W, ly + 8, f"Simulated minus real catch at the {float(rate)*100:g}% alarm line for each probe, with 95% intervals and real run 2 minus run 1", b)

wrap = lambda a, c: f'<div style="display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start"><div style="flex:1 1 300px;min-width:0;max-width:440px">{a}</div><div style="flex:1 1 300px;min-width:0;max-width:440px">{c}</div></div>'
p1 = wrap(panel1_one("reply", "Replies, averaged", "each run's mean score; alarm line from honest B ∪ C"),
          panel1_one("command", "Commands, end of call", "caught if any command fires; alarm line from honest B"))
p2 = wrap(panel2_one("0.01"), panel2_one("0.05"))
open("/private/tmp/claude-501/-Users-aswinsurya-probe-shift-viewer/b3b0fe45-ac17-4911-908e-445ff32d8b31/scratchpad/status/draftfig/p1.html", "w").write(p1)
open("/private/tmp/claude-501/-Users-aswinsurya-probe-shift-viewer/b3b0fe45-ac17-4911-908e-445ff32d8b31/scratchpad/status/draftfig/p2.html", "w").write(p2)
print("ok", len(p1), len(p2))
