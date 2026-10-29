import json
from decimal import Decimal, ROUND_HALF_UP
d='/private/tmp/claude-501/-Users-aswinsurya-probe-shift-viewer/b3b0fe45-ac17-4911-908e-445ff32d8b31/scratchpad/status/figx/'
D=json.load(open(d+'data.json')); A=open(d+'figA.html').read(); B=open(d+'figB.html').read()
def r1(x): return float(Decimal(repr(x)).quantize(Decimal("0.1"), ROUND_HALF_UP))
def sg(x): v=r1(x); return "0.0" if v==0 else f"{v:+.1f}".replace("-","−")
main=[k for k in D["sweep"] if k!="Positive control"]
okn=sum(1 for k in main if all(p["d"][1] is not None and p["d"][1]<=0<=p["d"][2] for p in D["sweep"][k]["points"]))
excl={k:[round(p["fpr"],4) for p in D["sweep"][k]["points"] if p["d"][1] is not None and not (p["d"][1]<=0<=p["d"][2])] for k in D["sweep"]}
print('okn',okn, excl)
# X range of upper limits across plotted rows
keys=["v6_with_strategy","v6_rerun_with_strategy","v7_pooled","a25_pooled"]
Us=[]
for lab in ("Replies, 1%","Replies, 5%","Commands, 1%","Commands, 5%"):
    for k in keys:
        v=D[lab].get(k)
        if v: Us.append(v["U"])
    v=(D[lab].get("served") or {}).get("A")
    if v: Us.append(v["U"])
print('U range', min(Us), max(Us))
NAMES={"sol_a":"GPT-5.6-Sol-trained","fibs":"Published deception","harmfulness":"Harmfulness","refusal":"Refusal","highstakes":"High-stakes"}
rows=[]
for lab in ("Replies, 1%","Replies, 5%","Commands, 1%","Commands, 5%"):
    for p,v in D[lab]["v6_per_probe"].items():
        a=v.get("sim_minus_mean_real|with_strategy"); b=v.get("reexecution_run2_minus_run1|with_strategy")
        if a: rows.append(f'<tr><td>{lab}</td><td>{NAMES.get(p,p)}</td><td>{sg(a[0])} [{sg(a[1])}, {sg(a[2])}]</td><td>{sg(b[0])} [{sg(b[1])}, {sg(b[2])}]</td></tr>')
tab=('<table class="small"><tr><th>Readout</th><th>Probe (eligible)</th><th>Simulated − mean of real, 85 with-strategy tasks [95%]</th><th>Real run 2 − run 1, same tasks [95%]</th></tr>'+''.join(rows)+'</table>')
open(d+'pertab.html','w').write(tab)
json.dump({"okn":okn,"excl":excl,"Umin":min(Us),"Umax":max(Us)},open(d+'meta.json','w'))
