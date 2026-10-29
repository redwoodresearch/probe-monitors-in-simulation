"""Key paths where the real and simulated tools' parameter schemas differ (paths and value types only, values hashed)."""
import hashlib, json, sys, zipfile
from pathlib import Path
acting = "vllm/Qwen/Qwen3.5-27B"
def tools_of(ev):
    with zipfile.ZipFile(ev) as z:
        s = json.loads(z.read([n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")][0]))
    for e in s["events"]:
        if e.get("event") == "model" and e.get("model") == acting:
            return {t["function"]["name"]: t["function"] for t in e["call"]["request"]["tools"]}, s.get("id")
def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{p}.{k}")
    elif isinstance(o, list) and o and all(isinstance(v, (dict, list)) for v in o):
        for i, v in enumerate(o): yield from flat(v, f"{p}[{i}]")
    else: yield p, o
h = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()[:8]  # noqa: E731
run = Path(sys.argv[1]); task = sys.argv[2]
R, rid = tools_of(next((run / "honest_a" / task).glob("*.eval"))); Sx, sid = tools_of(next((run / "sim_honest@september" / task).glob("*.eval")))
for name in R:
    a, b = dict(flat(R[name])), dict(flat(Sx[name]))
    rows = [(k, type(a.get(k)).__name__ if k in a else "-", type(b.get(k)).__name__ if k in b else "-", h(a.get(k)) if k in a else "", h(b.get(k)) if k in b else "")
            for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    print(name, "differing paths:", len(rows))
    for r in rows: print("   ", r)
