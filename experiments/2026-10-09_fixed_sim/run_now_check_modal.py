"""Audit follow-up (audit.md), read-only: on DEV runs (real e1 donors and simulated runs, original inputs), "now" under the frozen rule vs a
candidate fix that only lets September 14-21 stamps (any year) be "now"; offsets from donors under each."""
import json, random, sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import modal

app = modal.App("probe-shift_aswin-surya_sub-fixed-sim_kill-2026-11-09")
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations")
W = Path(__file__).resolve().parent if modal.is_local() else Path("/root")
image = modal.Image.debian_slim(python_version="3.13").add_local_file(W / "clock.py", "/root/fixed_sim/clock.py") if modal.is_local() else modal.Image.debian_slim(python_version="3.13")
DEV = {"auto_workflow", "file_indexing", "model_registry", "slack_clone", "web_scraping"}


@app.function(image=image, volumes={"/acts": acts}, cpu=2, memory=16384, timeout=600, retries=0)
def peek(cells: list[str]) -> dict:
    sys.path.insert(0, "/root/fixed_sim")
    import clock
    cells = set(cells)

    def now_new(texts):
        found = [s for t in texts for s in clock.stamps(t, clock.LS_YEAR) if s["kind"] != "FAILED" and s["utc"].month == 9 and 14 <= s["utc"].day <= 21]
        dts = [s["utc"] for s in found if s["kind"] in ("DATE_CMD", "ISO", "HTTP")] or [s["utc"] for s in found if s["kind"] == "LS"]
        if not dts:
            return None
        b = lambda d: d.replace(minute=0, second=0, microsecond=0)
        hour = Counter(map(b, dts)).most_common(1)[0][0]
        inside = sorted(d for d in dts if b(d) == hour)
        return inside[len(inside) // 2]

    rows = []
    for f in sorted(Path("/acts/2026-10-06_la_segment-end_qwen3-32b_v1/inputs").glob("shard_*.jsonl")):
        rows += [r for r in map(json.loads, f.read_text().splitlines()) if r["arm"] in ("real-attack-e1", "sim-attack") and r["cell"] in cells and r["environment"] in DEV]
    outs = lambda r: [m["content"] for m in r["messages"] if m["role"] == "tool" and isinstance(m["content"], str)]
    res = {}
    for name, fn in (("frozen", clock.now_of), ("sept14_21", now_new)):
        donors = {r["cell"]: n for r in rows if r["arm"] == "real-attack-e1" and (n := fn(outs(r))) is not None}
        c, sim_days, donor_days = Counter(), Counter(), Counter(n.strftime("%Y-%m-%d") for n in donors.values())
        for r in (r for r in rows if r["arm"] == "sim-attack"):
            now = fn(outs(r))
            if now is None:
                c["sim_no_now"] += 1
                continue
            _, dn = clock.donor_for(r["cell"], donors)
            off = abs((dn - now).total_seconds()) / 86400
            sim_days[now.strftime("%Y-%m-%d")] += 1
            c["sim_shifted"] += 1
            c["offset<=3d"] += off <= 3
            c["offset_about_1y"] += 360 <= off <= 370
            c["offset_other"] += 3 < off < 360 or off > 370
        res[name] = {"donors": len(donors), "donor_days": donor_days.most_common(6), "sim_days": sim_days.most_common(6), "counts": dict(c)}
    return res


@app.local_entrypoint()
def main():
    cells = json.loads((W.parents[1] / "splits/exclusions_v4.json").read_text())["paired_attack_cells"]["cells"]
    from zoneinfo import ZoneInfo
    out = peek.remote(cells)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(W / "results" / f"now_check_{stamp}.json", "x") as f:
        f.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
