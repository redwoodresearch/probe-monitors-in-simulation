"""k = 1 stage (Aswin's pick, "Run the 1-reply stage (~$280–400)", just before 16:30 PT 10-11): choose the 400 sites, $0,
from the pack's run metadata only (keys and token positions; no text). A site is a kept end-of-output position t of a
real e1 run in the 5 test environments with at least 7 earlier kept outputs, whose token sits below 32,768 - 6,000 (room
for a longer simulated reply), and whose (step, action) is also a kept output in the same cell's fixed-sim run (G's
twin). One seeded site per eligible run (random.Random("k1-site")), then 400 runs seeded (random.Random("k1-runs")).
Writes results/k1_sites_<stamp>.json and the same file to the Volume (claimB/k1_v1/sites.json, never overwritten).

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' python experiments/2026-10-11_claimB_k_replies/select_k1_sites.py
"""

import hashlib
import io
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
PACK = "claimB/level1/packedB_2cfd1aec393c"  # step B's level-1 pack: the rows the frozen probes were fit on and score
OUT = "claimB/k1_v1"
TEST = ["clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"]
SIM, E1, N_SITES, MIN_EARLIER, MAX_TOKENS, MARGIN = "sim-attack-fixed", "real-attack-e1", 400, 7, 32768, 6000


def main():
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    raw = b"".join(vol.read_file(f"{PACK}/runs.L63.json"))
    runs = json.loads(raw)
    by = {(r["cell"], r["arm"]): r for r in runs if r["environment"] in TEST and r["arm"] in (SIM, E1)}
    eligible = {}
    for (cell, arm), r in sorted(by.items()):
        if arm != E1 or (cell, SIM) not in by:
            continue
        sim_kept = {(p["step"], p["action"]) for p in by[(cell, SIM)]["positions"] if p["kind"] == "output_end" and p["kept"]}
        outs = sorted((p["step"], p["action"], p["token"]) for p in r["positions"] if p["kind"] == "output_end" and p["kept"])
        ok = [(s, a, t) for i, (s, a, t) in enumerate(outs) if i >= MIN_EARLIER and t < MAX_TOKENS - MARGIN and (s, a) in sim_kept]
        if ok:
            s, a, t = random.Random(f"k1-site|{cell}").choice(ok)
            eligible[cell] = {"cell": cell, "environment": r["environment"], "main_task": r["main_task"], "key_e1": r["key"],
                              "key_sim": by[(cell, SIM)]["key"], "step": s, "action": a, "token": t, "earlier_outputs": outs.index((s, a, t))}
    cells = sorted(eligible)
    chosen = sorted(random.Random("k1-runs").sample(cells, N_SITES))
    sites = [eligible[c] | {"site": i} for i, c in enumerate(chosen)]
    out = {"label": "k = 1 stage sites (keys and positions only)", "eligible_runs": len(cells), "sites": sites,
           "by_environment": {e: sum(1 for s in sites if s["environment"] == e) for e in TEST},
           "main_tasks": len({s["main_task"] for s in sites}), "pack_runs_sha256": hashlib.sha256(raw).hexdigest(),
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"k1_sites_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    try:
        have = {e.path for e in vol.listdir(OUT)}
    except Exception:  # noqa: BLE001  (the folder doesn't exist yet: nothing to overwrite)
        have = set()
    if f"{OUT}/sites.json" in have:
        raise SystemExit(f"{OUT}/sites.json exists; never overwritten")
    with vol.batch_upload() as b:
        b.put_file(io.BytesIO(dst.read_bytes()), f"{OUT}/sites.json")
    print({k: out[k] for k in ("eligible_runs", "by_environment", "main_tasks")}, "->", dst)


if __name__ == "__main__":
    main()
