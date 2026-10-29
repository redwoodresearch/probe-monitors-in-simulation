"""POST HOC descriptive (Appendix D, slot 22): run length vs attempt, from the committed records only (no new scan).
From behaviour_full_2026-10-16_025753.json (per_key: admitted runs' steps and attempt labels):
  (a) median and mean steps of attempted vs not-attempted (unclear or declined) runs, per arm, per set and pooled;
  (b) the share of admitted runs at or near the 200-step cap, at three fixed thresholds: >= 190, >= 195 and >= 200 steps
      (steps = assistant messages), per arm;
and from attempt_rate4_2026-10-16_030654.json (bc594d28's definition: every labelled run):
  (c) the non-attempt share per arm (1 - the attempt rate) and the paired sim - run1 (and run2 - run1) difference with its
      interval: the attempt-rate pairs negated (interval endpoints swapped and negated), so (c) cannot drift from that record.
All launched cells; the final cohort (09's exclusions file) is applied later by key. One Modal CPU container.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with "numpy==2.3.3" modal run \\
      experiments/2026-10-16_behaviour_full/run_cap_summary.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-behaviourfull_cap_kill-2026-11-16"
BEH = (HERE / "results" / "behaviour_full_2026-10-16_025753.json", "40ad66d382cd26a9")
RATE = (HERE.parent / "2026-10-15_attempt_rate" / "results" / "attempt_rate4_2026-10-16_030654.json", "a539b184f6e9e4dc")
ARMS, THRESHOLDS = ("sim", "run1", "run2"), (190, 195, 200)
SETS = ("S_old", "S_new", "S_newmains", "S_paper493")
RATE_VIEW = {"S_old": "T196", "S_new": "S_new", "S_newmains": "nm360", "S_paper493": "S_paper493", "pooled": "pooled"}

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


@app.function(image=image, cpu=2, memory=4096, timeout=600, retries=0)
def summarize(per_key: dict, rate: dict) -> dict:
    import numpy as np
    out = {}
    for view in (*SETS, "pooled"):
        sets = SETS if view == "pooled" else (view,)
        res = {"steps": {}, "near_cap": {}, "non_attempt_share": {}, "non_attempt_pairs": {}}
        for a in ARMS:
            rows = [v for k, v in per_key.items() if k.split("|")[0] in sets and k.split("|")[1] == a and v["admitted"]]
            for grp, want in (("attempted", ("attempted",)), ("not attempted", ("unclear", "declined"))):
                st = np.array([v["steps"] for v in rows if v["label"] in want], float)
                res["steps"][f"{a} {grp}"] = {"n": len(st), "median": float(np.median(st)) if len(st) else None, "mean": float(st.mean()) if len(st) else None}
            st = np.array([v["steps"] for v in rows], float)
            res["near_cap"][a] = {"n_admitted": len(st), **{f">= {t}": float((st >= t).mean()) if len(st) else None for t in THRESHOLDS},
                                  "max_steps": float(st.max()) if len(st) else None}
            pa = rate[RATE_VIEW[view]]["per_arm"][a]
            res["non_attempt_share"][a] = {"share": 1 - pa["rate"], "ci95": [1 - pa["rate_ci95"][1], 1 - pa["rate_ci95"][0]],
                                           "labelled": pa["labelled"], "unclear": pa["unclear"], "declined": pa["declined"]}
        for pk in ("sim - run1", "run2 - run1"):
            p = rate[RATE_VIEW[view]]["pairs"][pk]
            res["non_attempt_pairs"][pk] = {"diff": -p["diff"], "ci95": [-p["ci95"][1], -p["ci95"][0]], "n_pairs": p["n_pairs"], "mcnemar_p": p["mcnemar_p"]}
        out[view] = res
    return out


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no") or not git("branch", "-r", "--contains", "HEAD"):
        raise SystemExit("commit and push first")
    for p, want in (BEH, RATE):
        if not hashlib.sha256(p.read_bytes()).hexdigest().startswith(want):
            raise SystemExit(f"{p.name}: sha256 differs from its pin")
    beh, rate = json.loads(BEH[0].read_text()), json.loads(RATE[0].read_text())
    res = summarize.remote(beh["per_key"], rate)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "POST HOC descriptive; PRELIMINARY; Appendix D slot 22; all launched cells (the final cohort is applied later by key)",
           "git": git("rev-parse", "HEAD"), "stamp (PT)": stamp, "app": APP,
           "inputs": {"behaviour": {"file": BEH[0].name, "sha256_prefix": BEH[1]}, "attempt_rate": {"file": RATE[0].name, "sha256_prefix": RATE[1]}},
           "definitions": {"steps": "assistant messages (behaviour_full)", "not attempted": "label unclear or declined (admitted runs for (a) and (b))",
                           "near the cap": f"steps >= each of {THRESHOLDS}", "(c)": "1 - attempt rate over every labelled run (bc594d28); pairs negated"}} | {"results": res}
    dst = HERE / "results" / f"cap_summary_{stamp}.json"
    dst.open("x").write(json.dumps(rec, indent=1))
    lines = []
    for view, r in res.items():
        lines.append(f"[{view}]")
        for a in ARMS:
            s_a, s_n = r["steps"][f"{a} attempted"], r["steps"][f"{a} not attempted"]
            nc, na = r["near_cap"][a], r["non_attempt_share"][a]
            lines.append(f"  {a}: attempted n {s_a['n']} median {s_a['median']} mean {s_a['mean']:.1f}; not attempted n {s_n['n']} median {s_n['median']} "
                         f"mean {(s_n['mean'] or float('nan')):.1f}; near cap " + ", ".join(f">={t} {nc[f'>= {t}']:.3f}" for t in THRESHOLDS)
                         + f" (max {nc['max_steps']:.0f}); non-attempt share {na['share']:.3f} [{na['ci95'][0]:.3f}, {na['ci95'][1]:.3f}]")
        for pk, p in r["non_attempt_pairs"].items():
            lines.append(f"  non-attempt {pk}: {p['diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] (n {p['n_pairs']}, McNemar p {p['mcnemar_p']:.3g})")
    (HERE / "results" / f"cap_summary_{stamp}.printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("->", dst)
