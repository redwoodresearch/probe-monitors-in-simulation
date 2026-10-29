"""The completed-pass record of the replay (df's RULE @ 9445e96c6: Sol-answered steps are simulator-written and reported
descriptively per arm, per run and per path; the sensitivity that drops runs over 10% Sol-answered reads THIS file).

Reads the data Volume mounted READ-ONLY. For every one of the 135 ledgers: the digest chain re-verified exactly as
replay_core.Ledger does (sha256 of the previous digest + the row without its digest, keys sorted), the row count and the
last digest; output statuses below the eligible count (simulated_changed / simulated_identical from ok steps, kept_real_*,
kept_real_rejected, the rest beyond_limit), attempts by status, spend (cost and charge), and the Sol-answered calls by path
(net / refusal / block / fallback: calls in steps where that path's counter moved, and the counter sum). Counts, run keys
and digests only; no output text.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-15_q2_replay/record_replay.py --label t196_v1
"""

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-replay_record_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q2-onpolicysim_data_kill-2026-11-14"
MANIFESTS = {"phase1": ("604bc7c60", "experiments/2026-10-15_replay_sim_outputs/phase1_manifest.json",
                        "872ddc7c68bd856f1b21d3effb5f368068d058372cedc9d749532c57bc9e2ce7"),
             "placebo": ("604bc7c60", "experiments/2026-10-15_replay_sim_outputs/placebo_manifest.json",
                         "8eb57dae18ff40fd1c6da7fdc178868d8773415252a57fe4b83510c7ea58cf94")}
PATHS = {"net": "net_substitutions", "refusal": "refusal_substitutions", "block": "block_substitutions",
         "fallback": "fallback_substitutions"}
PT = ZoneInfo("America/Los_Angeles") if modal.is_local() else None  # the slim image has no tzdata

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2).read_only()


def h(s: str) -> str:  # replay_core.h
    return hashlib.sha256(s.encode()).hexdigest()


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=1, memory=2048, timeout=600, retries=0,
              volumes={"/data": vol})
def record(arm_of: dict, label: str) -> dict:
    root = Path("/data/replay") / label
    runs, arms = {}, {a: Counter() for a in sorted(set(arm_of.values()))}
    for rid in sorted(arm_of):
        f = root / "ledger" / f"{rid}.jsonl"
        rec = {"arm": arm_of[rid], "ledger": f.exists()}
        runs[rid] = rec
        if not f.exists():
            continue
        prev, ok_chain, rows = "", True, []
        for line in f.read_bytes().split(b"\n"):
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                ok_chain = False
                break
            d = r.pop("digest", None)
            if d != h(prev + json.dumps(r, sort_keys=True)):
                ok_chain = False
                break
            prev = d
            rows.append(r)
        head = rows[0] if rows else {}
        ok_chain = ok_chain and head.get("kind") == "header" and head.get("run") == rid
        st, att, paths, cost, charge, calls, sol_calls = {}, Counter(), Counter(), 0.0, 0.0, 0, 0
        for r in rows:
            k = r["kind"]
            if k == "result":
                att[r["status"]] += 1
                cost += r.get("cost_usd", 0.0)
                charge += r.get("charge_usd", 0.0)
                if r["status"] == "ok":
                    n, subs = len(r["js"]), r.get("subs", {})
                    calls += n
                    sol_calls += n if any(v > 0 for v in subs.values()) else 0
                    for p, key in PATHS.items():
                        paths[f"{p}_calls"] += n if subs.get(key, 0) > 0 else 0
                        paths[f"{p}_count"] += subs.get(key, 0)
                    st.update(zip(r["js"], r["statuses"]))
            elif k == "kept_real":
                st.update({j: r["status"] for j in r["js"]})
            elif k == "rejected":
                st.update({j: "kept_real_rejected" for j in r["js"]})
        statuses = Counter(st.get(j, "beyond_limit") for j in range(head.get("eligible", 0)))
        rec |= {"chain_ok": ok_chain, "rows": len(rows), "last_digest": prev, "K_r": head.get("K_r"), "eligible": head.get("eligible"),
                "done": any(r["kind"] == "done" for r in rows), "stops": [r["reason"] for r in rows if r["kind"] == "stop"],
                "statuses": dict(statuses), "attempts": dict(att), "cost_usd": round(cost, 4), "charged_usd": round(charge, 4),
                "simulated_calls": calls, "sol_answered_calls": sol_calls,
                "sol_answered_share": round(sol_calls / calls, 6) if calls else None, "sol_by_path": dict(paths)}
        a = arms[arm_of[rid]]
        a["runs"] += 1
        a["chain_ok"] += ok_chain
        a["done"] += rec["done"]
        a["stops"] += len(rec["stops"])
        a.update({f"status_{s}": n for s, n in statuses.items()})
        a.update({f"attempts_{s}": n for s, n in att.items()})
        a.update(paths)
        a["cost_usd"] += cost
        a["charged_usd"] += charge
        a["simulated_calls"] += calls
        a["sol_answered_calls"] += sol_calls
        a["runs_over_10pct_sol"] += 1 if calls and sol_calls / calls > 0.10 else 0
    out = {}
    for name, a in arms.items():
        processed = sum(v for k, v in a.items() if k.startswith("status_") and k != "status_beyond_limit")
        fb = a["status_kept_real_after_failure"] + a["status_kept_real_after_refusal"]
        out[name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sorted(a.items())} | {
            "processed_outputs": processed, "fallbacks": fb, "fallback_share": round(fb / processed, 6) if processed else None,
            "sol_answered_share": round(a["sol_answered_calls"] / a["simulated_calls"], 6) if a["simulated_calls"] else None,
            "runs_over_10pct_sol_keys": sorted(r for r, q in runs.items() if q["arm"] == name and q.get("simulated_calls")
                                               and q["sol_answered_calls"] / q["simulated_calls"] > 0.10)}
    ev = root / "pass_events.jsonl"
    events = [json.loads(x) for x in ev.read_text().splitlines()] if ev.exists() else []
    return {"label": label, "runs_total": len(runs), "ledgers": sum(q["ledger"] for q in runs.values()),
            "chain_ok_all": all(q.get("chain_ok") for q in runs.values()), "arms": out, "pass_events": events, "per_run": runs}


@app.local_entrypoint()
def main(label: str = "t196_v1"):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, cwd=HERE, check=True).stdout  # noqa: E731
    arm_of = {}
    for c, p, sha in MANIFESTS.values():
        b = git("show", f"{c}:{p}")
        assert hashlib.sha256(b).hexdigest() == sha, p
        arm_of |= {f"{r['world']}__{r['task']}": r["arm_of"] for r in json.loads(b)["runs"]}
    assert len(arm_of) == 135
    res = record.remote(arm_of, label)
    stamp = datetime.now(PT).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"pass_record_{label}_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps({"git": git("rev-parse", "HEAD").decode().strip(), "app": APP, "written_pt": stamp} | res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_run"})[:6000], "->", dst.name)
