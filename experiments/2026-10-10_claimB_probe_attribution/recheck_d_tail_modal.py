"""Review check for #529 (thread LWic): attribution step D's all-token numbers without the transcript's final tokens.
Reads the as-run per-token projection files (run_attribution_tokens_modal.py, sha256 5f981b6f…; every file's provenance
must name it) on Modal CPU and recomputes D's summaries three ways: all tokens (as reported), all tokens before the end
of the last tool call or output span (the dropped tail: the closing template tokens and any final assistant
text), and tool call/output tokens only (assistant prose excluded). The files must cover exactly the
planned D cells x both arms (cells.json). Writes <out-dir>/recheck_d_tail_<stamp>.json (numbers and provenance only).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run \\
      experiments/2026-10-10_claimB_probe_attribution/recheck_d_tail_modal.py --out-dir <results dir>
"""

import json

import modal

AS_RUN = "5f981b6f1a208e39cce74cbc57ed2d4d334fac9c95f5ee5c9df34adc4fcc655c"
SIM, REAL = "sim-attack-fixed", "real-attack-e1"
app = modal.App("probe-shift_aswin-surya_sub-claimb_review-checks_kill-2026-11-10")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


def cat(s):
    if s is None:
        return "agent text or template"
    return ("call " if s["kind"] == "call" else "") + (f"text_editor {s['command']}" if s["tool"] == "text_editor" else s["tool"])


@app.function(image=image, cpu=2, memory=8192, volumes={"/acts": vol}, timeout=900, retries=0)
def recheck() -> dict:
    from pathlib import Path

    import numpy as np

    import hashlib

    plan_b = Path("/acts/claimB/attribution_tokens_v1/cells.json").read_bytes()
    plan = json.loads(plan_b)
    runs = []
    for f in sorted(Path("/acts/claimB/attribution_tokens_v1/d").glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        if meta["provenance"]["script_sha256"] != AS_RUN:
            raise ValueError(f"{f.name} was not made by the as-run script")
        runs.append((meta, z["proj_L63"]))
    want = sorted((c, a) for c in plan["D"] for a in (SIM, REAL))
    have = sorted((m["cell"], m["arm"]) for m, _ in runs)
    if have != want:  # exactly the planned cells, both arms, once each
        raise ValueError(f"D files cover {len(have)} (cell, arm) pairs; the plan has {len(want)}")
    nr = {a: sum(m["arm"] == a for m, _ in runs) for a in (SIM, REAL)}
    out = {"runs": len(runs), "by_arm": nr, "complete": True, "cells_json_sha256": hashlib.sha256(plan_b).hexdigest(), "as_run_script_sha256": AS_RUN}
    for scope in ("all_tokens", "before_final_tokens", "content_tokens"):
        top_where, contrib, slip_top, tail_top = {SIM: {}, REAL: {}}, {SIM: {}, REAL: {}}, {SIM: [], REAL: []}, {SIM: [], REAL: []}
        for m, p in runs:
            owner = np.full(len(p), -1)
            for k, s in enumerate(m["spans"]):
                owner[s["span"][0]:s["span"][1]] = k
            last_end = max(s["span"][1] for s in m["spans"])
            tail = np.arange(len(p)) >= last_end  # the transcript's final tokens after the last tool call/output span
            use = {"all_tokens": np.ones(len(p), bool), "before_final_tokens": ~tail, "content_tokens": owner >= 0}[scope]
            top = np.argsort(np.where(use, p, -np.inf))[::-1][:20]
            for t in top:
                c = cat(m["spans"][owner[t]] if owner[t] >= 0 else None)
                top_where[m["arm"]][c] = top_where[m["arm"]].get(c, 0) + 1
            slip_top[m["arm"]].append(float(np.mean([owner[t] >= 0 and bool(m["spans"][owner[t]].get("slips")) for t in top])))
            tail_top[m["arm"]].append(float(np.mean(tail[top])))
            n = use.sum()
            for k in sorted(set(owner[use].tolist())):
                c = cat(m["spans"][k] if k >= 0 else None)
                contrib[m["arm"]][c] = contrib[m["arm"]].get(c, 0.0) + float(p[use & (owner == k)].sum() / n)
        cats = sorted(set(contrib[SIM]) | set(contrib[REAL]))
        g = {c: contrib[SIM].get(c, 0.0) / nr[SIM] - contrib[REAL].get(c, 0.0) / nr[REAL] for c in cats}
        total = sum(g.values())
        out[scope] = {"mean_projection_gap": total,
                      "gap_share_by_category": {c: v / total for c, v in sorted(g.items(), key=lambda x: -abs(x[1]))},
                      "top20_tokens_by_category": {a: {c: v / (20 * nr[a]) for c, v in sorted(w.items(), key=lambda x: -x[1])} for a, w in top_where.items()},
                      "top20_tokens_in_slip_outputs": {a: float(np.mean(v)) for a, v in slip_top.items()},
                      "top20_tokens_in_final_tokens": {a: float(np.mean(v)) for a, v in tail_top.items()}}
    return out


@app.local_entrypoint()
def main(out_dir: str):
    import hashlib
    import subprocess
    from datetime import datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo

    res = recheck.remote()
    here = Path(__file__)
    res["provenance"] = {"script_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
                         "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=here.parent).stdout.strip()}
    path = Path(out_dir) / f"recheck_d_tail_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
    with open(path, "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("runs", "by_arm", "complete")}), "->", path)
