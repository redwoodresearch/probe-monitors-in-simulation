"""Keys of the extra </think> tokens inside assistant turns (09's post-hoc sensitivity, via 4f). Token ids only; nothing decoded.

Same definition as run_think_extra.py: in an assistant turn (<|im_start|> 'assistant' .. next <|im_end|>/<|im_start|>/end),
expected </think> = 1 if the turn opens with <think> and holds one, else 0; the extra ones are every </think> after the
turn's first (or every one, in a turn without an opening <think>).
For each extra token in the served full staged epochs that layout A uses (6 worlds, the 846 scored runs):
- served position, and whether layout A contains it (A = served ids after the system splice, cut at the last needed request);
- (a) generated or render-time: the request whose LOGGED completion covers the position (logged prompt length <= pos < n,
  logged prompt + completion == the epoch's ids[:n]); none -> prompt/template ids added at render time;
- (b) reasoning or visible content: before or after the turn's first </think>;
- (c) tokens after it in the same turn (0 = at the turn end).

  uv run --no-project --with 'modal>=1.2' modal run experiments/2026-10-14_p2_served_recheck/run_think_keys.py
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-p2probes_servedrecheck_kill-2026-11-13"
STAGE = "served_2026-10-14_200651"
WORLDS = ("run1", "run2", "sim", "honest_a", "honest_b", "honest_c")
IM_START, IM_END, ASSISTANT, NL, THINK_O, THINK_C = 248045, 248046, 74455, 198, 248068, 248069

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.3")
ro = lambda n: modal.Volume.from_name(n).read_only()  # noqa: E731
VOLS = {"/acts": ro("probe-shift_aswin-surya_sub-p2served_acts_kill-2026-11-14"), "/q3": ro("probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"),
        "/cw": ro("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"), "/rl": ro("probe-shift_aswin-surya_sub-phase2_request-log_kill-2026-11-06")}
out_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13")


def _read(path, keep=None):
    import re

    rx = re.compile(r'"key":\s*"([^"]+)"')
    for line in open(path, "rb"):
        if keep is not None:
            m = rx.search(line[:2048].decode("utf-8", "replace"))
            if not m or m.group(1) not in keep:
                continue
        yield json.loads(line)


def extras(a):
    """[(pos, turn_start, turn_end, end_kind, opened, first_close_pos)] for every extra </think> in ids a."""
    import numpy as np

    starts = np.flatnonzero((a[:-1] == IM_START) & (a[1:] == ASSISTANT))
    bounds = np.flatnonzero((a == IM_START) | (a == IM_END))
    out = []
    for s in starts:
        nb = bounds[bounds > s]
        e = int(nb[0]) if len(nb) else len(a)
        kind = "end of ids" if not len(nb) else ("<|im_end|>" if a[e] == IM_END else "<|im_start|>")
        opened = len(a) > s + 3 and a[s + 2] == NL and a[s + 3] == THINK_O
        closes = [int(p) for p in np.flatnonzero(a[s + 2:e] == THINK_C) + s + 2]
        ex = closes[1:] if (opened and closes) else closes
        for p in ex:
            out.append((p, int(s), e, kind, bool(opened), closes[0] if closes else None))
    return out


@app.function(image=image, volumes=VOLS, cpu=2, memory=16384, timeout=600, retries=0)
def find_world(world: str, scored: list) -> list:
    import os

    import numpy as np

    scored = set(scored)
    A, need = {}, {}
    for r in _read(f"/acts/{STAGE}/A/inputs/{world}.jsonl"):
        rk = f"{world}|{r['task']}"
        if rk in scored:
            A[(r["key"], r["epoch"])] = (rk, r["task"], np.asarray(r["ids"]), r["served_staging"])
            need.setdefault(r["served_staging"], set()).add(r["key"])
    rows = []
    for st, keys in need.items():
        for f in sorted(os.listdir(f"/q3/{st}/inputs")):
            for r in _read(f"/q3/{st}/inputs/{f}", keep=keys):
                kk = (r["key"], r["epoch"])
                if kk not in A:
                    continue
                rk, task, a, _ = A[kk]
                s = np.asarray(r["ids"])
                ss = int(np.flatnonzero(s == IM_END)[0]) + 1
                sa = int(np.flatnonzero(a == IM_END)[0]) + 1
                cut = ss + len(a) - sa  # A's cut in served coordinates
                ex = extras(s)
                if not ex:
                    continue
                reqs = {}
                for c in r["calls"]:
                    reqs.setdefault(c["irid"], {"n": c["n"], "sha256": c["sha256"], "turns": set()})
                    if c.get("kind") in ("command", "call_end") and c.get("src_irid") == c["irid"] and c.get("turn") is not None:
                        reqs[c["irid"]]["turns"].add(c["turn"])
                for p, ts, te, kind, opened, fc in ex:
                    pa = p - ss + sa
                    in_a = p < cut and 0 <= pa < len(a) and int(a[pa]) == THINK_C
                    rows.append({"world": world, "run key": rk, "task": task, "served key": kk[0], "epoch": kk[1],
                                 "served file": f"{st}/{f}", "served pos": p, "served epoch length": len(s), "A cut (served coords)": cut,
                                 "in A": bool(in_a), "A pos": pa if in_a else None,
                                 "turn start": ts, "turn end": te, "turn ends with": kind, "turn opened": opened,
                                 "after the turn's first </think>": bool(fc is not None and p > fc),
                                 "tokens after it in the turn": te - p - 1,
                                 "next token is </think>": bool(p + 1 < len(s) and s[p + 1] == THINK_C),
                                 "requests": {i: {"n": v["n"], "sha256": v["sha256"], "turns": sorted(v["turns"])} for i, v in reqs.items()},
                                 "_ids_sha_by_n": {str(v["n"]): hashlib.sha256(s[:v["n"]].astype("<i4").tobytes()).hexdigest() for v in reqs.values()
                                                    if v["n"] <= len(s)}})
    return rows


@app.function(image=image, volumes=VOLS, cpu=2, memory=8192, timeout=300, retries=0)
def scan_log(fname: str, wanted: list) -> dict:
    import gzip
    import re

    W = set(wanted)
    rx = re.compile(rb'"request_id":\s*"([A-Za-z0-9]+)"')
    out = {}
    with gzip.open(f"/rl/{fname}", "rb") as f:
        while True:
            try:
                line = f.readline()
            except (EOFError, OSError):
                break
            if not line:
                break
            m = rx.search(line[:200])
            if not m or m.group(1).decode() not in W:
                continue
            r = json.loads(line)
            ch = (r.get("choices") or [{}])[0]
            if r.get("status") != 200 or not r.get("complete") or r.get("prompt_token_ids") is None or ch.get("token_ids") is None:
                continue
            p, c = r["prompt_token_ids"], ch["token_ids"]
            import numpy as np

            full = np.asarray(p + c, dtype="<i4")
            out.setdefault(r["request_id"], []).append({"lp": len(p), "n": len(full), "sha256": hashlib.sha256(full.tobytes()).hexdigest(),
                                                        "received": r.get("received"), "finish": ch.get("finish_reason"),
                                                        "think_close_in_completion": [i for i, t in enumerate(c) if t == THINK_C]})
    return out


@app.function(image=image, volumes=VOLS | {"/out": out_vol}, cpu=1, memory=8192, timeout=1200, retries=0)
def orchestrate(meta: dict) -> dict:
    import os
    from collections import Counter
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import numpy as np

    keys = sorted(set(map(str, np.load("/cw/served/A/p2/fibs__command.npz", allow_pickle=False)["key"])))
    calls = {w: find_world.spawn(w, [k for k in keys if k.startswith(w + "|")]) for w in WORLDS}
    rows = [x for w in WORLDS for x in calls[w].get()]
    wanted = sorted({i for x in rows for i in x["requests"]})
    files = sorted(f for f in os.listdir("/rl") if f.startswith("requests_") and "2026-10-08" <= f[9:19] <= "2026-10-11")
    logs = {}
    for fc in [scan_log.spawn(f, wanted) for f in files]:
        for i, xs in fc.get().items():
            logs.setdefault(i, []).extend(xs)
    for x in rows:
        p = x["served pos"]
        x["generated by"] = None
        x["log check"] = []
        for irid, v in x["requests"].items():
            recs = [r for r in logs.get(irid, []) if r["n"] == v["n"] and r["sha256"] == v["sha256"]]
            ok = len(recs) > 0 and x["_ids_sha_by_n"].get(str(v["n"])) == v["sha256"]
            x["log check"].append(bool(ok))
            if ok and recs[0]["lp"] <= p < v["n"]:
                off = p - recs[0]["lp"]
                x["generated by"] = {"request id": irid, "completion offset": off, "request turn": v["turns"],
                                     "logged completion has </think> at that offset": off in recs[0]["think_close_in_completion"],
                                     "finish_reason": recs[0]["finish"]}
        x["class (a)"] = "generated completion ids" if x["generated by"] else "prompt/template ids (render time)"
        x["class (b)"] = ("visible content (after the turn's first </think>)" if x["after the turn's first </think>"]
                          else ("no opening <think> in the turn" if not x["turn opened"] else "reasoning span"))
        x["class (c)"] = "at the turn end" if x["tokens after it in the turn"] == 0 else "more tokens follow in the turn"
        if not x["in A"]:
            x["why not in A"] = (f"beyond A's cut at served token {x['A cut (served coords)']}" if p >= x["A cut (served coords)"]
                                 else "inside A's range but A differs there")
        x["all staged requests of the epoch found in the logs"] = all(x["log check"])
        turns = sorted(t for v in x["requests"].values() for t in v["turns"])
        x["request turns contiguous"] = turns == list(range(turns[0], turns[0] + len(turns))) if turns else None
        del x["requests"], x["_ids_sha_by_n"], x["log check"]
    inA = [x for x in rows if x["in A"]]
    runs_A = {}
    for x in inA:
        d = runs_A.setdefault(x["run key"], {"run key": x["run key"], "task": x["task"], "epochs": set(), "extra </think> in A": 0})
        d["epochs"].add(x["epoch"])
        d["extra </think> in A"] += 1
    full_only = {}
    for x in rows:
        if not x["in A"]:
            d = full_only.setdefault((x["run key"], x["epoch"]), {"run key": x["run key"], "epoch": x["epoch"], "n": 0, "why": set()})
            d["n"] += 1
            d["why"].add(x["why not in A"])
    cnt = lambda xs, k: dict(Counter(x[k] for x in xs))  # noqa: E731
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "PRELIMINARY: keys of the extra </think> tokens (token ids only; no text decoded)", "stamp (PT)": stamp, **meta,
           "definition": "as run_think_extra.py; positions are served (vLLM) coordinates of the full staged epoch layout A uses",
           "log files scanned": len(files), "requests wanted": len(wanted), "requests found": sum(1 for i in wanted if i in logs),
           "A/B runs": [dict(d, epochs=sorted(d["epochs"])) for d in runs_A.values()],
           "counts in A by class": {k: cnt(inA, k) for k in ("class (a)", "class (b)", "class (c)", "world")},
           "counts in A: next token is </think>": cnt(inA, "next token is </think>"),
           "full-epoch-only extras": [dict(d, why=sorted(d["why"])) for d in full_only.values()],
           "counts full-epoch-only by class": {k: cnt([x for x in rows if not x["in A"]], k) for k in ("class (a)", "class (b)", "class (c)", "world")},
           "tokens": rows}
    os.makedirs("/out/2026-10-14_p2_served_recheck", exist_ok=True)
    path = f"/out/2026-10-14_p2_served_recheck/think_keys_{stamp}.json"
    blob = json.dumps(rec, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o)).encode()
    with open(path, "xb") as f:
        f.write(blob)
    out_vol.commit()
    return {"path": path, "sha256": hashlib.sha256(blob).hexdigest(), "blob": blob.decode()}


@app.local_entrypoint()
def main():
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=root).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--", str(HERE)], capture_output=True, text=True, check=True, cwd=root).stdout.strip():
        raise SystemExit("commit this folder before launch")
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    r = orchestrate.remote({"git": head, "script": "experiments/2026-10-14_p2_served_recheck/run_think_keys.py", "script_sha256": sha})
    (HERE / "results" / Path(r["path"]).name).write_text(r["blob"])
    print(json.dumps({"path": r["path"], "sha256": r["sha256"]}))
