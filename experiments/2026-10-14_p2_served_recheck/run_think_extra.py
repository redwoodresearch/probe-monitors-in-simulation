"""Extra </think> tokens inside assistant turns, per world, in the served ids and in the reader layout (09's follow-up via 4f).
Token ids only; nothing is decoded or printed beyond counts. Modal CPU, read-only inputs; the record is written in-job.

Definition (by token id):
- an assistant turn starts at <|im_start|> (248045) followed by "assistant" (74455) and runs to the next <|im_end|> (248046)
  or <|im_start|>, or to the end of the ids;
- it is "opened" if the token after "assistant\\n" is <think> (248068);
- expected </think> (248069) count per turn = 1 if the turn is opened and holds at least one </think>, else 0;
- extra </think> = (</think> tokens in the turn) - expected. Extra <think> (248068 beyond the opening one) is counted beside it.
Layouts counted, over the 846 scored runs (keys of the served npz):
- served (full): each staged served epoch's vLLM ids as staged (uncut), for the epochs layout A uses;
- A and B: the layouts' staged ids (A = served ids after the system splice, cut at the last needed request; B = A with the
  opening think blocks emptied);
- reader: the reader layout's staged ids (corrected B manifest bf847d42 epochs; honest C from q4f_c_v1), one epoch per run.

  uv run --no-project --with 'modal>=1.2' --with 'numpy>=2' modal run --detach experiments/2026-10-14_p2_served_recheck/run_think_extra.py
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-p2probes_servedrecheck_kill-2026-11-13"
STAGE = "served_2026-10-14_200651"
WORLDS = ("run1", "run2", "sim", "honest_a", "honest_b", "honest_c")
MANIFEST_B = ("46ca25fc", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_manifest_v1.json",
              "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc")
IM_START, IM_END, ASSISTANT, NL, THINK_O, THINK_C = 248045, 248046, 74455, 198, 248068, 248069
DEFINITION = ("assistant turn = from <|im_start|>(248045) 'assistant'(74455) to the next <|im_end|>(248046) or <|im_start|>, or the end "
              "of the ids; opened = the token after 'assistant\\n' is <think>(248068); expected </think>(248069) = 1 if opened and the turn "
              "holds at least one </think>, else 0; extra </think> = </think> count in the turn - expected; extra <think> = <think> "
              "count - (1 if opened). Runs/epochs affected = those with at least one extra </think>.")

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.3")
ro = lambda n: modal.Volume.from_name(n).read_only()  # noqa: E731
VOLS = {"/acts": ro("probe-shift_aswin-surya_sub-p2served_acts_kill-2026-11-14"), "/q3": ro("probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"),
        "/q4": ro("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"), "/cw": ro("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14")}
out_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13")


def extra_think(ids) -> dict:
    import numpy as np

    a = np.asarray(ids)
    starts = np.flatnonzero((a[:-1] == IM_START) & (a[1:] == ASSISTANT))
    bounds = np.flatnonzero((a == IM_START) | (a == IM_END))
    out = {"assistant turns": 0, "opened turns": 0, "extra </think>": 0, "extra <think>": 0, "turns with extra </think>": 0}
    for s in starts:
        nb = bounds[bounds > s]
        e = int(nb[0]) if len(nb) else len(a)
        body = a[s + 2:e]
        opened = len(a) > s + 3 and a[s + 2] == NL and a[s + 3] == THINK_O
        nc, no = int((body == THINK_C).sum()), int((body == THINK_O).sum())
        x = nc - (1 if opened and nc >= 1 else 0)
        out["assistant turns"] += 1
        out["opened turns"] += int(opened)
        out["extra </think>"] += x
        out["extra <think>"] += no - int(opened)
        out["turns with extra </think>"] += int(x > 0)
    return out


def _read(path, keep=None):
    import re

    rx = re.compile(r'"key":\s*"([^"]+)"')
    for line in open(path, "rb"):
        if keep is not None:
            m = rx.search(line[:2048].decode("utf-8", "replace"))
            if not m or m.group(1) not in keep:
                continue
        yield json.loads(line)


@app.function(image=image, volumes=VOLS, cpu=2, memory=16384, timeout=600, retries=0)
def count_world(world: str, scored: list, reader_sources: list) -> dict:
    import os
    from collections import Counter

    scored = set(scored)
    res = {}

    def add(layout, rk, ep_key, c):
        d = res.setdefault(layout, {"counts": Counter(), "runs": set(), "epochs": set(), "runs_seen": set(), "epochs_seen": 0})
        d["counts"].update(c)
        d["runs_seen"].add(rk)
        d["epochs_seen"] += 1
        if c["extra </think>"] > 0:
            d["runs"].add(rk)
            d["epochs"].add(ep_key)

    need = {}
    for L in ("A", "B"):
        for r in _read(f"/acts/{STAGE}/{L}/inputs/{world}.jsonl"):
            rk = f"{world}|{r['task']}"
            if rk not in scored:
                continue
            add(L, rk, (r["key"], r["epoch"]), extra_think(r["ids"]))
            if L == "A":
                need.setdefault(r["served_staging"], set()).add((r["key"], r["epoch"], rk))
    for st, eps in need.items():
        keys = {k for k, _, _ in eps}
        want = {(k, e): rk for k, e, rk in eps}
        for f in sorted(os.listdir(f"/q3/{st}/inputs")):
            for r in _read(f"/q3/{st}/inputs/{f}", keep=keys):
                rk = want.get((r["key"], r["epoch"]))
                if rk is not None:
                    add("served (full staged epochs)", rk, (r["key"], r["epoch"]), extra_think(r["ids"]))
    by_file = {}
    for s in reader_sources:
        if s["run_key"] in scored:
            by_file.setdefault(s["inputs"], {})[s["key"]] = s["run_key"]
    for f, m in by_file.items():
        for r in _read(f"/q4/{f}", keep=set(m)):
            add("reader", m[r["key"]], (r["key"], r.get("epoch")), extra_think(r["ids"]))
    out = {}
    for L, d in res.items():
        out[L] = dict(d["counts"]) | {"runs affected": len(d["runs"]), "epochs affected": len(d["epochs"]),
                                      "runs counted": len(d["runs_seen"]), "epochs counted": d["epochs_seen"]}
    return out


@app.function(image=image, volumes=VOLS | {"/out": out_vol}, cpu=1, memory=8192, timeout=900, retries=0)
def orchestrate(manifest_b: dict, meta: dict) -> dict:
    import os
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import numpy as np

    keys = sorted(set(map(str, np.load("/cw/served/A/p2/fibs__command.npz", allow_pickle=False)["key"])))
    srcs = {w: [] for w in WORLDS}
    for e in manifest_b["epochs"]:
        srcs[e["world"]].append({"run_key": f"{e['world']}|{e['task']}", "key": e["key"], "inputs": e["staged_inputs_file"]})
    for f in sorted(os.listdir("/q4/q4b_extract_c_v1/meta/honest_b")):
        m = json.load(open(f"/q4/q4b_extract_c_v1/meta/honest_b/{f}"))
        srcs["honest_c"].append({"run_key": f"honest_c|{m['task']}", "key": m["key"], "inputs": "q4f_c_v1/inputs/honest_b.jsonl"})
    calls = {w: count_world.spawn(w, [k for k in keys if k.startswith(w + "|")], srcs[w]) for w in WORLDS}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "PRELIMINARY: extra </think> token ids inside assistant turns, per world and layout (counts only)",
           "stamp (PT)": stamp, "definition": DEFINITION,
           "layouts": "served (full staged epochs) = vLLM ids as staged for the epochs layout A uses; A, B = staged layouts; reader = reader staged ids",
           "scored runs": len(keys), **meta, "by_world": {w: calls[w].get() for w in WORLDS}}
    os.makedirs("/out/2026-10-14_p2_served_recheck", exist_ok=True)
    path = f"/out/2026-10-14_p2_served_recheck/think_extra_{stamp}.json"
    blob = json.dumps(rec, indent=1).encode()
    with open(path, "xb") as f:
        f.write(blob)
    out_vol.commit()
    return {"path": path, "sha256": hashlib.sha256(blob).hexdigest(), "record": rec}


@app.local_entrypoint()
def main():
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=root).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--", str(HERE)], capture_output=True, text=True, check=True, cwd=root).stdout.strip():
        raise SystemExit("commit this folder before launch")
    mb = subprocess.run(["git", "show", f"{MANIFEST_B[0]}:{MANIFEST_B[1]}"], capture_output=True, check=True, cwd=root).stdout
    if hashlib.sha256(mb).hexdigest() != MANIFEST_B[2]:
        raise SystemExit("manifest sha256 differs")
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    r = orchestrate.remote(json.loads(mb), {"git": head, "script": "experiments/2026-10-14_p2_served_recheck/run_think_extra.py",
                                            "script_sha256": script_sha, "manifest_b_sha256": MANIFEST_B[2]})
    Path(HERE / "results" / Path(r["path"]).name).write_bytes(json.dumps(r["record"], indent=1).encode())
    print(json.dumps({"path": r["path"], "sha256": r["sha256"]}))
