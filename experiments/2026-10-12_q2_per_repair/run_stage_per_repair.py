"""Part B, staging (PLAN.md here; committed before it runs): each editor repair applied ALONE to the 939 simulated runs
of #36's base dump, with fix939's own two steps (../2026-10-08_q3_swap_cue/run_fix939_stage.py): the insert repair
`rerender_messages(m, "diff_with_note")` and the str_replace repair `swap_messages(m, "sr_gutter_no_newline")`. Writes two
new staged runs (only the runs each repair changes) and returns counts and keys only (no message text; failures are
re-raised as type and position only). One Modal CPU container.

Gates, in every simulated run: each repair's changed messages are exactly its rendered outputs; the two repairs' changed
messages are disjoint and their union equals fix939's (its rerender_report.json); every changed message is byte-identical
to fix939's staged message at that index; no run outside fix939's 823 changes.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-12_q2_per_repair/run_stage_per_repair.py --runs <#36 runs.json>
"""
import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).parent
SWAP = HERE.parent / "2026-10-07_q2_rendering_swap"
APP = "probe-shift_aswin-surya_sub-q2-perrepair_stage_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
BASE, FIX = "2026-10-06_la_segment-end_qwen3-32b_v1", "2026-10-11_q3_fix939_inputs_v1"
OUT = {"sr": ("2026-10-12_q2_perrepair_sr_inputs_v1", "sim-attack-sr-only"),
       "insert": ("2026-10-12_q2_perrepair_insert_inputs_v1", "sim-attack-insert-only")}
PIN = {"runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"}

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "pyyaml")
         .add_local_file(SWAP / "rerender.py", "/root/rerender.py")
         .add_local_file(SWAP / "str_replace_tells.py", "/root/str_replace_tells.py")
         .add_local_python_source("probe_shift"))


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=4, memory=32768, timeout=1800, retries=0)
def stage(keys: list[str]) -> dict:
    try:
        return _stage(keys)
    except Exception as e:  # sanitized and re-raised, never swallowed
        raise RuntimeError(f"stage failed: {type(e).__name__} at {getattr(e, 'where', 'unknown position')}") from None


def _stage(keys: list[str]) -> dict:
    import sys

    sys.path.insert(0, "/root")
    import rerender as rr
    import str_replace_tells as st

    for run, _ in OUT.values():
        if (Path("/acts") / run).exists():
            raise FileExistsError(f"{run} exists: a staged run is never overwritten")
    fix_report = json.loads((Path("/acts") / FIX / "rerender_report.json").read_text())
    fix_msgs = {}
    for line in (Path("/acts") / FIX / "inputs" / "shard_0.jsonl").read_text().splitlines():
        it = json.loads(line); fix_msgs[it["key"]] = it["messages"]
    want, staged, report = set(keys), {k: [] for k in OUT}, {k: {} for k in OUT}
    seen = set()
    for p in sorted((Path("/acts") / BASE / "inputs").glob("shard_*.jsonl")):
        with p.open() as fh:
            for ln, line in enumerate(fh):
                try:
                    it = json.loads(line)
                    if it["arm"] != "sim-attack" or it["key"] not in want:
                        continue
                    if it["key"] in seen:
                        raise ValueError("a run appears twice in the base inputs")
                    seen.add(it["key"])
                    msgs = it["messages"]
                    new_i, rec_i = rr.rerender_messages(msgs, "diff_with_note")
                    new_s, rec_s = st.swap_messages(msgs, "sr_gutter_no_newline")
                    ch = {"insert": [i for i, (a, b) in enumerate(zip(msgs, new_i)) if a != b],
                          "sr": [i for i, (a, b) in enumerate(zip(msgs, new_s)) if a != b]}
                    rendered = {"insert": sorted({r["msg"] for r in rec_i if r["category"] in rr.RENDERED}),
                                "sr": sorted({r["msg"] for r in rec_s if r["changed"]})}
                    if any(ch[k] != rendered[k] for k in ch) or len(new_i) != len(msgs) or len(new_s) != len(msgs):
                        raise ValueError("a repair's changed messages differ from its rendered outputs")
                    if set(ch["insert"]) & set(ch["sr"]):
                        raise ValueError("the two repairs change the same message")
                    union = sorted(set(ch["insert"]) | set(ch["sr"]))
                    fx = fix_report.get(it["key"], {}).get("changed_messages sim-attack-note-sr-both", [])
                    if union != fx:
                        raise ValueError("the two repairs' changed messages don't equal fix939's")
                    for k, new in (("insert", new_i), ("sr", new_s)):
                        if any(new[i] != fix_msgs[it["key"]][i] for i in ch[k]):
                            raise ValueError("a changed message differs from fix939's staged message")
                        if ch[k]:
                            run, arm = OUT[k]
                            staged[k].append(it | {"arm": arm, "messages": new})
                            report[k][it["key"]] = {f"changed_messages {arm}": ch[k]}
                except Exception as e:
                    e.where = f"{p.name} line {ln}"
                    raise
    if seen != want:
        raise ValueError(f"found {len(seen)} of {len(want)} simulated runs")
    if set(report["insert"]) | set(report["sr"]) != set(fix_report):
        raise ValueError("the runs changed by either repair are not fix939's changed runs")
    out = {}
    for k, (run, arm) in OUT.items():
        dst = Path("/acts") / run
        (dst / "inputs").mkdir(parents=True)
        body = "".join(json.dumps(x) + "\n" for x in sorted(staged[k], key=lambda x: x["key"]))
        (dst / "inputs" / "shard_0.jsonl").write_text(body)
        (dst / "rerender_report.json").write_text(json.dumps(report[k]))
        out[k] = {"staged run": run, "arm": arm, "runs changed": len(staged[k]), "messages changed": sum(len(v[f"changed_messages {arm}"]) for v in report[k].values()),
                  "inputs sha256": hashlib.sha256(body.encode()).hexdigest(), "keys sha256": hashlib.sha256(json.dumps(sorted(report[k])).encode()).hexdigest()}
    acts_vol.commit()
    return out | {"simulated runs read": len(seen), "fix939 changed runs": len(fix_report),
                  "runs changed by both repairs": len(set(report["insert"]) & set(report["sr"])),
                  "runs changed by the insert repair only": len(set(report["insert"]) - set(report["sr"])),
                  "runs changed by the str_replace repair only": len(set(report["sr"]) - set(report["insert"]))}


@app.local_entrypoint()
def main(runs: str):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if hashlib.sha256(Path(runs).read_bytes()).hexdigest() != PIN["runs.json"]:
        raise SystemExit("not the pinned #36 runs.json")
    rs = json.loads(Path(runs).read_text())
    keys = sorted(r["key"] for r in rs if r["arm"] == "sim-attack")
    if len(keys) != 939:
        raise SystemExit(f"{len(keys)} simulated runs, not 939")
    res = {"plan": "PLAN.md Part B (this folder)", "git": head} | stage.remote(keys)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"stage_per_repair_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1)); print(dest)
