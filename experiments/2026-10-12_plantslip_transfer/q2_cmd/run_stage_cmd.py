"""Planted-slip transfer, pairs P2 and P3 (RULE.md, Amendment 1): staging, shared by both readers. One Modal CPU container
plants both editor slips (`plant_both()` from ../plant_both.py) into the 939 real e1 runs behind forest rows A1 and A2,
and into the second-run control's e2 runs: route 3's 885 (the e2 runs with labelled matched calls; route 3 labelled them
inside its 10-10 job with labeler eb014307fd1aab31 at a85ee070, so no labeller runs here). It stages the changed runs
(arms `e1-planted`, `e2-planted`) plus 20 seeded unplanted runs of each (arms `e1-control`, `e2-control`, for the
reproduction gates) for the two extractor copies. Counts and keys only leave the container.

The source is the staged messages both readers' dumps read for the attack arms: `2026-10-06_la_segment-end_qwen3-32b_v1`
(Qwen3-32B's segment-end v1 dump, behind P3; Qwen3.5's original dump read the same). P2's features came from the v4
dump (and route 3's e2 scores the v4 dump), which read `2026-10-06_la_segment-end-v4_qwen3-32b_v1`; v4 added only honest
arms, and this checks that every staged e1 and e2 run's messages are identical in the two, aborting otherwise.

Plant reach, per RULE.md: runs changed, outputs changed per slip, and whether the appended trailing newline reaches the
token ids under each reader's chat template: the planted run's token ids against the same run with every appended
newline removed again (identical ids: the newline did not reach the prompt). These monitors read call_end, the agent's own call content, so the plant can move a call's readout only
through context: calls after the first planted output.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-12_plantslip_transfer/q2_cmd/run_stage_cmd.py --manifest <P3's Q2 manifest.json> \\
      --rows <P2's refit rows.json> --route3 <route3_scores_2026-10-10_232231.json>
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
EXP = HERE.parent.parent
FILES = {"rerender.py": EXP / "2026-10-07_q2_rendering_swap", "str_replace_tells.py": EXP / "2026-10-07_q2_rendering_swap",
         "fixes.py": EXP / "2026-10-09_fixed_sim", "plant_both.py": EXP / "2026-10-12_plantslip_transfer"}  # by name: in the container HERE is /root
APP = "probe-shift_aswin-surya_sub-q2-plantcmd_stage_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"  # read-only: the staged messages the dumps read
INPUTS = "probe-shift_aswin-surya_sub-q2-plantcmd_inputs_kill-2026-11-12"
SOURCE, SOURCE_V4, STAGED = "2026-10-06_la_segment-end_qwen3-32b_v1", "2026-10-06_la_segment-end-v4_qwen3-32b_v1", "2026-10-12_q2_plantcmd_e1_inputs_v1"
QWEN35 = ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654")
QWEN3 = ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137")
ARMS = {"real-attack-e1": ("e1-planted", "e1-control"), "real-attack-e2": ("e2-planted", "e2-control")}
SEED, N_CONTROLS, N_SHARD_FILES = "20261012", 20, 8
PIN = {"q2_manifest.json": "cd4ab3495ded77f92ca73d88e53a00d4572ebabc9089f5684db75ad8e801263a",
       "refit_qwen35_rows.json": "41a68be0b1d7ae045d573dce15ce2a4b3cfa349f915cc6babc7fd478f28ed8e4",
       "route3_scores_2026-10-10_232231.json": "cf33388724da001c54107cf22cc416582ee8f58dcf99b39db61c4ad25c5b62e6"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
inputs = modal.Volume.from_name(INPUTS, version=2, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2==3.1.6", "huggingface_hub").add_local_python_source("probe_shift")
for _n, _f in FILES.items():
    image = image.add_local_file(_f / _n, f"/root/{_f.name}/{_n}")


def _mods():
    import sys
    for folder in {f for f in FILES.values()}:
        for p in (f"/root/{folder.name}", str(folder)):
            if p not in sys.path:
                sys.path.insert(0, p)
    import fixes  # noqa: E402  (it puts the swap folder on the path)
    import plant_both
    import rerender
    return fixes, rerender, plant_both


@app.function(image=image, volumes={"/acts": acts, "/in": inputs}, cpu=16, memory=32768, timeout=3600, retries=0)
def stage(keys: dict, controls: dict) -> dict:
    try:
        return _stage(keys, controls)
    except ModuleNotFoundError as e:  # a module's name can't quote staged text
        raise RuntimeError(f"staging failed: ModuleNotFoundError: {e.name}") from None
    except Exception as e:  # nothing else from the exception (which may quote staged text) crosses the boundary
        raise RuntimeError(f"staging failed: {type(e).__name__}") from None


def _stage(keys, controls):
    from collections import Counter

    from transformers import AutoTokenizer

    from probe_shift.reader import _layout
    fixes, rerender, pb = _mods()
    toks = {"qwen3.5": AutoTokenizer.from_pretrained(QWEN35[0], revision=QWEN35[1]), "qwen3-32b": AutoTokenizer.from_pretrained(QWEN3[0], revision=QWEN3[1])}

    def load(run):
        out = {}
        for shard in sorted((Path("/acts") / run / "inputs").glob("shard_*.jsonl")):
            for line in shard.read_text().splitlines():
                it = json.loads(line)
                if it["arm"] in keys and it["key"] in keys[it["arm"]]:
                    out[(it["arm"], it["key"])] = it
        return out
    items, items_v4 = load(SOURCE), load(SOURCE_V4)
    want = {(a, k) for a, ks in keys.items() for k in ks}
    if set(items) != want or set(items_v4) != want:
        raise ValueError("a staged run is missing from the v1 or v4 staged inputs")
    if any(items_v4[x]["messages"] != items[x]["messages"] for x in want):
        raise ValueError("a run's messages differ between the v1 and v4 staged inputs")
    n = Counter()
    per_run, staged = [], []

    def ids(name, msgs):
        return _layout(toks[name], msgs)[2]["input_ids"]
    for arm, k in sorted(want):
        it, (pl_arm, _) = items[(arm, k)], ARMS[arm]
        new, changed = pb.plant_both(it["messages"], fixes, rerender)
        if not changed:
            n[f"{arm}: runs unchanged"] += 1
            continue
        diff = [i for i, (a, b) in enumerate(zip(it["messages"], new)) if a != b]
        assert len(new) == len(it["messages"]) and diff == [c[0] for c in changed]
        assert all({x for x in new[i] if new[i][x] != it["messages"][i][x]} == {"content"} for i in diff)
        n[f"{arm}: runs changed"] += 1; n[f"{arm}: outputs changed"] += len(changed)
        for c in changed:
            n[f"{arm}: outputs changed, {c[3]}"] += 1
        sr = [c[0] for c in changed if c[3] == "gutter+newline"]
        no_nl = [dict(m) for m in new]
        for i in sr:  # the same planted run with every appended newline removed again
            no_nl[i]["content"] = no_nl[i]["content"][:-1]
        reach, t = {}, {}
        for name in toks:
            t[name] = ids(name, new)
            reach[name] = bool(sr) and t[name] != ids(name, no_nl)
            n[f"{arm}: runs whose appended newline reaches the {name} token ids"] += reach[name]
        n[f"{arm}: runs with a planted str_replace output"] += bool(sr)
        tool_idx = [i for i, m in enumerate(it["messages"]) if m["role"] == "tool"]
        per_run.append({"arm": arm, "key": k, "outputs changed": len(changed), "slips": dict(Counter(c[3] for c in changed)),
                        "first changed (step, action)": list(changed[0][1:3]), "first changed is the first tool output": changed[0][0] == tool_idx[0],
                        "appended newline reaches the token ids": reach, "qwen35 tokens": len(t["qwen3.5"]), "qwen3 tokens": len(t["qwen3-32b"])})
        staged.append(it | {"arm": pl_arm, "messages": new, "tokens": len(t["qwen3.5"]), "tokens_qwen3": len(t["qwen3-32b"]), "plant": "both", "orig_arm": arm})
    for arm, ks in controls.items():
        for k in ks:
            it = items[(arm, k)]
            staged.append(it | {"arm": ARMS[arm][1], "tokens": len(ids("qwen3.5", it["messages"])), "tokens_qwen3": len(ids("qwen3-32b", it["messages"])),
                                "plant": None, "orig_arm": arm})
            n[f"{arm}: controls"] += 1
    d = Path("/in") / STAGED / "inputs"
    if d.exists() and any(d.iterdir()):
        raise FileExistsError(f"{STAGED} is already staged")
    d.mkdir(parents=True)
    shas = {}
    for s in range(N_SHARD_FILES):
        body = "".join(json.dumps(x) + "\n" for x in staged[s::N_SHARD_FILES]).encode()
        (d / f"shard_{s}.jsonl").write_bytes(body)
        shas[f"shard_{s}.jsonl"] = hashlib.sha256(body).hexdigest()
    inputs.commit()
    return {"counts": dict(n), "per run": per_run, "inputs_sha256": shas, "staged transcripts": len(staged),
            "qwen35 tokens staged": int(sum(x["tokens"] for x in staged)), "qwen3 tokens staged": int(sum(x["tokens_qwen3"] for x in staged))}


@app.local_entrypoint()
def main(manifest: str, rows: str, route3: str):
    import subprocess
    for f in [Path(__file__).resolve(), HERE.parent / "RULE.md", HERE / "PINS.md"] + [folder / name for name, folder in FILES.items()]:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    for p_, role in ((Path(manifest), "q2_manifest.json"), (Path(rows), "refit_qwen35_rows.json"), (Path(route3), "route3_scores_2026-10-10_232231.json")):
        if hashlib.sha256(p_.read_bytes()).hexdigest() != PIN[role]:
            raise SystemExit(f"{p_.name} is not the pinned {role}")
    e1 = sorted({r["key"] for r in json.loads(Path(manifest).read_text()) if r["arm"] == "real-attack-e1"})
    p2 = sorted({r["key"] for r in json.loads(Path(rows).read_text()) if r["arm"] == "real-attack-e1"})
    r3 = json.loads(Path(route3).read_text())
    if r3["labeler"] != "eb014307fd1aab31" or r3["labeler_commit"] != "a85ee070" or not r3["gate"]["passed"]:
        raise SystemExit("route 3's file is not the eb014307 labels with a passed gate")
    e2 = sorted({x["key"] for x in r3["rows"] if x["world"] == "e2"})
    if len(e1) != 939 or not set(p2) <= set(e1) or len(e2) != 885:
        raise SystemExit(f"{len(e1)} e1 runs (not 939), P2's runs not among them, or {len(e2)} route-3 e2 runs (not 885)")
    rank = lambda k: hashlib.sha256(f"{SEED}|{k}".encode()).hexdigest()  # noqa: E731
    controls = {"real-attack-e1": sorted(p2, key=rank)[:N_CONTROLS], "real-attack-e2": sorted(e2, key=rank)[:N_CONTROLS]}  # serve both pairs' gates
    res = stage.remote({"real-attack-e1": e1, "real-attack-e2": e2}, controls)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"rule": "RULE.md Amendment 1 (../)", "git": head, "pins sha256": PIN, "source": SOURCE, "staged": f"{INPUTS}:{STAGED}",
           "e1 runs (P3)": len(e1), "e1 runs (P2)": len(p2), "e2 runs (route 3's labelled set)": len(e2), "controls": controls} | res
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"stage_cmd_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    (HERE / "config").mkdir(exist_ok=True)
    with open(HERE / "config" / "gate_transcripts.json", "x") as fh:  # the extractor copies' `transcripts:` file: the 40 controls
        fh.write(json.dumps({"inputs_run": STAGED, "affected": [], "controls": [[ARMS[a][1], k] for a, ks in controls.items() for k in ks],
                             "inputs_sha256": res["inputs_sha256"]}, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per run"}, indent=1)); print(dest)
