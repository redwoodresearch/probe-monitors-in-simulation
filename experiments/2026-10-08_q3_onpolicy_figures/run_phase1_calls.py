"""D (amendments 15, 50), off-policy side: the call table for the pilot cells' Phase 1 runs. Modal CPU; no text out.
With --pilot (the 196-task redo, plan_D_redo_196 Amendment 3): any pilot file, plus Sol's epoch-2 real attack run
(the cell's `sol_e2`, real-attack-e2 in the dump) for D's real-vs-real control; a transcript the dump lacks is listed
per task and arm, not raised (that task then leaves D's cohort, or the control for e2).

For each of the 20 pilot cells (pilot_tasks_amendment9.json), Phase 1's two attack runs as Qwen3.5-27B read them in the
v4 all-layer dump: Sol's epoch-1 real attack run (the cell's `sol_e1` file and id) and the cell's one simulated attack
run (September's simulator; the dump holds exactly one per pilot cell, checked). Per call (the dump's call_end positions;
step = the assistant message's index, action = the call's index in it, as reader.segment_end_positions counts):
- tool, label (`calls.label_call`, eb014307; the terminal submit excluded; any other tool without a rule raises);
- rule 1's parameter (`reader.command_param`, #163 at a9f49c2) and the matched subset (b): the call's last argument is
  its own non-empty command parameter (amendment 15), where call_end reads the same token as the on-policy readout;
- its dump row (chunk, row_start + row) when the call_end token was kept.
The staged messages are read on Modal from the inputs the dump was made from; only labels, flags and row numbers return.

  PYTHONPATH=<#163 src> uv run --no-project --python 3.14 --with 'modal>=1.2' modal run run_phase1_calls.py
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3onpol_phase1calls_kill-2026-11-08"
DUMP_VOL, DUMP_RUN = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06", "2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1"
INPUTS_VOL, INPUTS_RUN = "probe-shift_aswin-surya_sub-la-activations", "2026-10-06_la_segment-end-v4_qwen3-32b_v1"
PILOT = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9.json"

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("pyyaml").add_local_python_source("probe_shift")


@app.function(image=image, cpu=4, memory=32768, timeout=1800, retries=0,
              volumes={"/dump": modal.Volume.from_name(DUMP_VOL, version=2).read_only(),
                       "/in": modal.Volume.from_name(INPUTS_VOL, version=2).read_only()})
def build(cells: list, e2: bool = False) -> dict:
    from probe_shift.calls import fingerprint, label_call
    from probe_shift.reader import command_param

    meta_dir = Path("/dump") / DUMP_RUN / "meta"
    arms = ["real-attack-e1", "sim-attack"] + (["real-attack-e2"] if e2 else [])
    files = {a: sorted(p.name for p in (meta_dir / a).glob("*.json")) for a in arms}
    want, missing = {}, []  # (arm, transcript) -> pilot task; (task, arm) the dump lacks
    for c in cells:
        found = {}
        for arm in arms:
            if arm == "sim-attack":
                hit = [f for f in files[arm] if f"__{c['cell']}:" in f]
            else:
                (run,) = [c["sol_e1"]] if arm == "real-attack-e1" else c["sol_e2"]
                stem = run["file"].removesuffix(".eval")
                hit = [f for f in files[arm] if f.startswith(stem + "__") and run["id"] in f]
            if len(hit) > 1 or (len(hit) == 0 and not e2):  # the 20-task table keeps its frozen strictness
                raise ValueError(f"{c['key']}: {len(hit)} {arm} transcripts in the dump")
            if hit:
                found[arm] = hit[0][:-5]
            else:
                missing.append([c["key"], arm])
        for arm, fn in found.items():
            if (arm, fn) in want:
                raise ValueError(f"{arm}/{fn} matches two tasks: {want[(arm, fn)]} and {c['key']}")
            want[(arm, fn)] = c["key"]
    metas = {(arm, fn): json.loads((meta_dir / arm / f"{fn}.json").read_text()) for arm, fn in want}
    by_key = {(arm, m["key"]): (arm, fn) for (arm, fn), m in metas.items()}
    msgs = {}
    for f in sorted((Path("/in") / INPUTS_RUN / "inputs").glob("shard_*.jsonl")):
        for line in f.open():
            it = json.loads(line)
            if (it["arm"], it["key"]) in by_key:
                msgs[by_key[(it["arm"], it["key"])]] = it["messages"]
    if len(msgs) != len(want):
        raise ValueError(f"{len(want) - len(msgs)} transcripts not found in the staged inputs")
    rows, skipped = [], Counter()
    for (arm, fn), m in metas.items():
        turns = [x.get("tool_calls") or [] for x in msgs[(arm, fn)] if x["role"] == "assistant"]
        for p in m["positions"]:
            if p["kind"] != "call_end":
                continue
            tc = turns[p["step"]][p["action"]]
            f_ = tc["function"]  # staged messages: {"function": {"name", "arguments"}}, as Q2's labeler reads them
            name, args = (f_["name"], f_.get("arguments")) if isinstance(f_, dict) else (f_, tc.get("arguments"))
            args = json.loads(args) if isinstance(args, str) else (args or {})
            if name != p["tool"]:
                raise ValueError(f"{fn} step {p['step']} action {p['action']}: message says {name}, positions say {p['tool']}")
            if name == "submit":
                skipped["terminal submit (amendment 7)"] += 1
                continue
            param = command_param(name, args)
            matched = bool(param) and list(args)[-1] == param
            rows.append({"arm": arm, "task": want[(arm, fn)], "transcript": fn, "step": p["step"], "action": p["action"], "tool": name,
                         "label": label_call(name, args)["label"], "param": param, "matched_subset": matched, "kept": p["kept"],
                         "chunk": m["chunk"], "row": None if p["row"] is None else int(m["row_start"]) + p["row"]})
    return {"rows": rows, "skipped": dict(skipped), "labeler": fingerprint(), "transcripts": sorted(f"{a}/{f}" for a, f in want),
            "missing": missing}


@app.local_entrypoint()
def main(pilot: str = PILOT):
    cells = json.loads((HERE.parents[1] / pilot).read_text())["cells"]
    e2 = pilot != PILOT
    r = build.remote(cells, e2)
    rows = r["rows"]
    summary = {"pilot": pilot, "dump": f"{DUMP_VOL}/{DUMP_RUN}", "inputs": f"{INPUTS_VOL}/{INPUTS_RUN}", "labeler": r["labeler"],
               "skipped": r["skipped"], "missing_transcripts": r["missing"],
               "transcripts": len(r["transcripts"]), "calls": len(rows),
               "by_arm": {a: {"calls": sum(x["arm"] == a for x in rows), "matched_subset": sum(x["arm"] == a and x["matched_subset"] for x in rows),
                              "matched_impactful": sum(x["arm"] == a and x["matched_subset"] and x["label"] == "impactful" for x in rows),
                              "matched_read_only": sum(x["arm"] == a and x["matched_subset"] and x["label"] == "read_only" for x in rows),
                              "not_kept": sum(x["arm"] == a and not x["kept"] for x in rows),
                              "matched_by_tool": dict(Counter(x["tool"] for x in rows if x["arm"] == a and x["matched_subset"])),
                              "unmatched_by_tool": dict(Counter(x["tool"] for x in rows if x["arm"] == a and not x["matched_subset"]))}
                          for a in sorted({x["arm"] for x in rows})}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"phase1_calls_{'pilot' if not e2 else Path(pilot).stem}_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps({"summary": summary, "transcripts": r["transcripts"], "rows": rows}))
    print(json.dumps(summary, indent=1))
    print("->", path)
