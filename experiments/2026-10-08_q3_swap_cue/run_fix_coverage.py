"""POST HOC coverage (2026-10-11): how much of the 939-cell tool-output gap could the swap's two fixes reach? Modal CPU,
pennies. Keys and numbers only: no message text leaves the container.

The two fixes are the swap's both-fixes arm (`sim-attack-note-sr-both`, variant diff_with_note+sr_gutter_no_newline in
../2026-10-07_q2_rendering_swap):
- insert: a text_editor insert output that rerender.py can rebuild (category exact / approx_window / approx_numbering)
  keeps the simulator's diff and gets the real tool's review note appended, without the trailing newline;
- str_replace: a text_editor str_replace output that parses as a plain diff gets the real tool's numbered gutter
  (`{n:>4}|{sign} {code}`) and loses its trailing newline.
Everything else is unchanged.

For every simulated attack run in #36's base dump, the container applies those fixes exactly as run_swap.stage does,
checks that only the fixed outputs changed, and returns per run the (step, action) of each changed output. Gate: for
the 217 runs already staged by the swap (its combined and extension staged runs), the changed messages must equal the
swap's own rerender reports. Locally, the changed outputs are matched to #36's kept tool-output readouts (tables.npz),
so outputs past the 32k cut, which no readout reads, don't count.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run \
      experiments/2026-10-08_q3_swap_cue/run_fix_coverage.py --scores-dir <#36 run_scores_2026-10-06_181548>
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
SWAP = HERE.parent / "2026-10-07_q2_rendering_swap"
APP = "probe-shift_aswin-surya_sub-q3swapcue_fixcover_kill-2026-11-10"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
BASE, STAGED, ARM = "2026-10-06_la_segment-end_qwen3-32b_v1", ["2026-10-08_q2_swap_combined_inputs_v1", "2026-10-08_q2_swap_ext_inputs_v1"], "sim-attack-note-sr-both"

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "pyyaml")
         .add_local_file(SWAP / "rerender.py", "/root/rerender.py")
         .add_local_file(SWAP / "str_replace_tells.py", "/root/str_replace_tells.py")
         .add_local_python_source("probe_shift"))


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=4, memory=32768, timeout=1800, retries=0)
def cover(keys: list[str]) -> dict:
    """Any failure is re-raised as its type name and position only: an exception's own text could carry message text."""
    try:
        return _cover(keys)
    except Exception as e:  # noqa: BLE001 — sanitized and re-raised, never swallowed
        raise RuntimeError(f"cover failed: {type(e).__name__} at {getattr(e, 'where', 'unknown position')}") from None


def _cover(keys: list[str]) -> dict:
    import sys

    sys.path.insert(0, "/root")
    import rerender as rr
    import str_replace_tells as st

    want, per = set(keys), {}
    for p in sorted((Path("/acts") / BASE / "inputs").glob("shard_*.jsonl")):
        with p.open() as fh:
            for ln, line in enumerate(fh):
                try:
                    it = json.loads(line)
                except Exception as e:
                    e.where = f"{p.name} line {ln}"
                    raise
                if it["arm"] != "sim-attack" or it["key"] not in want:
                    continue
                if it["key"] in per:
                    raise ValueError("a simulated run appears twice in the base inputs")
                try:
                    new, rec_i = rr.rerender_messages(it["messages"], "diff_with_note")  # the order run_swap.stage uses
                    new, rec_s = st.swap_messages(new, "sr_gutter_no_newline")
                except Exception as e:
                    e.where = f"{p.name} line {ln}"
                    raise
                ins = [r for r in rec_i if r["category"] in rr.RENDERED]
                srs = [r for r in rec_s if r["changed"]]
                changed = [i for i, (a, b) in enumerate(zip(it["messages"], new)) if a != b]
                if changed != sorted({r["msg"] for r in ins} | {r["msg"] for r in srs}) or len(new) != len(it["messages"]):
                    raise ValueError(f"{it['key']}: changed messages differ from the fixed outputs")
                per[it["key"]] = {"insert": [[r["step"], r["action"]] for r in ins], "str_replace": [[r["step"], r["action"]] for r in srs],
                                  "changed_messages": changed, "tool_messages": sum(m["role"] == "tool" for m in it["messages"]),
                                  "insert_categories": dict(Counter(r["category"] for r in rec_i)),
                                  "str_replace_categories": dict(Counter(r["category"] for r in rec_s))}
    if set(per) != want:
        raise ValueError(f"found {len(per)} of {len(want)} simulated runs in the base inputs")
    gate, seen = {}, set()
    for s, n_want in zip(STAGED, (100, 117)):
        rep = json.loads((Path("/acts") / s / "rerender_report.json").read_text())
        if len(rep) != n_want or not set(rep) <= set(per) or set(rep) & seen:
            raise ValueError(f"{s}: its report's runs are not {n_want} distinct base runs")
        seen |= set(rep)
        both = sorted(rep)
        bad = [k for k in both if rep[k][f"changed_messages {ARM}"] != per[k]["changed_messages"]]
        gate[s] = {"runs_compared": len(both), "mismatches": len(bad)}
        if bad:
            raise ValueError(f"{s}: {len(bad)} runs' changed messages differ from the swap's own report")
    return {"per": per, "gate": gate}


@app.local_entrypoint()
def main(scores_dir: str):
    import numpy as np

    d = Path(scores_dir)
    runs = json.loads((d / "runs.json").read_text())
    t = np.load(d / "tables.npz")
    sim = [i for i, r in enumerate(runs) if r["arm"] == "sim-attack"]
    keys = [runs[i]["key"] for i in sim]
    res = cover.remote(keys)
    per = res["per"]
    kept = {}
    for ri, s, a in zip(t["output_end__run"], t["output_end__step"], t["output_end__action"]):
        kept.setdefault(int(ri), set()).add((int(s), int(a)))
    rows = []
    for i in sim:
        k, p = runs[i]["key"], per[runs[i]["key"]]
        kk = kept.get(i, set())
        if len(kk) != runs[i]["n_output_end_kept"]:
            raise ValueError(f"{k}: {len(kk)} kept output readouts in tables.npz vs {runs[i]['n_output_end_kept']} in runs.json")
        ins, srs = {tuple(x) for x in p["insert"]} & kk, {tuple(x) for x in p["str_replace"]} & kk
        rows.append({"cell": runs[i]["cell"], "main_task": runs[i]["main_task"], "kept_outputs": len(kk),
                     "kept_insert_changed": len(ins), "kept_str_replace_changed": len(srs),
                     "any_changed": len(p["changed_messages"]), "tool_messages": p["tool_messages"],
                     "changed_not_read": len({tuple(x) for x in p["insert"] + p["str_replace"]} - kk)})
    n = len(rows)
    tot_kept = sum(r["kept_outputs"] for r in rows)
    ci = sum(r["kept_insert_changed"] for r in rows)
    cs = sum(r["kept_str_replace_changed"] for r in rows)
    cells_any = sum(1 for r in rows if r["kept_insert_changed"] + r["kept_str_replace_changed"] > 0)
    frac = np.array([(r["kept_insert_changed"] + r["kept_str_replace_changed"]) / r["kept_outputs"] for r in rows if r["kept_outputs"]])
    out = {"base": BASE, "fixes": "diff_with_note (inserts) + sr_gutter_no_newline (str_replace), as the swap's both-fixes arm",
           "gate_vs_swap_reports": res["gate"], "simulated_runs": n,
           "cells_with_a_changed_kept_output": cells_any, "share_of_cells": cells_any / n,
           "cells_with_a_changed_insert": sum(1 for r in rows if r["kept_insert_changed"]),
           "cells_with_a_changed_str_replace": sum(1 for r in rows if r["kept_str_replace_changed"]),
           "kept_tool_outputs": tot_kept, "changed_kept_outputs": ci + cs, "changed_kept_inserts": ci, "changed_kept_str_replaces": cs,
           "share_of_kept_outputs": (ci + cs) / tot_kept,
           "per_run_changed_share": {"median": float(np.median(frac)), "p90": float(np.percentile(frac, 90)), "mean": float(frac.mean())},
           "runs_with_any_changed_message_including_past_the_cut": sum(1 for r in rows if r["any_changed"]),
           "changed_outputs_without_a_kept_readout": sum(r["changed_not_read"] for r in rows),
           "insert_categories": dict(sum((Counter(p["insert_categories"]) for p in per.values()), Counter())),
           "str_replace_categories": dict(sum((Counter(p["str_replace_categories"]) for p in per.values()), Counter()))}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"fix_coverage_POSTHOC_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(out | {"per_run": rows}, indent=1))
    print(json.dumps(out, indent=1))
    print(path.name)
