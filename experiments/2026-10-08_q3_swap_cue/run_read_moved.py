"""Control 5: read the tool outputs whose detector score moved most between as simulated and both fixes, and the
outputs of runs that flipped from missed to caught. Modal CPU (reads the staged texts on the Volume); no model calls.

Picks, from the saved per-call scores (tool-output readout, L63, outer model):
- the N outputs with the largest |both fixes − as simulated| score change, over the swap's 100 runs and the extension;
- every run whose frozen catch at 5% goes from below 0.5 (as simulated) to 0.5 or above (both fixes), with its most
  moved output by the outer model's score (catch comes from the nested models, whose per-output scores are not saved).
For each picked output it returns both texts and what differs: the review note appended to an insert output, the
str_replace output in the real format, no change in the text itself (the score moved through context), or anything
else (a pipeline bug). The texts go to git-ignored data/; results/ gets the classification and scores only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.2' --with numpy modal run \
      experiments/2026-10-08_q3_swap_cue/run_read_moved.py
"""

import gzip
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent  # also valid inside the Modal container (/root)
APP = "probe-shift_aswin-surya_sub-q3swapcue_score_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
N_TOP = 20
SETS = {  # set: (per-call scores, per-run catch, V1 staged inputs and arm, both-fixes staged inputs and arm)
    "swap100": ("swap_frozen_calls_2026-10-08_044006.jsonl.gz", "swap_frozen_runs_2026-10-08_044006.jsonl",
                ("2026-10-07_q2_swap_inputs_v1", "sim-attack"), ("2026-10-08_q2_swap_combined_inputs_v1", "sim-attack-note-sr-both")),
    "ext117": ("swap_frozen_calls_2026-10-08_130738.jsonl.gz", "swap_frozen_runs_2026-10-08_130738.jsonl",
               ("2026-10-06_la_segment-end_qwen3-32b_v1", "sim-attack"), ("2026-10-08_q2_swap_ext_inputs_v1", "sim-attack-note-sr-both")),
}
MANIFEST = "swap_ext_manifest_2026-10-08.json"  # #217: the extension's scoring files are on S3, not in git
REVIEW = "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=2, memory=16384, timeout=1200, volumes={"/acts": acts_vol.read_only()})
def read(requests: list) -> list:
    """requests: [set, key, step, action, v1 (inputs, arm), fixed (inputs, arm)]; returns both outputs and the call"""
    want = {}
    for r in requests:
        for inputs, arm in (r[4], r[5]):
            want.setdefault(inputs, set()).add((arm, r[1]))
    staged = {}
    for inputs, keys in want.items():
        for shard in sorted((Path("/acts") / inputs / "inputs").glob("shard_*.jsonl")):
            for line in shard.read_text().splitlines():
                it = json.loads(line)
                if (it["arm"], it["key"]) in keys:
                    staged[(inputs, it["arm"], it["key"])] = it["messages"]

    def output(msgs, step, action):
        s, a, call = -1, 0, None
        for m in msgs:
            if m["role"] == "assistant":
                s, a, calls = s + 1, 0, m.get("tool_calls") or []
            elif m["role"] == "tool":
                if (s, a) == (step, action):
                    c = calls[a]["function"] if a < len(calls) else {}
                    args = c.get("arguments")
                    args = json.loads(args) if isinstance(args, str) else (args or {})
                    return m["content"] if isinstance(m["content"], str) else json.dumps(m["content"]), c.get("name"), args.get("command")
                a += 1
        raise KeyError((step, action))

    out = []
    for r in requests:
        v1, tool, command = output(staged[(r[4][0], r[4][1], r[1])], r[2], r[3])
        fx, _, _ = output(staged[(r[5][0], r[5][1], r[1])], r[2], r[3])
        out.append({"set": r[0], "key": r[1], "step": r[2], "action": r[3], "tool": tool, "command": command, "v1": v1, "fixed": fx})
    return out


def kind(o):
    """the change must be exactly q2's: the note appended to an insert output (diff_with_note), or q2's str_replace
    rendering of this very output (str_replace_tells.variant_output, sr_gutter_no_newline)"""
    sys.path.insert(0, str(REPO / "experiments/2026-10-07_q2_rendering_swap"))
    import str_replace_tells as st

    if o["v1"] == o["fixed"]:
        return "unchanged text (context)"
    if o["tool"] == "text_editor" and o["command"] == "insert" and o["fixed"] == o["v1"].rstrip("\n") + "\n" + REVIEW:
        return "insert: review note appended"
    if o["tool"] == "text_editor" and o["command"] == "str_replace" and o["fixed"] == st.variant_output(o["v1"], "sr_gutter_no_newline")[0]:
        return "str_replace: real format"
    return "other change (unexpected)"


def local(name: str) -> Path:
    """results/<name>; if it is not in git, fetched once from S3 through #217's manifest into git-ignored data/ and
    checked by sha256."""
    import subprocess

    path = HERE / "results" / name
    if path.exists():
        return path
    path = REPO / "data" / "q3_swap_cue" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = next((f for f in json.loads((HERE / "results" / MANIFEST).read_text())["files"] if f["file"] == name), None)
    if entry is None:
        raise FileNotFoundError(f"{name}: not in results/ and not in {MANIFEST}")
    if not path.exists():
        subprocess.run(["aws", "s3", "cp", "--profile", "redwood-sandbox", "--quiet", entry["s3"], str(path)], check=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:  # a fresh download or the cached copy
        raise ValueError(f"{name}: {path} differs from {MANIFEST}'s sha256; delete it to fetch again")
    return path


EDITOR_COMMANDS = {"view", "create", "str_replace", "insert", "undo_edit"}  # only these reach results/: a shell command is text


@app.local_entrypoint()
def main():
    picks, flips, scores = [], [], {}
    for name, (calls_f, runs_f, v1, fx) in SETS.items():
        calls = {}
        with gzip.open(local(calls_f), "rt") as f:
            for line in f:
                c = json.loads(line)
                if c["readout"] == "output_end" and c["label"] in ("V1", "COMB"):
                    calls.setdefault((c["key"], c["step"], c["action"]), {})[c["label"]] = c["score"]
        for (k, s, a), v in calls.items():
            if "V1" in v and "COMB" in v:
                scores[(name, k, s, a)] = (v["V1"], v["COMB"], v["COMB"] - v["V1"])
        runs = {}
        for line in local(runs_f).read_text().splitlines():
            r = json.loads(line)
            runs.setdefault(r["key"], {})[r["label"]] = r["caught"]["output_end"]["0.05"]
        for k, v in runs.items():
            if v.get("V1") is not None and v.get("COMB") is not None and v["V1"] < 0.5 <= v["COMB"]:
                best = max(((s, a) for (n, kk, s, a) in scores if n == name and kk == k), key=lambda sa: abs(scores[(name, k, *sa)][2]))  # largest absolute movement
                flips.append([name, k, *best, list(v1), list(fx), v["V1"], v["COMB"]])
    top = sorted(scores, key=lambda x: -abs(scores[x][2]))[:N_TOP]
    for name, k, s, a in top:
        picks.append([name, k, s, a, list(SETS[name][2]), list(SETS[name][3])])
    texts = read.remote(picks + [f[:6] for f in flips])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rows = []
    for i, o in enumerate(texts):
        group = "top_moved" if i < len(picks) else "flipped_run"
        v1s, fxs, d = scores[(o["set"], o["key"], o["step"], o["action"])]
        rows.append({"group": group, "set": o["set"], "key": o["key"], "step": o["step"], "action": o["action"], "tool": o["tool"],
                     "command": o["command"] if o["tool"] == "text_editor" and o["command"] in EDITOR_COMMANDS else None,
                     "change": kind(o), "score_v1": v1s, "score_fixed": fxs, "delta": d,
                     "chars_v1": len(o["v1"]), "chars_fixed": len(o["fixed"]),
                     **({"catch_v1": flips[i - len(picks)][6], "catch_fixed": flips[i - len(picks)][7]} if group == "flipped_run" else {})})
    data = REPO / "data" / f"read_moved_{stamp}.jsonl"  # the texts: git-ignored
    data.parent.mkdir(exist_ok=True)
    with data.open("x") as f:
        f.writelines(json.dumps(o) + "\n" for o in texts)
    summary = {"stamp": stamp, "n_top": N_TOP, "n_flipped_runs": len(flips),
               "change_counts": {g: dict(Counter(r["change"] for r in rows if r["group"] == g)) for g in ("top_moved", "flipped_run")},
               "texts_file": {"path": str(data.relative_to(REPO)), "sha256": hashlib.sha256(data.read_bytes()).hexdigest()},
               "rows": rows}
    with (HERE / "results" / f"read_moved_{stamp}.json").open("x") as f:
        f.write(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
