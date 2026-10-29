"""POST HOC follow-up to the H3 plant-1 diagnosis (0d's Codex review, item 1; under the standing under-$5 rule): in
which pairs does the re-made plant-1 copy differ from applying the four fixes directly to the staged planted copy, and
why? One Modal CPU container, keys and numbers only (no transcript or command text: only message indices, the editor
operation name, booleans and line indices). For all 150 test plant-1 pairs, with H3's code (run_h3_small_modal.py at
23b5c37a) and B-dev's evening:

- the message indices where the re-made copy and the directly evened copy differ (is it only the planted output?);
- the planted output's tool and editor operation ("create", "view", ... or none for bash);
- which fixes fire on the twin's planted output and on the planted copy's (create newline strip, view empty-last-line
  strip, error-line move), and whether the planted line itself matches the error-line pattern;
- the planted line's index in the re-made and the directly evened output, against the staged after_line.
Category per differing pair: the error-line move reorders around the line; the line itself counted as an error line;
the create or view strip interacting with the line's position; or other.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy \\
      modal run experiments/2026-10-11_claimB_h3_fix_to_plateau/run_POSTHOC_plant1_followup_modal.py
"""

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE) if modal.is_local() else "/root")
import run_h3_small_modal as R  # noqa: E402

H3_RUNNER_SHA = "04c3953b139ec5a32cc2e822ec8d4310aa3583384950a0010ae15bf0d048b77e"
PL1 = "sim-attack-fixed-plant1"
app = modal.App("probe-shift_aswin-surya_sub-claimb-h3p1follow_kill-2026-11-11")
image = R.image.add_local_file(HERE / "run_h3_small_modal.py", "/root/run_h3_small_modal.py")


@app.function(image=image, cpu=4, memory=16384, volumes={"/acts": R.B.P.vol}, timeout=20 * 60, retries=0)
def follow() -> dict:
    from probe_shift.scrub import STDERR_LINE

    if R.SCRIPT_SHA != H3_RUNNER_SHA:
        raise ValueError("H3's runner here differs from the one that ran")
    R.check_b()
    M, B = R.M, R.B
    runs = R.staged()
    man = json.loads((Path("/acts") / R.STAGED_A / "manifest.json").read_text())["plant1"]
    sites, line = {x["key"]: x for x in man["sites"]}, man["line"]
    err_line = re.compile(STDERR_LINE.pattern.replace(r")\n?", ")"), re.I)

    def fired(text, tool, cmd):
        out = {"create_strip": cmd == "create" and text != text.rstrip("\n"), "view_strip": False, "error_move": False}
        if cmd == "view":
            body = text.rstrip("\n")
            out["view_strip"] = B.EMPTY_LAST.sub("", body) != body
        if tool != "text_editor":
            err = [err_line.fullmatch(x) is not None for x in text.split("\n")]
            moved = "\n".join([x for x, e in zip(text.split("\n"), err) if e] + [x for x, e in zip(text.split("\n"), err) if not e])
            out["error_move"] = any(err) and not all(err) and moved != text
        return out
    recs = []
    for cell in sorted(c for (c, a) in runs if a == PL1 and runs[(c, a)]["environment"] in R.TEST):
        twin, copy = runs[(cell, M.SIM)], runs[(cell, PL1)]
        site = sites[twin["key"]]
        i, al = site["msg"], site["after_line"]
        ec, _ = R.evened_version(runs, cell, PL1)
        ed, _, _ = B.evened(copy["messages"])
        diff = [k for k in range(max(len(ec), len(ed))) if k >= len(ec) or k >= len(ed) or ec[k] != ed[k]]
        tool, cmd = B.calls_of(copy["messages"]).get(i, (None, None))
        tool = (tool or "").rsplit("/", 1)[-1]
        f_twin, f_copy = fired(twin["messages"][i]["content"], tool, cmd), fired(copy["messages"][i]["content"], tool, cmd)
        pos_e = [k for k, x in enumerate(ec[i]["content"].split("\n")) if x == line]
        pos_d = [k for k, x in enumerate(ed[i]["content"].split("\n")) if x == line]
        line_is_error = err_line.fullmatch(line) is not None
        if not diff:
            cat = "same"
        elif diff != [i]:
            cat = "other outputs differ"
        elif line_is_error and f_copy["error_move"]:
            cat = "the line itself moved as an error line"
        elif f_twin["error_move"] or f_copy["error_move"]:
            cat = "error-line move reorders around the line"
        elif f_copy["create_strip"] or f_copy["view_strip"]:
            cat = "create/view strip interacts with the line's position"
        else:
            cat = "other"
        recs.append({"cell": cell, "environment": twin["environment"], "diff_messages": diff, "only_planted_output": diff == [i] if diff else None,
                     "tool": tool, "editor_op": cmd if tool.endswith("text_editor") else None, "fixes_on_twin_output": f_twin, "fixes_on_copy_output": f_copy,
                     "line_is_error_line": line_is_error, "line_index_remade": pos_e, "line_index_direct": pos_d, "after_line": al, "category": cat})
    return {"pairs": recs}


@app.local_entrypoint()
def main():
    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    res = follow.remote()
    P = res["pairs"]
    cats = {}
    for r in P:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    diffs = [r for r in P if r["category"] != "same"]
    summary = {"pairs": len(P), "categories": cats, "differing_pairs": len(diffs),
               "differing_only_in_planted_output": sum(1 for r in diffs if r["only_planted_output"]),
               "differing_by_environment": {e: sum(1 for r in diffs if r["environment"] == e) for e in sorted({r["environment"] for r in P})},
               "differing_by_tool": {t: sum(1 for r in diffs if r["tool"] == t) for t in sorted({r["tool"] for r in diffs})}}
    res |= {"label": "POST HOC: plant-1 re-made vs directly evened copies (follow-up item 1; keys and numbers only)", "summary": summary,
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"plant1_followup_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=1))
    for r in diffs:
        print(r["environment"], r["category"], "tool", r["tool"], "op", r["editor_op"], "diff", r["diff_messages"], "idx remade/direct/after", r["line_index_remade"], r["line_index_direct"], r["after_line"])
    print("->", dst)
