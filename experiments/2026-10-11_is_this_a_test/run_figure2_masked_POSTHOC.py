"""Figure 2 (appendix), masked: POST-HOC illustration selected for success-header matches. A copy of
run_figure2_masked.py (1f276cf5) with two changes, made after seeing that script's as-run output, in which both runs'
first insert output was the same out-of-range error message. The as-run figure (figure2_masked.md) stays unchanged
beside this one. This figure illustrates the editor's format only, not typical simulator fidelity.

Approval. 0d, received by df before 10:15 PT 2026-10-12, verbatim: "go on the $0.01 follow-up. It falls under Aswin's 09:16:13 Figure 2
approval ("masked skeletons only, nothing leaves Modal") and his standing "<$5 just run" rule. Aswin is told and can
veto."

The two changes, applied identically to the real and the simulated run:
- **Insert selection:** the first insert output whose first line (literally line 1) fully matches an editor success header
  (HEADERS). If a run has none, that panel is reported as none. The task, the two runs and the str_replace selection
  are as in run_figure2_masked.py: the task is the first eligible one by sha256("20261012|" + task), and its hash must
  equal the as-run one (ASRUN_TASK), else the job stops.
- **Diff file names:** under a diff header, the two lines right after it are kept when each exactly equals its fixed
  template (`--- old_version`, `+++ edited_version`), even if the rest of the diff fails validation.
Masking, trimming, the leak check and what leaves the container are otherwise unchanged (see run_figure2_masked.py).

  PYTHONPATH=src uv run --no-project --with modal modal run experiments/2026-10-11_is_this_a_test/run_figure2_masked_POSTHOC.py
"""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-itat_fig2posthoc_kill-2026-11-12"
LA, INPUTS = "probe-shift_aswin-surya_sub-la-activations", "2026-10-06_la_segment-end-v4_qwen3-32b_v1"
SEED, KEEP, DEADLINE, ASRUN_TASK = "20261012", 7, 780, "4b32b95aa68c"

app = modal.App(APP)
la = modal.Volume.from_name(LA, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12")


HEADERS = [  # whole-line template structures, with the path groups masked: (pattern, gutter kind)
    (re.compile(r"^(The file )(\S+)( has been edited\. Here's the result of running `cat -n` on a snippet of the edited file:)$"), "cat"),
    (re.compile(r"^(The file )(\S+)( has been edited\. Here's the result of running `cat -n` on a snippet of )(\S+)(:)$"), "cat"),
    (re.compile(r"^(The file )(\S+)( has been edited\. Here's a diff showing the changes made:)$"), "diff")]
FOOTERS = {"Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary.",
           "Review the changes and make sure they are as expected. Edit the file again if necessary."}
DIFF_FILES = ["--- old_version", "+++ edited_version"]  # control_tower's diff display: fixed file names
X = lambda s: "".join("x" if ch.isalnum() else ch for ch in s)  # every letter or digit to "x"
CAT = re.compile(r"^ *([1-9][0-9]{0,6})\t")  # Inspect's cat -n: f"{n:6}\t"
HUNK = re.compile(r"^@@ -([0-9]{1,7}),([0-9]{1,7}) \+([0-9]{1,7}),([0-9]{1,7}) @@$")
DIFF = re.compile(r"^ *([1-9][0-9]{0,6})\|([+ -]) ")  # control_tower: f"{n:>4}|{sign} "


def cat_gutters(lines):
    """Gutter end per line if every line is a canonical f"{n:6}\t" with consecutive n, else None."""
    ms = [CAT.match(x) for x in lines]
    if not lines or not all(m and m.group(0) == f"{int(m.group(1)):6}\t" for m in ms):
        return None
    ns = [int(m.group(1)) for m in ms]
    return [m.end() for m in ms] if ns == list(range(ns[0], ns[0] + len(ns))) else None


def diff_gutters(lines):
    """Kept prefix length per line if the lines are the fixed file names, then hunks whose headers are canonical (a start of 0 only with
    length 0) and whose line numbers count up exactly as the hunk header says, else None."""
    if lines[:2] != DIFF_FILES or len(lines) < 3:
        return None
    keep, src, tgt, left = [len(DIFF_FILES[0]), len(DIFF_FILES[1])], 0, 0, (0, 0)
    for x in lines[2:]:
        h, d = HUNK.match(x), DIFF.match(x)
        if h:
            src, ls, tgt, lt = map(int, h.groups())
            if left != (0, 0) or x != f"@@ -{src},{ls} +{tgt},{lt} @@" or (ls and not src) or (lt and not tgt):
                return None
            left = (ls, lt)
            keep.append(len(x))
        elif d and d.group(0) == f"{int(d.group(1)):>4}|{d.group(2)} ":
            n, sign = int(d.group(1)), d.group(2)
            if sign == "+" and n == tgt and left[1] > 0:
                tgt, left = tgt + 1, (left[0], left[1] - 1)
            elif sign == "-" and n == src and left[0] > 0:
                src, left = src + 1, (left[0] - 1, left[1])
            elif sign == " " and n == src and left[0] > 0 and left[1] > 0:
                src, tgt, left = src + 1, tgt + 1, (left[0] - 1, left[1] - 1)
            else:
                return None
            keep.append(d.end())
        else:
            return None
    return keep if left == (0, 0) and len(keep) > 2 else None


def skeleton(text):
    """Template text survives only as a whole matching line: a header with its paths masked, an exact review note, or
    the diff's fixed file names and hunk headers. Gutters survive only under a matching header, and only if every
    line between validates (canonical width, numbers counting up). Everything else alphanumeric becomes "x"."""
    body = text.split("\n")
    if text.endswith("\n"):
        body = body[:-1]
    filled = [i for i, x in enumerate(body) if x.strip()]
    out = [X(x) for x in body]
    if filled:
        first, last = filled[0], filled[-1]
        m, kind = next(((m, k) for m, k in ((pat.match(body[first]), k) for pat, k in HEADERS) if m), (None, None))
        if m:
            out[first] = "".join(X(g) if i % 2 == 1 else g for i, g in enumerate(m.groups()))  # odd groups are paths
            footer = last != first and body[last] in FOOTERS
            if footer:
                out[last] = body[last]
            inner = list(range(first + 1, last if footer else last + 1))
            keep = (cat_gutters if kind == "cat" else diff_gutters)([body[i] for i in inner])
            for i, k in zip(inner, keep or []):
                out[i] = body[i][:k] + X(body[i][k:])
            if kind == "diff" and not keep:  # post-hoc: the fixed file names alone, on exact whole-line equality
                for i, name in zip(inner[:2], DIFF_FILES):
                    if body[i] == name:
                        out[i] = name
    if len(out) > 2 * KEEP + 1:
        out = out[:KEEP] + ["…"] + out[-KEEP:]
    return "\n".join(out) + "\n" + ("⏎ [the output ends with a newline]" if text.endswith("\n") else "∅ [no trailing newline]")


def leaks(skel):
    """How many lines of a finished skeleton hold a letter or digit other than "x", outside an exact review note or
    diff file name, a header with masked path groups, a canonical hunk header, or a canonical gutter."""
    dirty = lambda s: any(ch.isalnum() and ch != "x" for ch in s)
    bad = 0
    for line in skel.split("\n")[:-1]:  # the last line is the trailing-newline marker
        if line in FOOTERS or line in DIFF_FILES:
            continue
        m = next((m for m in (pat.match(line) for pat, _ in HEADERS) if m), None)
        if m:
            bad += any(dirty(g) for i, g in enumerate(m.groups()) if i % 2 == 1)
            continue
        h = HUNK.match(line)
        if h and line == "@@ -{},{} +{},{} @@".format(*map(int, h.groups())):  # numbers only, canonical
            continue
        c, d = CAT.match(line), DIFF.match(line)
        if c and c.group(0) == f"{int(c.group(1)):6}\t":
            line = line[c.end():]
        elif d and d.group(0) == f"{int(d.group(1)):>4}|{d.group(2)} ":
            line = line[d.end():]
        bad += dirty(line)
    return bad


def outputs(msgs, command):
    """(step, output text) for each text_editor call with this command and its tool output, in order."""
    out, step = [], -1
    for j, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        step += 1
        calls = m.get("tool_calls") or []
        tools = []
        for k in range(j + 1, len(msgs)):  # the tool messages right after this turn, one per call
            if msgs[k]["role"] != "tool":
                break
            tools.append(k)
        if len(tools) != len(calls):  # unpaired calls and outputs: skip the turn rather than guess
            continue
        for c, k in zip(calls, tools):
            if msgs[k].get("tool_name") not in (None, c["function"]["name"]) or (c.get("id") and msgs[k].get("tool_call_id") not in (None, c["id"])):
                continue
            args = c["function"].get("arguments") or {}
            args = json.loads(args) if isinstance(args, str) else args
            if c["function"]["name"] == "text_editor" and args.get("command") == command and isinstance(msgs[k].get("content"), str):
                out.append((step, msgs[k]["content"]))
    return out


@app.function(image=image, volumes={"/in": la}, cpu=(2, 2), memory=(8192, 8192), timeout=600, startup_timeout=120,
              retries=0)  # limits equal the requests
def figure2(own_sha: str) -> dict:
    import traceback

    null = os.open(os.devnull, os.O_WRONLY)  # no run text in logs or errors
    os.dup2(null, 1), os.dup2(null, 2)
    try:
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != own_sha:
            raise ValueError("the deployed script is not the launching commit's")
        return _figure2()
    except BaseException as e:
        lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("run_figure2_masked_POSTHOC.py")]
        raise RuntimeError(f"figure2 failed: {type(e).__name__} at run_figure2_masked_POSTHOC.py lines {lines}") from None


def _figure2():
    from collections import Counter

    real_arm, sim_arm = "real-honest", "sim-honest"
    runs, arms = {}, Counter()
    for f in sorted((Path("/in") / INPUTS / "inputs").glob("shard_*.jsonl")):
        for r in map(json.loads, f.read_text().splitlines()):
            if r["arm"] in (real_arm, sim_arm):  # honest runs only; other arms are never read further
                arms[r["arm"]] += 1
                runs.setdefault((r["arm"], r["main_task"]), []).append(r)
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
    counts = {"real honest": arms[real_arm], "simulated honest": arms[sim_arm]}  # constant labels only
    if not (arms[real_arm] and arms[sim_arm]):
        return {"honest runs": counts, "stopped": "the honest arms are not named as expected"}
    tasks = sorted({t for a, t in runs if a == sim_arm} & {t for a, t in runs if a == real_arm},
                   key=lambda t: hashlib.sha256(f"{SEED}|{t}".encode()).hexdigest())
    for t in tasks:
        real = sorted(runs[(real_arm, t)], key=lambda r: r["key"])[0]
        sim = sorted(runs[(sim_arm, t)], key=lambda r: r["key"])[0]
        got = {(arm, cmd): outputs(r["messages"], cmd) for arm, r in (("real", real), ("simulated", sim)) for cmd in ("str_replace", "insert")}
        if all(got.values()):  # the as-run eligibility, so the same task and runs
            if h(t) != ASRUN_TASK:
                return {"honest runs": counts, "stopped": "the eligible task is not the as-run one"}
            success = lambda text: any(p.match(text.split("\n")[0]) for p, _ in HEADERS)  # the literal first line
            pick = {k: v[0] if k[1] == "str_replace" else next((o for o in v if success(o[1])), None) for k, v in got.items()}
            skels = {k: skeleton(v[1]) for k, v in pick.items() if v}
            if any(leaks(s) for s in skels.values()):
                raise ValueError("a skeleton kept a content character")
            fig = {f"{arm} {cmd}": {"run key sha256": h((real if arm == "real" else sim)["key"]), "step": pick[(arm, cmd)][0],
                                    "skeleton": skels[(arm, cmd)]} if pick[(arm, cmd)] else
                   {"run key sha256": h((real if arm == "real" else sim)["key"]), "none": "no insert output with an editor success header"}
                   for arm, cmd in got}
            return {"honest runs": counts, "task sha256": h(t), "selection": "post-hoc illustration selected for success-header matches",
                    "figure": fig}
    return {"honest runs": counts, "stopped": "no paired task has both commands in both runs"}


@app.local_entrypoint()
def main():
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", __file__], capture_output=True, cwd=ROOT).returncode == 0
    if st or not tracked:
        raise SystemExit("commit first: the run must come from committed code")
    call = figure2.spawn(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:  # a 13-min deadline for the whole job, restarts included; the app stops when this launcher exits
        out = call.get(timeout=DEADLINE) | {"git": git}
    except (TimeoutError, modal.exception.TimeoutError):  # get() raises the builtin in modal 1.6.1
        call.cancel()
        raise SystemExit(f"stopped at the {DEADLINE} s deadline")
    Path(ROOT / ".run_cache" / "is_this_a_test").mkdir(parents=True, exist_ok=True)
    (ROOT / ".run_cache" / "is_this_a_test" / f"figure2_posthoc_{git[:8]}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "figure"}, indent=1))
