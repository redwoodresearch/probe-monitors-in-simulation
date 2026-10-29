"""Addendum 1's state-compatibility audit (PLAN.md, @ f8c0777c), score-blind: messages, metadata and text only, never a
probe score. 0d's GO for the audit only, under Aswin's standing under-$5 rule; no pair is read.

One Modal CPU container reads step B's level-1 staged inputs (the runs H1 matched) and the level-1 pack's positions
(keys, kept flags; no scores). It rebuilds H1's identical-call matching (`run_h1_test_swap_modal.py` `cells` @ c1bb941b:
same cell, same function and sorted-JSON arguments hash, k-th occurrence to k-th, bash-family tools), with "both slots
have a stored row" taken from the pack's kept output_end positions instead of stored scores, then applies:
  A2   identical ordered write signatures before the slot (H1's TE_WRITES / BASH_WRITE, `run_h1_swap_modal.py` @ 87ada211);
  A2b  each earlier write's reply identical (level-1 text);
  A3   the two slot replies differ as strings;
  A4   neither slot reply is empty (after strip).
Returns pair IDs (cell, arms, message indices, step/action), the first failed criterion, per-criterion failure counts and
coverage. No text, and no hash of any reply or message, leaves the container; the input files are identified only by
their whole-file sha256 (provenance). It fails, with counts only, unless it reproduces H1's 257 sim|e1 and 296 e2|e1
candidates.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run \\
      experiments/2026-10-11_claimB_swap_changed/run_stepfork_audit_modal.py
"""

import hashlib
import json
import re
from collections import Counter, defaultdict

import modal

STAGED = "2026-10-10_la_fixed-sim-v2_probe-inputs-b-l1_v1"  # step B's level-1 staged inputs (meaning control's STAGED_B)
PACK = "claimB/level1/packedB_2cfd1aec393c/runs.L63.json"
TEST = ["clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"]
SIM, E1, E2 = "sim-attack-fixed", "real-attack-e1", "real-attack-e2"
H1_CANDIDATES = {"sim|e1": 257, "e2|e1": 296}  # h1_test_swap_cells_2026-10-11_143409.json @ f2ba9ac3, pair_counts
TE_WRITES = {"create", "str_replace", "insert"}  # copied from run_h1_swap_modal.py @ 87ada211, line 41
BASH_WRITE = re.compile(r"sed -i|\btee\b|cat\s*>|cat\s*<<|\bpatch\b|(?:echo|printf)[^|\n]*>\s*[^&\s]")  # line 42

app = modal.App("probe-shift_aswin-surya_sub-claimb-sfaudit_kill-2026-11-11")
image = modal.Image.debian_slim(python_version="3.12")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)


def text(c):
    return c if isinstance(c, str) else json.dumps(c, sort_keys=True)


def tool_calls(msgs):
    """Per tool message index: (function, arguments hash, write signatures before it, write replies before it, reply).
    The first three as run_h1_swap_modal.tool_calls (lines 54-68 @ 87ada211), which kept only a write count."""
    out, pending, sigs, replies = {}, [], [], []
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            pending = list(m.get("tool_calls") or [])
        elif m["role"] == "tool" and pending:
            fn = pending.pop(0)["function"]
            a = fn.get("arguments")
            a = json.loads(a) if isinstance(a, str) else a or {}
            name = fn["name"]
            key = hashlib.sha256((name + " " + json.dumps(a, sort_keys=True)).encode()).hexdigest()
            reply = text(m.get("content"))
            out[i] = (name, key, tuple(sigs), tuple(replies), reply)
            cmd = str(a.get("cmd") or a.get("command") or "") if isinstance(a, dict) else ""
            is_write = a.get("command") in TE_WRITES if name == "text_editor" else name.endswith("bash") and bool(BASH_WRITE.search(cmd))
            if is_write:
                sigs.append(key)
                replies.append(reply)
    return out


def step_action(msgs):
    """tool-message index -> (step, action), the reader's counting (run_h1_swap_modal.step_action @ 87ada211)."""
    out, step, action = {}, -1, 0
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            step, action = step + 1, 0
        elif m["role"] == "tool":
            out[i] = (step, action)
            action += 1
    return out


@app.function(image=image, cpu=1, memory=8192, volumes={"/acts": vol}, timeout=20 * 60, retries=0)
def audit() -> dict:
    from pathlib import Path

    shards = sorted(Path(f"/acts/{STAGED}/inputs").glob("shard_*.jsonl"))
    if not shards:
        raise ValueError("no staged input shards")
    sha = {"pack": hashlib.sha256(Path(f"/acts/{PACK}").read_bytes()).hexdigest()}
    runs = {}
    for f in shards:
        raw = f.read_bytes()
        sha[f.name] = hashlib.sha256(raw).hexdigest()
        for line in raw.decode().splitlines():
            r = json.loads(line)
            if r["environment"] in TEST and r["arm"] in (SIM, E1, E2):
                runs[(r["cell"], r["arm"])] = r
    kept = {(r["cell"], r["arm"], p["step"], p["action"]) for r in json.loads(Path(f"/acts/{PACK}").read_text())
            if r["environment"] in TEST for p in r["positions"] if p["kind"] == "output_end" and p["kept"]}
    pairs, dropped, first_fail, any_fail = defaultdict(list), Counter(), defaultdict(Counter), defaultdict(Counter)
    for cell in sorted({c for c, _ in runs}):
        if not all((cell, a) in runs for a in (SIM, E1, E2)):
            continue
        calls = {a: tool_calls(runs[(cell, a)]["messages"]) for a in (SIM, E1, E2)}
        sa = {a: step_action(runs[(cell, a)]["messages"]) for a in (SIM, E1, E2)}
        for a, name in ((SIM, "sim|e1"), (E2, "e2|e1")):
            qa, qb = defaultdict(list), defaultdict(list)
            for i, c in sorted(calls[a].items()):
                qa[c[1]].append(i)
            for j, c in sorted(calls[E1].items()):
                qb[c[1]].append(j)
            for k in sorted(qa.keys() & qb.keys()):
                for i, j in zip(qa[k], qb[k]):
                    if not calls[a][i][0].endswith("bash"):
                        continue
                    if (cell, a, *sa[a][i]) not in kept or (cell, E1, *sa[E1][j]) not in kept:
                        dropped[name] += 1
                        continue
                    x, y = calls[a][i], calls[E1][j]
                    fails = [c for c, bad in (("A2_write_signatures", x[2] != y[2]), ("A2b_write_replies", x[2] == y[2] and x[3] != y[3]),
                                              ("A3_slot_replies_identical", x[4] == y[4]), ("A4_empty_reply", not x[4].strip() or not y[4].strip())) if bad]
                    for c in fails:
                        any_fail[name][c] += 1
                    first_fail[name][fails[0] if fails else "pass"] += 1
                    pairs[name].append({"cell": cell, "main_task": cell.split(":")[0], "environment": runs[(cell, a)]["environment"],
                                        "arm_a": a, "i_a": i, "sa_a": list(sa[a][i]), "i_e1": j, "sa_e1": list(sa[E1][j]),
                                        "pass": not fails, "first_fail": fails[0] if fails else None})
    cand = {k: len(v) for k, v in pairs.items()}
    if cand != H1_CANDIDATES:
        raise ValueError(f"candidates {cand} differ from H1's {H1_CANDIDATES} (dropped {dict(dropped)})")
    out = {"inputs_sha256": sha, "candidates": cand, "dropped_no_kept_row": dict(dropped),
           "first_fail": {k: dict(v) for k, v in first_fail.items()}, "any_fail": {k: dict(v) for k, v in any_fail.items()}, "pairs": pairs}
    for name, ps in pairs.items():
        ok = [p for p in ps if p["pass"]]
        out.setdefault("survivors", {})[name] = {"pairs": len(ok), "main_tasks": len({p["main_task"] for p in ok}),
                                                  "environments": dict(Counter(p["environment"] for p in ok)),
                                                  "pairs_per_task_sum_m2_over_n": (sum(v * v for v in Counter(p["main_task"] for p in ok).values()) / len(ok)) if ok else None}
    return out


@app.local_entrypoint()
def main():
    import subprocess
    from datetime import datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo

    here = Path(__file__).parent
    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    res = audit.remote() | {"label": "addendum 1 state-compatibility audit (score-blind; IDs and counts only)",
                            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=here).stdout.strip(),
                            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = here / "results" / f"stepfork_audit_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(res))
    print(json.dumps({k: res[k] for k in ("candidates", "dropped_no_kept_row", "first_fail", "any_fail", "survivors")}, indent=1))
    print("->", dst)
