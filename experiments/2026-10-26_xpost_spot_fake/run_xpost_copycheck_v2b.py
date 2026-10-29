"""X post spot-the-fake v2, copy-check addendum (v2b): the drawn pair's simulated output B against EVERY `ls -la /app`-style listing in
the simulator's reference run. NOT RUN.

Why: v2's copy check matched only the byte-identical command `ls -la /app` and found COMMAND NOT IN REFERENCE; the reference probably
ran `ls -la /app/`, which prints the same listing. Scope, from Writing's message of 2026-10-26 ~20:55 PT (Aswin's request through a
reviewer; kit section 1a item 7), fixed before any output is read:
  - in each reference candidate v2 checked for the drawn task, find every shell call whose command is an `ls` whose target is `/app` or
    `/app/` and that has both -l and -a in any order (-la, -al, -l -a);
  - for each, read only that call's output and diff it against B with v2's visible-line rule (visible_diff: right-strip, drop blank
    lines, compare by position, each extra line counts as one);
  - report keys and counts only: the matches, each one's step index and number of differing visible lines (0 = identical), the
    reference run key, the script commit.
Interpretation (written before the run):
  - "an ls command": the whole shell command, with no shell operator, redirection, substitution or glob character, split with shlex, is
    `ls` plus its arguments, parsed GNU-style: short-option clusters are pooled, so -la, -al, -l -a and -lah all count; option values
    (-w 80, --block-size 1) are skipped; `--` ends the options; other long options are ignored, and neither --all nor -A counts as -a.
    The only non-option argument must be exactly `/app` or `/app/`.
  - Commands that do not parse that way but contain an `ls` word and the text `/app` (e.g. `cd /app && ls -la`, `ls -la /app | head`)
    are NOT in scope; they are counted and listed by index as near misses, so a miss is visible. Their outputs are not read.
  - "step index": tool_call_index = the call's 0-based position among all of the run's tool calls, in order; message_index = the
    position of the assistant message that issued it.
  - An unanswered matching call is reported as unanswered (no diff).
B is re-read from the v2 record's simulated run and must hash to the v2 record's digests (command and sim_output) before anything else
is compared. Reuse, by import and unchanged: run_xpost_spot_fake_v2.py (honest_gate, build_pool, load, checked_messages, call_fields,
shell_cmd, call_output, read_run, visible_diff, u8) at the blob pinned below. No reference or simulated text leaves the container; the
laptop writes one keys-and-counts record create-only to results/ and prints one status line.

  cd <worktree> && uv run --no-project --python 3.12 --with modal==1.6.1 modal run \\
      experiments/2026-10-26_xpost_spot_fake/run_xpost_copycheck_v2b.py --go
SPEND, worst-case bound: 1 call x (startup_timeout 300 + timeout 1800 + scaledown_window 2) s = 2,102 s at 2 cores and 8 GiB =
2,102 x $0.00004396 = $0.09240392. Typical: under $0.01.
"""

import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # /root in the container: no parents[N] there
sys.path.insert(0, str(HERE))
import run_xpost_spot_fake_v2 as V  # noqa: E402

APP = "probe-shift_aswin-surya_sub-xpost_copycheck-v2b_kill-2026-11-25"
DIR = V.DIR
V2_SHA256 = "684f683ea032e2a91b55d171fc6e69ed5f54c6803a5b04794685d7547edf7c18"  # the v2 record's script.sha256 (blob 98e6dd33 at aa0af62ad)
V2_RECORD = HERE / "results" / "xpost_spotfake_draw_v2_2026-10-26_163406.json"
TARGETS = ("/app", "/app/")

image = V.image.add_local_file(HERE / "run_xpost_spot_fake_v2.py", "/root/run_xpost_spot_fake_v2.py")
app = modal.App(APP, image=image)


SHELL_META = re.compile(r"[;&|<>`$(){}\[\]*?~\\\n]")  # any of these: not a plain ls command (Codex v2b x1)
LONG_WITH_ARG = {"--block-size", "--format", "--hide", "--ignore", "--indicator-style", "--quoting-style", "--sort", "--time",
                 "--time-style", "--tabsize", "--width"}  # GNU ls long options whose value may be the next word
SHORT_WITH_ARG = "ITw"  # GNU ls short options that take a value (rest of the cluster, else the next word)


def in_scope(cmd: str):
    """The matched target (/app or /app/) if the command is one plain `ls` with -l and -a and exactly that one target, else None.
    GNU-style parsing: short clusters pooled; option values skipped; `--` ends options; --all is not counted as -a (rule: -a)."""
    if SHELL_META.search(cmd):
        return None
    try:
        toks = shlex.split(cmd)
    except ValueError:
        return None
    if not toks or toks[0] != "ls":
        return None
    letters, targets, i, opts_done = set(), [], 1, False
    while i < len(toks):
        a = toks[i]
        i += 1
        if opts_done or not a.startswith("-") or a == "-":
            targets.append(a)
        elif a == "--":
            opts_done = True
        elif a.startswith("--"):
            i += a in LONG_WITH_ARG  # "--opt value"; "--opt=value" is one word
        else:
            for j, ch in enumerate(a[1:]):
                if ch in SHORT_WITH_ARG:
                    i += j == len(a) - 2  # the value is the next word only when the option ends the cluster
                    break
                letters.add(ch)
    return targets[0] if {"l", "a"} <= letters and len(targets) == 1 and targets[0] in TARGETS else None


def near_miss(cmd: str) -> bool:
    """Not in scope, but an `ls` word and the text /app both appear: counted and listed by index; its output is not read."""
    return "/app" in cmd and "ls" in re.split(r"[\s;&|()]+", cmd)


def scan_reference(s: dict, sim_text: str) -> dict:
    msgs = V.checked_messages(s)
    n_calls, matches, misses = 0, [], []
    for i, m in enumerate(msgs):
        calls = m.get("tool_calls") if m["role"] == "assistant" else None
        if calls is None or calls == []:
            continue
        if not isinstance(calls, list):
            raise V.X.Gate("an assistant message's tool_calls are malformed")
        for tc in calls:
            fn, args, cid = V.call_fields(tc)
            k, n_calls = n_calls, n_calls + 1
            c = V.shell_cmd(fn, args)
            if c is None:
                continue
            tgt = in_scope(c)
            if tgt is not None:
                out = V.call_output(msgs, i, cid)
                row = {"tool_call_index": k, "message_index": i, "target": tgt,
                       "command_sha256": hashlib.sha256(V.u8(c)).hexdigest(), "answered": out["answered"]}
                if out["answered"]:
                    idx = V.visible_diff(sim_text, out["text"])
                    row |= {"n_differing": len(idx), "differing_indices": idx, "identical": not idx, "error_field": out["error_field"]}
                matches.append(row)
            elif near_miss(c):
                misses.append({"tool_call_index": k, "message_index": i})
    return {"n_tool_calls": n_calls, "n_matches": len(matches), "matches": matches, "n_near_misses": len(misses), "near_misses": misses}


@app.function(**V.FN)
def check(p: dict) -> dict:
    try:
        t = p["task"]
        V.honest_gate([t], p["run1_launches"])
        sim = V.read_run(t["sim"], "sim")
        if sim.get("first") != "shell" or not sim.get("answered"):
            raise SystemExit("STOP: the simulated run's first call is not an answered shell call")
        got = {"command": hashlib.sha256(V.u8(sim["command"])).hexdigest(), "sim_output": hashlib.sha256(V.u8(sim["text"])).hexdigest()}
        if any(got[k] != p["digests"][k] for k in got):
            raise SystemExit("STOP: B does not hash to the v2 record's digests")
        out = []
        for c in t["refs"]["candidates"]:
            row = {"eval_sha256": c["eval_sha256"], "pinned": c["pinned"], "records": c["records"]}
            loaded = None
            for k in c["keys"]:
                try:
                    loaded = (k, *V.load({"key": k, "eval_sha256": c["eval_sha256"], "task": t["task"], "pinned": c["pinned"]}, "ref"))
                    break
                except V.X.Gate as e:
                    if not str(e).startswith("0 .eval files"):
                        raise
            if loaded is None:
                raise V.X.Gate(f"none of the candidate's {len(c['keys'])} keys has its .eval under the eval roots")
            key, vol, path, s = loaded
            out.append(row | {"reference_run_key": key, "volume": vol, "eval": path} | scan_reference(s, sim["text"]))
        return {"b_digests_match_v2": True, "candidates": out}
    except SystemExit as e:
        raise SystemExit(str(e)) from None
    except V.X.Gate as e:
        raise SystemExit(f"gate: {e}") from None
    except Exception as e:  # noqa: BLE001 -- a parse error can quote source text: only its type leaves
        raise SystemExit(f"stopped: {type(e).__name__} (message withheld)") from None


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True, text=True).stdout.strip()


@app.local_entrypoint()
def main(go: bool = False):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if not go:
        raise SystemExit("--go only (after review)")
    me = f"{DIR}/{Path(__file__).name}"
    if not git("ls-files", me) or subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", me]).returncode != 0:
        raise SystemExit("commit this script before launch: the record names the commit it ran from")
    if hashlib.sha256((HERE / "run_xpost_spot_fake_v2.py").read_bytes()).hexdigest() != V2_SHA256:
        raise SystemExit("run_xpost_spot_fake_v2.py differs from the script the v2 draw ran")
    v2 = json.loads(V2_RECORD.read_text())
    d = v2["draw"]
    if (d["task"], d["simulated_is"], v2.get("status")) != ("t384_tiktok", "B", "drawn"):
        raise SystemExit("STOP: the v2 record is not the drawn t384_tiktok pair with the simulated output as B")
    pool = V.build_pool()
    t = next(x for x in pool["tasks"] if x["task"] == d["task"])
    if t["sim"]["key"] != d["simulated_run_key"] or t["real"]["key"] != d["real_run_key"] or \
            [c["eval_sha256"] for c in t["refs"]["candidates"]] != [c["eval_sha256"] for c in v2["copy_check"]["candidates"]]:
        raise SystemExit("STOP: the rebuilt pool's task entry differs from the v2 record (keys or reference candidates)")
    res = check.remote({"task": t, "run1_launches": pool["run1_launches"], "digests": d["digests"]})
    rec = {"what": "x-post spot-the-fake v2b copy-check addendum: keys and counts only (no text)",
           "stamp_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"),
           "scope": "every ls with -l and -a whose only target is /app or /app/, per v2 reference candidate; visible-line diff vs B",
           "v2_record": {"path": f"{DIR}/results/{V2_RECORD.name}", "sha256": hashlib.sha256(V2_RECORD.read_bytes()).hexdigest()},
           "task": d["task"], "simulated_is": d["simulated_is"],
           "simulated_run_key": d["simulated_run_key"],
           "any_identical": any(m.get("identical") for c in res["candidates"] for m in c["matches"]), **res,
           "script": {"path": me, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "head": git("rev-parse", "HEAD"),
                      "committed_at_head": True, "app": APP}}  # gated at launch
    out = HERE / "results" / f"xpost_copycheck_v2b_{rec['stamp_pt']}.json"
    ks = V.write_new(out, rec, 0o644)
    print(json.dumps({"status": "DONE", "any_identical": rec["any_identical"],
                      "n_matches": [c["n_matches"] for c in res["candidates"]], "keys_record": str(out), "sha256": ks}), flush=True)
