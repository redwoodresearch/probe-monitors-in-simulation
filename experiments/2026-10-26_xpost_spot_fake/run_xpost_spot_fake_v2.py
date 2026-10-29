"""X post "spot the fake", rule v2: ONE real-versus-simulated pair of HONEST-run first tool outputs, plus a copy check. NOT RUN.

Rule: x_post_kit_drawrule_v2_frozen_160633.md section 1a (RULE_FILE, sha256 RULE_SHA256, checked at launch). It is rule v1 plus the
v1 NOTE's clarifications, with v1's draw void (a trailing blank line counted as a differing line). This is
run_xpost_spot_fake.py (v1 @ 46fad2a50) with only these changes:
  1. Differ counts VISIBLE lines (visible_diff): each line is right-stripped, blank lines are dropped, the rest are compared
     position by position, and each extra line in the longer output counts as one. At least 2 are required. The 4-12 line and
     <= 64 character limits still apply to the RAW outputs, as in v1.
  2. Seed label "x-post spot-the-fake v2"; the same sort, take-first and A/B rule (rng.random() < 0.5 -> the REAL output is "A").
  3. COPY CHECK, always run on the drawn pair, inside the render call (copy_check).
     - The reference is the real honest run the simulator was given for the task. It is the sha256 PINNED by the sim launches'
       committed reference records (POINTER_RECORDS): the T196 merged references (references[task].sha256), and the generation
       manifests of the +311 / newmains / sp493 / t196fill launches (pinned cells' eval_sha256).
     - A task with no pinned record falls back to the registry's honest_a row(s) for it (pools_ep2 excluded, as for the pool).
     - Every candidate is honest-gated (honest_a, a real-honest file name, a run-1 launch, no "attack") and path-checked before
       any read, exactly like the real side. One exception: a pinned sha256's byte-identical 'references/<task>.eval' copy is
       also accepted, but only for a pinned sha256.
     - In each candidate, the FIRST tool call whose shell command is byte-identical to the drawn command gives the reference
       output. The SIMULATED output is diffed against it with the visible-line rule.
     - Per candidate, the result is IDENTICAL, DIFFERS IN N LINES (indices only), or COMMAND NOT IN REFERENCE. Rarer results are
       REFERENCE CALL UNANSWERED and UNREADABLE (a structural reason; nothing is read past a failed gate).
     - With several candidates (REFERENCE AMBIGUOUS) every one is checked, per Writing's ~16:20 change. Overall:
       copy_identical_to_any is true, false, or null (no candidate, or an unreadable one and no IDENTICAL).
     - No reference text leaves the container.
  4. NO redraws. An empty eligible pool writes the keys record and stops cleanly, with no draw.
Kept from v1: honest_gate before any read; locate's pre-read path gate (byte-identical copies allowed); the real exit code bound to
the first call's own execution window; fixed tool categories; malformed -> unreadable -> STOP before the draw.
Text. scan returns keys, counts, booleans and fixed categories only. render returns the drawn command, its two outputs and the copy
check's results (no reference text), after a hash check against scan. The laptop writes the text ONLY to a create-only file under
TEXT_DIR (outside git). It writes a keys-and-counts record (no text) create-only to results/, and prints one status line. Nothing
scans for credentials: Aswin reads both outputs by eye before posting.
Reuse, by import and unchanged: run_tables_v81.py at blob 88589b50 (EVAL_ROOTS, VOLS) and run_examples_v81.py at blob f191a247
(sample_of / read_member, no_attachment_refs, Gate). Both are copied from PIN_SOURCE and checked by blob at launch.

  cd <worktree> && uv run --no-project --python 3.12 --with modal==1.6.1 modal run \\
      experiments/2026-10-26_xpost_spot_fake/run_xpost_spot_fake_v2.py --go
Run it without --detach: the laptop's entrypoint writes both files. It should take about 5 minutes.
SPEND, worst-case bound: 37 calls (36 scan chunks of 24 of the 864 tasks, plus 1 render) x (startup_timeout 300 + timeout 1800 +
scaledown_window 2) s = 77,774 s at 2 CPU cores and 8 GiB. At Modal list prices ($0.0000131 per core-second, $0.00000222 per
GiB-second) that is 77,774 x $0.00004396 = $3.41894504. Typical: about $0.4 ($0.5 with a 30% buffer). 0d writes the SPEND row.
"""

import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # /root in the container: no parents[N] there
sys.path.insert(0, str(HERE))
import run_tables_v81 as RT  # noqa: E402  (blob 88589b50: EVAL_ROOTS, VOLS)

X = RT.X  # run_examples_v81, blob f191a247: sample_of, read_member, no_attachment_refs, Gate

APP = "probe-shift_aswin-surya_sub-xpost_spotfake-v2_kill-2026-11-25"
DIR = "experiments/2026-10-26_xpost_spot_fake"
RULE_FILE = Path("/Users/aswinsurya/probe-shift-writing-handoff-2026-10-15/w23/x_post/x_post_kit_drawrule_v2_frozen_160633.md")
RULE_SHA256 = "828731f774d1497b70733aa379799678aed4a2182db17178bdd6ddde1c98a458"
PIN_SOURCE = "origin/aswin/routec-fa-honest-render @ 0bb2083224e3e51d2d3713d37fd65fa734df7304, experiments/2026-10-16_examples_v81/"
PIN_BLOBS = {"run_tables_v81.py": "88589b507a8439d4abb85be40a1d60f4590fd34d", "run_examples_v81.py": "f191a2477226f868633ec2c5f6092ebb85d48fea"}
REGISTRY = ("1a842a85635bd63a8bfc60071195584834117c11", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_2026-10-17_233212.json",
            "1e1868586fe8a786f6e1094b9d5b62e362ec0e2ce241e9b0103d6584af351126")  # (commit, path, sha256)
POINTER_RECORDS = {  # the sim-honest launches' committed reference pins (keys and sha256s only)
    "merged_refs": ("572f67dbb64dab626338e92d6ca8833d38d44bc4",
                    "experiments/2026-10-14_q3_q35_simhonest_purchase/results/merged_refs_launch_2026-10-14.json",
                    "26487d40031765d91f195f6086247522c1b2e4a3235902d2a8036a7a4d31409b"),  # (commit, path, sha256)
    "generation_manifests": ("7935e9e0be05170e8946112a1e21fa039d2db951", "experiments/2026-10-15_q3_simhonest_311/results", 36,
                             "297e1c8ae966983f32ab41576b8aa07abd5a4675692920364d615cf1f92478bb")}  # (commit, dir, n manifest_simh_*.json,
#   sha256 of their bytes concatenated in name order)
EXCLUDED_INVENTORIES = {"pools_ep2": "epoch-2 replicate honest runs (honest_b_e2 / honest_a), not run 1's own launches"}
REAL_WORLD, SIM_WORLD, REF_WORLD = "honest_b", "sim_honest", "honest_a"
SHELL_TOOL = "bash"  # control_tower bash_with_timeout's registered name, @tool(name="bash") (control_tower 3.0.0, tools/bash_tools.py L17)
ROLES = ("system", "user", "assistant", "tool")  # Inspect's four ChatMessage roles
LABEL = b"x-post spot-the-fake v2"
SEED = int(hashlib.sha256(LABEL).hexdigest(), 16)
MIN_LINES, MAX_LINES, MAX_WIDTH, TABSIZE, MIN_DIFF = 4, 12, 64, 4, 2
CHUNK = 24  # tasks per scan call (2 .eval files each)
TEXT_DIR = Path("/Users/aswinsurya/probe-shift-reruns-2026-10-20/xpost_draw")
STEPS = ("pairs_found", "both_first_calls_shell", "commands_identical", "lines_and_width_fit", "differ_ge2_visible_lines", "no_error",
         "real_exit_recorded")
PARSE = {
    "first tool call": "the messages must be a list of dicts, each with one of Inspect's four roles. The first call is tool_calls[0] of "
                       "the first assistant message whose tool_calls is a non-empty list (the walk in "
                       "analyze_POSTHOC_streaks_errors.transcript, experiments/2026-10-08_q3_onpolicy_figures @ 378c9af57; sce "
                       "src/sim/adapters/inspect_eval.py _tool_calls @ e561f936). That call must be a dict with a non-empty string "
                       "function, a dict of arguments and a non-empty string id. Anything else is MALFORMED (unreadable, STOP). No "
                       "such assistant message means 'no tool call' (category 'absent')",
    "shell command": "function == 'bash' with a non-empty string arguments['cmd'], else arguments['command']: the value the tool runs "
                     "(control_tower bash_tools.py L42, `actual = cmd or command`). A bash call without one is 'not shell'. "
                     "Byte-identical means equal UTF-8 bytes. Other arguments (timeout) are not compared",
    "output (stdout)": "the ONE role-'tool' message whose tool_call_id is the call's id. Several such messages are MALFORMED. The "
                       "content must be a string, or a list of {'type': 'text', 'text': <string>} parts joined by '\\n' as Inspect's "
                       "ChatMessage.text does; any other structure is MALFORMED. When the error field is set (it must be a dict whose "
                       "message, if present, is a string; else MALFORMED), the output is error.message, falling back to the content "
                       "(sce inspect_eval.tool_result_text @ e561f936). Real: Inspect's bash returns stderr + '\\n' + stdout when "
                       "stderr is non-empty (inspect_ai 0.3.251 tool/_tools/_execute.py L108-112), so this is the merged text the "
                       "agent saw. Simulated: the simulator's text, returned unchanged (sce src/runner/simulate/tools.py _wrap_tool "
                       "@ e561f936), with no stdout/stderr split",
    "unanswered": "no tool message answers the call. This is legitimate only if no assistant message follows the call's message (the "
                  "run ended mid-call); the pair then fails the lines-and-width step (counted). If the run went on, it is MALFORMED",
    "exit code (real side only)": "the call's execution window runs from the ONE model event whose output.choices[*].message.tool_calls "
                                  "carries the call's id (the path run_examples_v81.arguments_at reads, L336-343) up to the next model "
                                  "event; nested events are included. The events must be a list. Zero or several issuing model events "
                                  "are MALFORMED. In the window, the sandbox events with action 'exec' and cmd 'bash --login -c ' + "
                                  "shlex.quote(command) are counted; that is the form Inspect logs (inspect_ai 0.3.251 "
                                  "util/_sandbox/events.py L96-110) for bash's exec(['bash', '--login', '-c', command]) (_execute.py "
                                  "L105-107). Exactly one, with an integer result, is the recorded exit code. A matching event without an "
                                  "integer result is MALFORMED. None or several means 'not recorded'. A later repeat of the command, "
                                  "outside the window, never counts",
    "simulated exit code": "not read: the simulator records none (NOTE §1)",
    "lines and width": "on the RAW output: one trailing '\\n' removed, split on '\\n' (empty output has 0 lines); "
                       "width = len(line.expandtabs(4))",
    "differ (rule v2)": "visible lines only: split on '\\n', each line right-stripped (str.rstrip()), lines empty after that dropped; "
                        "positions 0 .. max(n_real, n_sim) - 1 where the visible lines differ or one side has none. At least 2. The "
                        "raw positional count (v1's rule) is kept per task as raw_lines_differing, for audit only",
    "error": "real: error field set, OR whitespace-only output, OR a recorded exit code other than 0. Simulated: error field set, OR "
             "whitespace-only output (NOTE §1)",
    "real exit recorded": "after the no-error step, a pair whose real exit code is 'not recorded' is excluded and counted (NOTE §3 as "
                          "amended by Writing, 10-26)",
    "tool category": "only fixed categories leave the container: 'bash', 'other' (any other function name) or 'absent' (no call)",
    "candidate paths": "for each EVAL_ROOT x {key path, 'runs/' + key path}, as run_tables_v81.find_eval_v2, before any read: the resolved "
                       "path must be inside the resolved root; its path relative to the root must equal the literal one (so no "
                       "symlink redirects it); it must pass the side's honest layout and file-name check; it must carry no 'attack'; "
                       "and it must be a regular file. Then it is read and its sha256 compared. Byte-identical copies with the "
                       "pinned sha256 are allowed; the first by (volume, path) is used",
    "reference candidates": "pinned: every sha256 the sim launches' committed records pin for the task (merged_refs references[task]; "
                            "generation-manifest cells with status 'pinned'), each with its keys in this order: the registry rows "
                            "carrying that sha256, then the record's reference_source and reference_copy paths as keys (from the launch "
                            "folder before 'honest_a' or 'references'). A recorded eval_task must be '...-real-honest-...'. Fallback, "
                            "only if no record pins the task: the registry's honest_a rows for it (pools_ep2 excluded)",
    "copy check": "per candidate: the first key whose .eval is found with the sha256 (each key path-gated before any read), its one "
                  "sample, the FIRST tool call (any assistant message, any position in tool_calls) that is a shell call whose command "
                  "is byte-identical to the drawn command, that call's output as above, then the visible-line diff of the SIMULATED "
                  "output against it",
}
AMBIGUITIES = {
    "stdout": "real tool text merges stderr into stdout; simulated text has no split. Both sides use the text the agent saw (NOTE §2)",
    "seed": "int(sha256(b'x-post spot-the-fake v2').hexdigest(), 16) (rule v2 §1a, as v1's NOTE §4 for v1)",
    "A/B bit": "rng.random() < 0.5 -> real is A (NOTE §5). For the same state that equals rng.getrandbits(1) == 0 (both read the top "
               "bit of the next 32-bit output)",
    "tabs": "str.expandtabs(4), with tab stops every 4 columns, not each tab replaced by 4 spaces (NOTE §6)",
    "lines": "the raw lines (limits) drop one trailing newline; the visible lines (differ, copy check) are right-stripped and "
             "drop blank lines; both are compared by position, not by an alignment diff",
    "task key": "the inventory row's `task` field. Two honest_b runs sit in '<task>_redraw1' folders under that task",
    "pool scope": "every pinned inventory except pools_ep2. This includes S_paper493 and pools_fill sim-honest rows (pools_fill is "
                  "labelled 'A25 fill-in (POST HOC)')",
    "reference copies": "for t389 / t309 / t325 the registry's only honest_a row, and for some sp493 / t196fill cells the record's only "
                        "path, is the launch's 'references/<task>.eval' copy, whose file name has no '-real-honest-'. It is accepted "
                        "only for a sha256 a committed record pins as the task's reference (records vouch eval_task '-real-honest-')",
    "redraw": "none (rule v2 §1a: no redraws of any kind)",
}

image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("zstandard==0.25.0")
         .add_local_file(HERE / "run_tables_v81.py", "/root/run_tables_v81.py")
         .add_local_file(HERE / "run_examples_v81.py", "/root/run_examples_v81.py"))
app = modal.App(APP, image=image)
VOL_MOUNTS = ("/q4", "/acts", "/q3")
if {"/" + r.split("/")[1] for r in RT.EVAL_ROOTS} != set(VOL_MOUNTS):
    raise SystemExit("VOL_MOUNTS must be exactly the mounts of run_tables_v81.EVAL_ROOTS")
vols = {k: modal.Volume.from_name(RT.VOLS[k]).read_only() for k in VOL_MOUNTS}
FN = dict(cpu=(2, 2), memory=(8192, 8192), timeout=1800, startup_timeout=300, scaledown_window=2, retries=0, max_containers=40,
          volumes=vols)


def layout_ok(parts, side: str, task: str = "", pinned: bool = False) -> bool:
    """The side's honest layout for a key's or a resolved path's parts: <launch>/<arm folder>/<task folder>/<file>.eval, or (side
    'ref', pinned sha256 only) <launch>/references/<task>.eval. No part may be empty, '.' or '..', and none may carry 'attack'."""
    parts = list(parts)
    if not parts or any(p in ("", ".", "..") or "attack" in p.lower() for p in parts) or not parts[-1].endswith(".eval"):
        return False
    if side == "ref":
        if "honest_c" in parts[0] or "ep2" in parts[0]:
            return False
        if len(parts) == 3:
            return pinned and parts[1] == "references" and parts[2] == f"{task}.eval"
        return len(parts) == 4 and parts[1] == REF_WORLD and "-real-honest-" in parts[3]
    if len(parts) != 4:
        return False
    if side == "real":
        return parts[1] == REAL_WORLD and "-real-honest-" in parts[3] and "honest_c" not in parts[0] and "ep2" not in parts[0]
    return parts[1].startswith(SIM_WORLD) and "-sim-honest-" in parts[3] and "simh" in parts[0]


def sha_ok(sha) -> bool:
    return isinstance(sha, str) and len(sha) == 64 and all(ch in "0123456789abcdef" for ch in sha)


def honest_gate(tasks: list, run1_launches) -> None:
    """Keys only, every run and every reference candidate key of every task, before any .eval is opened. Any miss STOPs; the
    message names the task, side and check. Pattern: run_fa_honest_render_routec.honest_gate @ 0bb208322, narrowed to honest_b in
    run 1's launches, sim_honest in simh launches, and honest_a (or a pinned references/ copy) in run 1's launches."""
    r1, seen = set(run1_launches), set()
    for t in tasks:
        if not isinstance(t.get("task"), str) or not t["task"] or t["task"] in seen:
            raise SystemExit("STOP: a task key is missing or repeated")
        seen.add(t["task"])
        for side, world in (("real", REAL_WORLD), ("sim", SIM_WORLD)):
            r, w = t[side], f"{t['task']} / {side}"
            key, inv = r.get("key"), r.get("inventories") or []
            if not isinstance(key, str) or not key.endswith("#e0"):
                raise SystemExit(f"STOP: {w}: the run key is not an epoch-1 (#e0) key")
            if any("attack" in str(s).lower() for s in (key, r.get("eval_rel"), r.get("launch"), r.get("arm"), *inv)):
                raise SystemExit(f"STOP: {w}: an attack marker in the key / .eval path / launch / arm / inventory")
            parts = key.removeprefix("runs/").split("#e")[0].split("/")
            if not layout_ok(parts, side) or r.get("world") != world or r.get("task") != t["task"]:
                raise SystemExit(f"STOP: {w}: the key's layout, world or task is not its side's honest one")
            if not inv or set(inv) & set(EXCLUDED_INVENTORIES):
                raise SystemExit(f"STOP: {w}: no inventory, or an excluded inventory")
            if r.get("eval_rel") is not None and str(r["eval_rel"]).removeprefix("runs/") != "/".join(parts[1:]):
                raise SystemExit(f"STOP: {w}: the inventory's .eval path is not the key's")
            if not sha_ok(r.get("eval_sha256")):
                raise SystemExit(f"STOP: {w}: no .eval sha256")
            arm_ok = r.get("arm") == REAL_WORLD if side == "real" else str(r.get("arm")).startswith(SIM_WORLD)
            if not arm_ok or (side == "real" and parts[0] not in r1):
                raise SystemExit(f"STOP: {w}: not its side's honest arm, or the real run is not in a run-1 launch")
        refs = t.get("refs")
        if not isinstance(refs, dict) or not isinstance(refs.get("candidates"), list):
            raise SystemExit(f"STOP: {t['task']} / ref: no reference candidate list")
        for c in refs["candidates"]:
            if not sha_ok(c.get("eval_sha256")) or not c.get("keys") or any("attack" in str(x).lower() for x in c.get("records") or []):
                raise SystemExit(f"STOP: {t['task']} / ref: a candidate without a sha256 or keys, or with an attack marker")
            for k in c["keys"]:
                if not isinstance(k, str) or not k.endswith("#e0") or "attack" in k.lower():
                    raise SystemExit(f"STOP: {t['task']} / ref: a candidate key is not an epoch-1 honest key")
                parts = k.removeprefix("runs/").split("#e")[0].split("/")
                if not layout_ok(parts, "ref", t["task"], c.get("pinned") is True) or parts[0] not in r1:
                    raise SystemExit(f"STOP: {t['task']} / ref: a candidate key fails the reference layout or run-1 launch check")


# ---------- container: locate, read and judge ----------


def locate(r: dict, side: str) -> tuple:
    """(volume name, path in the volume, bytes) of the run's .eval: run_tables_v81.find_eval_v2's search (blob 88589b50 L241-257,
    the same roots and the same two relative paths; byte-identical copies with the sha256 allowed, first by (volume, path)), with
    PARSE['candidate paths'] checked on each candidate BEFORE any of its bytes are read."""
    base = r["key"].split("#e")[0].removeprefix("runs/")
    hits = []
    for root in RT.EVAL_ROOTS:
        for rel in (base, "runs/" + base):
            pth = Path(root) / rel
            if not os.path.lexists(pth):
                continue
            if not pth.exists():
                raise X.Gate("a candidate .eval path is a dangling symlink")
            root_res, res = Path(root).resolve(strict=True), pth.resolve(strict=True)
            if root_res not in res.parents:
                raise X.Gate("a candidate .eval path resolves outside its volume root")
            got = res.relative_to(root_res).parts
            if got != Path(rel).parts:
                raise X.Gate("a candidate .eval path is redirected by a symlink")
            if not layout_ok(got[1:] if got[0] == "runs" else got, side, r.get("task", ""), r.get("pinned") is True) \
                    or "attack" in str(res).lower() or not res.is_file():
                raise X.Gate("a resolved candidate .eval path fails the side's honest layout / file-name check")
            b = res.read_bytes()
            if hashlib.sha256(b).hexdigest() == r["eval_sha256"]:
                mount = "/" + root.split("/")[1]
                hits.append((RT.VOLS[mount], str(pth).removeprefix(mount + "/"), b))
    if not hits:  # byte-identical copies (same pinned sha256, every copy gated above before its read) are allowed: take the first by (volume, path)
        raise X.Gate("0 .eval files with the identity's sha256 under the eval roots")
    return sorted(hits, key=lambda h: (h[0], h[1]))[0]


def load(r: dict, side: str) -> tuple:
    """(volume, path, sample): locate (pre-read path gate, sha256), then its one epoch-1 sample (run_examples_v81.sample_of: zstd
    members, attachments inlined, epoch checked)."""
    vol, path, eb = locate(r, side)
    snap = Path("/tmp") / f"xpost_{r['eval_sha256']}.eval"
    snap.write_bytes(eb)
    del eb
    try:
        s, _member = X.sample_of(snap, 0)  # every key is #e0 (gated); the member name never leaves the container
    finally:
        snap.unlink(missing_ok=True)
    return vol, path, s


def walk(events):
    for ev in events:
        if not isinstance(ev, dict):
            raise X.Gate("an event is not a dict")
        yield ev
        sub = ev.get("events")
        if sub is not None:
            if not isinstance(sub, list):
                raise X.Gate("an event's nested events are not a list")
            yield from walk(sub)  # older Inspect nests a tool call's sandbox events inside its tool event


def call_ids(ev: dict) -> set:
    """The tool-call ids in a model event's output (run_examples_v81.arguments_at's path, L336-343). Structure is checked; a model
    event without output (e.g. a failed generate) has none."""
    out = ev.get("output")
    if out is None:
        return set()
    if not isinstance(out, dict) or not (out.get("choices") is None or isinstance(out.get("choices"), list)):  # Codex x2 (4): no defaulting of malformed values
        raise X.Gate("a model event's output is malformed")
    ids = set()
    for ch in out.get("choices") or []:
        msg = ch.get("message") if isinstance(ch, dict) else None
        tcs = msg.get("tool_calls") if isinstance(msg, dict) else None
        if not isinstance(msg, dict) or not (tcs is None or isinstance(tcs, list)) or not all(isinstance(tc, dict) for tc in tcs or []):
            raise X.Gate("a model event's choice is malformed")
        ids |= {tc.get("id") for tc in tcs or []}
    return ids


def exit_record(s: dict, cid: str, cmd: str) -> dict:
    """The real first call's exit status, bound to its execution window: PARSE['exit code (real side only)']."""
    if not isinstance(s.get("events"), list):
        raise X.Gate("the sample's events are not a list")
    evs = list(walk(s["events"]))
    issued = [i for i, ev in enumerate(evs) if ev.get("event") == "model" and cid in call_ids(ev)]
    if len(issued) != 1:
        raise X.Gate(f"{len(issued)} model events issued the first call")
    end = next((j for j in range(issued[0] + 1, len(evs)) if evs[j].get("event") == "model"), len(evs))
    want = "bash --login -c " + shlex.quote(cmd)
    hits = [ev.get("result") for ev in evs[issued[0] + 1:end]
            if ev.get("event") == "sandbox" and ev.get("action") == "exec" and ev.get("cmd") == want]
    if any(not isinstance(h, int) or isinstance(h, bool) for h in hits):
        raise X.Gate("an in-window exec record has no integer result")
    return {"exit_code": hits[0] if len(hits) == 1 else None, "exec_events": len(hits)}


def checked_messages(s: dict) -> list:
    msgs = s.get("messages")
    if not isinstance(msgs, list) or not all(isinstance(m, dict) and m.get("role") in ROLES for m in msgs):
        raise X.Gate("the messages are not a list of role-tagged dicts")
    return msgs


def call_fields(tc) -> tuple:
    """(function, arguments, id) of one tool call; anything else is MALFORMED."""
    fn, args, cid = (tc.get("function"), tc.get("arguments"), tc.get("id")) if isinstance(tc, dict) else (None, None, None)
    if not isinstance(fn, str) or not fn or not isinstance(args, dict) or not isinstance(cid, str) or not cid:
        raise X.Gate("a tool call lacks a function name, an argument dict or an id")
    X.no_attachment_refs(args)
    return fn, args, cid


def shell_cmd(fn: str, args: dict):
    cmd = args.get("cmd") or args.get("command")
    return cmd if fn == SHELL_TOOL and isinstance(cmd, str) and cmd else None


def call_output(msgs: list, at: int, cid: str) -> dict:
    """The output of the call issued by message `at`: PARSE['output (stdout)'] and PARSE['unanswered']."""
    ans = [m for m in msgs if m["role"] == "tool" and m.get("tool_call_id") == cid]
    if len(ans) > 1:
        raise X.Gate(f"{len(ans)} tool messages answer one call")
    if not ans:
        if any(m["role"] == "assistant" for m in msgs[at + 1:]):
            raise X.Gate("a call has no tool message although the run went on")
        return {"answered": False}
    c, err = ans[0].get("content"), ans[0].get("error")
    if isinstance(c, str):
        text = c
    elif isinstance(c, list) and all(isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str) for p in c):
        text = "\n".join(p["text"] for p in c)
    else:
        raise X.Gate("an output's content is neither a string nor a list of text parts")
    if err is not None:
        if not isinstance(err, dict) or not isinstance(err.get("message", ""), str):
            raise X.Gate("an output's error field is malformed")
        text = err.get("message") or text
    if text.startswith("attachment://"):
        raise X.Gate("unresolved attachment reference in an output")
    return {"answered": True, "text": text, "error_field": err is not None}


def first_output(s: dict, side: str) -> dict:
    """The run's first tool call and its output: PARSE. Malformed data raises X.Gate (unreadable, STOP). Text stays in the returned
    dict, which never leaves scan."""
    msgs = checked_messages(s)
    at = tc = None
    for i, m in enumerate(msgs):
        calls = m.get("tool_calls") if m["role"] == "assistant" else None
        if calls is None or calls == []:
            continue
        if not isinstance(calls, list) or not isinstance(calls[0], dict):
            raise X.Gate("an assistant message's tool_calls are malformed")
        at, tc = i, calls[0]
        break
    if tc is None:
        return {"first": "no tool call", "tool": "absent"}
    fn, args, cid = call_fields(tc)
    cat, cmd = ("bash" if fn == SHELL_TOOL else "other"), shell_cmd(fn, args)
    if cmd is None:
        return {"first": "not shell", "tool": cat}
    out = {"first": "shell", "tool": cat, "command": cmd} | call_output(msgs, at, cid)
    return out | (exit_record(s, cid, cmd) if side == "real" and out["answered"] else {})


def reference_output(s: dict, cmd: str):
    """The output of the reference run's FIRST shell call whose command is byte-identical to `cmd`, or None if it has none."""
    msgs = checked_messages(s)
    for i, m in enumerate(msgs):
        calls = m.get("tool_calls") if m["role"] == "assistant" else None
        if calls is None or calls == []:
            continue
        if not isinstance(calls, list):
            raise X.Gate("an assistant message's tool_calls are malformed")
        for tc in calls:
            fn, args, cid = call_fields(tc)
            c = shell_cmd(fn, args)
            if c is not None and u8(c) == u8(cmd):
                return call_output(msgs, i, cid)
    return None


def read_run(r: dict, side: str) -> dict:
    vol, path, s = load(r, side)
    return {"volume": vol, "eval": path} | first_output(s, side)


def lines(text: str) -> list:
    body = text[:-1] if text.endswith("\n") else text
    return body.split("\n") if body else []


def visible(text: str) -> list:
    """Rule v2 §1a.1: each line right-stripped, lines empty after that dropped."""
    return [ln.rstrip() for ln in text.split("\n") if ln.rstrip()]


def visible_diff(a: str, b: str) -> list:
    """Positions (indices into the visible lines) where they differ; each extra line in the longer output counts as one."""
    va, vb = visible(a), visible(b)
    return [i for i in range(max(len(va), len(vb))) if i >= len(va) or i >= len(vb) or va[i] != vb[i]]


def u8(s: str) -> bytes:
    return s.encode("utf-8", "surrogatepass")


def judge(real: dict, sim: dict) -> dict:
    """Section 1a's steps for one task, in STEPS order (each flag implies the previous). Keys, counts, booleans and fixed categories
    only: real_first / sim_first are 'shell' / 'not shell' / 'no tool call'; real_tool / sim_tool are 'bash' / 'other' / 'absent'."""
    v = {"real_first": real["first"], "sim_first": sim["first"], "real_tool": real["tool"], "sim_tool": sim["tool"],
         "both_shell": real["first"] == "shell" and sim["first"] == "shell"}
    v["commands_identical"] = v["both_shell"] and u8(real["command"]) == u8(sim["command"])
    v["both_answered"] = v["commands_identical"] and real["answered"] and sim["answered"]
    if not v["both_answered"]:
        return v | {"fit": False, "differ": False, "no_error": False, "real_exit_recorded": False}
    ln = {"real": lines(real["text"]), "sim": lines(sim["text"])}
    for k, x in (("real", real), ("sim", sim)):
        v |= {f"{k}_lines": len(ln[k]), f"{k}_max_width": max((len(s.expandtabs(TABSIZE)) for s in ln[k]), default=0),
              f"{k}_visible_lines": len(visible(x["text"])), f"{k}_error_field": x["error_field"], f"{k}_empty": not x["text"].strip()}
    v |= {"real_exit_code": real["exit_code"], "real_exec_events": real["exec_events"]}
    v["fit"] = all(MIN_LINES <= v[f"{k}_lines"] <= MAX_LINES and v[f"{k}_max_width"] <= MAX_WIDTH for k in ln)
    n = max(len(ln["real"]), len(ln["sim"]))
    v["raw_lines_differing"] = sum(i >= len(ln["real"]) or i >= len(ln["sim"]) or ln["real"][i] != ln["sim"][i] for i in range(n))
    v["visible_lines_differing"] = len(visible_diff(real["text"], sim["text"]))
    v["differ"] = v["fit"] and v["visible_lines_differing"] >= MIN_DIFF
    real_err = v["real_error_field"] or v["real_empty"] or real["exit_code"] not in (None, 0)
    sim_err = v["sim_error_field"] or v["sim_empty"]
    v["no_error"] = v["differ"] and not real_err and not sim_err
    v["real_exit_recorded"] = v["no_error"] and real["exit_code"] is not None
    return v


def digests(real: dict, sim: dict) -> dict:
    return {k: hashlib.sha256(u8(s)).hexdigest() for k, s in (("command", real["command"]), ("real_output", real["text"]),
                                                              ("sim_output", sim["text"]))}


def copy_check(task: str, refs: dict, cmd: str, sim_text: str) -> dict:
    """Rule v2 §1a.3 with Writing's ~16:20 change: PARSE['reference candidates'] and PARSE['copy check']. Every candidate is checked;
    only results, counts, indices, keys and sha256s are returned (no reference text)."""
    out = []
    for c in refs["candidates"]:
        row = {"eval_sha256": c["eval_sha256"], "pinned": c["pinned"], "records": c["records"], "keys_tried": c["keys"]}
        try:
            got = None
            for k in c["keys"]:
                try:
                    got = (k, *load({"key": k, "eval_sha256": c["eval_sha256"], "task": task, "pinned": c["pinned"]}, "ref"))
                    break
                except X.Gate as e:
                    if not str(e).startswith("0 .eval files"):
                        raise
            if got is None:
                raise X.Gate(f"none of the candidate's {len(c['keys'])} keys has its .eval under the eval roots")
            key, vol, path, s = got
            ref = reference_output(s, cmd)
            row |= {"reference_run_key": key, "volume": vol, "eval": path}
            if ref is None:
                row["result"] = "COMMAND NOT IN REFERENCE"
            elif not ref["answered"]:
                row["result"] = "REFERENCE CALL UNANSWERED"
            else:
                idx = visible_diff(sim_text, ref["text"])
                row |= {"result": "IDENTICAL" if not idx else f"DIFFERS IN {len(idx)} LINES", "n_differing": len(idx),
                        "differing_indices": idx, "reference_error_field": ref["error_field"]}
        except X.Gate as e:  # fixed structural reasons only; nothing past a failed gate was read
            row["result"] = f"UNREADABLE (gate: {e})"
        except Exception as e:  # noqa: BLE001 -- a parse error can quote source text: only its type leaves
            row["result"] = f"UNREADABLE ({type(e).__name__}, message withheld)"
        out.append(row)
    res = [r["result"] for r in out]
    any_identical = "IDENTICAL" in res
    definite = bool(out) and not any(r.startswith("UNREADABLE") or r == "REFERENCE CALL UNANSWERED" for r in res)  # Codex v2 x1: no comparison -> inconclusive
    return {"reference_source": refs["source"], "n_candidates": len(out),
            "status": ("NO REFERENCE CANDIDATE" if not out else "ONE REFERENCE" if len(out) == 1
                       else f"REFERENCE AMBIGUOUS ({len(out)}): every candidate checked"),
            "candidates": out, "copy_identical_to_any": True if any_identical else (False if definite else None)}


@app.function(**FN)
def scan(payload: dict) -> list:
    """One chunk of tasks: gate, read both runs, judge. Unreadable or malformed runs are counted with a structural reason (they STOP
    the job before the draw), never dropped."""
    honest_gate(payload["tasks"], payload["run1_launches"])  # again here, before any .eval is opened
    out = []
    for t in payload["tasks"]:
        row, got, bad = {"task": t["task"], "real_key": t["real"]["key"], "sim_key": t["sim"]["key"]}, {}, []
        for side in ("real", "sim"):
            try:
                got[side] = read_run(t[side], side)
            except X.Gate as e:  # fixed structural reasons only (counts, check names)
                bad.append({"side": side, "reason": f"gate: {e}"})
            except Exception as e:  # noqa: BLE001 -- a parse error can quote source text: only its type leaves (run_tables_v81 job)
                bad.append({"side": side, "reason": f"{type(e).__name__} (message withheld)"})
        if bad:
            out.append(row | {"unreadable": bad})
            continue
        v = judge(got["real"], got["sim"])
        row |= {"real_eval": [got["real"]["volume"], got["real"]["eval"]], "sim_eval": [got["sim"]["volume"], got["sim"]["eval"]]} | v
        if v["real_exit_recorded"]:
            row["digests"] = digests(got["real"], got["sim"])
        out.append(row)
    return out


def _render(p: dict) -> dict:
    t = p["task"]
    honest_gate([t], p["run1_launches"])
    real, sim = read_run(t["real"], "real"), read_run(t["sim"], "sim")
    if not judge(real, sim)["real_exit_recorded"] or digests(real, sim) != p["digests"]:
        raise SystemExit("STOP: the drawn pair does not re-derive to the scan's eligible pair (verdict or digests differ)")
    return {"command": real["command"], "real_output": real["text"], "simulated_output": sim["text"],
            "copy_check": copy_check(t["task"], t["refs"], real["command"], sim["text"])}


@app.function(**FN)
def render(payload: dict) -> dict:
    """The drawn task only: re-gate, re-read, re-judge, hash-check against scan, run the copy check, then return the command, the two
    outputs and the copy check's results."""
    err = None
    try:
        return _render(payload)
    except SystemExit as e:
        err = str(e)
    except X.Gate as e:
        err = f"gate: {e}"
    except Exception as e:  # noqa: BLE001 -- only the type crosses back (run_tables_v81 job)
        err = f"stopped: {type(e).__name__} (message withheld)"
    raise SystemExit(err)


def draw(eligible: list) -> dict:
    """Section 1a step 3, keys only."""
    keys = sorted(eligible)
    if not keys or len(set(keys)) != len(keys):
        raise SystemExit("STOP: the eligible task keys are empty or repeated")
    rng = random.Random(SEED)
    order = list(keys)
    rng.shuffle(order)
    real_is_a = rng.random() < 0.5
    return {"task": order[0], "shuffled_position": 0, "sorted_position": keys.index(order[0]), "n_eligible": len(keys),
            "real_is": "A" if real_is_a else "B", "simulated_is": "B" if real_is_a else "A"}


# ---------- laptop: pool (git only), launch, write ----------


def git(*a) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout


def pinned(commit: str, path: str, sha: str) -> bytes:
    b = git("show", f"{commit}:{path}")
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit(f"{path} @ {commit}: sha256 differs from its pin")
    return b


def manifest_sha(r: dict, cache: dict):
    """A row's .eval sha256 from its routing record's .eval manifest, by the key's runs/-normalized path: run_tables_v81 eval_sha_of
    (blob 88589b50 L1523-1535), plus a STOP when one manifest lists a path with two sha256s."""
    rt = (r.get("uncut_planned") or {}).get("routing") or ""
    if " @ " not in rt:
        return None
    if rt not in cache:
        rp, rc = rt.split(" @ ")
        mm = json.loads(git("show", f"{rc}:{rp}")).get("manifest") or ""
        files = {}
        if " @ " in mm:
            mp, mc = mm.split(" @ ")
            for f in json.loads(git("show", f"{mc}:{mp}")).get("files", []):
                if files.setdefault(f["path"].removeprefix("runs/"), f["sha256"]) != f["sha256"]:
                    raise SystemExit(f"STOP: routing manifest {mp}: one path with two sha256s")
        cache[rt] = files
    return cache[rt].get(r["key"].split("#e")[0].removeprefix("runs/"))


def ref_key(path: str) -> str:
    """A recorded devbox path's run key: from the launch folder before its 'honest_a' or 'references' folder, plus '#e0'."""
    segs = path.split("/")
    i = next((j for j, s in enumerate(segs) if s in (REF_WORLD, "references")), None)
    if i is None or i < 1:
        raise SystemExit("STOP: a pinned reference path has no launch / honest_a or references folder")
    return "/".join(segs[i - 1:]) + "#e0"


def reference_pointers() -> tuple:
    """task -> {sha256: {"keys": [...], "records": [...]}} from POINTER_RECORDS (keys and sha256s only), and each record file's
    sha256. A recorded eval_task must name a real honest log."""
    ptr, files = {}, {}

    def add(task, sha, paths, rec, eval_task):
        if not isinstance(task, str) or not task or not sha_ok(sha):
            raise SystemExit(f"STOP: {rec}: a pinned reference without a task or a sha256")
        if eval_task is not None and "-real-honest-" not in str(eval_task):
            raise SystemExit(f"STOP: {rec}: a pinned reference that is not a real honest log")
        e = ptr.setdefault(task, {}).setdefault(sha, {"keys": [], "records": set()})
        e["keys"] += [ref_key(p) for p in paths if isinstance(p, str) and p]
        e["records"].add(rec)

    c, p, h = POINTER_RECORDS["merged_refs"]
    files[p] = h
    for task, e in sorted(json.loads(pinned(c, p, h))["references"].items()):
        add(task, e.get("sha256"), (e.get("source"), e.get("copy")), Path(p).name, None)
    c, d, n, h = POINTER_RECORDS["generation_manifests"]
    names = sorted(x for x in git("ls-tree", "--name-only", f"{c}:{d}").decode().split("\n")
                   if x.startswith("manifest_simh_") and x.endswith(".json"))
    blobs = [git("show", f"{c}:{d}/{x}") for x in names]
    if len(names) != n or hashlib.sha256(b"".join(blobs)).hexdigest() != h:
        raise SystemExit("the sim launches' generation manifests differ from their pin")
    for x, b in zip(names, blobs):
        files[f"{d}/{x}"] = hashlib.sha256(b).hexdigest()
        for cell in json.loads(b)["cells"]:
            if cell.get("status") == "pinned":
                add(cell.get("task"), cell.get("eval_sha256"), (cell.get("reference_source"), cell.get("reference_copy")), x,
                    cell.get("eval_task"))
    for t in ptr.values():
        for e in t.values():
            e["keys"], e["records"] = list(dict.fromkeys(e["keys"])), sorted(e["records"])
    return ptr, files


def build_pool() -> dict:
    """Every honest_b / sim_honest row of the registry's pinned inventories (pools_ep2 excluded), deduplicated by runs/-normalized key
    (all inventories must agree on world, task and .eval sha256), grouped by task; each task's reference candidates (PARSE). A run1
    row contributes ONLY its key's launch folder (for the run-1-launch check); nothing else of an attack row is kept."""
    inf = json.loads(pinned(*REGISTRY))
    rows_by, run1, runs, cache, pins, sha_src = Counter(), set(), {}, {}, {}, Counter()
    for name, e in sorted(inf["inventories"].items()):
        j = json.loads(pinned(e["commit"], e["path"], e["sha256"]))
        pins[name] = {k: e[k] for k in ("commit", "path", "sha256")}
        for r in j.get("runs") or []:
            world = r.get("world") if isinstance(r, dict) and isinstance(r.get("key"), str) else None
            rows_by[f"{name} | {world}"] += 1
            if world == "run1":
                run1.add(r["key"].removeprefix("runs/").split("/")[0])
            if world not in (REAL_WORLD, SIM_WORLD, REF_WORLD) or name in EXCLUDED_INVENTORIES:
                continue
            have = {s: h for s, h in (("inventory", r.get("eval_sha256")), ("routing manifest", manifest_sha(r, cache))) if h}
            if len(set(have.values())) != 1:
                raise SystemExit(f"STOP: {r['key']}: .eval sha256 sources {'disagree' if have else 'missing'}")
            sha_src[f"{world} | " + " + ".join(sorted(have))] += 1
            ent = {"key": r["key"], "world": world, "arm": r.get("arm"), "task": r.get("task"), "launch": r.get("launch"),
                   "eval_rel": r.get("eval_rel"), "eval_sha256": next(iter(have.values()))}
            old = runs.setdefault(r["key"].removeprefix("runs/"), ent | {"inventories": []})
            if any(old[k] != ent[k] for k in ("world", "task", "eval_sha256")):
                raise SystemExit(f"STOP: {r['key']}: two inventories disagree on world / task / .eval sha256")
            old["inventories"].append(name)
    by_task = {}
    for e in runs.values():
        if not isinstance(e["task"], str) or not e["task"]:
            raise SystemExit(f"STOP: {e['key']}: no task key")
        by_task.setdefault(e["task"], {"real": [], "sim": [], "ref": []})[
            {REAL_WORLD: "real", SIM_WORLD: "sim", REF_WORLD: "ref"}[e["world"]]].append(e)
    multi = [t for t, d in by_task.items() if len(d["real"]) > 1 or len(d["sim"]) > 1]
    if multi:
        raise SystemExit(f"STOP: {len(multi)} tasks have more than one {REAL_WORLD} or {SIM_WORLD} run; the rule names one each")
    ptr, pfiles = reference_pointers()

    def refs_of(task: str, ha: list) -> dict:
        if task in ptr:
            return {"source": "pinned by the sim launches' reference records", "candidates": [
                {"eval_sha256": sha, "pinned": True, "records": e["records"],
                 "keys": list(dict.fromkeys([r["key"] for r in ha if r["eval_sha256"] == sha] + e["keys"]))}
                for sha, e in sorted(ptr[task].items())]}
        return {"source": "registry honest_a rows (no reference record pins this task)", "candidates": [
            {"eval_sha256": r["eval_sha256"], "pinned": False, "records": sorted(r["inventories"]), "keys": [r["key"]]} for r in ha]}

    tasks = [{"task": t, "real": d["real"][0], "sim": d["sim"][0], "refs": refs_of(t, d["ref"])}
             for t, d in sorted(by_task.items()) if d["real"] and d["sim"]]
    counts = {"inventory_rows_by_world": dict(sorted(rows_by.items())), "sha256_sources": dict(sha_src),
              "unique_runs": dict(Counter(e["world"] for e in runs.values())),
              "tasks": {"both": len(tasks), "real_only": sum(bool(d["real"]) and not d["sim"] for d in by_task.values()),
                        "sim_only": sum(bool(d["sim"]) and not d["real"] for d in by_task.values())},
              "task_folder_not_task": sum(t[s]["key"].removeprefix("runs/").split("/")[2] != t["task"] for t in tasks for s in ("real", "sim")),
              "run1_launches": len(run1),
              "reference_candidates": {"by_source": dict(Counter(t["refs"]["source"] for t in tasks)),
                                       "n_candidates": dict(Counter(str(len(t["refs"]["candidates"])) for t in tasks)),
                                       "pinned_sha_matching_a_registry_honest_a_row": sum(
                                           any(c["pinned"] and c["keys"][0] in {r["key"] for r in by_task[t["task"]]["ref"]}
                                               for c in t["refs"]["candidates"]) for t in tasks)}}
    return {"tasks": tasks, "run1_launches": sorted(run1), "counts": counts, "pins": pins, "pointer_files": pfiles}


def step_counts(rows: list) -> dict:
    ok = [r for r in rows if "unreadable" not in r]
    c = {"pairs_found": len(rows), "pairs_readable": len(ok), "both_first_calls_shell": sum(r["both_shell"] for r in ok),
         "commands_identical": sum(r["commands_identical"] for r in ok), "lines_and_width_fit": sum(r["fit"] for r in ok),
         "differ_ge2_visible_lines": sum(r["differ"] for r in ok), "no_error": sum(r["no_error"] for r in ok),
         "real_exit_recorded": sum(r["real_exit_recorded"] for r in ok)}
    if list(c) != [STEPS[0], "pairs_readable", *STEPS[1:]]:
        raise SystemExit("step_counts and STEPS differ")
    ans = [r for r in ok if r["both_answered"]]
    fit = [r for r in ok if r["fit"]]
    diag = {"first_call": {s: dict(Counter(f"{r[f'{s}_first']} | {r[f'{s}_tool']}" for r in ok)) for s in ("real", "sim")},
            "identical_but_unanswered": sum(r["commands_identical"] and not r["both_answered"] for r in ok),
            "answered_pairs": len(ans),
            "among_fit_raw_ge2_lines": sum(r["raw_lines_differing"] >= MIN_DIFF for r in fit),
            "among_fit_raw_ge2_but_visible_lt2": sum(r["raw_lines_differing"] >= MIN_DIFF > r["visible_lines_differing"] for r in fit),
            "among_fit_visible_lines_differing": {str(k): n for k, n in sorted(Counter(min(r["visible_lines_differing"], 5) for r in fit).items())},
            "real_exec_events_in_window_among_answered": {str(k): n for k, n in sorted(Counter(min(r["real_exec_events"], 2) for r in ans).items())},
            "error_field_among_answered": {s: sum(r[f"{s}_error_field"] for r in ans) for s in ("real", "sim")},
            "excluded_no_real_exit_record": c["no_error"] - c["real_exit_recorded"],
            "unreadable_runs": dict(Counter(f"{b['side']} | {b['reason']}" for r in rows for b in r.get("unreadable", [])))}
    return {"steps": c, "diagnostics": diag}


def write_new(path: Path, obj: dict, mode: int) -> str:
    """Create-only (O_EXCL, no symlink follow); returns the written bytes' sha256."""
    b = (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, "wb") as fh:
        fh.write(b)
    return hashlib.sha256(b).hexdigest()


@app.local_entrypoint()
def main(go: bool = False):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if not go:
        raise SystemExit("--go only (after review and 0d's go)")
    if hashlib.sha256(RULE_FILE.read_bytes()).hexdigest() != RULE_SHA256:
        raise SystemExit("the frozen rule file differs from its pinned sha256")
    for name, blob in PIN_BLOBS.items():
        if git("hash-object", "--no-filters", f"{DIR}/{name}").decode().strip() != blob:
            raise SystemExit(f"{name} is not the pinned blob {blob[:8]}")
    if REPO.resolve() in TEXT_DIR.resolve().parents or not TEXT_DIR.parent.is_dir():
        raise SystemExit("TEXT_DIR must sit outside the repository, under an existing folder")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    me = f"{DIR}/{Path(__file__).name}"
    script = {"path": me, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "head": git("rev-parse", "HEAD").decode().strip(),
              "committed_at_head": subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", me]).returncode == 0
              and bool(git("ls-files", me).strip()), "pinned_blobs": PIN_BLOBS, "pinned_from": PIN_SOURCE,
              "python": sys.version.split()[0], "modal": modal.__version__, "app": APP}

    pool = build_pool()
    honest_gate(pool["tasks"], pool["run1_launches"])
    chunks = [pool["tasks"][i:i + CHUNK] for i in range(0, len(pool["tasks"]), CHUNK)]
    rows = [row for part in scan.map([{"tasks": c, "run1_launches": pool["run1_launches"]} for c in chunks]) for row in part]
    if [r["task"] for r in rows] != [t["task"] for t in pool["tasks"]]:
        raise SystemExit("the scan's rows differ from the pool's tasks")
    eligible = sorted(r["task"] for r in rows if r.get("real_exit_recorded"))
    sc = step_counts(rows)
    rec = {"what": "x-post spot-the-fake pair draw v2: keys and counts only (no text)", "stamp_pt": stamp,
           "rule": {"file": str(RULE_FILE), "sha256": RULE_SHA256, "section": "1a", "note": f"{DIR}/NOTE_xpost_spot_fake_v2.md"},
           "label": LABEL.decode(), "seed": {"rule": "int(sha256(label).hexdigest(), 16)", "hex": format(SEED, "064x")},
           "a_rule": "after the shuffle, rng.random() < 0.5 -> the real output is A", "step_order": list(STEPS),
           "registry": {"inputs_full": dict(zip(("commit", "path", "sha256"), REGISTRY)), "inventories": pool["pins"],
                        "excluded": EXCLUDED_INVENTORIES},
           "reference_records": {"pins": {k: list(v) for k, v in POINTER_RECORDS.items()}, "files_sha256": pool["pointer_files"]},
           "pool": pool["counts"], **sc, "eligible": eligible,
           "eligible_sha256": hashlib.sha256(json.dumps(eligible).encode()).hexdigest(),
           "per_task": [{k: v for k, v in r.items() if k != "digests"} for r in rows],
           "parse": PARSE, "ambiguities": AMBIGUITIES, "script": script}
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    keys_path = res / f"xpost_spotfake_draw_v2_{stamp}.json"
    n_bad = sc["steps"]["pairs_found"] - sc["steps"]["pairs_readable"]
    if n_bad:
        ks = write_new(keys_path, rec | {"status": f"STOPPED before the draw: {n_bad} tasks with an unreadable or malformed run"}, 0o644)
        print(json.dumps({"status": "STOPPED", "keys_record": str(keys_path), "keys_sha256": ks}), flush=True)
        raise SystemExit(1)
    if not eligible:  # rule v2 §1a: no redraws of any kind; an empty pool is a result, not an error
        ks = write_new(keys_path, rec | {"status": "no eligible pair: no draw"}, 0o644)
        print(json.dumps({"status": "NO ELIGIBLE PAIR", "keys_record": str(keys_path), "keys_sha256": ks}), flush=True)
        return

    d = draw(eligible)
    row = next(r for r in rows if r["task"] == d["task"])
    t = next(x for x in pool["tasks"] if x["task"] == d["task"])
    got = render.remote({"task": t, "run1_launches": pool["run1_launches"], "digests": row["digests"]})
    cc = got["copy_check"]
    real_a = d["real_is"] == "A"
    TEXT_DIR.mkdir(mode=0o700, exist_ok=True)
    text_path = TEXT_DIR / f"xpost_spotfake_pair_v2_{stamp}.json"
    ts = write_new(text_path, {
        "what": "x-post spot-the-fake pair v2: the drawn command and its two honest-run outputs, with the copy check. Read by eye before "
                "posting",
        "rule": rec["rule"], "label": rec["label"], "task": d["task"], "real_run_key": t["real"]["key"], "simulated_run_key": t["sim"]["key"],
        "draw": d, "command": got["command"],
        "A": {"is": "real" if real_a else "simulated", "output": got["real_output"] if real_a else got["simulated_output"]},
        "B": {"is": "simulated" if real_a else "real", "output": got["simulated_output"] if real_a else got["real_output"]},
        "copy_check": cc, "keys_record": str(keys_path)}, 0o600)
    ks = write_new(keys_path, rec | {"status": "drawn", "draw": d | {"real_run_key": t["real"]["key"], "simulated_run_key": t["sim"]["key"],
                                                                      "digests": row["digests"]},
                                     "copy_check": cc, "reference_run_keys": [c.get("reference_run_key") for c in cc["candidates"]],
                                     "text_file": {"path": str(text_path), "sha256": ts}}, 0o644)
    print(json.dumps({"text_file": str(text_path), "text_sha256": ts, "keys_record": str(keys_path), "keys_sha256": ks}), flush=True)
