"""1e-4 real honest runs (RULE.md, Amendment 7): 09's cue_D flag per run, counts only. A copy of the D path of 09's
run_cue_cause.py @ 9167ad76f: `segments` and `struct_of_segments` verbatim (the flag: `user_before_last_tool`, a non-tool
user turn after the first one, before the last tool output, on the conversation rendered with Qwen3-32B's template, as
lines 169-170 and 214-217), and the raw-message variant `msg_cue_d` by the same logic (lines 171-185, keeping only the
flag), required to agree. Changed:
the messages come from each .eval directly, as d9's staging made them for 09's D rows (stage_onpol.py: `transcripts.
reader_messages(sample, its own system prompt)`, before any scrub); the runs come from an upload manifest; Modal CPU.
Targets: `b`, the positive control (batch B's honest_a and honest_b .eval files on 2e's Volume, read only; the cue_D task
sets must equal 09's `cue_lists` for honest_a and honest_b, apart from runs the staging excludes: errored runs whose acting-model events and assistant messages differ in
number, stage_q4b's own test), and `rh`
(this launch, k of n per replicate). Keys, tasks and booleans only; no text leaves a container (errors give their type).

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run \\
      experiments/2026-10-14_q3_q35_realhonest_1e4/cue_d_rh.py --target b|rh
"""
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q3realhonest_cued_kill-2026-11-14"
MINE = "probe-shift_aswin-surya_sub-q3realhonest_data_kill-2026-11-14"
SRC_C = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
B_MAIN = ["2026-10-09_000523_rerun_strict", "2026-10-09_173815_scaleup_canary", "2026-10-09_182505_scaleup", "2026-10-10_001314_scaleup_topup"]
TARGETS = {"b": {"manifest": "manifest_q4b_eval_upload_v1.json", "cache": "/src/evals_v1", "launches": B_MAIN, "arms": ("honest_a", "honest_b")},
           "rh": {"manifest": "manifest_rh1e4_eval_upload_v1.json", "cache": "/ev/evals_v1",
                  "launches": ["2026-10-14_135334_honest_rh980", "2026-10-14_160335_honest_rh_relaunch"],  # both launches (Amendment 16)
                  "arms": ("honest_b",)}}
CUE_RESULT = ("0798ff6d", "experiments/2026-10-14_format_smell_09/results/cue_cause_2026-10-14_103513.json")  # 09's as-run result

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.13").uv_pip_install("transformers==5.19.0", "jinja2", "zstandard", "numpy", "boto3")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift")
         .add_local_file(HERE / "evalzip.py", "/root/repo/q4/evalzip.py").add_local_file(HERE / "stage_q4b.py", "/root/repo/q4/stage_q4b.py"))
VOLS = {"/ev": modal.Volume.from_name(MINE, version=2).read_only(), "/src": modal.Volume.from_name(SRC_C, version=2).read_only(),
        "/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only()}


def h16(s) -> str:
    return hashlib.sha256((s if isinstance(s, str) else json.dumps(s, sort_keys=True)).encode()).hexdigest()[:16]


def segments(text: str):
    """[(kind, think, body)] from rendered chat text; raises on a malformed turn boundary or an unknown role."""
    segs = text.split("<|im_start|>")[1:]
    if not segs:
        raise ValueError("malformed rendering")
    out = []
    for j, seg in enumerate(segs):
        n_end = seg.count("<|im_end|>")
        if (j < len(segs) - 1 and (n_end != 1 or not seg.rstrip("\n").endswith("<|im_end|>"))) or n_end > 1:
            raise ValueError("malformed turn boundary")
        role, _, body = seg.partition("\n")
        body = body.split("<|im_end|>")[0]
        if role == "assistant":
            k = "A"
        elif role == "system":
            k = "S"
        elif role in ("user", "tool"):
            k = "T" if body.lstrip().startswith("<tool_response>") else "U"
            if role == "tool" and k == "U":
                raise ValueError("tool role without a tool_response")
        else:
            raise ValueError("unknown role")
        out.append((k, k == "A" and body.lstrip().startswith("<think>"), body))
    return out


def struct_of_segments(segs, n_tokens=None) -> dict:
    kinds = [k for k, _, _ in segs]
    a = [i for i, k in enumerate(kinds) if k == "A"]
    if not a:
        raise ValueError("no assistant turn")
    lt = max((i for i, k in enumerate(kinds) if k == "T"), default=-1)
    la = a[-1]
    users = [i for i, k in enumerate(kinds) if k == "U"]
    think = [i for i in a if segs[i][1]]
    no_call = [i for i in a if "<tool_call>" not in segs[i][2]]
    empty = [i for i in no_call if not re.sub(r"^\s*<think>\s*</think>", "", segs[i][2]).strip()]
    appended = []  # tool segments with text after the last </tool_response>
    for i, (k, _, body) in enumerate(segs):
        if k == "T" and not body.rstrip().endswith("</tool_response>"):
            tail = body[body.rfind("</tool_response>") + len("</tool_response>"):].strip() if "</tool_response>" in body else body.strip()
            appended.append({"hash": h16(tail), "after_last_assistant": i > la, "index": i, "of": len(segs)})
    sys_b = next((b for k, _, b in segs if k == "S"), None)
    return {"n_assistant": len(a), "n_tool_turns": kinds.count("T"), "n_user": len(users), "n_tool_calls": sum(segs[i][2].count("<tool_call>") for i in a),
            "n_assistant_no_call": len(no_call), "n_assistant_empty": len(empty), "n_think": len(think), "n_tokens": n_tokens,
            "zero_think": not think, "think_after_last_tool_only": all(i > lt for i in think), "user_before_last_tool": any(i < lt for i in users[1:]),
            "ends_on": kinds[-1], "appended": appended, "sys_hash": h16(sys_b) if sys_b is not None else None,
            "first_user_hash": h16(segs[users[0]][2]) if users else None, "user_hashes": [h16(segs[i][2].strip()) for i in users[1:]]}


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes=VOLS)
def flags(cache: str, items: list) -> list:
    """Per (arm, task, launch/rel): errored, user_before_last_tool (rendered) and msg_cue_d (raw), or the error type."""
    import sys
    import zipfile

    sys.path.insert(0, "/root/repo/q4")
    from collections import Counter

    import evalzip  # noqa: F401  (zstd .eval members on Python 3.13)
    import stage_q4b as S  # 2e's, unchanged: the staging's own exclusion test (index_sample, reader_map)
    from transformers import AutoTokenizer

    from probe_shift import transcripts
    from probe_shift.models import QWEN3_32B
    from probe_shift.reader import prepare

    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"], local_files_only=True)
    out = []
    for arm, task, path in items:
        row = {"arm": arm, "task": task, "key": f"{path}#e0"}
        try:
            with zipfile.ZipFile(Path(cache) / path) as z:  # stage_q4b.load
                (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
                s = transcripts.resolve_attachments(json.loads(z.read(n)))
            # stage_q4b.process / stage_onpol.process: a run whose acting-model events and assistant messages differ in number
            # is excluded when it errored ("errored run: events and messages misaligned"), so 09's D rows never held it; such a
            # run is recorded as excluded, before any rendering (fix after the first positive-control run)
            misaligned = len(S.reader_map(s["messages"])[0]) != len(S.index_sample(s, Counter()))
            if misaligned and s.get("error"):
                out.append(row | {"errored": True, "staging_excluded": True})
                continue
            msgs = transcripts.reader_messages(s, transcripts.system_prompt(s))
            st = struct_of_segments(segments(tok.apply_chat_template(prepare(tok, msgs), tokenize=False, enable_thinking=False)))
            roles = [m["role"] for m in msgs]
            lt = max((i for i, r in enumerate(roles) if r == "tool"), default=-1)
            ui = [i for i, r in enumerate(roles) if r == "user"]
            nudges = []
            ui = [i for i in ui if not (isinstance(msgs[i].get("content"), str) and msgs[i]["content"].lstrip().startswith("<tool_response>"))]
            for i in ui[1:]:
                nudges.append({"before_last_tool": i < lt})
            row |= {"errored": bool(s.get("error")), "staging_excluded": False,
                    "user_before_last_tool": st["user_before_last_tool"], "msg_cue_d": any(x["before_last_tool"] for x in nudges)}
        except Exception as e:  # the error type only, never text
            row |= {"error": type(e).__name__}
        out.append(row)
    return out


@app.local_entrypoint()
def main(target: str):
    from collections import Counter

    T = TARGETS[target]
    man = json.loads((HERE / "config" / T["manifest"]).read_text())
    items = sorted((f["arm_dir"], f["task"], f"{f['launch']}/{f['rel']}") for f in man["files"]
                   if f["launch"] in T["launches"] and f["arm_dir"] in T["arms"])
    if not items or len({k for _, _, k in items}) != len(items) or len({(a, t) for a, t, _ in items}) != len(items):
        raise SystemExit(f"cue_D: {len(items)} selected runs, not a non-empty set of unique runs and (arm, task) pairs")
    if target == "rh":  # Codex round 22: the whole counted set, 196 per replicate, or stop
        from collections import Counter as _C
        per_rep = _C(t.rsplit("_", 1)[1] for _, t, _ in items)
        if len(items) != 980 or any(per_rep[f"r{i}"] != 196 for i in range(1, 6)):
            raise SystemExit(f"cue_D: {len(items)} selected runs, per replicate {dict(per_rep)}: not the 980 counted keys")
    chunks = [items[i:i + 25] for i in range(0, len(items), 25)]
    rows = [r for part in flags.map([T["cache"]] * len(chunks), chunks) for r in part]
    bad = [r for r in rows if "error" in r]
    excluded = [r for r in rows if r.get("staging_excluded")]  # as the staging (and so 09's D rows): not flagged
    ok = [r for r in rows if "error" not in r and not r.get("staging_excluded")]
    group = (lambda r: r["arm"]) if target == "b" else (lambda r: r["task"].rsplit("_", 1)[1])
    per = {}
    for r in ok:
        c = per.setdefault(group(r), Counter())
        c["runs"] += 1
        c["cue_D"] += r["user_before_last_tool"]
        c["cue_D_raw"] += r["msg_cue_d"]
        c["agree"] += r["user_before_last_tool"] == r["msg_cue_d"]
        c["errored"] += r["errored"]
        c["staging_excluded"] += r["staging_excluded"]
        c["cue_D_not_errored"] += r["user_before_last_tool"] and not r["errored"]
        c["runs_not_errored"] += not r["errored"]
    res = {"target": target, "runs": len(rows), "staging_excluded": len(excluded), "errors": dict(Counter(r["error"] for r in bad)), "per": {k: dict(v) for k, v in sorted(per.items())},
           "agree_all": sum(r["user_before_last_tool"] == r["msg_cue_d"] for r in ok), "of": len(ok)}
    if target == "b":  # the positive control: 09's cue_D task sets for honest_a and honest_b
        ref = json.loads(subprocess.run(["git", "show", f"{CUE_RESULT[0]}:{CUE_RESULT[1]}"], capture_output=True, text=True, check=True,
                                        cwd=REPO).stdout)["analysis"]["cue_lists"]["cue_D"]
        cmp = {}
        for arm in T["arms"]:
            theirs = {t for w, t in ref if w == arm}
            mine = {r["task"] for r in ok if r["arm"] == arm and r["user_before_last_tool"]}
            extra = mine - theirs
            excused = all(r["staging_excluded"] for r in ok if r["arm"] == arm and r["task"] in extra)  # 09's D never held them
            cmp[arm] = {"09": len(theirs), "mine": len(mine), "missing": sorted(theirs - mine), "extra": sorted(extra),
                        "extra_all_staging_excluded": excused, "passes": not (theirs - mine) and excused}
        res["positive_control"] = cmp
    res["rows"] = rows
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"cue_d_{target}_{stamp}.json"
    dest.open("x").write(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    print("->", dest)
    if bad or res["agree_all"] != res["of"] or (target == "b" and not all(v["passes"] for v in res["positive_control"].values())):
        raise SystemExit("cue_D: errors, a rendered/raw disagreement, or a failed positive control")
