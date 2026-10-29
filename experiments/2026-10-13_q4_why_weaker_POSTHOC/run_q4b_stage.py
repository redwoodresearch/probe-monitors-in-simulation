"""Q4 (B), RULE.md Amendment 5: stage Qwen3.5-27B's own runs in #36's reader format on Modal CPU, with stage_q4b.py (d9's
stage_crossread.py with the Amendment 5 diff). Sets `main` (run 1, simulated attack, simulated honest, honest A and B) and
`run2`. Parts run in parallel (one container per part); each part writes its staged epochs and per-run records to the
Volume; `assemble` joins them into inputs/<arm>.jsonl per set (sorted by key), with their sha256s and the coverage record.
Gate in `assemble`: the structural counts of 3 staged runs per arm, rendered as staged, match the reader pattern of Sol's
runs (empty think blocks only, at most one per assistant turn, no <tools> block, no submit row), else nothing is published. Counts only; no
transcript text is printed, logged or returned (failures re-raise as the run's file and the error type).

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_q4_why_weaker_POSTHOC/run_q4b_stage.py [--limit N]
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q4own_stage_kill-2026-11-13"
VOL = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
HF = "probe-shift_aswin-surya_hf-cache"
OUT = {"main": "q4b_main_v1", "run2": "q4b_run2_v1"}
PARTS = 48
LABEL_FILES = ["experiments/2026-10-08_q3_onpolicy_figures/results/phase2_attempt_labels_2026-10-09_000523_rerun_strict_selfreport_2026-10-10_111809.jsonl",
               "experiments/2026-10-08_q3_onpolicy_figures/results/phase2_attempt_labels_2026-10-09_173815_scaleup_canary_selfreport_2026-10-10_111817.jsonl",
               "experiments/2026-10-08_q3_onpolicy_figures/results/phase2_attempt_labels_2026-10-10_001314_scaleup_topup_selfreport_2026-10-10_111821.jsonl",
               "experiments/2026-10-08_q3_onpolicy_figures/results/phase2_attempt_labels_2026-10-10_001314_second_attack_2026-10-10_111840.jsonl"]
LABEL_S3 = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad/crossread_labels"
                "/phase2_attempt_labels_2026-10-09_182505_scaleup_selfreport_2026-10-10_111838.jsonl")
MODS = ["experiments/2026-10-09_fixed_sim/fixes.py", "experiments/2026-10-12_plantslip_transfer/plant_both.py",
        "experiments/2026-10-07_q2_rendering_swap/rerender.py", "experiments/2026-10-07_q2_rendering_swap/str_replace_tells.py",
        "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"]

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
image = (modal.Image.debian_slim(python_version="3.13").uv_pip_install("transformers==5.19.0", "numpy", "jinja2", "zstandard", "boto3", "pyyaml")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift")
         .add_local_file(HERE / "stage_q4b.py", "/root/repo/q4/stage_q4b.py").add_local_file(HERE / "evalzip.py", "/root/repo/q4/evalzip.py")
         .add_local_file(HERE / "results" / "manifest_q4b_eval_upload_v1.json", "/root/repo/manifest_q4b_eval_upload_v1.json")
         .add_local_file(LABEL_S3, f"/root/repo/labels/{LABEL_S3.name}"))
for m in MODS + LABEL_FILES:
    image = image.add_local_file(REPO / m, f"/root/repo/{m}")


def _mod():
    import sys

    sys.path.insert(0, "/root/repo/q4")
    import stage_q4b as S

    return S


def _setup(S, set_name):
    from collections import defaultdict

    from transformers import AutoTokenizer

    if not S.fingerprint().startswith(S.LABELER):
        raise ValueError(f"labeler fingerprint {S.fingerprint()[:8]} is not {S.LABELER}")
    tok = AutoTokenizer.from_pretrained(S.QWEN35_27B["name"], revision=S.QWEN35_27B["revision"], local_files_only=True)
    folds = S.frozen_folds(json.loads(S.T196.read_text()))
    counted = S.counted_attempts(set_name)
    prompts = defaultdict(dict)
    for (arm_dir, task), c in counted.items():
        if arm_dir in ("honest_a", "honest_b"):
            try:
                hs = S.load(c["eval"])
                prompts[task][arm_dir] = (S.transcripts.system_prompt(hs), S.first_user_sha(hs))
            except Exception as e:  # never let a message carry transcript text
                raise S.Withheld(f"{c['launch']}/{c['rel']}: {type(e).__name__} (message withheld)") from None
    return tok, folds, counted, prompts


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes={"/ev": vol, "/hf": modal.Volume.from_name(HF).read_only()})
def part(set_name: str, k: int, n_parts: int, limit: int, out: str) -> dict:
    from collections import Counter

    S = _mod()
    tok, folds, counted, prompts = _setup(S, set_name)
    todo = sorted(x for x in counted if x[0] in S.SETS[set_name] and counted[x]["launch"] in ([S.RUN2] if set_name == "run2" else S.MAIN))
    if limit:
        todo = [x for x in todo if sum(1 for y in todo[: todo.index(x)] if y[0] == x[0]) < limit]  # the first `limit` runs per arm
    mine, cov, eps, recs = todo[k::n_parts], Counter(), [], []
    for arm_dir, task in mine:
        c = counted[(arm_dir, task)]
        try:
            ep, rec = S.process(c, arm_dir, task, set_name, prompts, tok, folds, cov)
        except Exception as e:  # never let a message carry transcript text: the error type and code locations only
            import traceback

            where = [f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in traceback.extract_tb(e.__traceback__)[-4:]]
            raise S.Withheld(f"{c['launch']}/{c['rel']}: {type(e).__name__} at {where} (message withheld)") from None
        recs.append({"arm_dir": arm_dir, **rec, "run": f"{c['launch']}/{c['rel']}"})
        if ep is not None:
            eps.append(ep)
    d = Path("/ev") / out / "parts"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"part{k:03d}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    (d / f"part{k:03d}.recs.json").write_text(json.dumps({"recs": recs, "counters": dict(cov), "no_eval": S.NO_EVAL}))
    vol.commit()
    return {"part": k, "runs": len(mine), "staged": len(eps)}


@app.function(image=image, cpu=4, memory=32768, timeout=1800, retries=0, volumes={"/ev": vol, "/hf": modal.Volume.from_name(HF).read_only()})
def assemble(set_name: str, n_parts: int, stamp: str, out: str) -> dict:
    from collections import defaultdict

    import numpy as np
    from transformers import AutoTokenizer

    S = _mod()
    vol.reload()
    root = Path("/ev") / out
    eps, per_run, counters = defaultdict(list), defaultdict(list), defaultdict(int)
    for k in range(n_parts):
        for line in (root / "parts" / f"part{k:03d}.jsonl").read_text().splitlines():
            e = json.loads(line)
            eps[e["arm"]].append(e)
        r = json.loads((root / "parts" / f"part{k:03d}.recs.json").read_text())
        for rec in r["recs"]:
            per_run[rec.pop("arm_dir")].append(rec)
        for kk, v in r["counters"].items():
            counters[kk] += v
    # the format gate: 3 staged runs per arm, counts on the staged ids only
    tok = AutoTokenizer.from_pretrained(S.QWEN35_27B["name"], revision=S.QWEN35_27B["revision"], local_files_only=True)
    sid = {s: tok.convert_tokens_to_ids(s) for s in ("<|im_start|>", "<think>", "</think>", "<tool_call>", "<tool_response>")}
    asst = tok.encode("assistant", add_special_tokens=False)[0]
    tools_seq = tok.encode("<tools>", add_special_tokens=False)
    gate = {}
    for arm, lst in eps.items():
        rows = []
        for e in sorted(lst, key=lambda e: e["key"])[:3]:
            ids = np.asarray(e["ids"])
            st = np.flatnonzero(ids == sid["<|im_start|>"])
            n_asst = int(sum(1 for s in st if s + 1 < len(ids) and ids[s + 1] == asst))
            has_tools = any(np.array_equal(ids[i:i + len(tools_seq)], tools_seq) for i in range(min(len(ids), 4000)))
            submit = sum(1 for c in e["calls"] if c.get("name") == S.transcripts.RUNNER_SUBMIT)
            opens, closes = np.flatnonzero(ids == sid["<think>"]), np.flatnonzero(ids == sid["</think>"])
            nonempty = sum(1 for o, c in zip(opens, closes) if c - o > 3) if len(opens) == len(closes) else -1
            rows.append({"assistant_turns": n_asst, "think_open": len(opens), "think_close": len(closes), "think_blocks_with_content": nonempty,
                         "tools_block": bool(has_tools), "submit_rows": submit})
        ok = all(r["tools_block"] is False and r["submit_rows"] == 0 and r["think_open"] == r["think_close"]
                 and r["think_blocks_with_content"] == 0 and r["think_open"] <= r["assistant_turns"] for r in rows)
        gate[arm] = {"runs": rows, "passes": ok}
    record = {"set": set_name, "stamp": stamp, "cut": S.CUT, "scrub_level": S.SCRUB_LEVEL, "model": S.QWEN35_27B,
              "labeler_fingerprint": S.fingerprint(), "counters": dict(counters), "format_gate": gate,
              "coverage": S.summarize(per_run, S.attempt_labels(), set_name), "per_run": per_run}
    if not all(g["passes"] for g in gate.values()):
        (root / f"stage_FAILED_{stamp}.json").write_text(json.dumps(record, indent=1))
        vol.commit()
        return {"STOP": "format gate failed", "format_gate": gate}
    d = root / "inputs"
    if d.exists() and any(d.iterdir()):
        raise FileExistsError(f"{d} exists; stage under a new name")
    d.mkdir(parents=True, exist_ok=True)
    record["staged_inputs_sha256"], record["epochs"] = {}, {}
    for arm, lst in sorted(eps.items()):
        b = "".join(json.dumps(e) + "\n" for e in sorted(lst, key=lambda e: e["key"])).encode()
        (d / f"{arm}.jsonl").write_bytes(b)
        record["staged_inputs_sha256"][f"{arm}.jsonl"] = hashlib.sha256(b).hexdigest()
        record["epochs"][arm] = len(lst)
    (root / f"stage_{stamp}.json").write_text(json.dumps(record, indent=1))
    vol.commit()
    return {k: record[k] for k in ("set", "epochs", "staged_inputs_sha256", "format_gate")} | {"coverage": record["coverage"]}


@app.local_entrypoint()
def main(limit: int = 0):
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {}
    for set_name in OUT:
        n = 4 if limit else PARTS
        name = OUT[set_name] + (f"_pilot{limit}" if limit else "")
        parts = list(part.starmap([(set_name, k, n, limit, name) for k in range(n)]))
        out[set_name] = {"out": name, "parts": parts, "assembled": assemble.remote(set_name, n, stamp, name)}
    (HERE / "results").mkdir(exist_ok=True)
    dest = HERE / "results" / f"q4b_stage{'_limit' + str(limit) if limit else ''}_{stamp}.json"
    dest.open("x").write(json.dumps(out, indent=1, default=str))
    print(json.dumps({s: {"epochs": v["assembled"].get("epochs"), "gate": {a: g["passes"] for a, g in v["assembled"]["format_gate"].items()},
                          "stop": v["assembled"].get("STOP")} for s, v in out.items()}, indent=1))
    print(dest.name)
