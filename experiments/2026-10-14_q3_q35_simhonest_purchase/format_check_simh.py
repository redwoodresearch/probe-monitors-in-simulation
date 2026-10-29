"""Simulated-honest purchase (RULE.md amendment 12): 0d's format assertion, NOT written here. A copy of the real-honest
fork's `format_check` (and its `_setup`, `_mod` and `--format-n` entry) from
experiments/2026-10-14_q3_q35_realhonest_1e4/run_rh_stage.py @ 597e97b7c (aswin/q3-q35-realhonest-1e4, its Amendment 12;
validated both ways there: positive on batch C's corrected re-stage, results/format_check_c_2026-10-14_164553.json, 5 of 5
pass; negative on batch B's uncorrected staging of 2 cue_D runs, results/format_check_negative_b_2026-10-14_164821.json, both
fail) whose only changes are inputs: this job's Volume and .eval cache (/ev/simh_evals_v1), the group's launches and upload
manifest (results/manifest_simh_eval_upload_<SIMH_GROUP>.json), the staged file simh_q4f_v1_<SIMH_GROUP>/inputs/
sim_honest@september.jsonl, the arm sim_honest@september in place of honest_b, stage_simh.py in place of stage_q4b.py, the
negative-control branch dropped (cited, not rerun), and the function's resource limits. For A (SIMH_SET=atk2, arm
sim_attack@september; d9 approved, amendment 14) one more INPUT: the system prompt the messages are rebuilt with is the one
the stager actually used (the honest donor's), captured from stage_simh.process; honest groups keep the run's own.
Counts only; no text leaves the container.

  SIMH_GROUP=<g> [SIMH_SET=simh|atk2] SIMH_LAUNCHES=<launch>[,...] PYTHONPATH=src uv run --no-project --python 3.12 \
      --with modal==1.6.1 modal run <this file> --format-n <n>
"""

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q3simh_fmtcheck_kill-2026-11-13"
VOL = "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"
HF = "probe-shift_aswin-surya_hf-cache"
GROUP = os.environ["SIMH_GROUP"]
LAUNCHES = [x for x in os.environ["SIMH_LAUNCHES"].split(",") if x]
SET = os.environ.get("SIMH_SET", "simh")
ARM = {"simh": "sim_honest@september", "atk2": "sim_attack@september"}[SET]
TARGET = {"launch": LAUNCHES, "manifest": f"manifest_simh_eval_upload_{GROUP}.json", "cache": "/ev/simh_evals_v1",
          "out": f"simh_q4f_v1_{GROUP}"}
COND = "{%- if loop.index0 > ns.last_query_index %}"
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
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "SIMH_GROUP": GROUP, "SIMH_SET": SET, "SIMH_LAUNCHES": ",".join(LAUNCHES)}).add_local_python_source("probe_shift")
         .add_local_file(HERE / "stage_simh.py", "/root/repo/q4/stage_simh.py").add_local_file(HERE / "evalzip.py", "/root/repo/q4/evalzip.py")
         .add_local_file(HERE / "results" / TARGET["manifest"], f"/root/repo/{TARGET['manifest']}")
         .add_local_file(LABEL_S3, f"/root/repo/labels/{LABEL_S3.name}"))
for m in MODS + LABEL_FILES:
    image = image.add_local_file(REPO / m, f"/root/repo/{m}")
VOLS = {"/ev": vol, "/hf": modal.Volume.from_name(HF).read_only()}


def _mod():
    import sys

    sys.path.insert(0, "/root/repo/q4")
    import stage_simh as S

    return S


def _setup(S, set_name, target):
    from collections import defaultdict

    from transformers import AutoTokenizer

    T = TARGET
    launches = T["launch"] if isinstance(T["launch"], list) else [T["launch"]]
    S.MAIN, S.SETS = launches, dict(S.SETS) | {SET: [ARM]}  # this group's launches and arm
    S.UPLOAD_MANIFEST = Path(f"/root/repo/{T['manifest']}")
    S.CACHE = Path(T["cache"])
    if not S.fingerprint().startswith(S.LABELER):
        raise ValueError(f"labeler fingerprint {S.fingerprint()[:8]} is not {S.LABELER}")
    tok = AutoTokenizer.from_pretrained(S.QWEN35_27B["name"], revision=S.QWEN35_27B["revision"], local_files_only=True)
    tok_orig = AutoTokenizer.from_pretrained(S.QWEN35_27B["name"], revision=S.QWEN35_27B["revision"], local_files_only=True)
    if tok.chat_template.count(COND) != 1:
        raise ValueError("the template's last-query condition is not where Amendment 9 expects it")
    tok.chat_template = tok.chat_template.replace(COND, "{%- if true %}")  # Amendment 9: no later message changes a turn
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
    return (tok, tok_orig), folds, counted, prompts


def _system_used(S, c, task, prompts, tok, folds) -> str:
    """The system prompt stage_simh.process actually renders the run with (A: the honest donor's), captured from its
    reader_messages call; an input of the check, not a change to its logic."""
    from collections import Counter

    seen, rm = [], S.transcripts.reader_messages

    def capture(s, system, *a, **kw):
        seen.append(system)
        return rm(s, system, *a, **kw)

    S.transcripts.reader_messages = capture
    try:
        S.process(c, ARM, task, SET, prompts, tok, folds, Counter())
    finally:
        S.transcripts.reader_messages = rm
    return seen[0]


@app.function(image=image, cpu=(2.0, 2.0), memory=(16384, 16384), timeout=3600, retries=0, startup_timeout=300,
              scaledown_window=60, volumes=VOLS)
def format_check(target: str, n: int = 5, seed: int = 20261014, template: str = "fixed", tasks: list | None = None) -> dict:
    """0d's explicit format assertion (Amendment 12), counts only, on n staged runs drawn with `seed` from the target's staged
    honest_b.jsonl. Per run, the messages are rebuilt as stage_q4b.process builds them for an honest run (reader messages with
    the run's own system prompt, scrub level 1) and rendered with the corrected template:
      - prefix independence: for every assistant turn, the render of the messages up to that turn is an exact prefix of the
        full render (no later message changes an earlier turn);
      - think blocks: one per assistant turn and every one empty (the corrected template forces them; none is stripped);
      - the cut: the staged ids equal the full render's first 32,768 ids (the cut comes after a per-prefix-correct render)."""
    import random
    import re as _re
    from collections import Counter  # noqa: F401  (as in the original)

    S = _mod()
    (tok_fixed, tok_orig), folds, counted, prompts = _setup(S, SET, target)
    tok = tok_orig if template == "original" else tok_fixed  # "original": the negative control, the bug must show
    vol.reload()
    staged = Path("/ev") / TARGET["out"] / "inputs" / f"{ARM}.jsonl"
    lines = Path(staged).read_bytes().decode().splitlines()
    by_run = {f"{c['launch']}/{c['rel']}#e0": c for (arm, _t), c in counted.items() if arm == ARM}
    if tasks:  # the negative control's named runs (09's cue_D tasks in batch B's honest_b)
        want = {f"{c['launch']}/{c['rel']}#e0" for (arm, t), c in counted.items() if arm == ARM and t in tasks}
        picks = [i for i, line in enumerate(lines) if json.loads(line)["key"] in want]
    else:
        picks = random.Random(seed).sample(range(len(lines)), n)
    rows = []
    for i in sorted(picks):
        ep = json.loads(lines[i])
        c = by_run[ep["key"]]
        try:
            s = S.load(c["eval"])
            system = S.transcripts.system_prompt(s) if SET == "simh" else _system_used(S, c, ep["task"], prompts, tok, folds)
            msgs = S.scrubbed(S.transcripts.reader_messages(s, system), S.SCRUB_LEVEL)
            full = S.reader.segment_end_positions(tok, msgs, None)["ids"]
            text = tok.apply_chat_template(S.reader.prepare(tok, msgs), tokenize=False, enable_thinking=False)
            turns = [j for j, m in enumerate(msgs) if m["role"] == "assistant"]
            bad_prefix = sum(full[:len(p)] != p for p in (S.reader.segment_end_positions(tok, msgs[:j + 1], None)["ids"] for j in turns))
            thinks = _re.findall(r"<think>(.*?)</think>", text, flags=_re.S)
            rows.append({"line": i, "assistant_turns": len(turns), "prefix_mismatches": bad_prefix, "think_blocks": len(thinks),
                         "nonempty_think_blocks": sum(bool(t.strip()) for t in thinks), "staged_len": len(ep["ids"]),
                         "full_len": len(full), "cut_equal": ep["ids"] == full[:S.CUT]})
        except Exception as e:  # the error type only, never text
            rows.append({"line": i, "error": type(e).__name__})
    ok = all("error" not in r and r["prefix_mismatches"] == 0 and r["think_blocks"] == r["assistant_turns"]
             and r["nonempty_think_blocks"] == 0 and r["cut_equal"] for r in rows)
    return {"target": target, "template": template, "seed": seed, "n": n, "tasks": tasks, "rows": rows, "passes": ok}


@app.local_entrypoint()
def main(format_n: int = 5):
    import subprocess

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    r = format_check.remote(GROUP, format_n)  # 0d's format assertion on a seeded sample of the target's staged runs (Amendment 12)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results" / f"format_check_simh_{GROUP}_{stamp}.json").open("x").write(json.dumps(r | {"git": head}, indent=1))
    print(json.dumps(r))
    if not r["passes"]:
        raise SystemExit("format check FAILED")
