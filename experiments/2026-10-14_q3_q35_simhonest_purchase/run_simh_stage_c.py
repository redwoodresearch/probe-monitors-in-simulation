"""Simulated-honest purchase (RULE.md amendment 11): the CORRECTED staging. A copy of 2e's run_q4f_stage_c.py @ 50a7664ea
(the canonical corrected re-stager: the Qwen3.5 template's last-query condition forced true, so no later message changes
an earlier turn; each run rendered twice and compared; a key gate) whose only changes are:
- names and Volume: this job's app; the .eval files and outputs on probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13
  (stage_simh.py: stage_q4b.py with CACHE /ev/simh_evals_v1); output simh_q4f_v1_<SIMH_GROUP>;
- inputs: the set SIMH_SET ("simh": sim_honest@september; "atk2": sim_attack@september), the launches SIMH_LAUNCHES
  (for "atk2" also the launches of its donor honest_a/honest_b runs), and the group's upload manifest
  results/manifest_simh_eval_upload_<SIMH_GROUP>.json; c_gate counts only the staged arm's manifest files;
- 8 parts; CPU and memory limits equal to requests, startup_timeout 300, scaledown 60 (the gates' spend bound);
- main exits nonzero when the assembly stops (the record is kept). 0d's format assertion is NOT here: it is the
  real-honest fork's, copied as format_check_simh.py (RULE.md amendment 12), run after this staging.
Every failure exits nonzero after its record is written.

  SIMH_GROUP=<g> SIMH_SET=simh|atk2 SIMH_LAUNCHES=<launch>[,...] PYTHONPATH=src uv run --no-project --python 3.12 \
      --with modal==1.6.1 modal run --detach <this file>

run_q4f_stage_c.py's docstring follows.

Q4, RULE.md Amendment 11 (batch C): stage batch C (launch 2026-10-11_112714_honest_c, 196 counted real honest runs, arm
folder honest_b) in #36's reader format with the CORRECTED render. A copy of run_q4f_stage_fixed.py (B's corrected
re-stager) whose only changes are:
- the set `c`: stage_q4b's launch list and set table point at batch C (S.MAIN = [C], S.SETS["c"] = ["honest_b"]), and its
  counted attempts come from `results/manifest_q4f_c_eval_upload_v1.json` (make_q4f_c_upload_list.py, B's rules);
- each run is rendered twice, with the fixed template (staged) and with the original one (compared only): per key, whether
  the render changes the staged ids, and whether an unchanged think-less epoch is header-at-cut (step 1's rule);
- `c_gate` in place of `compare`: the staged keys plus the runs the staging excludes (each with its recorded reason) equal
  the manifest's 196 counted keys exactly, and every unchanged think-less epoch is header-at-cut;
- new app, output and result names; no compare-only path.

run_q4b_stage.py's docstring follows.

Q4 (B), RULE.md Amendment 5: stage Qwen3.5-27B's own runs in #36's reader format on Modal CPU, with stage_q4b.py (d9's
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
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q3simh_stagec_kill-2026-11-13"
VOL = "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"
HF = "probe-shift_aswin-surya_hf-cache"
GROUP, SET = os.environ["SIMH_GROUP"], os.environ["SIMH_SET"]
LAUNCHES = [x for x in os.environ["SIMH_LAUNCHES"].split(",") if x]
ARMS = {"simh": ["sim_honest@september"], "atk2": ["sim_attack@september"]}
OUT = {SET: f"simh_q4f_v1_{GROUP}"}
C_MANIFEST = f"manifest_simh_eval_upload_{GROUP}.json"
COND = "{%- if loop.index0 > ns.last_query_index %}"
PARTS = 8
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
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "SIMH_GROUP": GROUP, "SIMH_SET": SET,
               "SIMH_LAUNCHES": ",".join(LAUNCHES)}).add_local_python_source("probe_shift")
         .add_local_file(HERE / "stage_simh.py", "/root/repo/q4/stage_simh.py").add_local_file(HERE / "evalzip.py", "/root/repo/q4/evalzip.py")
         .add_local_file(HERE / "results" / C_MANIFEST, f"/root/repo/{C_MANIFEST}")
         .add_local_file(LABEL_S3, f"/root/repo/labels/{LABEL_S3.name}"))
for m in MODS + LABEL_FILES:
    image = image.add_local_file(REPO / m, f"/root/repo/{m}")


def _mod():
    import sys

    sys.path.insert(0, "/root/repo/q4")
    import stage_simh as S

    return S


def _setup(S, set_name):
    from collections import defaultdict

    from transformers import AutoTokenizer

    S.MAIN, S.SETS = LAUNCHES, dict(S.SETS) | {SET: ARMS[SET]}  # this group's launches and arm
    S.UPLOAD_MANIFEST = Path(f"/root/repo/{C_MANIFEST}")
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


@app.function(image=image, cpu=(2.0, 2.0), memory=(8192, 8192), timeout=1800, retries=0, startup_timeout=300, scaledown_window=60,
              volumes={"/ev": vol, "/hf": modal.Volume.from_name(HF).read_only()})
def part(set_name: str, k: int, n_parts: int, limit: int, out: str) -> dict:
    from collections import Counter

    S = _mod()
    (tok, tok_orig), folds, counted, prompts = _setup(S, set_name)
    im, think, asst = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<think>"), tok.encode("assistant", add_special_tokens=False)[0]

    def pattern(ids):  # step 1's rule: per kept assistant header, a <think> within its next 3 tokens
        st = [i for i, t in enumerate(ids) if t == im and i + 1 < len(ids) and ids[i + 1] == asst]
        return st, [think in ids[i + 2:i + 5] for i in st]

    todo = sorted(x for x in counted if x[0] in S.SETS[set_name] and counted[x]["launch"] in ([S.RUN2] if set_name == "run2" else S.MAIN))
    if limit:
        todo = [x for x in todo if sum(1 for y in todo[: todo.index(x)] if y[0] == x[0]) < limit]  # the first `limit` runs per arm
    mine, cov, eps, recs = todo[k::n_parts], Counter(), [], []
    for arm_dir, task in mine:
        c = counted[(arm_dir, task)]
        try:
            ep, rec = S.process(c, arm_dir, task, set_name, prompts, tok, folds, cov)
            ep0, _ = S.process(c, arm_dir, task, set_name, prompts, tok_orig, folds, Counter())  # the original render, compared only
        except Exception as e:  # never let a message carry transcript text: the error type and code locations only
            import traceback

            where = [f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in traceback.extract_tb(e.__traceback__)[-4:]]
            raise S.Withheld(f"{c['launch']}/{c['rel']}: {type(e).__name__} at {where} (message withheld)") from None
        if (ep is None) != (ep0 is None):
            raise ValueError(f"{c['launch']}/{c['rel']}: staged under one render only")
        render = None
        if ep is not None:
            st, th = pattern(ep0["ids"])
            render = {"changed": ep["ids"] != ep0["ids"], "orig_thinkless": bool(th and not all(th)),
                      "orig_header_at_cut": bool(st) and not th[-1] and all(th[:-1]) and len(ep0["ids"]) - st[-1] <= 3}
        recs.append({"arm_dir": arm_dir, **rec, "run": f"{c['launch']}/{c['rel']}", "render": render})
        if ep is not None:
            eps.append(ep)
    d = Path("/ev") / out / "parts"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"part{k:03d}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    (d / f"part{k:03d}.recs.json").write_text(json.dumps({"recs": recs, "counters": dict(cov), "no_eval": S.NO_EVAL}))
    vol.commit()
    return {"part": k, "runs": len(mine), "staged": len(eps)}


def empty_think_block(tok) -> list:
    """The template's empty think block exactly as it emits it for a reasoning-free assistant turn, tokenizer-encoded
    (AMENDMENT_cut_think_2026-10-14.md; never a hand-typed list)."""
    probe = tok.apply_chat_template([{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}], tokenize=False)
    i = probe.index("<think>")
    j = probe.index("</think>", i) + len("</think>")
    if probe[i + len("<think>"):j - len("</think>")].strip():
        raise ValueError("the template's think block for a reasoning-free turn is not empty")
    return tok.encode(probe[i:j], add_special_tokens=False)


def cut_think(ids, opens, closes, empty, cut) -> bool:
    """AMENDMENT_cut_think_2026-10-14.md: True only when the epoch is exactly `cut` tokens, its final <think> comes after
    its last </think>, and the tail from that <think> is a STRICT prefix of `empty`."""
    if len(ids) != cut or not len(opens) or (len(closes) and closes[-1] > opens[-1]):
        return False
    tail = [int(x) for x in ids[int(opens[-1]):]]
    return len(tail) < len(empty) and tail == [int(x) for x in empty[:len(tail)]]


def gate_row(ids, calls, sid, asst, tools_seq, empty, cut, submit) -> dict:
    """One staged epoch's format-gate counts; the final <think> leaves the counts only under cut_think (amendment)."""
    import numpy as np

    ids = np.asarray(ids)
    st = np.flatnonzero(ids == sid["<|im_start|>"])
    n_asst = int(sum(1 for s in st if s + 1 < len(ids) and ids[s + 1] == asst))
    has_tools = any(np.array_equal(ids[i:i + len(tools_seq)], tools_seq) for i in range(min(len(ids), 4000)))
    n_submit = sum(1 for c in calls if c.get("name") == submit)
    opens, closes = np.flatnonzero(ids == sid["<think>"]), np.flatnonzero(ids == sid["</think>"])
    flag = cut_think(ids, opens, closes, empty, cut)
    if flag:
        opens = opens[:-1]
    nonempty = sum(1 for o, c in zip(opens, closes) if c - o > 3) if len(opens) == len(closes) else -1
    return {"assistant_turns": n_asst, "think_open": len(opens), "think_close": len(closes), "think_blocks_with_content": nonempty,
            "tools_block": bool(has_tools), "submit_rows": n_submit, "think_open_truncated_at_cut": flag}


def gate_passes(r) -> bool:
    return (r["tools_block"] is False and r["submit_rows"] == 0 and r["think_open"] == r["think_close"]
            and r["think_blocks_with_content"] == 0 and r["think_open"] <= r["assistant_turns"])


@app.function(image=image, cpu=(4.0, 4.0), memory=(32768, 32768), timeout=1800, retries=0, startup_timeout=300,
              scaledown_window=60, volumes={"/ev": vol, "/hf": modal.Volume.from_name(HF).read_only()})
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
    empty = empty_think_block(tok)  # AMENDMENT_cut_think_2026-10-14.md
    gate = {}
    for arm, lst in eps.items():
        rows = [gate_row(e["ids"], e["calls"], sid, asst, tools_seq, empty, S.CUT, S.transcripts.RUNNER_SUBMIT)
                for e in sorted(lst, key=lambda e: e["key"])[:3]]
        ok = all(gate_passes(r) for r in rows)
        flagged = sum(cut_think(e["ids"], np.flatnonzero(np.asarray(e["ids"]) == sid["<think>"]),
                                np.flatnonzero(np.asarray(e["ids"]) == sid["</think>"]), empty, S.CUT) for e in lst)
        gate[arm] = {"runs": rows, "passes": ok, "think_open_truncated_at_cut_runs": int(flagged), "epochs": len(lst)}
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


@app.function(image=image, cpu=(2.0, 2.0), memory=(8192, 8192), timeout=1800, retries=0, startup_timeout=300,
              scaledown_window=60, volumes={"/ev": vol})
def c_gate(out: str) -> dict:
    """Staged keys + excluded runs == the manifest's counted keys; per key, the render comparison."""
    vol.reload()
    man = json.loads(Path(f"/root/repo/{C_MANIFEST}").read_text())
    want = {f"{f['launch']}/{f['rel']}#e0" for f in man["files"] if f["arm_dir"] in ARMS[SET]}  # not donor honest files
    staged = [json.loads(line)["key"] for line in (Path("/ev") / out / "inputs" / f"{ARMS[SET][0]}.jsonl").open()]
    recs = [r for p in sorted((Path("/ev") / out / "parts").glob("part*.recs.json")) for r in json.loads(p.read_text())["recs"]]
    if len({r["run"] for r in recs}) != len(recs):  # Codex: duplicates would collapse in the dictionaries below
        raise ValueError(f"c gate: {len(recs) - len({r['run'] for r in recs})} duplicate run records")
    excluded = {f"{r['run']}#e0": r["excluded"] for r in recs if "excluded" in r}
    render = {f"{r['run']}#e0": r["render"] for r in recs if r.get("render")}
    if len(staged) != len(set(staged)) or set(staged) & set(excluded) or set(staged) | set(excluded) != want or set(render) != set(staged):
        raise ValueError(f"c gate: staged {len(staged)}, excluded {len(excluded)}, manifest {len(want)}, rendered {len(render)}")
    changed = sorted(k for k, v in render.items() if v["changed"])
    unchanged_thinkless = sorted(k for k, v in render.items() if not v["changed"] and v["orig_thinkless"])
    bad = [k for k in unchanged_thinkless if not render[k]["orig_header_at_cut"]]
    if bad:
        raise ValueError(f"c gate: {len(bad)} unchanged think-less epochs are not header-at-cut")
    return {"manifest_keys": len(want), "staged": len(staged), "excluded": excluded, "render_changed": changed,
            "render_changed_with_thinkless_orig": sum(render[k]["orig_thinkless"] for k in changed), "unchanged_header_at_cut": unchanged_thinkless}


@app.local_entrypoint()
def main(limit: int = 0):
    import subprocess

    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    head = g("rev-parse", "HEAD")  # read once, before any spawn
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"git": head}
    for set_name in OUT:
        n = 4 if limit else PARTS
        name = OUT[set_name] + (f"_pilot{limit}" if limit else "")
        parts = list(part.starmap([(set_name, k, n, limit, name) for k in range(n)]))
        out[set_name] = {"out": name, "parts": parts, "assembled": assemble.remote(set_name, n, stamp, name)}
        if not limit and not out[set_name]["assembled"].get("STOP"):  # Amendment 11's gate
            out[set_name]["c_gate"] = c_gate.remote(name)
    (HERE / "results").mkdir(exist_ok=True)
    dest = HERE / "results" / f"simh_stage_c_{GROUP}{'_limit' + str(limit) if limit else ''}_{stamp}.json"
    dest.open("x").write(json.dumps(out, indent=1, default=str))
    print("git", head)
    for s_, v in out.items():
        if s_ in OUT:
            gate = v.get("c_gate", {})
            print(s_, json.dumps({"epochs": v["assembled"].get("epochs"), "format_gate": {a: x["passes"] for a, x in v["assembled"]["format_gate"].items()},
                                  "stop": v["assembled"].get("STOP"), "staged_inputs_sha256": v["assembled"].get("staged_inputs_sha256"),
                                  "manifest_keys": gate.get("manifest_keys"), "staged": gate.get("staged"), "excluded": gate.get("excluded"),
                                  "render_changed": len(gate.get("render_changed", [])), "unchanged_header_at_cut": len(gate.get("unchanged_header_at_cut", []))}))
    print(dest.name)
    for s_ in OUT:  # fail closed (the record above is kept); a failed key gate raises in c_gate
        if out[s_]["assembled"].get("STOP"):
            raise SystemExit(f"{s_}: staging FAILED (format gate); see {dest.name}")
