"""1e-4 real honest runs (RULE.md here, Amendment 6): a copy of 2e's run_q4f_stage_c.py (aswin/q4-why-weaker @ 50a7664ea) whose
only changes are names and inputs: this job's app and Volume; two targets, `c` (batch C again, from 2e's Volume read only:
the positive control, which must reproduce batch C's staged honest_b.jsonl byte for byte, sha256 513f621a...) and `rh` (this
job's launch, 2026-10-14_135334_honest_rh980, its .eval files on this job's Volume under evals_v1/, its upload manifest from
make_rh_upload_list.py); for `rh` the frozen folds also carry each task's replicate keys (<key>_r1.._r5, the base key's
fold); `add_repro` writes the extraction's per-shard reference arm (repro-c). stage_q4b.py and evalzip.py are 2e's,
unchanged. run_q4f_stage_c.py's docstring follows.

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
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q3realhonest_stage_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q3realhonest_data_kill-2026-11-14"  # this job's .eval files (evals_v1/<launch>/) and staged sets
SRC_C = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"  # 2e's Volume, read only: batch C's .eval files
HF = "probe-shift_aswin-surya_hf-cache"
TARGETS = {"c": {"launch": "2026-10-11_112714_honest_c", "manifest": "manifest_q4f_c_eval_upload_v1.json", "cache": "/src/evals_v1",
                 "out": "rh_c_restage_v1", "expect_sha256": "513f621a9e71e91f8188cccc76cb222e9c9f4acb5a91fc90c45a283964729b31"},
           "rh": {"launch": ["2026-10-14_135334_honest_rh980", "2026-10-14_160335_honest_rh_relaunch"], "manifest": "manifest_rh1e4_eval_upload_v1.json", "cache": "/ev/evals_v1",
                  "out": "rh1e4_stage_v1", "expect_sha256": None}}
# the format check's negative control (Amendment 12): batch B's UNCORRECTED staging (2e's q4b_main_v1, old render) and .eval files,
# 2e's Volume read only; set "main" as in 2e's B manifest
B_NEG = {"launch": ["2026-10-09_000523_rerun_strict", "2026-10-09_173815_scaleup_canary", "2026-10-09_182505_scaleup", "2026-10-10_001314_scaleup_topup"],
         "manifest": "manifest_q4b_eval_upload_v1.json", "cache": "/src/evals_v1", "staged": "/src/q4b_main_v1/inputs/honest_b.jsonl", "set": "main"}
CUE_RESULT = ("0798ff6d", "experiments/2026-10-14_format_smell_09/results/cue_cause_2026-10-14_103513.json")  # 09's cue_D lists
REPRO = {"seed": 20261014, "copies_per_shard": 1}  # add_repro: batch C's longest staged epoch and one drawn by seed, once per shard
COND = "{%- if loop.index0 > ns.last_query_index %}"
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
         .add_local_file(LABEL_S3, f"/root/repo/labels/{LABEL_S3.name}"))
for t in [*TARGETS.values(), B_NEG]:  # a target's manifest exists once it is made (rh: after the runs)
    if (HERE / "config" / t["manifest"]).exists():
        image = image.add_local_file(HERE / "config" / t["manifest"], f"/root/repo/{t['manifest']}")
for m in MODS + LABEL_FILES:
    image = image.add_local_file(REPO / m, f"/root/repo/{m}")
VOLS = {"/ev": vol, "/src": modal.Volume.from_name(SRC_C, version=2).read_only(), "/hf": modal.Volume.from_name(HF).read_only()}


def _mod():
    import sys

    sys.path.insert(0, "/root/repo/q4")
    import stage_q4b as S

    return S


def _setup(S, set_name, target):
    from collections import defaultdict

    from transformers import AutoTokenizer

    T = B_NEG if target == "b" else TARGETS[target]
    launches = T["launch"] if isinstance(T["launch"], list) else [T["launch"]]  # rh: launch 1 and its relaunch (Amendment 9)
    S.MAIN, S.SETS = launches, dict(S.SETS) | {"c": ["honest_b"]}  # Amendment 11: batch C (here: the target's launches)
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
    if target == "rh":  # replicate keys carry their base task's fold
        folds |= {f"{k}_r{i}": v for k, v in list(folds.items()) for i in range(1, 6)}
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


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes=VOLS)
def part(set_name: str, k: int, n_parts: int, limit: int, out: str, target: str) -> dict:
    from collections import Counter

    S = _mod()
    (tok, tok_orig), folds, counted, prompts = _setup(S, set_name, target)
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


@app.function(image=image, cpu=4, memory=32768, timeout=1800, retries=0, volumes=VOLS)
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


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes={"/ev": vol})
def c_gate(out: str, target: str) -> dict:
    """Staged keys + excluded runs == the manifest's counted keys; per key, the render comparison."""
    vol.reload()
    man = json.loads(Path(f"/root/repo/{TARGETS[target]['manifest']}").read_text())
    want = {f"{f['launch']}/{f['rel']}#e0" for f in man["files"]}
    staged = [json.loads(line)["key"] for line in (Path("/ev") / out / "inputs" / "honest_b.jsonl").open()]
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


@app.function(image=image, cpu=2, memory=16384, timeout=1800, retries=0, volumes={"/ev": vol})
def add_repro(shards: int) -> dict:
    """The extraction's per-shard reference arm: batch C's longest staged epoch (32,768 tokens or the longest there is) and one
    drawn with REPRO's seed, from the `c` re-stage (its honest_b.jsonl must still hash to batch C's), each copied once per
    extraction shard under its own key (<key>~ref<j>) and arm "repro-c", into the `rh` set's inputs. Equal-length copies sort
    next to each other in the extractor's (-length, key) order and so are dealt one to each shard."""
    import random

    vol.reload()
    c_in = Path("/ev") / TARGETS["c"]["out"] / "inputs" / "honest_b.jsonl"
    raw = c_in.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TARGETS["c"]["expect_sha256"]:
        raise ValueError("the c re-stage no longer hashes to batch C's staged honest_b.jsonl")
    eps = sorted((json.loads(line) for line in raw.decode().splitlines()), key=lambda e: (-len(e["ids"]), e["key"]))
    picks = [eps[0], random.Random(REPRO["seed"]).choice(eps[1:])]
    dest = Path("/ev") / TARGETS["rh"]["out"] / "inputs" / "repro-c.jsonl"
    if dest.exists():
        raise FileExistsError(f"{dest} exists")
    lines = [json.dumps(e | {"key": f"{e['key']}~ref{j:03d}", "arm": "repro-c"}) + "\n" for e in picks for j in range(shards * REPRO["copies_per_shard"])]
    b = "".join(lines).encode()
    dest.write_bytes(b)
    vol.commit()
    return {"repro_sources": [e["key"] for e in picks], "lengths": [len(e["ids"]) for e in picks], "lines": len(lines),
            "sha256": hashlib.sha256(b).hexdigest()}


@app.function(image=image, cpu=2, memory=16384, timeout=1800, retries=0, volumes={"/ev": vol})
def make_ref_subset() -> dict:
    """The positive control's extraction inputs: from the `c` re-stage (still batch C's bytes), the 16 longest staged epochs (one
    per shard of batch C's 16-shard extraction, q4b_extract_c_v1: each shard's longest) and 4 more drawn with REPRO's seed, their
    lines byte for byte, sorted by key, to rh_c_ref_v1/inputs/honest_b.jsonl."""
    import random

    vol.reload()
    raw = (Path("/ev") / TARGETS["c"]["out"] / "inputs" / "honest_b.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != TARGETS["c"]["expect_sha256"]:
        raise ValueError("the c re-stage no longer hashes to batch C's staged honest_b.jsonl")
    lines = raw.decode().splitlines(keepends=True)
    order = sorted(range(len(lines)), key=lambda i: (-len(json.loads(lines[i])["ids"]), json.loads(lines[i])["key"]))
    pick = order[:16] + random.Random(REPRO["seed"]).sample(order[16:], 4)
    chosen = sorted((lines[i] for i in pick), key=lambda x: json.loads(x)["key"])
    dest = Path("/ev") / "rh_c_ref_v1" / "inputs" / "honest_b.jsonl"
    if dest.exists():
        raise FileExistsError(f"{dest} exists")
    dest.parent.mkdir(parents=True)
    b = "".join(chosen).encode()
    dest.write_bytes(b)
    vol.commit()
    return {"epochs": len(chosen), "lengths": sorted((len(json.loads(x)["ids"]) for x in chosen), reverse=True), "sha256": hashlib.sha256(b).hexdigest()}


@app.function(image=image, cpu=2, memory=16384, timeout=3600, retries=0, volumes=VOLS)
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
    from collections import Counter

    S = _mod()
    (tok_fixed, tok_orig), _, counted, _ = _setup(S, B_NEG["set"] if target == "b" else "c", target)
    tok = tok_orig if template == "original" else tok_fixed  # "original": the negative control, the bug must show
    vol.reload()
    staged = B_NEG["staged"] if target == "b" else Path("/ev") / TARGETS[target]["out"] / "inputs" / "honest_b.jsonl"
    lines = Path(staged).read_bytes().decode().splitlines()
    by_run = {f"{c['launch']}/{c['rel']}#e0": c for (arm, _t), c in counted.items() if arm == "honest_b"}
    if tasks:  # the negative control's named runs (09's cue_D tasks in batch B's honest_b)
        want = {f"{c['launch']}/{c['rel']}#e0" for (arm, t), c in counted.items() if arm == "honest_b" and t in tasks}
        picks = [i for i, line in enumerate(lines) if json.loads(line)["key"] in want]
    else:
        picks = random.Random(seed).sample(range(len(lines)), n)
    rows = []
    for i in sorted(picks):
        ep = json.loads(lines[i])
        c = by_run[ep["key"]]
        try:
            s = S.load(c["eval"])
            msgs = S.scrubbed(S.transcripts.reader_messages(s, S.transcripts.system_prompt(s)), S.SCRUB_LEVEL)
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
def main(limit: int = 0, target: str = "c", repro_shards: int = 0, ref_subset: bool = False, format_n: int = 0, negative: bool = False,
         c_gate_only: bool = False):
    if negative:  # the format check's negative control: 2 seeded cue_D runs of batch B, original template, uncorrected staging
        import random
        import subprocess as sp

        lists = json.loads(sp.run(["git", "show", f"{CUE_RESULT[0]}:{CUE_RESULT[1]}"], capture_output=True, text=True, check=True, cwd=REPO).stdout)
        cue = sorted(t for w, t in lists["analysis"]["cue_lists"]["cue_D"] if w == "honest_b")
        tasks = random.Random(REPRO["seed"]).sample(cue, 2)
        r = format_check.remote("b", 2, REPRO["seed"], "original", tasks)
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        (HERE / "results" / f"format_check_negative_b_{stamp}.json").open("x").write(json.dumps(r, indent=1))
        print(json.dumps(r))
        # Codex round 16: the bug must SHOW on both named runs (no error rows, prefix mismatches or stripped think blocks)
        shows = len(r["rows"]) == 2 and all("error" not in x and (x["prefix_mismatches"] > 0 or x["think_blocks"] < x["assistant_turns"]) for x in r["rows"])
        if not shows:
            raise SystemExit("negative control GATE FAILED: the format check did not show the bug on both runs (blind, or an error)")
        return
    if format_n:  # 0d's format assertion on a seeded sample of the target's staged runs (Amendment 12)
        r = format_check.remote(target, format_n)
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        (HERE / "results" / f"format_check_{target}_{stamp}.json").open("x").write(json.dumps(r, indent=1))
        print(json.dumps(r))
        if not r["passes"]:
            raise SystemExit("format check GATE FAILED")
        return
    if c_gate_only:  # a resumed staging whose client dropped: the c_gate again over the staged inputs on the Volume
        g = c_gate.remote(TARGETS[target]["out"], target)
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        (HERE / "results" / f"rh_stage_{target}_cgate_{stamp}.json").open("x").write(json.dumps(g, indent=1, default=str))
        print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in g.items()}))
        return
    if repro_shards:  # after the rh staging: the reference arm only
        print(json.dumps(add_repro.remote(repro_shards)))
        return
    if ref_subset:  # after the c staging: the positive control's extraction inputs only
        print(json.dumps(make_ref_subset.remote()))
        return
    OUT = {"c": TARGETS[target]["out"]}
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
        parts = list(part.starmap([(set_name, k, n, limit, name, target) for k in range(n)]))
        out[set_name] = {"out": name, "parts": parts, "assembled": assemble.remote(set_name, n, stamp, name)}
        if not limit and not out[set_name]["assembled"].get("STOP"):  # Amendment 11's gate
            out[set_name]["c_gate"] = c_gate.remote(name, target)
            want = TARGETS[target]["expect_sha256"]  # target c: the positive control, byte for byte
            if want:
                got = out[set_name]["assembled"].get("staged_inputs_sha256", {}).get("honest_b.jsonl")
                out[set_name]["positive_control"] = {"want": want, "got": got, "passes": got == want}
    (HERE / "results").mkdir(exist_ok=True)
    dest = HERE / "results" / f"rh_stage_{target}{'_limit' + str(limit) if limit else ''}_{stamp}.json"
    dest.open("x").write(json.dumps(out, indent=1, default=str))
    print("git", head)
    for s_, v in out.items():
        if s_ in OUT:
            gate = v.get("c_gate", {})
            print(s_, json.dumps({"epochs": v["assembled"].get("epochs"), "format_gate": {a: x["passes"] for a, x in v["assembled"]["format_gate"].items()},
                                  "stop": v["assembled"].get("STOP"), "staged_inputs_sha256": v["assembled"].get("staged_inputs_sha256"),
                                  "manifest_keys": gate.get("manifest_keys"), "staged": gate.get("staged"), "excluded": gate.get("excluded"),
                                  "render_changed": len(gate.get("render_changed", [])), "unchanged_header_at_cut": len(gate.get("unchanged_header_at_cut", [])),
                                  "positive_control": v.get("positive_control")}))
    print(dest.name)
    for s_ in OUT:  # Codex round 6: a failed gate or positive control fails the command, after the record is saved
        v = out[s_]
        if v["assembled"].get("STOP") or (not limit and "c_gate" not in v):
            raise SystemExit(f"{s_}: staging GATE FAILED ({v['assembled'].get('STOP')}); see {dest.name}")
        if v.get("positive_control") and not v["positive_control"]["passes"]:
            raise SystemExit(f"positive control GATE FAILED: staged honest_b.jsonl {v['positive_control']['got']} != batch C's {v['positive_control']['want']}")
