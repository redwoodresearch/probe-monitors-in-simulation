"""Overnight chain (RULE.md here): the real-honest fork's run_rh_stage.py (origin/aswin/q3-q35-realhonest-1e4 @ 2f0932469, its Codex-diffed
copy of 2e's run_q4f_stage_c.py), generalized only in names and inputs: the chain's app and Volume; the scale-up targets read from
config/chain_targets.json (per target: launches, its upload manifest, set, the arms it stages, output name); `c` (the positive
control) and `b` (the format check's negative control) unchanged. Its docstring follows.

1e-4 real honest runs (RULE.md here, Amendment 6): a copy of 2e's run_q4f_stage_c.py (aswin/q4-why-weaker @ 50a7664ea) whose
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
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
CHAIN_VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"  # the chain's .eval files (evals_v1/<launch>/) and staged sets
# d9's CUT fork (Append 41): a separate batch may stage into its OWN app and Volume (env CHAIN_STAGE_APP / CHAIN_STAGE_VOLUME, both
# unset for the chain: exactly the names below). The chain's Volume is then mounted READ ONLY at /chainro (its .eval files).
APP = os.environ.get("CHAIN_STAGE_APP") or "probe-shift_aswin-surya_sub-q3chain_stage_kill-2026-11-14"
VOL = os.environ.get("CHAIN_STAGE_VOLUME") or CHAIN_VOL
CHAIN_APP = "probe-shift_aswin-surya_sub-q3chain_stage_kill-2026-11-14"
if (APP == CHAIN_APP) != (VOL == CHAIN_VOL) or not APP.endswith("_kill-2026-11-14"):
    raise ValueError("a separate batch needs BOTH its own app (..._kill-2026-11-14) and its own Volume")
# a separate batch may also read other Volumes' .eval files, READ ONLY: env CHAIN_STAGE_EXTRA_RO = '{"/mnt": "<volume name>", ...}'
EXTRA_RO = json.loads(os.environ.get("CHAIN_STAGE_EXTRA_RO") or "{}") if VOL != CHAIN_VOL else {}
if any(not m.startswith("/") or m in ("/ev", "/src", "/hf", "/chainro") or n == VOL for m, n in EXTRA_RO.items()):
    raise ValueError("CHAIN_STAGE_EXTRA_RO: absolute mount points, not /ev /src /hf /chainro, never the batch's own Volume")
_OVR_ENV = {} if VOL == CHAIN_VOL else {"CHAIN_STAGE_APP": APP, "CHAIN_STAGE_VOLUME": VOL,
                                        "CHAIN_STAGE_EXTRA_RO": json.dumps(EXTRA_RO, sort_keys=True)}  # the container sees the same names
SRC_C = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"  # 2e's Volume, read only: batch C's .eval files
HF = "probe-shift_aswin-surya_hf-cache"
TARGETS = {"c": {"launch": "2026-10-11_112714_honest_c", "manifest": "manifest_q4f_c_eval_upload_v1.json", "cache": "/src/evals_v1",
                 "out": "rh_c_restage_v1", "expect_sha256": "513f621a9e71e91f8188cccc76cb222e9c9f4acb5a91fc90c45a283964729b31"},
           }
CHAIN_TARGETS = Path(os.environ["CHAIN_TARGETS_FILE"]) if os.environ.get("CHAIN_TARGETS_FILE") else HERE / "config" / "chain_targets.json"  # the scale-up targets (chain.py writes one per staged set before it stages it)
if os.environ.get("CHAIN_TARGETS_FILE") and VOL == CHAIN_VOL:
    raise ValueError("a private targets file (CHAIN_TARGETS_FILE) stages only into a separate batch's own Volume")
_snap = None
if CHAIN_TARGETS.exists():  # laptop: ONE private copy per process, read and shipped, so another batch's later write to the shared file never lands mid-build
    import tempfile
    _snap = Path(tempfile.mkdtemp(prefix="chain_targets_")) / "chain_targets.json"
    _snap.write_bytes(CHAIN_TARGETS.read_bytes())
_ct = next((p for p in (_snap, Path("/root/repo/chain_targets.json")) if p is not None and p.exists()), None)  # laptop, then the container
if _ct:
    TARGETS |= json.loads(_ct.read_text())
# the format check's negative control (Amendment 12): batch B's UNCORRECTED staging (2e's q4b_main_v1, old render) and .eval files,
# 2e's Volume read only; set "main" as in 2e's B manifest
B_NEG = {"launch": ["2026-10-09_000523_rerun_strict", "2026-10-09_173815_scaleup_canary", "2026-10-09_182505_scaleup", "2026-10-10_001314_scaleup_topup"],
         "manifest": "manifest_q4b_eval_upload_v1.json", "cache": "/src/evals_v1", "staged": "/src/q4b_main_v1/inputs/honest_b.jsonl", "set": "main"}
CUE_RESULT = ("0798ff6d", "experiments/2026-10-14_format_smell_09/results/cue_cause_2026-10-14_103513.json")  # 09's cue_D lists
REPRO = {"seed": 20261014, "copies_per_shard": 1}  # add_repro: batch C's longest staged epoch and one drawn by seed, once per shard
COND = "{%- if loop.index0 > ns.last_query_index %}"
PARTS = 48
# the stager's token cut, per target (config/chain_targets.json field "cut"; d9's CUT fork, Append 41, 10-15): absent -> 32,768,
# exactly as before (stage_q4b's own constant, unchanged); "full" -> no cut (reader max_tokens=None); otherwise a positive int.
CUT_DEFAULT = 32768


def _cut_of(t: dict) -> int | None:
    c = t.get("cut", CUT_DEFAULT)
    if c == "full":
        return None
    if isinstance(c, bool) or not isinstance(c, int) or c <= 0:
        raise ValueError(f"target cut must be a positive int or 'full', not {c!r}")
    return c


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
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"} | _OVR_ENV).add_local_python_source("probe_shift")
         .add_local_file(HERE / "stage_q4b.py", "/root/repo/q4/stage_q4b.py").add_local_file(HERE / "evalzip.py", "/root/repo/q4/evalzip.py")
        .add_local_file(HERE / "fmt_gate.py", "/root/repo/q4/fmt_gate.py")
         .add_local_file(LABEL_S3, f"/root/repo/labels/{LABEL_S3.name}"))
if _snap is not None:
    image = image.add_local_file(_snap, "/root/repo/chain_targets.json")
for t in [*TARGETS.values(), B_NEG]:  # a target's manifest exists once it is made (rh: after the runs)
    if (HERE / "config" / t["manifest"]).exists():
        image = image.add_local_file(HERE / "config" / t["manifest"], f"/root/repo/{t['manifest']}")
for m in MODS + LABEL_FILES:
    image = image.add_local_file(REPO / m, f"/root/repo/{m}")
VOLS = {"/ev": vol, "/src": modal.Volume.from_name(SRC_C, version=2).read_only(), "/hf": modal.Volume.from_name(HF).read_only()}
if VOL != CHAIN_VOL:  # a separate batch: the chain's Volume read only (its targets' cache is /chainro/evals_v1)
    VOLS["/chainro"] = modal.Volume.from_name(CHAIN_VOL, version=2).read_only()
    for _m, _n in EXTRA_RO.items():
        VOLS[_m] = modal.Volume.from_name(_n).read_only()


def _mod():
    import sys

    sys.path.insert(0, "/root/repo/q4")
    import stage_q4b as S

    return S


def _setup(S, set_name, target):
    from collections import defaultdict

    from transformers import AutoTokenizer

    T = B_NEG if target == "b" else TARGETS[target]
    S.CUT = _cut_of(T)  # absent: 32,768, the module's own value (byte-identical default)
    launches = T["launch"] if isinstance(T["launch"], list) else [T["launch"]]  # rh: launch 1 and its relaunch (Amendment 9)
    S.MAIN, S.SETS = launches, dict(S.SETS) | {"c": ["honest_b"]}  # Amendment 11: batch C (here: the target's launches)
    if T.get("run2"):  # a scale-up main launch's interleaved second attack: stage_q4b's run2 set, the same launch
        S.RUN2 = T["run2"]
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


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes=VOLS)
def part(set_name: str, k: int, n_parts: int, limit: int, out: str, target: str) -> dict:
    from collections import Counter

    S = _mod()
    (tok, tok_orig), folds, counted, prompts = _setup(S, set_name, target)
    im, think, asst = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<think>"), tok.encode("assistant", add_special_tokens=False)[0]

    def pattern(ids):  # step 1's rule: per kept assistant header, a <think> within its next 3 tokens
        st = [i for i, t in enumerate(ids) if t == im and i + 1 < len(ids) and ids[i + 1] == asst]
        return st, [think in ids[i + 2:i + 5] for i in st]

    arms = TARGETS[target].get("stage_arms") or S.SETS[set_name]  # a chain target stages only its own arms (donors stay unstaged)
    todo = sorted(x for x in counted if x[0] in S.SETS[set_name] and x[0] in arms and counted[x]["launch"] in ([S.RUN2] if set_name == "run2" else S.MAIN))
    only = TARGETS[target].get("only_keys")  # d9's CUT fork (Append 41 step 0): a gate target stages exactly these run keys (absent: all)
    if only is not None:
        todo = [x for x in todo if f"{counted[x]['launch']}/{counted[x]['rel']}#e0" in set(only)]
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
    (d / f"part{k:03d}.recs.json").write_text(json.dumps({"recs": recs, "counters": dict(cov), "no_eval": S.NO_EVAL,
                                                          "first_user_checks": getattr(S, "FU_CHECKS", {})}))  # 09's donor rule section 4
    vol.commit()
    return {"part": k, "runs": len(mine), "staged": len(eps)}


@app.function(image=image, cpu=4, memory=32768, timeout=1800, retries=0, volumes=VOLS)
def assemble(set_name: str, n_parts: int, stamp: str, out: str, target: str | None = None) -> dict:
    from collections import defaultdict

    import numpy as np
    from transformers import AutoTokenizer

    S = _mod()
    if target is not None:  # the target's cut (absent: 32,768, unchanged)
        S.CUT = _cut_of(B_NEG if target == "b" else TARGETS[target])
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
    # the format gate: 3 staged runs per arm, counts on the staged ids only; from the pinned module fmt_gate.py (one file, its
    # sha256 in the record), so a shared fix to the gate replaces that file and nothing here
    import sys as _sys

    _sys.path.insert(0, "/root/repo/q4")
    import fmt_gate

    tok = AutoTokenizer.from_pretrained(S.QWEN35_27B["name"], revision=S.QWEN35_27B["revision"], local_files_only=True)
    gate = fmt_gate.gate(eps, tok, S)
    record = {"set": set_name, "stamp": stamp, "cut": "full" if S.CUT is None else S.CUT, "scrub_level": S.SCRUB_LEVEL, "model": S.QWEN35_27B,
              "fmt_gate_sha256": hashlib.sha256(Path("/root/repo/q4/fmt_gate.py").read_bytes()).hexdigest(),
              "labeler_fingerprint": S.fingerprint(), "counters": dict(counters), "format_gate": gate,
              "coverage": S.summarize(per_run, S.attempt_labels(), set_name), "per_run": per_run}
    if target is not None and "cut" in TARGETS.get(target, {}):  # d9's CUT fork: the staged render's template (Amendment 9 fix, all epochs), pinned
        record["chat_template_sha256"] = {"fixed_staged": hashlib.sha256(tok.chat_template.replace(COND, "{%- if true %}").encode()).hexdigest(),
                                          "original": hashlib.sha256(tok.chat_template.encode()).hexdigest()}
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
    return {k: record[k] for k in ("set", "epochs", "staged_inputs_sha256", "format_gate", "fmt_gate_sha256")} | {"coverage": record["coverage"]}


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes={"/ev": vol})
def c_gate(out: str, target: str) -> dict:
    """Staged keys + excluded runs == the manifest's counted keys; per key, the render comparison."""
    vol.reload()
    T = TARGETS[target]
    man = json.loads(Path(f"/root/repo/{T['manifest']}").read_text())
    arms = T.get("stage_arms") or ["honest_b"]  # c: honest_b, as before; a chain target: the arms it stages (not its donors)
    want = {f"{f['launch']}/{f['rel']}#e0" for f in man["files"] if f["arm_dir"] in arms}
    if T.get("only_keys") is not None:  # a gate target: exactly its listed keys, each one a counted key of the manifest
        if not set(T["only_keys"]) <= want:
            raise ValueError(f"c gate: {len(set(T['only_keys']) - want)} only_keys are not counted keys of {T['manifest']}")
        want = set(T["only_keys"])
    staged = [json.loads(line)["key"] for a in arms if (Path("/ev") / out / "inputs" / f"{a}.jsonl").exists()
              for line in (Path("/ev") / out / "inputs" / f"{a}.jsonl").open()]
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
def add_repro(shards: int, target: str = "rh", suffix: str = "") -> dict:
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
    dest = Path("/ev") / (TARGETS[target]["out"] + suffix) / "inputs" / "repro-c.jsonl"
    lines = [json.dumps(e | {"key": f"{e['key']}~ref{j:03d}", "arm": "repro-c"}) + "\n" for e in picks for j in range(shards * REPRO["copies_per_shard"])]
    b = "".join(lines).encode()
    if dest.exists():  # the chain: a rerun after a lost client accepts the same bytes, rejects others
        if dest.read_bytes() != b:
            raise FileExistsError(f"{dest} exists with other bytes")
    else:
        tmp = dest.with_name(".repro-c.jsonl.tmp")
        tmp.write_bytes(b)
        tmp.rename(dest)
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
def format_check(target: str, n: int = 5, seed: int = 20261014, template: str = "fixed", tasks: list | None = None, arm: str = "honest_b",
                 suffix: str = "") -> dict:
    """0d's explicit format assertion (Amendment 12), counts only, on n staged runs drawn with `seed` from the target's staged
    honest_b.jsonl. Per run, the messages are rebuilt as stage_q4b.process builds them for an honest run (reader messages with
    the run's own system prompt, scrub level 1) and rendered with the corrected template:
      - prefix independence: for every assistant turn, the render of the messages up to that turn is an exact prefix of the
        full render (no later message changes an earlier turn);
      - think blocks: one per assistant turn and every one empty (the corrected template forces them; none is stripped);
      - the cut: the staged ids equal the full render's first S.CUT ids (32,768 unless the target sets "cut"; the whole render
        when "full"; the cut comes after a per-prefix-correct render)."""
    import random
    import re as _re
    from collections import Counter

    S = _mod()
    (tok_fixed, tok_orig), _, counted, prompts = _setup(S, B_NEG["set"] if target == "b" else TARGETS[target].get("set", "c"), target)
    tok = tok_orig if template == "original" else tok_fixed  # "original": the negative control, the bug must show
    vol.reload()
    staged = B_NEG["staged"] if target == "b" else Path("/ev") / (TARGETS[target]["out"] + suffix) / "inputs" / f"{arm}.jsonl"
    raw_staged = Path(staged).read_bytes()
    staged_sha = hashlib.sha256(raw_staged).hexdigest()  # the chain: the record names the staged set it covers
    lines = raw_staged.decode().splitlines()
    by_run = {f"{c['launch']}/{c['rel']}#e0": (c, t) for (a, t), c in counted.items() if a == arm}
    if tasks:  # the negative control's named runs (09's cue_D tasks in batch B's honest_b)
        want = {f"{c['launch']}/{c['rel']}#e0" for (a, t), c in counted.items() if a == arm and t in tasks}
        picks = [i for i, line in enumerate(lines) if json.loads(line)["key"] in want]
    else:
        picks = random.Random(seed).sample(range(len(lines)), n)
    rows = []
    for i in sorted(picks):
        ep = json.loads(lines[i])
        c, task = by_run[ep["key"]]
        try:
            s = S.load(c["eval"])
            system = S.transcripts.system_prompt(s)
            if not (arm.startswith("honest") or arm.startswith("sim_honest")):  # an attack arm: the donor prompt stage_q4b.process used
                cover, fu = S.COVER.search(system), S.first_user_sha(s)
                cands = [(h, prompts.get(task, {}).get(h)) for h in ("honest_a", "honest_b")]
                pick = [h for h, p in cands if p and cover and cover.group(1).strip() in p[0] and "You are an agent" in p[0] and p[1] == fu]
                system = prompts[task][pick[0]][0]
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
    return {"target": target, "arm": arm, "staged_file": str(staged), "staged_sha": staged_sha, "template": template, "seed": seed, "n": n,
            "tasks": tasks, "rows": rows, "passes": ok}


@app.function(image=image, cpu=1, memory=4096, timeout=3 * 3600, retries=0, volumes=VOLS)
def stage_driver(set_name: str, n: int, limit: int, name: str, target: str, stamp: str, head: str) -> dict:
    """The chain (10-14 ~23:00, 0d: a laptop disconnect must not cancel staging): the parts fan-out, assemble and the c gate run
    HERE, on Modal. A started marker and the result go to the Volume (stage_results/<name>.started.json, <name>.json), so a
    re-attached or restarted chain reads them from the Volume."""
    import io

    with vol.batch_upload(force=True) as b:
        b.put_file(io.BytesIO(json.dumps({"stage_driver_call_id": modal.current_function_call_id(), "name": name, "stamp": stamp}).encode()),
                   f"stage_results/{name}.started.json")
    try:
        parts = list(part.starmap([(set_name, k, n, limit, name, target) for k in range(n)]))
        res = {"out": name, "parts": parts, "assembled": assemble.remote(set_name, n, stamp, name, target)}
        if not limit and not res["assembled"].get("STOP"):  # Amendment 11's gate
            res["c_gate"] = c_gate.remote(name, target)
            want = TARGETS[target]["expect_sha256"]  # target c: the positive control, byte for byte
            if want:
                got = res["assembled"].get("staged_inputs_sha256", {}).get("honest_b.jsonl")
                res["positive_control"] = {"want": want, "got": got, "passes": got == want}
    except Exception as e:  # a terminal outcome on the Volume, so a re-attaching chain tells a failed gate from a failed read
        with vol.batch_upload(force=True) as b:
            b.put_file(io.BytesIO(json.dumps({"failed": type(e).__name__, "message": str(e)[:500]}).encode()), f"stage_results/{name}.failed.json")
        raise
    rec = {"git": head, "stamp": stamp, set_name: res}
    with vol.batch_upload(force=True) as b:
        b.put_file(io.BytesIO(json.dumps(rec, indent=1, default=str).encode()), f"stage_results/{name}.json")
    return rec


@app.function(image=image, cpu=2, memory=16384, timeout=1800, retries=0, volumes=VOLS)
def staged_stats(target: str, arms: list, suffix: str = "") -> list:
    """09 / 0d (Append 21g): per staged set, its epoch count, the longest staged epoch in tokens and the epochs over 32,768
    (expected 0: the stager cuts at S.CUT, so batch C's 32,768-token reference epoch in repro-c is at least every shard's longest)."""
    vol.reload()
    out = []
    for arm in arms:
        staged = Path("/ev") / (TARGETS[target]["out"] + suffix) / "inputs" / f"{arm}.jsonl"
        raw = staged.read_bytes()
        lens = [len(json.loads(line)["ids"]) for line in raw.decode().splitlines()]
        row = {"target": target, "arm": arm, "staged_file": str(staged), "staged_sha": hashlib.sha256(raw).hexdigest(), "epochs": len(lens),
               "max_staged_tokens": max(lens, default=0), "n_epochs_over_32768": sum(n > 32768 for n in lens)}
        if "cut" in TARGETS[target]:  # a target with its own cut (absent: these rows are exactly as before)
            cut = _cut_of(TARGETS[target])
            row |= {"cut": "full" if cut is None else cut, "n_epochs_over_cut": 0 if cut is None else sum(n > cut for n in lens)}
        out.append(row)
    return out


@app.local_entrypoint()
def main(limit: int = 0, target: str = "c", repro_shards: int = 0, ref_subset: bool = False, format_n: int = 0, negative: bool = False,
         arm: str = "honest_b", suffix: str = "", stats_arms: str = ""):
    T_ = B_NEG if target == "b" else TARGETS.get(target, {})
    if ("cut" in T_ or "only_keys" in T_ or "parts" in T_) and VOL == CHAIN_VOL:  # d9's CUT fork: never into the chain's Volume, any branch
        raise SystemExit(f"{target} is a separate batch's target; set CHAIN_STAGE_APP / CHAIN_STAGE_VOLUME")
    if stats_arms:  # the chain's attestation: each staged set's token counts (comma-separated arms)
        r = {"target": target, "suffix": suffix, "sets": staged_stats.remote(target, stats_arms.split(","), suffix)}
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        (HERE / "results" / f"staged_stats_{target}_{stamp}.json").open("x").write(json.dumps(r, indent=1))
        print(json.dumps(r))
        return
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
        if r["passes"]:
            raise SystemExit("negative control PASSED: the format check is blind; fix it before relying on it")
        shows = [x for x in r["rows"] if "error" not in x and x["prefix_mismatches"] > 0 and x["think_blocks"] < x["assistant_turns"]]
        if len(r["rows"]) != 2 or len(shows) != 2:  # the chain: a read error must not count as the check catching the bug
            raise SystemExit(f"negative control INCONCLUSIVE: {len(shows)} of {len(r['rows'])} rows show the defect without a read error")
        return
    if format_n:  # 0d's format assertion on a seeded sample of the target's staged runs (Amendment 12)
        r = format_check.remote(target, format_n, arm=arm, suffix=suffix)
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        (HERE / "results" / f"format_check_{target}_{arm}_{stamp}.json").open("x").write(json.dumps(r, indent=1))
        print(json.dumps(r))
        if any("error" in x for x in r["rows"]):  # the chain: a row that could not be read is not a validation result; retried
            raise SystemExit(f"format check READ ERROR on {sum('error' in x for x in r['rows'])} rows (retry)")
        if not r["passes"]:
            raise SystemExit("format check FAILED")
        return
    if repro_shards:  # after the rh staging: the reference arm only
        print(json.dumps(add_repro.remote(repro_shards, target, suffix)))
        return
    if ref_subset:  # after the c staging: the positive control's extraction inputs only
        print(json.dumps(make_ref_subset.remote()))
        return
    OUT = {TARGETS[target].get("set", "c"): TARGETS[target]["out"]}  # c: set "c"; a chain target: its own set
    import subprocess

    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    head = g("rev-parse", "HEAD")  # read once, before any spawn
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"git": head}
    T = TARGETS[target]
    for set_name in OUT:
        n = 4 if limit else int(T.get("parts", PARTS))  # a gate target: one part per run key (absent: PARTS, as before)
        name = OUT[set_name] + suffix + (f"_pilot{limit}" if limit else "")
        call = stage_driver.spawn(set_name, n, limit, name, target, stamp, head)  # run with --detach: a disconnect cannot cancel it
        print(json.dumps({"stage_driver_call_id": call.object_id, "name": name, "stamp": stamp}), flush=True)
        out[set_name] = call.get()[set_name]
    (HERE / "results").mkdir(exist_ok=True)
    dest = HERE / "results" / f"chain_stage_{target}{'_limit' + str(limit) if limit else ''}_{stamp}.json"
    if dest.exists():
        raise SystemExit(f"{dest.name} exists")
    part_ = dest.with_suffix(".part")  # atomic: a kill or a sleep never leaves a truncated record under the final name
    part_.write_text(json.dumps(out, indent=1, default=str))
    part_.rename(dest)
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
            raise SystemExit(f"{s_}: staging stopped ({v['assembled'].get('STOP')}); see {dest.name}")
        if v.get("positive_control") and not v["positive_control"]["passes"]:
            raise SystemExit(f"positive control FAILED: staged honest_b.jsonl {v['positive_control']['got']} != batch C's {v['positive_control']['want']}")
