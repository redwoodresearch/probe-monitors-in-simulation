"""On-policy title cells (RULE.md here), POST HOC: stage Qwen3.5-27B's own runs as extractor inputs for the Qwen3-32B and
Seed-OSS-36B readers, in #36's reader format, on Modal CPU. A copy of 2e's run_q4b_stage.py (aswin/q4-why-weaker @ 0bf6ec95)
driving stage_onpol.py (2e's stage_q4b.py with Qwen3-32B's command path and the scrub-0 messages returned). Changed:
  - the tokenizer: Qwen3-32B at its pinned revision, transformers 4.56.2 (the cross-read's staging and d9's all-layers
    extractor, run_extract_all_layers.py @ 82c87660, both used 4.56.2);
  - outputs: per run, an all-layers extractor item (#36's v4 item fields; `messages` = stage_onpol's reader messages at
    scrub level 0, #36's text) written to <STAGED>/inputs/shard_<i>.jsonl on the activation Volume (one arm per file), and
    its Qwen3-32B positions (`reader.segment_end_positions` on those messages, the 32,768 cut: exactly what the
    extractor's meta will hold) as <QMETA>/meta/<arm>/<stem>.json, the keep flags Seed's extractor reads;
  - gates in `assemble`, before anything is published (else a FAILED record and nothing written):
      counts: run 1 attack 193, run 2 attack 192, simulated attack 191, simulated honest 19, honest A 191, honest B 195;
      G0 (the cross-read reproduction): every staged epoch's scrub-1 Qwen3-32B token ids, as stage_onpol computes them,
      equal the cross-read's staged ids (xread_main_v1, xread_run2_v1; files pinned by sha256) for the same key, every
      key on both sides (the simulated honest arm has no cross-read counterpart and is counted apart);
  - reproduction runs added to the inputs for the extraction gates: G1, 8 real honest runs drawn with seed 20261014 from
    #36's 990 valid real honest runs with a meta in d9's mean-pool dump (2026-10-12_la_mean-pool36_qwen3-32b_l7_v1),
    copied from that dump's own inputs (arm repro-real-honest; stage_fresh_modal.py's way); G2, 8 run-1 attack runs drawn
    with the same seed, their messages at scrub level 1 (arm repro-xread-s1), for the activation check against the
    cross-read's rows. Their tokens are -1..-16, so they sort last and the extractor deals them to different shards.
Counts and hashes only leave the containers; no transcript text is printed, logged or returned.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-14_q3_onpolicy_titlecells/run_onpol_stage.py
"""

import hashlib
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q3onpol_stage_kill-2026-11-13"
EVALS = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"  # 2e's uploaded .eval files (read only)
XREAD = "probe-shift_aswin-surya_sub-q3xread_acts_kill-2026-11-13"  # the cross-read's staged inputs (read only)
ACTS = "probe-shift_aswin-surya_sub-la-activations"  # extractor inputs and the Qwen3-32B position metas (new run names only)
HF = "probe-shift_aswin-surya_hf-cache"
STAGED = "2026-10-14_q3onpol_q35own_inputs_v1"
QMETA = "2026-10-14_q3onpol_q35own_qmeta_v1"
D9 = ("2026-10-12_la_mean-pool36_qwen3-32b_l7_v1", "real-honest")
SETS = {"main": 48, "run2": 16}  # parts per set
ARM = {("main", "attack"): "q35-real-attack-1", ("run2", "attack"): "q35-real-attack-2", ("main", "sim_attack@september"): "q35-sim-attack",
       ("main", "sim_honest@september"): "q35-sim-honest", ("main", "honest_a"): "q35-real-honest-a", ("main", "honest_b"): "q35-real-honest-b"}
EXPECT = {"q35-real-attack-1": 193, "q35-real-attack-2": 192, "q35-sim-attack": 191, "q35-sim-honest": 19, "q35-real-honest-a": 191, "q35-real-honest-b": 195}
XREAD_FILES = {("main", "attack"): ("xread_main_v1/inputs/attack.jsonl", "74a09fe26a7e0a012f5a24a7a4e4260921d18f7d196c7b0cadd04e1e6c255180"),
               ("main", "honest_a"): ("xread_main_v1/inputs/honest_a.jsonl", "b5c4b17706e80c97b1354cad0c91e48869d04c9ea52f7cfd3db3eefce0296996"),
               ("main", "honest_b"): ("xread_main_v1/inputs/honest_b.jsonl", "5c7c854e66e9c45f7f331c8ec15bd4a2e42812e23363dcd2172e0ee8b0a30c5d"),
               ("main", "sim_attack@september"): ("xread_main_v1/inputs/sim_attack@september.jsonl", "98a40a683f36448d820ebf8767614f0dc7d9d5b955cbb460559bd9fbd3be851c"),
               ("run2", "attack"): ("xread_run2_v1/inputs/attack.jsonl", "b0a5b546557191613f8d72e45ebee19e1b798b09672ffb78a11ce9d400d0e56d")}
UPLOAD_MANIFEST = ("manifest_q4b_eval_upload_v1.json", "3b767d29011052603857fee3104e64890b586ac69c653c4cda92ebad9c532fd9")  # 2e's, bf9b297f
N_REPRO, SEED, CUT = 8, 20261014, 32768
LABEL_FILES = ["experiments/2026-10-08_q3_onpolicy_figures/results/phase2_attempt_labels_2026-10-09_000523_rerun_strict_selfreport_2026-10-10_111809.jsonl",
               "experiments/2026-10-08_q3_onpolicy_figures/results/phase2_attempt_labels_2026-10-09_173815_scaleup_canary_selfreport_2026-10-10_111817.jsonl",
               "experiments/2026-10-08_q3_onpolicy_figures/results/phase2_attempt_labels_2026-10-10_001314_scaleup_topup_selfreport_2026-10-10_111821.jsonl",
               "experiments/2026-10-08_q3_onpolicy_figures/results/phase2_attempt_labels_2026-10-10_001314_second_attack_2026-10-10_111840.jsonl"]
LABEL_S3 = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad/crossread_labels"
                "/phase2_attempt_labels_2026-10-09_182505_scaleup_selfreport_2026-10-10_111838.jsonl")
MODS = ["experiments/2026-10-09_fixed_sim/fixes.py", "experiments/2026-10-12_plantslip_transfer/plant_both.py",
        "experiments/2026-10-07_q2_rendering_swap/rerender.py", "experiments/2026-10-07_q2_rendering_swap/str_replace_tells.py",
        "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"]
RUNS36 = "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"

app = modal.App(APP)
evals = modal.Volume.from_name(EVALS, version=2).read_only()
xread = modal.Volume.from_name(XREAD, version=2).read_only()
acts = modal.Volume.from_name(ACTS, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==4.56.2", "numpy", "jinja2", "zstandard==0.25.0", "boto3", "pyyaml")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift")
         .add_local_file(HERE / "stage_onpol.py", "/root/repo/q4/stage_q4b.py").add_local_file(HERE / "evalzip.py", "/root/repo/q4/evalzip.py")
         .add_local_file(HERE / UPLOAD_MANIFEST[0], "/root/repo/manifest_q4b_eval_upload_v1.json")
         .add_local_file(REPO / RUNS36, f"/root/repo/{RUNS36}")
         .add_local_file(LABEL_S3, f"/root/repo/labels/{LABEL_S3.name}"))
for m in MODS + LABEL_FILES:
    image = image.add_local_file(REPO / m, f"/root/repo/{m}")


class Gate(Exception):
    """A failed check; its message holds counts only."""


def _mod():
    import sys

    sys.path.insert(0, "/root/repo/q4")
    import stage_q4b as S  # stage_onpol.py, mounted under stage_q4b's name (its REPO and module paths are unchanged)

    if "command_positions(tok, msgs, CUT, cv)" not in Path(S.__file__).read_text():
        raise Gate("the mounted stager is not stage_onpol.py")
    return S


def filename(key: str) -> str:  # the extractors' file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def msha(msgs):
    return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()


def _tok():
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN3_32B
    return AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"], local_files_only=True)


def _setup(S, set_name):
    from collections import defaultdict

    if not S.fingerprint().startswith(S.LABELER):
        raise ValueError(f"labeler fingerprint {S.fingerprint()[:8]} is not {S.LABELER}")
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
    return folds, counted, prompts


def item_key(ep_key: str) -> str:
    """The extractor key: the staged epoch key ("<launch>/<rel>#e0") with no path separators (meta file stems)."""
    return ep_key.replace("/", "__").replace("#", "__")


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0,
              volumes={"/ev": evals, "/hf": modal.Volume.from_name(HF).read_only(), "/acts": acts})
def part(set_name: str, k: int, n_parts: int, limit: int) -> dict:
    from collections import Counter

    from probe_shift import reader, transcripts
    S = _mod()
    tok = _tok()
    folds, counted, prompts = _setup(S, set_name)
    lab = S.attempt_labels()  # {(world, task): label}; world run1 / run2 / sim (claim C's map)
    todo = sorted(x for x in counted if x[0] in S.SETS[set_name] and counted[x]["launch"] in ([S.RUN2] if set_name == "run2" else S.MAIN))
    if limit:
        todo = [x for x in todo if sum(1 for y in todo[: todo.index(x)] if y[0] == x[0]) < limit]
    mine, cov, out, recs = todo[k::n_parts], Counter(), [], []
    for arm_dir, task in mine:
        c = counted[(arm_dir, task)]
        try:
            ep, rec = S.process(c, arm_dir, task, set_name, prompts, tok, folds, cov)
            recs.append({"arm_dir": arm_dir, **rec, "run": f"{c['launch']}/{c['rel']}"})
            if ep is None:
                continue
            m0 = ep.pop("_messages0")
            s = S.load(c["eval"])
            md = s.get("metadata") or {}
            env = md.get("environment") or md.get("env")
            side = md.get("side_task")
            seg0 = reader.segment_end_positions(tok, m0, max_tokens=CUT)
            arm = ARM[(set_name, arm_dir)]
            key = item_key(ep["key"])
            item = {"key": key, "arm": arm, "cohort": "q35own", "file": c["rel"], "sample_id": s["id"], "epoch": s["epoch"],
                    "environment": env.get("id") if isinstance(env, dict) else env, "main_task": task,
                    "side_task": side.get("id") if isinstance(side, dict) else side, "cell": task, "sim_reference": None,
                    "tokens": seg0["n_tokens"], "truncated": seg0["truncated"], "n_submit_calls": transcripts.submit_calls(s),
                    "messages": m0, "staged_epoch_key": ep["key"], "fold": rec.get("fold"), "set": set_name, "arm_dir": arm_dir,
                    "attempt_label": lab.get(({"q35-real-attack-1": "run1", "q35-real-attack-2": "run2", "q35-sim-attack": "sim"}.get(arm), task))}
            out.append({"item": item, "qmeta": {"key": key, "arm": arm, "positions": seg0["positions"]},
                        "g0": {"set": set_name, "arm_dir": arm_dir, "key": ep["key"], "ids_sha": S.ids_sha(ep["ids"]), "n": len(ep["ids"])}})
        except Gate:
            raise
        except Exception as e:  # never let a message carry transcript text: the error type and code locations only
            import traceback

            where = [f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in traceback.extract_tb(e.__traceback__)[-4:]]
            raise S.Withheld(f"{c['launch']}/{c['rel']}: {type(e).__name__} at {where} (message withheld)") from None
    d = Path("/acts") / STAGED / "_parts"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{set_name}_part{k:03d}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    (d / f"{set_name}_part{k:03d}.recs.json").write_text(json.dumps({"recs": recs, "counters": dict(cov), "no_eval": S.NO_EVAL}))
    acts.commit()
    return {"set": set_name, "part": k, "runs": len(mine), "staged": len(out)}


@app.function(image=image, cpu=8, memory=65536, timeout=3600, retries=0,
              volumes={"/ev": evals, "/hf": modal.Volume.from_name(HF).read_only(), "/acts": acts, "/xr": xread})
def assemble(parts: dict, stamp: str, repro_pool: list) -> dict:
    from collections import Counter, defaultdict

    from probe_shift import reader
    S = _mod()
    acts.reload()
    root = Path("/acts") / STAGED
    rows, recs, counters = [], defaultdict(list), Counter()
    for set_name, n in parts.items():
        for k in range(n):
            rows += [json.loads(l) for l in (root / "_parts" / f"{set_name}_part{k:03d}.jsonl").read_text().splitlines()]
            r = json.loads((root / "_parts" / f"{set_name}_part{k:03d}.recs.json").read_text())
            for rec in r["recs"]:
                recs[(set_name, rec.pop("arm_dir"))].append(rec)
            counters.update(r["counters"])
    items = [x["item"] for x in rows]
    if len({(x["arm"], x["key"]) for x in items}) != len(items):
        raise Gate("duplicate staged runs")
    by_arm = Counter(x["arm"] for x in items)
    gates = {"counts": {a: [by_arm.get(a, 0), want] for a, want in EXPECT.items()}}
    ok_counts = all(v[0] == v[1] for v in gates["counts"].values()) and set(by_arm) == set(EXPECT)

    # G0: the scrub-1 ids against the cross-read's staged epochs
    mine = defaultdict(dict)
    for x in rows:
        g = x["g0"]
        mine[(g["set"], g["arm_dir"])][g["key"]] = g["ids_sha"]
    g0 = {}
    for sa, (rel, want) in XREAD_FILES.items():
        b = (Path("/xr") / rel).read_bytes()
        if hashlib.sha256(b).hexdigest() != want:
            raise Gate(f"{rel}: the cross-read's staged file differs from its pin")
        theirs = {e["key"]: S.ids_sha(e["ids"]) for e in map(json.loads, b.decode().splitlines())}
        ours = mine.get(sa, {})
        g0[f"{sa[0]}/{sa[1]}"] = {"cross-read epochs": len(theirs), "staged here": len(ours), "keys only there": len(set(theirs) - set(ours)),
                                  "keys only here": len(set(ours) - set(theirs)),
                                  "ids differ": sum(1 for k in set(theirs) & set(ours) if theirs[k] != ours[k]),
                                  "ids identical": sum(1 for k in set(theirs) & set(ours) if theirs[k] == ours[k])}
    ok_g0 = all(v["keys only there"] == 0 and v["keys only here"] == 0 and v["ids differ"] == 0 and v["ids identical"] == v["cross-read epochs"] > 0
                for v in g0.values())
    gates["G0 (scrub-1 ids vs the cross-read, by set/arm)"] = g0
    gates["simulated honest (no cross-read counterpart)"] = len(mine.get(("main", "sim_honest@september"), {}))

    # reproduction runs
    rng = random.Random(SEED)
    dump, src_arm = D9
    cfg = json.loads((Path("/acts") / dump / "config.json").read_text())
    have = {p.stem for p in (Path("/acts") / dump / "meta" / src_arm).glob("*.json")}
    pool = set(repro_pool)
    src = {}
    for p in sorted((Path("/acts") / cfg["inputs_from"] / "inputs").glob("shard_*.jsonl")):
        for line in p.read_text().splitlines():
            it = json.loads(line)
            if it["arm"] == src_arm and filename(it["key"]) in have and it["key"] in pool:
                src[filename(it["key"])] = it
    repro = [src[s_] | {"arm": "repro-real-honest", "tokens": -i, "source_tokens": src[s_]["tokens"]}
             for i, s_ in enumerate(rng.sample(sorted(src), N_REPRO), 1)]
    att = sorted((x for x in items if x["arm"] == "q35-real-attack-1"), key=lambda x: x["key"])
    repro += [x | {"arm": "repro-xread-s1", "messages": S.scrubbed(x["messages"], 1), "tokens": -(N_REPRO + i), "source_tokens": x["tokens"]}
              for i, x in enumerate(rng.sample(att, N_REPRO), 1)]
    tok = _tok()
    qmeta = [x["qmeta"] for x in rows]
    for x in repro:  # their Qwen3-32B positions, as for the staged runs (Seed reads repro-real-honest's)
        if x["arm"] == "repro-real-honest":
            qmeta.append({"key": x["key"], "arm": x["arm"], "positions": reader.segment_end_positions(tok, x["messages"], max_tokens=CUT)["positions"]})
    ok = ok_counts and ok_g0
    record = {"stamp": stamp, "staged": STAGED, "qmeta": QMETA, "cut": CUT, "scrub level of the extractor inputs": 0,
              "reader tokenizer": "Qwen/Qwen3-32B (probe_shift.models.QWEN3_32B), transformers 4.56.2", "gates": gates, "gates pass": ok,
              "counters": dict(counters), "coverage (stage_q4b's summary, scrub-1 rows)": {f"{s}/{a}": S.summarize({a: v}, S.attempt_labels(), s)
                                                                                         for (s, a), v in sorted(recs.items())},
              "reproduction draw": {"seed": SEED, "repro-real-honest": {"dump": dump, "dump arm": src_arm, "inputs_from": cfg["inputs_from"],
                                                                          "pool": len(src), "keys": sorted(x["key"] for x in repro if x["arm"] == "repro-real-honest")},
                                    "repro-xread-s1": sorted(x["staged_epoch_key"] for x in repro if x["arm"] == "repro-xread-s1")}}
    if not ok:
        (root / f"stage_FAILED_{stamp}.json").write_text(json.dumps(record, indent=1, default=str))
        acts.commit()
        return {"mode": "stage", "STOP": "a staging gate failed", "gates": gates, "gates pass": False}
    d = root / "inputs"
    if d.exists() and any(d.iterdir()):
        raise FileExistsError(f"{d} exists: a staged run is never overwritten")
    d.mkdir(parents=True)
    shards, runs_by_arm = {}, {}
    for i, a in enumerate(sorted({x["arm"] for x in items + repro})):
        rs = sorted((x for x in items + repro if x["arm"] == a), key=lambda x: x["key"])
        body = "".join(json.dumps(x) + "\n" for x in rs).encode()
        (d / f"shard_{i}.jsonl").write_bytes(body)
        shards[f"inputs/shard_{i}.jsonl"] = {"arm": a, "runs": len(rs), "sha256": hashlib.sha256(body).hexdigest()}
        runs_by_arm[a] = [x["key"] for x in rs]
    q = Path("/acts") / QMETA
    if q.exists():
        raise FileExistsError(f"{QMETA} exists: never overwritten")
    for m in qmeta:
        p = q / "meta" / m["arm"] / f"{filename(m['key'])}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(m))
    record |= {"shards_files": shards, "runs": [{"arm": x["arm"], "key": x["key"], "messages_sha256": msha(x["messages"])}
                                                for x in sorted(items + repro, key=lambda x: (x["arm"], x["key"]))],
               "qmeta files": len(qmeta)}
    rb = json.dumps(record, default=str).encode()
    (root / "stage_record.json").write_bytes(rb)
    acts.commit()
    return {"mode": "stage", "staged": STAGED, "qmeta": QMETA, "gates pass": True, "gates": gates, "shards_files": shards, "runs by arm": runs_by_arm,
            "record_sha256": hashlib.sha256(rb).hexdigest(), "reproduction draw": record["reproduction draw"],
            "tokens": {a: {"total": sum(x["tokens"] for x in items if x["arm"] == a), "max": max((x["tokens"] for x in items if x["arm"] == a), default=0),
                           "over 32k": sum(x["truncated"] for x in items if x["arm"] == a)} for a in EXPECT},
            "coverage": record["coverage (stage_q4b's summary, scrub-1 rows)"], "counters": record["counters"]}


@app.function(image=image, cpu=2, memory=8192, timeout=3 * 3600, retries=0, volumes={"/acts": acts})
def orchestrate(git: str, stamp: str, pool36: list) -> dict:
    """Runs the parts and the assembly from inside Modal (the laptop client may go away) and writes the result (counts and
    hashes only) beside the staged run as stage_result_<stamp>.json."""
    out = {"git": git, "made": stamp, "parts": {}}
    for set_name, n in SETS.items():
        out["parts"][set_name] = list(part.starmap([(set_name, k, n, 0) for k in range(n)]))
    out |= assemble.remote(SETS, stamp, pool36)
    acts.reload()
    d = Path("/acts") / STAGED
    d.mkdir(parents=True, exist_ok=True)
    (d / f"stage_result_{stamp}.json").write_text(json.dumps(out, indent=1, default=str))
    acts.commit()
    return out


@app.local_entrypoint()
def main():
    for f in (Path(__file__), HERE / "stage_onpol.py"):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"commit {f.name} before running it")
    if hashlib.sha256((HERE / UPLOAD_MANIFEST[0]).read_bytes()).hexdigest() != UPLOAD_MANIFEST[1]:
        raise SystemExit("the upload manifest is not 2e's (bf9b297f)")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    pool36 = [r["key"] for r in json.loads((REPO / RUNS36).read_text()) if r["arm"] == "real-honest"]
    call = orchestrate.spawn(git, stamp, pool36)
    print(f"staging orchestrated on Modal (function call {call.object_id}); result at {STAGED}/stage_result_{stamp}.json on {ACTS}", flush=True)
