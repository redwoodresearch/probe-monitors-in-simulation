"""09's section 8 replay check (RULE.md @ ee026b8ad, "Stager donor rule, replacement wording"): the historical and the amended stager
on identical pinned inputs, over the complete own-world input inventory of every IL snapshot-1 set (routing 1 and 2), excluded epochs
and no-eval statuses included. Keys, hashes, decisions and reason strings only; never transcript text.

- Historical implementation: stage_q4b.py @ 9be2fa9b6 (byte-identical to cf7cd2b1e), instrumented by inserting ONE line after the pick
  that records each attack epoch's donor (no predicate changed; the patch is asserted, recorded, and its log list is set from outside).
- Amended implementation: stage_q4b.py in this commit (sections 1-6), which records its donor decisions in DONOR_LOG.
- Both run through stage_chain.py's own _setup (counted attempts sha-checked against the frozen manifest, donor prompts) and the same
  process() call the staging parts make, on the same image; the Volume is mounted read only.
- The authoritative staged set of each target comes from its pinned routing file (staging-retry resolution preserved); the original
  per-run records are that set's parts/*.recs.json.
- Checks (any failure blocks every paired rerun launch and goes to 09):
  1. completeness: per target, the replayed runs = the manifest's counted runs in the target's arms = the original records' runs, with no
     duplicates; the no-eval statuses equal the manifest's and the original parts';
  2. the historical replay reproduces each original decision and reason string;
  3. per epoch, identical staged/not-staged decisions and selected donor identities (arm, path, sha256; null if none), and no errors;
  4. every changed exclusion string maps by exact epoch identity (path, epoch, .eval sha256, world, cell) to one row of the corrected
     exclusion record @ 99d8b5ac3 (file checked against that commit's blob), with 09's ruling attached;
  5. gate (b) on this inventory: per cell, run3 vs sim2 own attack system-prompt sha256; a mismatch fails; every cell is listed as
     compared, not comparable (an epoch without a prompt) or single-arm. sim1 comparisons stay in il_gate_b @ ac0768ab4.
Run: PYTHONPATH=src modal run experiments/2026-10-14_q3_overnight_chain/replay_stager.py::replay --out <json> --record <corrected record>
     --rulings <rulings json> --routing <chain routing dir> [--parts 6]"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_chain as SC  # noqa: E402  (its app, image, targets and _setup: the staging code's own)

app = SC.app
image = SC.image.add_local_file(HERE / "stage_chain.py", "/root/stage_chain.py")
VOLS = {"/ev": SC.vol.read_only(), "/src": SC.VOLS["/src"], "/hf": SC.VOLS["/hf"]}
HIST_COMMIT = "9be2fa9b6"
RECORD_COMMIT = "99d8b5ac3"
RECORD_PATH = "experiments/2026-10-14_q3_overnight_chain/results/il_donor_check/il_exclusions_corrected_snap1_2026-10-15_012623.json"
PICK = '        pick = [h for h, p in cands if p and cover and cover.group(1).strip() in p[0] and "You are an agent" in p[0] and p[1] == fu]\n'
LOG = '        _DONOR_LOG.append({"run": f"{c[\'launch\']}/{c[\'rel\']}", "donor": pick[0] if pick else None})  # replay instrumentation only\n'
BATCH = {"il311_q1": "il-q1_s1", "il311_q2": "il-q2_s1", "il311_q3": "il-q3_s1", "il311_q4": "il-q4_s1", "ilnm_s1": "il-s1_s1", "ilnm_s2": "il-s2_s1",
         "ilold_old": "il-old_s1"}
TARGETS = [f"{p}_{w}_s1" for p in BATCH for w in ("run3", "sim2")]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


TNAME = re.compile(r"^(?P<pfx>il311_q[1-4]|ilnm_s[1-5]|ilold_old)_(?P<w>run3|sim2)(?:_s(?P<k>[0-9]+))?$")


def stem(t: str) -> str:
    return TNAME.match(t).group("pfx")


def batch_of(t: str) -> str:  # the chain batch that staged target t (snapshot k, or the family's final)
    m = TNAME.match(t)
    fam = BATCH[m.group("pfx")].rsplit("_", 1)[0]
    return f"{fam}_s{m.group('k')}" if m.group("k") else fam


@app.function(image=image, cpu=2, memory=16384, timeout=3600, retries=0, volumes=VOLS)
def original_records(staged_set: str) -> dict:
    """The staged set's own per-run records and no-eval lists, every part, with each parts file's sha256."""
    out = {"recs": [], "no_eval": [], "files": {}}
    for p in sorted((Path("/ev") / staged_set / "parts").glob("part*.recs.json")):
        raw = p.read_bytes()
        out["files"][p.name] = sha(raw)
        d = json.loads(raw)
        out["recs"] += [{"run": r["run"], "arm_dir": r.get("arm_dir"), "decision": "not staged" if r.get("excluded") else "staged", "reason": r.get("excluded")}
                        for r in d["recs"]]
        out["no_eval"] += [x if isinstance(x, str) else json.dumps(x, sort_keys=True) for x in d.get("no_eval", [])]
    return out


@app.function(image=image, cpu=2, memory=16384, timeout=3600, retries=0, volumes=VOLS)
def replay_part(target: str, k: int, n_parts: int, hist_src: str) -> dict:
    import importlib.util
    import traceback
    from collections import Counter

    sys.path.insert(0, "/root/repo/q4")
    Path("/tmp/hist").mkdir(exist_ok=True)
    Path("/tmp/hist/stage_q4b_hist.py").write_text(hist_src)
    spec = importlib.util.spec_from_file_location("stage_q4b_hist", "/tmp/hist/stage_q4b_hist.py")
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    A = SC._mod()
    T = SC.TARGETS[target]
    set_name = T.get("set", "c")
    out = {"target": target, "part": k, "rows": {}, "no_eval": {}, "own_prompt": {}, "inventory": {}}
    for name, M in (("hist", H), ("amended", A)):
        M.NO_EVAL.clear()
        (tok, _), folds, counted, prompts = SC._setup(M, set_name, target)
        man = {f["path"]: f["sha256"] for f in json.loads(M.UPLOAD_MANIFEST.read_text())["files"]}
        arms = T.get("stage_arms") or M.SETS[set_name]
        todo = sorted(x for x in counted if x[0] in M.SETS[set_name] and x[0] in arms and counted[x]["launch"] in M.MAIN)
        out["inventory"][name] = len(todo)
        rows = []
        for arm_dir, task in todo[k::n_parts]:
            c = counted[(arm_dir, task)]
            run = f"{c['launch']}/{c['rel']}"
            log = []
            if name == "hist":
                M._DONOR_LOG = log
            else:
                M.DONOR_LOG = log
            row = {"run": run, "launch": c["launch"], "arm_dir": arm_dir, "task": task, "sha256": man.get(run), "epoch": "e0"}
            try:
                ep, rec = M.process(c, arm_dir, task, set_name, prompts, tok, folds, Counter())
                row |= {"decision": "staged" if ep is not None else "not staged", "reason": rec.get("excluded"), "error": None}
            except Exception as e:  # never a message: the class and code locations only
                where = [f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in traceback.extract_tb(e.__traceback__)[-4:]]
                row |= {"decision": "error", "reason": None, "error": f"{type(e).__name__} at {where}"}
            picks = [x["donor"] for x in log if x["run"] == run]
            donor = picks[-1] if picks else None
            row["donor_log_entries"] = len(picks)
            if donor is not None:
                dp = f"{counted[(donor, task)]['launch']}/{counted[(donor, task)]['rel']}"
                row["donor"] = {"arm": donor, "path": dp, "sha256": man.get(dp)}
            else:
                row["donor"] = None
            if name == "amended":
                row["first_user_checks"] = A.FU_CHECKS.get(run)
                sp = M.transcripts.system_prompt(M.load(c["eval"]))  # gate (b): this epoch's own system prompt
                out["own_prompt"][run] = sha(sp.encode()) if sp else None
            rows.append(row)
        out["rows"][name] = rows
        out["no_eval"][name] = sorted(x if isinstance(x, str) else json.dumps(x, sort_keys=True) for x in M.NO_EVAL)
    return out


@app.local_entrypoint()
def replay(out: str, record: str, rulings: str, routing: str, parts: int = 6, targets: str = "", map_only: bool = False):
    """map_only (09 @ 527337ee3 section 3): replay other historically staged IL sets to map their reason strings by exact epoch
    identity; decisions, donors, completeness and errors are checked exactly as in the section 8 run, but a changed reason needs no
    row in the snapshot-1 corrected record (this output is that mapping)."""
    global TARGETS
    if targets:
        TARGETS = targets.split(",")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout  # noqa: E731
    hist = git("show", f"{HIST_COMMIT}:experiments/2026-10-14_q3_overnight_chain/stage_q4b.py")
    if hist.count(PICK) != 1:
        raise SystemExit("the historical pick line is not where expected")
    inst = hist.replace(PICK, PICK + LOG)
    if inst.count("_DONOR_LOG") != 1 or len(inst.splitlines()) != len(hist.splitlines()) + 1:
        raise SystemExit("instrumentation is not exactly one inserted line")
    problems = []
    rec_raw = Path(record).read_bytes()
    if sha(rec_raw) != sha(git("show", f"{RECORD_COMMIT}:{RECORD_PATH}").encode()):
        problems.append("the corrected record file differs from its committed blob @ 99d8b5ac3")
    staged_set, routing_sha = {}, {}
    for t in TARGETS:
        rf = Path(routing) / f"{batch_of(t)}__{t}.json"
        staged_set[t] = json.loads(rf.read_text())["staged_set"]  # the authoritative (retry-resolved) staged set
        routing_sha[rf.name] = sha(rf.read_bytes())
    orig = dict(zip(TARGETS, original_records.map([staged_set[t] for t in TARGETS])))
    res = list(replay_part.starmap([(t, k, parts, inst) for t in TARGETS for k in range(parts)]))
    rows = {}
    for name in ("hist", "amended"):
        for r in res:
            for x in r["rows"][name]:
                key = (r["target"], x["arm_dir"], x["task"], x["launch"], x["run"], x["sha256"], x["epoch"])
                if (name, key) in rows:
                    problems.append(f"duplicate {name} row {key}")
                rows[(name, key)] = x
    keys = {k for n, k in rows if n == "hist"}
    if keys != {k for n, k in rows if n == "amended"}:
        problems.append("hist and amended row sets differ")
    inventory = {}
    for t in TARGETS:  # check 1: completeness against the manifest and the original records
        man = json.loads((HERE / "config" / SC.TARGETS[t]["manifest"]).read_text())
        arms = SC.TARGETS[t].get("stage_arms")
        want = sorted(f["path"] for f in man["files"] if f["arm_dir"] in arms)
        got = sorted(k[4] for k in keys if k[0] == t)
        o_runs = [r["run"] for r in orig[t]["recs"]]
        inv = {r_["inventory"]["hist"] for r_ in res if r_["target"] == t} | {r_["inventory"]["amended"] for r_ in res if r_["target"] == t}
        man_ne = sorted(json.dumps(x, sort_keys=True) if not isinstance(x, str) else x for x in man["counted_attempts_without_eval"]["main"])
        rep_ne = {tuple(r_["no_eval"][n]) for r_ in res if r_["target"] == t for n in ("hist", "amended")}
        orig_ne = sorted(set(orig[t]["no_eval"]))
        inventory[t] = {"manifest_runs": len(want), "replayed": len(got), "original_records": len(o_runs), "original_unique": len(set(o_runs)),
                        "no_eval_manifest": len(man_ne), "no_eval_original": len(orig_ne), "staged_set": staged_set[t], "parts_files": orig[t]["files"]}
        if want != got or sorted(set(o_runs)) != got or len(o_runs) != len(set(o_runs)) or inv != {len(want)}:
            problems.append(f"{t}: inventory mismatch {inventory[t]}")
        if rep_ne != {tuple(man_ne)} or orig_ne != man_ne:
            problems.append(f"{t}: no-eval statuses differ (manifest {len(man_ne)}, original {len(orig_ne)}, replay {[len(x) for x in rep_ne]})")
    original = {(t, r["run"]): r for t in TARGETS for r in orig[t]["recs"]}
    rec_rows = {}
    for x in json.loads(rec_raw)["rows"]:
        k = (x["run_key"], x["eval_sha256"], x["world"], x["key"].split("|")[1])
        if k in rec_rows:
            problems.append(f"duplicate corrected-record row {k}")
        rec_rows[k] = x
    rul = json.loads(Path(rulings).read_text())["rulings"]
    rl_ruling = next(r for r in rul if r["key"] == "sim2 x 66")

    def ruling_for(x):
        if x["proposed_class"].startswith("RateLimitError"):
            return rl_ruling
        m = [r for r in rul if r["key"].startswith(x["key"] + " ")]
        return m[0] if len(m) == 1 else None

    table, changed = [], []
    for key in sorted(keys):
        h, a = rows[("hist", key)], rows[("amended", key)]
        o = original.get((key[0], key[4]))
        same_orig = o is not None and o["decision"] == h["decision"] and o["reason"] == h["reason"]
        if not same_orig:
            problems.append(f"historical replay differs from the original staging: {key[0]} {key[4]}")
        if h["error"] or a["error"]:
            problems.append(f"replay error: {key[0]} {key[4]} hist {h['error']} amended {a['error']}")
        if h["decision"] != a["decision"] or h["donor"] != a["donor"]:
            problems.append(f"decision or donor differs: {key[0]} {key[4]}")
        if h["donor_log_entries"] > 1 or a["donor_log_entries"] > 1:
            problems.append(f"donor log ambiguous: {key[0]} {key[4]}")
        if a.get("first_user_checks") and not all(x["equal"] for x in a["first_user_checks"]):
            problems.append(f"first-user consistency check failed: {key[0]} {key[4]}")
        if h["reason"] != a["reason"]:
            world = "run3" if key[1] == "attack" else "sim2"
            x = rec_rows.get((f"{key[4]}#{key[6]}", key[5], world, key[2]))
            ru = ruling_for(x) if x else None
            changed.append({"key": list(key), "hist_reason": h["reason"], "amended_reason": a["reason"],
                            "corrected_record": None if x is None else {k_: x[k_] for k_ in ("key", "run_key", "eval_sha256", "proposed_class", "terminal_error_class",
                                                                                                 "n_messages", "assistant_steps", "has_system_prompt")},
                            "ruling_09": ru})
            if (x is None or ru is None) and not map_only:
                problems.append(f"changed reason without an exact corrected-record row and ruling: {key[0]} {key[4]}")
        table.append({"key": list(key), "original": o, "hist": h, "amended": a, "same_as_original": same_orig})
    own = {run: s for r in res for run, s in r["own_prompt"].items()}
    gate_b = {"scope": "run3 vs sim2 own attack system prompts per IL snapshot-1 cell; sim1 pairs in il_gate_b_2026-10-15_020046.json @ ac0768ab4",
              "cells": []}
    by_cell = {}
    for key in keys:  # per family and cell: a cell's run3 and sim2 may sit in different snapshots of one launch; each run is staged once
        w = "run3" if key[1] == "attack" else "sim2"
        d = by_cell.setdefault((stem(key[0]), key[2]), {})
        if w in d:
            problems.append(f"gate (b): {w}|{key[2]} appears in two selected sets of {stem(key[0])}")
        d[w] = {"run": key[4], "sha256": key[5], "prompt_sha256": own.get(key[4]), "set": key[0]}
    for (st_, cell), d in sorted(by_cell.items()):
        if len(d) < 2:
            status = "single-arm"
        elif d["run3"]["prompt_sha256"] is None or d["sim2"]["prompt_sha256"] is None:
            status = "not comparable: no system prompt"
        else:
            status = "equal" if d["run3"]["prompt_sha256"] == d["sim2"]["prompt_sha256"] else "MISMATCH"
        gate_b["cells"].append({"batch": st_, "cell": cell, "arms": d, "status": status})
        if status == "MISMATCH":
            problems.append(f"gate (b) mismatch: {st_} {cell}")
    gate_b["counts"] = {s: sum(c["status"] == s for c in gate_b["cells"]) for s in ("equal", "MISMATCH", "not comparable: no system prompt", "single-arm")}
    pins = {"historical": {"commit": HIST_COMMIT, "sha256": sha(hist.encode()), "instrumented_sha256": sha(inst.encode()), "inserted_line": LOG.strip()},
            "amended": {"path": "experiments/2026-10-14_q3_overnight_chain/stage_q4b.py", "sha256": sha((HERE / "stage_q4b.py").read_bytes())},
            "stage_chain": {"sha256": sha((HERE / "stage_chain.py").read_bytes())}, "replay": {"sha256": sha(Path(__file__).read_bytes())},
            "commit": git("rev-parse", "HEAD").strip(), "dirty": bool(git("status", "--porcelain", "--untracked-files=no").strip()),
            "targets_config": {t: SC.TARGETS[t] for t in TARGETS},
            "targets_config_file": {"path": "experiments/2026-10-14_q3_overnight_chain/config/chain_targets.json", "sha256": sha(SC._ct.read_bytes())}, "manifests": {t: sha((HERE / "config" / SC.TARGETS[t]["manifest"]).read_bytes()) for t in TARGETS},
            "routing": routing_sha, "record": {"path": RECORD_PATH, "commit": RECORD_COMMIT, "sha256": sha(rec_raw)},
            "rulings": {"path": rulings, "sha256": sha(Path(rulings).read_bytes())}}
    counts = {"rows": len(keys), "staged_hist": sum(rows[("hist", k)]["decision"] == "staged" for k in keys),
              "staged_amended": sum(rows[("amended", k)]["decision"] == "staged" for k in keys), "same_as_original": sum(x["same_as_original"] for x in table),
              "same_decision_and_donor": sum(rows[("hist", k)]["decision"] == rows[("amended", k)]["decision"] and rows[("hist", k)]["donor"] == rows[("amended", k)]["donor"] for k in keys),
              "donor_null_both": sum(rows[("hist", k)]["donor"] is None and rows[("amended", k)]["donor"] is None for k in keys), "changed_reasons": len(changed)}
    doc = {"rule": "09 RULE.md @ ee026b8ad, section 8", "pins": pins, "targets": TARGETS, "inventory": inventory, "counts": counts, "gate_b": gate_b,
           "changed_reasons": changed, "problems": problems, "pass": not problems, "rows": table}
    Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(json.dumps({"counts": counts, "gate_b": gate_b["counts"], "problems": problems[:25], "n_problems": len(problems), "pass": not problems}, indent=1))
    if problems:
        sys.exit(2)
