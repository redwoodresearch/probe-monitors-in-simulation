"""Overnight chain (RULE.md here): 09's approach provenance (Append 21 section 5) for one simulated batch, on Modal CPU over the
relayed .evals. Hashes only: no message text is returned, stored or printed.

Per cell of the batch, the three ATTACK runs (honest arms are exempt): run1 = the main launch's attack `<cell>`, run2 = its
attack `<cell>_r2`, sim = the follow-up's sim_attack@september `<cell>`. Each counted .eval is hashed by hash_eval() from
experiments/2026-10-14_q3_q35_simhonest_purchase/check_approach_hashes.py @ 8ad72f4a (its lines 33-66: sha, msg_sha,
hash_eval), vendored byte for byte (vendored/approach_hash_8ad72f4a_L33-66.py, sha256 checked here): the sample's
metadata attack_approach and the first system and user messages the policy's first model call was SENT, read with
resolve_attachments=True. Only the per-cell grouping and the output shape are new (that script's main() assumes the 196
layout: run2 in a separate *_second_attack launch, PHASE2_CELLS on box paths).

Output per cell: {"cell", "run1": {...}, "run2": {...}, "sim": {...}, "equal_approach", "equal_system_prompt",
"equal_first_user", "first_user_empty", "staged_shas": [run1, run2, sim attack-arm staged shas], "task_file": {...},
"pass"}; each world: approach_sha256, system_prompt_sha256, first_user_sha256, policy_calls, launch_task, eval, eval_sha256.
A cell passes when all three worlds were hashed (one policy call or more) and their approach and system-prompt hashes are
equal. The task-file comparison (the main launch.json's approach_sha256 / sol_system_sha256 / sol_input_sha256 for the
launcher task) is recorded beside it, not in `pass`. The tool definitions differ between worlds by design (not compared).

  modal run experiments/2026-10-14_q3_overnight_chain/provenance_chain.py --spec <spec json> --out <file>
"""
import hashlib
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q3chain_prov_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
HASHER, HASHER_SHA = "approach_hash_8ad72f4a_L33-66.py", "76d800e8df13f9be1babe398edd8bdcd986e7769a4260d679b30f3d1e828d1eb"
INSPECT = "inspect-ai==0.3.276"  # the repo's uv.lock pin
WORLDS = ("run1", "run2", "sim")
SRC_VOL = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"  # 2e's Volume (the T196 .evals, read only): the ug mode's S_old artifacts
UG_WORLDS = ("run1", "run2", "sim1", "sim_ug")
app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.14").pip_install(INSPECT)  # 3.14: zstd .eval members
         .add_local_dir(HERE / "vendored", "/root/vendored"))


def load_hasher():
    src = Path("/root/vendored") / HASHER
    if hashlib.sha256(src.read_bytes()).hexdigest() != HASHER_SHA:
        raise ValueError("the vendored hash_eval differs from 8ad72f4a's lines 33-66")
    ns = {"hashlib": hashlib, "json": json}
    exec(src.read_text(), ns)
    return ns["hash_eval"]


def hash_one(job: tuple) -> dict:
    path, want_sha, policy = job
    p = Path("/ev/evals_v1") / path
    if hashlib.sha256(p.read_bytes()).hexdigest() != want_sha:
        raise ValueError(f"{path}: differs from its manifest")
    try:
        return load_hasher()((str(p), policy))
    except Exception as e:  # identifiers and the exception type only: a reader's validation error can quote message text
        raise RuntimeError(f"{path}: {type(e).__name__} (message withheld)") from None


@app.function(image=image, cpu=8, memory=16384, timeout=1800, retries=0, volumes={"/ev": modal.Volume.from_name(VOL, version=2).read_only()})
def provenance(spec: dict) -> dict:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    main = json.loads((Path("/ev/evals_v1") / spec["main_run"] / "launch.json").read_text())
    policy = {"run1": main["policy"], "run2": main["policy"], "sim": spec.get("sim_policy") or main["policy"]}
    tasks = {t["key"]: {k: t.get(k) for k in ("approach_sha256", "sol_system_sha256", "sol_input_sha256")} for t in main.get("tasks", [])}
    jobs = [(c["cell"], w, (c[w]["path"], c[w]["sha256"], policy[w])) for c in spec["cells"] for w in WORLDS if c.get(w)]
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("fork")) as pool:  # 3.14 defaults to forkserver
        hashed = dict(zip([(c, w) for c, w, _ in jobs], pool.map(hash_one, [j for _, _, j in jobs])))
    empty = hashlib.sha256(b"").hexdigest()
    cells = []
    for c in spec["cells"]:
        row = {"cell": c["cell"]}
        for w in WORLDS:
            h = hashed.get((c["cell"], w))
            row[w] = None if h is None else {"approach_sha256": h["approach_sha256"], "system_prompt_sha256": h["sent_system_sha256"],
                                             "first_user_sha256": h["sent_user_sha256"], "policy_calls": h["policy_calls"],
                                             "launch_task": c[w]["launch_task"], "eval": c[w]["path"], "eval_sha256": c[w]["sha256"]}
        got = [row[w] for w in WORLDS if row[w] is not None and row[w]["policy_calls"] > 0]
        full = len(got) == 3
        row["equal_approach"] = full and len({x["approach_sha256"] for x in got}) == 1 and got[0]["approach_sha256"] is not None
        row["equal_system_prompt"] = full and len({x["system_prompt_sha256"] for x in got}) == 1 and got[0]["system_prompt_sha256"] is not None
        row["equal_first_user"] = full and len({x["first_user_sha256"] for x in got}) == 1
        row["first_user_empty"] = full and all(x["first_user_sha256"] == empty for x in got)
        row["staged_shas"] = c["staged_shas"]
        tf = {}
        for w in WORLDS:
            t = tasks.get(c[w]["launch_task"]) if c.get(w) else None
            if row[w] is not None and t is not None:
                tf[w] = {"approach_matches": row[w]["approach_sha256"] == t["approach_sha256"],
                         "system_prompt_matches": row[w]["system_prompt_sha256"] == t["sol_system_sha256"],
                         "first_user_matches": row[w]["first_user_sha256"] == t["sol_input_sha256"]}
            else:
                tf[w] = None  # no hashed run, or no task record for the launcher task in the main launch.json
        row["task_file"] = tf
        row["pass"] = row["equal_approach"] and row["equal_system_prompt"]
        cells.append(row)
    return {"cells": cells, "policy": policy, "main_launch": spec["main_run"]}


@app.function(image=image, cpu=8, memory=16384, timeout=1800, retries=0, volumes={"/ev": modal.Volume.from_name(VOL, version=2).read_only()})
def provenance_pairwise(spec: dict) -> dict:
    """09's amendment A (interleaved launches): per row (a run3 or sim2 run of this batch, key <world>|<cell>), every attack arm of the
    cell that is relayed (run1, run2, sim1, run3, sim2) hashed by the same vendored hash_eval, and every PAIR compared."""
    import itertools
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    # the policy comes from the spec; the main's launch.json only as a fallback, and never for the T196 launch (main_run "@index:...",
    # its donors are a merged reference index, not a launch)
    pol = spec.get("policy") or json.loads((Path("/ev/evals_v1") / spec["main_run"] / "launch.json").read_text())["policy"]
    jobs = sorted({(w["path"], w["sha256"]) for r in spec["rows"] for w in r["worlds"].values()})
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("fork")) as pool:
        hashed = dict(zip([j[0] for j in jobs], pool.map(hash_one, [(path, sha, pol) for path, sha in jobs])))
    rows = []
    for r in spec["rows"]:
        hw = {w: {"approach_sha256": hashed[v["path"]]["approach_sha256"], "system_prompt_sha256": hashed[v["path"]]["sent_system_sha256"],
                  "first_user_sha256": hashed[v["path"]]["sent_user_sha256"], "policy_calls": hashed[v["path"]]["policy_calls"], "launch": v["launch"],
                  "launch_task": v["launch_task"], "eval": v["path"], "eval_sha256": v["sha256"]} for w, v in r["worlds"].items()}
        ok = [w for w, h in hw.items() if h["policy_calls"] > 0]
        pairs = {f"{a}~{b}": {"equal_approach": hw[a]["approach_sha256"] == hw[b]["approach_sha256"] and hw[a]["approach_sha256"] is not None,
                              "equal_system_prompt": hw[a]["system_prompt_sha256"] == hw[b]["system_prompt_sha256"] and hw[a]["system_prompt_sha256"] is not None}
                 for a, b in itertools.combinations(sorted(ok), 2)}
        rows.append({"key": r["key"], "cell": r["cell"], "subject": r["subject"], "worlds": hw, "pairs": pairs, "no_policy_call": sorted(set(hw) - set(ok)),
                     "staged_shas": r["staged_shas"],
                     "pass": r["subject"] in ok and all(v["equal_approach"] and v["equal_system_prompt"] for k, v in pairs.items() if r["subject"] in k.split("~"))})
    return {"rows": rows, "policy": pol}


def hash_at(job: tuple) -> dict:
    root, path, want_sha, policy = job
    p = Path(root) / path
    if hashlib.sha256(p.read_bytes()).hexdigest() != want_sha:
        raise ValueError(f"{path}: differs from its pinned sha256")
    try:
        return load_hasher()((str(p), policy))
    except Exception as e:  # identifiers and the exception type only
        raise RuntimeError(f"{path}: {type(e).__name__} (message withheld)") from None


@app.function(image=image, cpu=8, memory=16384, timeout=1800, retries=0,
              volumes={"/ev": modal.Volume.from_name(VOL, version=2).read_only(), "/src": modal.Volume.from_name(SRC_VOL, version=2).read_only()})
def provenance_ug(spec: dict) -> dict:
    """09's ug note @ 48363962a Q1 (Append 34 §7, full): per staged ug cell, run1, run2, the counted sim1 (the comparability map's exact
    artifacts, FINAL's) and sim_ug hashed by the same vendored hash_eval: approach and the attacker system prompt the first policy call was
    SENT. A cell passes only when all four worlds were hashed with a policy call and every world's approach and system-prompt hashes are
    equal (a missing artifact, an empty prompt or a mismatch fails; nothing falls back)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    jobs = sorted({(w["root"], w["path"], w["sha256"], spec["policy"]) for r in spec["rows"] for w in r["worlds"].values() if w})  # one policy, as the IL check
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("fork")) as pool:
        hashed = dict(zip([(j[0], j[1]) for j in jobs], pool.map(hash_at, jobs)))
    cells = []
    for r in spec["rows"]:
        row = {"cell": r["cell"], "origin": r["origin"]}
        for w in UG_WORLDS:
            v = r["worlds"].get(w)
            h = hashed.get((v["root"], v["path"])) if v else None
            row[w] = None if h is None else {"approach_sha256": h["approach_sha256"], "system_prompt_sha256": h["sent_system_sha256"],
                                             "system_prompt_len": h["sent_system_len"], "first_user_sha256": h["sent_user_sha256"],
                                             "policy_calls": h["policy_calls"], "eval": v["key"], "eval_sha256": v["sha256"]}
            if v is None:
                row[f"{w}_null_reason"] = r.get(f"{w}_null_reason") or "no artifact"
        # a world counts only with a policy call and a NON-EMPTY sent system prompt (four empty prompts hash equal and must not pass)
        got = [row[w] for w in UG_WORLDS if row[w] is not None and row[w]["policy_calls"] > 0 and (row[w]["system_prompt_len"] or 0) > 0]
        row["empty_or_missing_prompt"] = [w for w in UG_WORLDS if row[w] is None or row[w]["policy_calls"] == 0 or not (row[w]["system_prompt_len"] or 0) > 0]
        full = len(got) == len(UG_WORLDS)
        row["equal_approach"] = full and len({x["approach_sha256"] for x in got}) == 1 and got[0]["approach_sha256"] is not None
        row["equal_system_prompt"] = full and len({x["system_prompt_sha256"] for x in got}) == 1 and got[0]["system_prompt_sha256"] is not None
        row["equal_first_user"] = full and len({x["first_user_sha256"] for x in got}) == 1
        row["staged_shas"] = r["staged_shas"]
        row["pass"] = row["equal_approach"] and row["equal_system_prompt"]
        cells.append(row)
    return {"cells": cells, "policy": spec["policy"]}


@app.local_entrypoint()
def main(spec: str, out: str):
    s = json.loads(Path(spec).read_text())
    if s.get("mode") == "ug":  # the ungrounded arm (09's ug note @ 48363962a Q1): four worlds per cell, two Volumes
        r = provenance_ug.remote(s)
        import subprocess

        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
        doc = {"batch": s["batch"], "pool": s["pool"], "mode": "ug", "ug_launch": s["sim_run"], "comparability_map": s["comparability_map"],
               "hasher": f"experiments/2026-10-14_q3_q35_simhonest_purchase/check_approach_hashes.py@8ad72f4a L33-66 (vendored, sha256 {HASHER_SHA[:16]})",
               "checker": f"experiments/2026-10-14_q3_overnight_chain/provenance_chain.py@{commit}", "inspect_ai": INSPECT, "policy": r["policy"],
               "staged_shas": sorted({x for c in r["cells"] for x in c["staged_shas"] if x}), "covers": {"cells": [c["cell"] for c in r["cells"]]}, "cells": r["cells"],
               "counts": {"cells": len(r["cells"]), "pass": sum(c["pass"] for c in r["cells"]), "equal_approach": sum(c["equal_approach"] for c in r["cells"]),
                          "equal_system_prompt": sum(c["equal_system_prompt"] for c in r["cells"])},
               "pass": bool(r["cells"]) and all(c["pass"] for c in r["cells"])}
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.open("x").write(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({"file": p.name, "pass": doc["pass"], "counts": doc["counts"]}))
        return
    if s.get("mode") == "pairwise":  # an interleaved batch (09's amendment A)
        r = provenance_pairwise.remote(s)
        import subprocess

        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
        doc = {"batch": s["batch"], "pool": s["pool"], "quarter": s["q"], "snapshot": s.get("snap"), "main_launch": s["main_run"], "il_launch": s["sim_run"], "mode": "pairwise",
               "hasher": f"experiments/2026-10-14_q3_q35_simhonest_purchase/check_approach_hashes.py@8ad72f4a L33-66 (vendored, sha256 {HASHER_SHA[:16]})",
               "checker": f"experiments/2026-10-14_q3_overnight_chain/provenance_chain.py@{commit}", "inspect_ai": INSPECT, "policy": r["policy"],
               "staged_shas": sorted({x for row in r["rows"] for x in row["staged_shas"].values() if x}), "covers": {"cells": sorted({row["cell"] for row in r["rows"]}),
               "keys": [row["key"] for row in r["rows"]]}, "rows": r["rows"],
               "counts": {"rows": len(r["rows"]), "pass": sum(row["pass"] for row in r["rows"]),
                          "pairs": sum(len(row["pairs"]) for row in r["rows"]), "pairs_equal": sum(v["equal_approach"] and v["equal_system_prompt"] for row in r["rows"] for v in row["pairs"].values())},
               "pass": bool(r["rows"]) and all(row["pass"] for row in r["rows"])}
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.open("x").write(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({"file": p.name, "pass": doc["pass"], "counts": doc["counts"]}))
        return
    r = provenance.remote(s)
    import subprocess

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    doc = {"batch": s["batch"], "pool": s["pool"], "quarter": s["q"], "snapshot": s.get("snap"), "main_launch": s["main_run"], "sim_launch": s["sim_run"],
           "hasher": f"experiments/2026-10-14_q3_q35_simhonest_purchase/check_approach_hashes.py@8ad72f4a L33-66 (vendored, sha256 {HASHER_SHA[:16]})",
           "checker": f"experiments/2026-10-14_q3_overnight_chain/provenance_chain.py@{commit}", "inspect_ai": INSPECT, "policy": r["policy"],
           "staged_shas": sorted({x for c in r["cells"] for x in c["staged_shas"] if x}), "covers": {"cells": [c["cell"] for c in r["cells"]]}, "cells": r["cells"],
           "counts": {"cells": len(r["cells"]), "pass": sum(c["pass"] for c in r["cells"]), "equal_approach": sum(c["equal_approach"] for c in r["cells"]),
                      "equal_system_prompt": sum(c["equal_system_prompt"] for c in r["cells"]), "first_user_empty": sum(c["first_user_empty"] for c in r["cells"])},
           "pass": bool(r["cells"]) and all(c["pass"] for c in r["cells"])}
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.open("x").write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"file": p.name, "pass": doc["pass"], "counts": doc["counts"]}))
