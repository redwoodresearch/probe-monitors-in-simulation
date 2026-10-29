"""IL resolution index (d9 10:3x, for 09's IL-analysis fork: Append 32 co-primary sim2 - run3 on Revision 6s). Keys only:
no outcome field is read (IL embargo); run states, manifests, staging gates and supersession bindings only.

One row per registered IL cell x world (run3, sim2) of every IL launch (il_q1..q4, il_old, il_s1, il_s2, il_s3, il_s5,
il_s4d). For each row: the authoritative launch_id (bare), run key, counted manifest (path@sha), scorer folder, stage,
attempt label, status, replacement_kind and the superseded originals. Resolution: the original, unless a pinned
supersession record of the chain designates a replacement; a later designation (rank r < r2 < r3 < c) retires an earlier one.

  python make_il_resolution_index.py --chain-commit <sha> --out <json>   (exit 1 if any check fails)

Sources: the chain branch at --chain-commit (results/supersession/*.json, config/manifest_*.json, results/chain_stage_*.json,
results/hygiene/: each original family's latest run table run_table_<family>[_vN].json (a .SUPERSEDED.md marker retires a
version), per-batch labels labels_il_<batch>.json, per-batch attestations attestations_il_<batch>[_vN].json (highest version),
input_index_retries_2026-10-15.json), this folder's registered task files and rerun/completion registrations. A row is
"scored" only if its run is in a counted manifest, not excluded at staging, and every check of the attestation of its staged
set passes. 09's run_failed rulings for staged-out runs are cited by the commit that added each ruling to
aswin/convergence-audit-09's RULE.md.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
C = "experiments/2026-10-14_q3_overnight_chain"
WORLD_ARM = {"run3": "attack", "sim2": "sim_attack@september"}
ARM_WORLD = {v: k for k, v in WORLD_ARM.items()}
# launch key -> (chain family prefix, original launch id, registered IL task file, run-table batch or None)
LAUNCHES = {
    "q1": ("il311_q1", "2026-10-14_il_q1", "config/tasks_il_q1_all.json", "il-q1"),
    "q2": ("il311_q2", "2026-10-14_il_q2", "config/tasks_il_q2_all.json", "il-q2"),
    "q3": ("il311_q3", "2026-10-14_il_q3", "config/tasks_il_q3_all.json", "il-q3"),
    "q4": ("il311_q4", "2026-10-14_il_q4", "config/tasks_il_q4_all.json", "il-q4"),
    "old": ("ilold_old", "2026-10-14_il_old", "config/tasks_il_old_all.json", "il-old"),
    "s1": ("ilnm_s1", "2026-10-14_il_s1", "config/tasks_il_s1.json", "il-s1"),
    "s2": ("ilnm_s2", "2026-10-14_il_s2", "config/tasks_il_s2.json", "il-s2"),
    "s3": ("ilnm_s3", "2026-10-14_il_s3", "config/tasks_il_s3.json", None),
    "s5": ("ilnm_s5", "2026-10-14_il_s5", "config/tasks_il_s5.json", None),
    "s4d": ("ilnm_s4d", "2026-10-14_il_s4d", "config/tasks_il_s4d.json", None),
}
RANK = {"original": 0, "rlr": 1, "rlr2": 2, "rlr3": 3, "completion": 4}
FOLDER_RE = re.compile(r"^(?:il311_q[1-4](?:r|r2|r3|c)?|ilnm_s(?:[1-5]|4d)(?:r|r2|r3)?|ilold_old(?:r|r3|c)?)_(?:run3(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__attack|sim2(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_attack)$")
# 09's rulings on staged-out replacement runs (aswin/convergence-audit-09, experiments/2026-10-14_priority1_spec_09/RULE.md;
# commit = the one that added the ruling heading)
RULINGS = {
    ("2026-10-15_il_q3_rlr3", "sim2", "t478_media_processing"): ("run_failed (OutputContractError)", "Append 35 ruling t478 @ b321d2aad"),
    ("2026-10-15_il_q3c", "sim2", "t318_file_indexing"): ("run_failed (EmptyPolicyCompletionError)", "Append 35 ruling t318 @ 4f06d384c"),
    ("2026-10-15_il_oldc", "sim2", "t66_file_indexing"): ("run_failed (EmptyPolicyCompletionError)", "Append 35 ruling t66 @ 3a6ac11da"),
    ("2026-10-15_il_q1_rlr3", "sim2", "t237_media_processing"): ("run_failed (OutputContractError)", "Append 35 ruling t237 @ 2f3437700"),
    ("2026-10-15_il_q1c", "run3", "t360_shared_dotfiles"): ("run_failed (EmptyPolicyCompletionError)", "Append 35 ruling t360 @ 2d20dd751"),
    ("2026-10-15_il_q2c", "run3", "t434_slack_clone"): ("run_failed (EmptyPolicyCompletionError)", "Append 35 ruling t434 @ 2b54906c3"),
}
# 09's rulings on ORIGINAL runs (aswin/convergence-audit-09 RULE.md; commit = the one that added the ruling). status None keeps
# "staging blocker" (the run itself is complete); otherwise 09's class replaces it and the stager string is kept as staging_reason.
_EMPTY = "excluded at staging (empty staged set): chain il_final_run3_empty_staged_sets_2026-10-15.json @ f803fa61f"
ORIG_RULINGS = {
    ("2026-10-14_il_q1", "run3", "t389_model_registry"): ("run_failed (EmptyPolicyCompletionError)", "09 @ f8736f041 section 3: 1-message EmptyPolicyCompletionError, causal class unknown, not rerun-eligible"),
    ("2026-10-14_il_q3", "sim2", "t386_slack_clone"): ("run_failed (EmptyPolicyCompletionError)", "09 @ f8736f041 section 3: 1-message EmptyPolicyCompletionError, causal class unknown, not rerun-eligible"),
    ("2026-10-14_il_q3", "sim2", "t290_web_scraping"): ("run_failed (OutputContractError)", "09 @ f8736f041 section 2: OutputContractError at the first step, causal class unknown"),
    ("2026-10-14_il_q4", "sim2", "t375_media_processing"): ("run_failed (OutputContractError)", "09 @ f8736f041 section 2: OutputContractError at the first step, causal class unknown"),
    ("2026-10-14_il_s2", "run3", "nm171_shared_dotfiles"): ("run_failed (EmptyPolicyCompletionError)", "09 @ 1bc718e98: 1-message EmptyPolicyCompletionError, donors usable unknown, not rerun-eligible"),
    ("2026-10-14_il_s1", "run3", "nm209_slack_clone"): (None, "09 @ f8736f041 section 4: staging blocker, no usable donor; the run itself is complete"),
    ("2026-10-14_il_s1", "sim2", "nm209_slack_clone"): (None, "09 @ 527337ee3 item 2: staging blocker, no usable donor; same donor basis as run3|nm209"),
    ("2026-10-14_il_old", "run3", "t139_model_registry"): (None, "09 @ f8736f041 section 5: staging blocker, no usable donor (no donor row in the il_old reference index)"),
    ("2026-10-14_il_old", "run3", "t98_model_registry"): (None, "09 Append 35 ruling t98 @ 221bb94ea: staging blocker, no usable donor in the pinned reference"),
    ("2026-10-14_il_s2", "run3", "nm140_model_registry"): (None, "09 @ 0d8e1273e item 1: staging blocker, no usable donor (every listed donor has an empty system prompt)"),
    **{("2026-10-14_il_s1", "run3", c): ("run_failed (no policy call)", f"09 @ 0d8e1273e item 2: causal class unknown, not rerun-eligible, no IL pair; {_EMPTY}") for c in ("nm208_slack_clone", "nm213_slack_clone")},
    **{("2026-10-14_il_s2", "run3", c): ("run_failed (no policy call)", f"09 @ 87f9ec7b4 item 4: causal class unknown, not rerun-eligible; {_EMPTY}") for c in ("nm222_slack_clone", "nm241_slack_clone")},
    ("2026-10-14_il_s5", "run3", "nm203_slack_clone"): ("run_failed (no policy call)", "09 Append 32/35 notes @ 2d0ccfe05 section 1: causal class unknown, not rerun-eligible, no IL pair; evidence evidence_il_s5_nm203_2026-10-15.json @ 2ea2869d6"),
    **{("2026-10-14_il_s3", "run3", c): ("run_failed (no policy call)", "09 @ 51a612b29 section 1: causal class unknown, not rerun-eligible") for c in ("nm106_media_processing", "nm219_slack_clone")},
    ("2026-10-14_il_s3", "run3", "nm141_model_registry"): ("run_failed (EmptyPolicyCompletionError)", "09 @ 51a612b29 section 2: causal class unknown, not rerun-eligible, no IL pair"),
}
# originals with no counted .eval (counted attempt without .eval): 09 @ 0d8e1273e item 3
NO_EVAL_RULINGS = {("2026-10-14_il_old", "run3", c): ("run_failed (no policy call)", "09 @ 0d8e1273e item 3: counted attempt without .eval, causal class unknown, not rerun-eligible") for c in ("t88_model_registry", "t169_clinical_trial")}
EXPECT_RLR_CELLS = 70
EXPECT_TIER3 = {"il311_q3c": 16, "ilold_oldc": 52, "il311_q1c": 17, "il311_q2c": 17, "il311_q4c": 21}  # v5: q4c routed (21 = config/completion/ilc_supersession_tier3_q4.json rows)
PENDING = {}  # v5 (d9 ~12:10): q4c routed on the chain (attestations @ 605b7e1e7, supersession v2 @ 78a5b7ebd); no launch pending


def git_show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=HERE).stdout


def git_ls(commit, path):
    out = subprocess.run(["git", "ls-tree", "--full-tree", "--name-only", commit, path + "/"], capture_output=True, text=True, check=True, cwd=HERE).stdout
    return [p.split("/")[-1] for p in out.split()]


def bare(launch_id):
    return launch_id[5:] if launch_id.startswith("runs/") else launch_id  # 09 @ c065957e2: strip one leading runs/


def kind_of(family):
    m = re.fullmatch(r"(il311_q[1-4]|ilnm_s(?:[1-5]|4d)|ilold_old)(r|r2|r3|c)?", family)
    assert m, family
    return {None: "original", "r": "rlr", "r2": "rlr2", "r3": "rlr3", "c": "completion"}[m.group(2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-commit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ended-unrouted", default=None, help="JSON {launch_id, finished, launch_json_sha256, read, keys: {run_key: {state, attempts}}} for a "
                    "launch that has ended but is not routed yet (09 @ Append 35 section 2: resolved by launch, status 'ended; awaiting routing')")
    a = ap.parse_args()
    CH = subprocess.run(["git", "rev-parse", a.chain_commit], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    errors, sources = [], {}

    def chain_json(path):
        raw = git_show(CH, f"{C}/{path}")
        sources[path] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    # 1. the universe: registered IL cells x world
    universe = {}
    for h, (fam, orig, tf, _) in LAUNCHES.items():
        t = json.loads((HERE / tf).read_text())
        cells = [c["key"] for c in t["cells"]]  # the task key (e.g. t441_media_processing, nm343_web_scraping), as in the chain's records
        assert len(cells) == len(set(cells)), tf
        universe[h] = cells
    # 2. counted manifests (v2 where a .SUPERSEDED.md marker or the input index says so)
    names = git_ls(CH, f"{C}/config")
    superseded = {n[:-len(".SUPERSEDED.md")] for n in names if n.endswith(".SUPERSEDED.md")}
    idx = chain_json("results/hygiene/input_index_retries_2026-10-15.json")
    auth_folder = {e["target"]: e["authoritative"].get("scorer_folder") for e in idx["entries"] if e["authoritative"].get("scorer_folder")}
    auth_set = {e["target"]: e["authoritative"].get("staged_set") for e in idx["entries"] if e["authoritative"].get("staged_set")}
    auth_manifest = {e["target"]: e["authoritative"]["manifest"].split("/")[-1] for e in idx["entries"] if e["authoritative"].get("manifest")}
    manifests, key_manifest, key_sha, path_key = {}, defaultdict(list), defaultdict(set), {}
    for n in names:
        m = re.fullmatch(r"manifest_((?:il311_q[1-4]|ilnm_s(?:[1-5]|4d)|ilold_old)(?:r|r2|r3|c)?)_(run3|sim2)(_s\d+)?(_v2)?\.json", n)
        if not m:
            continue
        stem = n[:-5]
        if stem in superseded:
            continue
        fam, world, snap, v2 = m.groups()
        target = f"{fam}_{world}{snap or ''}"
        if v2 is None and f"{stem}_v2.json" in names:
            if stem not in superseded and auth_manifest.get(target) != f"{stem}_v2.json":
                errors.append(f"{stem}_v2.json exists but neither a .SUPERSEDED.md marker nor the input index makes it authoritative")
            continue  # its v2 is authoritative
        if target in auth_manifest and auth_manifest[target] != n:
            errors.append(f"input index names {auth_manifest[target]} for {target}, but {n} would be counted")
        folder = auth_folder.get(target, f"{target}_v1__{'attack' if world == 'run3' else 'sim_attack'}")
        if not FOLDER_RE.match(folder):
            errors.append(f"scorer folder {folder} does not match the registered regex")
        mj = chain_json(f"config/{n}")
        manifests[n] = {"family": fam, "world": world, "target": target, "folder": folder, "sha256": sources[f"config/{n}"]}
        for x in mj["files"]:
            if x["arm_dir"] == WORLD_ARM[world]:
                k = (bare(x["launch"]), world, x["task"])
                if n not in key_manifest[k]:
                    key_manifest[k].append(n)
                key_sha[k + (n,)].add(x["sha256"])
                path_key[bare(f"{x['launch']}/{x['rel']}")] = k  # the epoch's file -> its key (a redraw's dir is <task>_redrawN)
    # 3. staging gates: the authoritative staging record per target (the input index's set, else the latest _v1 run)
    stage_by_set = {}
    for n in sorted(git_ls(CH, f"{C}/results")):
        if not n.startswith("chain_stage_") or not n.endswith(".json"):
            continue
        s = chain_json(f"results/{n}").get("main")
        if not isinstance(s, dict) or "c_gate" not in s or "out" not in s:
            continue
        shas = s.get("assembled", {}).get("staged_inputs_sha256") or {}
        stage_by_set[s["out"]] = (n, s["c_gate"]["excluded"], sorted(shas.values()) if isinstance(shas, dict) else [shas])  # latest record per set wins
    excluded = {}
    for n, mi in manifests.items():
        st = auth_set.get(mi["target"], f"{mi['target']}_v1")
        if st not in stage_by_set:
            mi["staging"] = None
            continue
        rec, ex, staged_shas = stage_by_set[st]
        mi["staging"] = {"staged_set": st, "record": rec, "sha256": sources[f"results/{rec}"], "staged_inputs_sha256": staged_shas}
        for ep, reason in ex.items():
            k = path_key.get(bare(ep.split("#")[0]))  # some stagers write runs/<launch>/...
            if k is None:
                errors.append(f"{rec}: excluded epoch {ep} matches no counted manifest file")
                continue
            excluded[k] = (reason, n)
        if not staged_shas:
            mi["staging"]["empty_staged_set"] = True
    # 4. supersession records (v2 replaces v1), designations by key
    sup_names = git_ls(CH, f"{C}/results/supersession")
    desig = defaultdict(list)
    fam_records = {}
    for n in sup_names:
        if not n.startswith("supersession_") or (not n.endswith("_v2.json") and n.replace(".json", "_v2.json") in sup_names):
            continue
        r = chain_json(f"results/supersession/{n}")
        fam = r.get("rerun_family") or r.get("completion_family")
        if r.get("pass") is not True or r.get("errors"):
            errors.append(f"{n}: upstream record not passing (pass={r.get('pass')}, errors={len(r.get('errors') or [])})")
        kind = kind_of(fam)
        staged_out = {x["key"] for x in r.get("staged_out_rows") or []}
        fam_records[fam] = (n, r)
        for x in r["records"]:
            rep = x["replacement"]
            if rep.get("state") not in ("done", "no policy call", "failed", "not run"):  # Codex r33c: every record, unconditionally
                errors.append(f"{n}: {x['key']} unknown replacement state {rep.get('state')!r}")
            desig[(x["arm"], x["cell"])].append({
                "kind": kind, "family": fam, "launch_id": bare(rep["launch_id"]), "run_key": rep["run_key"], "state": rep.get("state"),
                "original_launch_id": bare(x["original"]["launch_id"]), "original_state": x["original"].get("state"),
                "original_manifests": x["original"].get("chain_manifest") or [], "reason": x.get("rerun_reason") or x.get("tier"),
                "supersedes_original_run": x.get("supersedes_original_run"), "staged": x["key"] not in staged_out,
                "record": f"{C}/results/supersession/{n}@{sources['results/supersession/' + n][:12]}"})
    # 4b. a registered completion with no chain record yet (q4c, running): its per-arm map designates it (pending)
    ended = json.loads(Path(a.ended_unrouted).read_text()) if a.ended_unrouted else None
    if ended and not ended.get("finished"):
        errors.append(f"--ended-unrouted {ended.get('launch_id')}: launch.json has no finished value")  # Codex r37: fail closed
    for h in ("q1", "q2", "q3", "q4", "old"):
        fam = f"{LAUNCHES[h][0]}c"
        if fam in fam_records:
            continue
        p = f"config/completion/ilc_supersession_perarm_{h}.json"
        raw = (HERE / p).read_bytes()
        for x in json.loads(raw)["rows"]:
            desig[(x["arm"], x["cell"])].append({
                "kind": "completion", "family": fam, "launch_id": bare(x["authoritative"]["launch"]), "run_key": x["authoritative"]["run_key"],
                "state": ((ended["keys"].get(x["authoritative"]["run_key"]) or {}).get("state", "missing") if ended and bare(x["authoritative"]["launch"]) == ended["launch_id"] else "pending"),
                "original_launch_id": bare(x["original"]["launch"]), "original_state": x["original"].get("state"),
                "original_manifests": [], "reason": x.get("tier"), "supersedes_original_run": None, "staged": True,
                "record": f"experiments/2026-10-15_q3_interleaved_run3_sim2/{p}@{hashlib.sha256(raw).hexdigest()[:12]} (per-arm map; no chain record yet)"})
    # 5. original run states: the chain's IL run table (7 launches) and box reads (s3, s5, s4d)
    hyg = git_ls(CH, f"{C}/results/hygiene")
    hyg_sup = {n[:-len(".SUPERSEDED.md")] for n in hyg if n.endswith(".SUPERSEDED.md")}

    def latest(prefix):  # the highest version of <prefix>[_vN].json not retired by a .SUPERSEDED.md marker
        cands = [n for n in hyg if re.fullmatch(re.escape(prefix) + r"(_v\d+)?\.json", n) and n[:-5] not in hyg_sup]
        ver = lambda n: int(re.search(r"_v(\d+)\.json$", n).group(1)) if re.search(r"_v\d+\.json$", n) else 1  # noqa: E731
        return max(cands, key=ver) if cands else None
    orig_state, run_tables = {}, {}
    for h, (fam, orig, _, _) in LAUNCHES.items():
        n = latest(f"run_table_{fam}")
        if n is None:
            errors.append(f"no run table for {fam}")
            continue
        run_tables[fam] = n
        for r in chain_json(f"results/hygiene/{n}")["cells"]:
            orig_state[(bare(r["launch_id"]), r["world"], r["cell"])] = {"status": r["status"], "first_start": r["first_start"], "stage": r["stage"]}

    def batch_of(target):  # il311_q1_run3_s1 -> il-q1_s1; ilold_oldc_sim2 -> il-oldc; ilnm_s3_run3 -> il-s3
        m = re.fullmatch(r"(?:il311_|ilnm_|ilold_)(q[1-4]|s(?:[1-5]|4d)|old)(r|r2|r3|c)?_(?:run3|sim2)(_s\d+)?", target)
        return f"il-{m.group(1)}{m.group(2) or ''}{m.group(3) or ''}"
    label_cache, attest_cache = {}, {}

    def label_for(batch, key):
        if batch not in label_cache:
            n = latest(f"labels_il_{batch}")
            label_cache[batch] = (n, chain_json(f"results/hygiene/{n}")["labels"] if n else None)
        n, L = label_cache[batch]
        return (L.get(key) if L is not None else None), n

    def attest_for(batch, staged_shas):
        if batch not in attest_cache:
            n = latest(f"attestations_il_{batch}")
            attest_cache[batch] = (n, chain_json(f"results/hygiene/{n}")["attestations"] if n else None)
        n, A = attest_cache[batch]
        if A is None or not staged_shas or any(sh not in A for sh in staged_shas):
            return None, n
        if any(not A[sh] for sh in staged_shas):
            return ["no checks recorded"], n
        return sorted({c["check"] for sh in staged_shas for c in A[sh] if c.get("pass") is not True}), n
    # 6. rows
    rows, final_launch = [], {}
    for h, (fam, orig, tf, batch) in LAUNCHES.items():
        for cell in universe[h]:
            for world in ("run3", "sim2"):
                ds = sorted([d for d in desig.get((world, cell), []) if d["original_launch_id"] == orig], key=lambda d: RANK[d["kind"]])
                os_ = orig_state.get((orig, world, cell))
                if os_ is None:
                    errors.append(f"{orig} {world}|{cell}: no original run state")
                    continue
                if ds:
                    d = ds[-1]
                    auth, run_key, kind = d["launch_id"], d["run_key"], d["kind"]
                else:
                    d, auth, run_key, kind = None, orig, f"{WORLD_ARM[world]}/{cell}", "original"
                final_launch[(h, world, cell)] = auth
                mans = key_manifest.get((auth, world, cell), [])
                if len(mans) > 1:
                    errors.append(f"{auth} {world}|{cell}: in {len(mans)} counted manifests {mans}")
                man = mans[0] if mans else None
                ex = excluded.get((auth, world, cell))
                ruling = RULINGS.get((auth, world, cell))
                label, label_file, attest_fail, attest_file = None, None, None, None
                if man:
                    bt = batch_of(manifests[man]["target"])
                    label, label_file = label_for(bt, f"{world}|{cell}")
                    attest_fail, attest_file = attest_for(bt, (manifests[man].get("staging") or {}).get("staged_inputs_sha256"))
                if ended and auth == ended["launch_id"]:
                    st = d["state"] if d else None
                    if st == "done":
                        status = "ended; awaiting routing"
                    elif st == "not run":
                        status = "never started"
                    elif st in ("no policy call", "failed"):
                        status = f"run_failed ({st}); awaiting routing"
                    else:
                        status = "error"
                        errors.append(f"{auth} {world}|{cell}: launch.json state {st!r} for the designated run key")
                    cite = f"{auth} launch.json (sha256 {ended['launch_json_sha256'][:16]}..., finished {ended['finished']}, read {ended['read']}); resolved by launch, routing pending as a data update"
                elif auth in PENDING:
                    status, cite = PENDING[auth], None
                elif man and ex and manifests[man]["staging"].get("empty_staged_set") and not ruling:
                    orr = ORIG_RULINGS.get((auth, world, cell))
                    status = (orr[0] if orr and orr[0] else f"staging blocker: {ex[0]} (empty staged set)")
                    cite = orr[1] if orr else f"stage record of {ex[1]} (no 09 class ruling found)"
                elif man and not manifests[man].get("staging"):
                    status, cite = "pending: counted but not yet staged at the chain commit", None
                elif man and not ex and attest_fail is None:
                    status, cite = "pending: staged set not attested at the chain commit", attest_file
                elif man and not ex and attest_fail:
                    status, cite = f"attestation failing: {', '.join(attest_fail)}", attest_file
                elif ruling:
                    status, cite = ruling
                    if not ex or (d and d["staged"]):
                        errors.append(f"{auth} {world}|{cell}: ruling cited but the run is not staged out")
                elif d is not None:
                    if d["state"] == "done" and man and not ex:
                        status, cite = "scored", None
                    elif d["state"] == "done" and ex:
                        status, cite = f"staging blocker: {ex[0]}", f"stage record of {ex[1]}"
                    elif d["state"] == "done":
                        status, cite = "error", None
                        errors.append(f"{auth} {world}|{cell}: replacement done but in no counted manifest")
                    elif d["state"] in ("no policy call", "failed"):
                        status, cite = "run_failed", None
                    else:
                        status, cite = ("never started" if d["state"] == "not run" else d["state"]), None
                else:
                    s = os_["status"]
                    if s in ("ran", "done"):
                        if label == "run_failed":
                            status, cite = "run_failed", label_file
                        elif man and not ex:
                            status, cite = "scored", None
                        elif ex:
                            orr = ORIG_RULINGS.get((auth, world, cell))
                            status = orr[0] if orr and orr[0] else f"staging blocker: {ex[0]}"
                            cite = orr[1] if orr else f"stage record of {ex[1]} (no 09 class ruling found)"
                        else:
                            status, cite = "error", None
                            errors.append(f"{auth} {world}|{cell}: ran but in no counted manifest")
                    elif s == "no policy call" or "no .eval" in s:
                        nr = NO_EVAL_RULINGS.get((auth, world, cell))
                        status, cite = nr if nr else (f"run_failed ({s})", f"{run_tables.get(fam)} status (no 09 class ruling found)")
                    elif s == "not run":
                        started = os_.get("first_start") is not None or os_.get("attempts", 0) > 0
                        status, cite = ("not run" if started else "never started"), None
                    else:
                        status, cite = "error", None
                        errors.append(f"{auth} {world}|{cell}: unknown run-table status {s!r}")
                sup = []
                for prev in ([{"kind": "original", "launch_id": orig, "state": os_["status"]}] + ds[:-1]) if d else []:
                    pm = key_manifest.get((prev["launch_id"], world, cell), [])
                    sup.append({"launch_id": prev["launch_id"], "kind": prev["kind"], "state": prev.get("state"),
                                "manifest": [f"{m}@{manifests[m]['sha256'][:12]}" for m in pm],
                                "folder": [manifests[m]["folder"] for m in pm],
                                "reason": d["reason"] if prev["kind"] == "original" else f"retired by {d['kind']}",
                                "record": d["record"]})
                rows.append({"launch": h, "cell": cell, "world": world, "key": f"{world}|{cell}", "stage": os_.get("stage"),
                             "launch_id": auth, "run_key": run_key, "replacement_kind": kind,
                             "counted_manifest": f"{C}/config/{man}@{manifests[man]['sha256'][:12]}" if man else None,
                             "scorer_folder": manifests[man]["folder"] if man else None,
                             "staging_record": (manifests[man].get("staging") or {}).get("record") if man else None,
                             "attempt_label": label, "label_file": label_file, "attestation_file": attest_file,
                             "status": status, "status_cite": cite, "staging_reason": ex[0] if ex else None,
                             "run_table_status": os_["status"] if kind == "original" else None, "superseded_originals": sup})
    # 7. cross-checks (fail closed)
    for (h, world, cell), auth in final_launch.items():
        other = final_launch.get((h, "sim2" if world == "run3" else "run3", cell))
        if other is not None and other != auth:
            errors.append(f"mixed pair {h} {cell}: {world} {auth} vs {other}")
    for (arm, cell), ds in desig.items():
        fams = Counter(d["family"] for d in ds)
        if any(v > 1 for v in fams.values()):
            errors.append(f"{arm}|{cell}: listed twice in one family {fams}")
    rlr_cells = {(r["launch"], r["cell"]) for r in rows if r["replacement_kind"] in ("rlr", "rlr2", "rlr3")}
    registered_rlr = set()
    for h in ("q1", "q2", "q3", "q4", "old", "s1", "s2"):
        for b in json.loads((HERE / f"config/ilrr_rerun_of_{h}.json").read_text())["rerun_of"]:
            registered_rlr.add((h, b["cell"]))
    if rlr_cells != registered_rlr or len(registered_rlr) != EXPECT_RLR_CELLS:
        errors.append(f"rate-limit cells: index {len(rlr_cells)}, registered {len(registered_rlr)} (expected {EXPECT_RLR_CELLS}); "
                      f"only in index {sorted(rlr_cells - registered_rlr)[:5]}, only registered {sorted(registered_rlr - rlr_cells)[:5]}")
    tier3 = {}
    for fam, exp in EXPECT_TIER3.items():
        n, r = fam_records[fam]
        tier3[fam] = sum(1 for x in r["records"] if x["arm"] == "run3" and x.get("supersedes_original_run"))
        if tier3[fam] != exp:
            errors.append(f"{fam}: {tier3[fam]} superseded run3 originals, expected {exp}")
    for fam, (n, r) in fam_records.items():  # staged-out rows must name records whose replacement ran
        rk = {x["key"]: x for x in r["records"]}
        for y in r.get("staged_out_rows") or []:
            if y["key"] not in rk or rk[y["key"]]["replacement"].get("state") != "done":
                errors.append(f"{n}: staged_out_rows key {y['key']} is not a record with a done replacement")
    for fam, (n, r) in fam_records.items():  # each record's binding = the resolved row, unless retired by a later record
        for x in r["records"]:
            row = next((y for y in rows if y["cell"] == x["cell"] and y["world"] == x["arm"] and bare(x["original"]["launch_id"]) == LAUNCHES[y["launch"]][1]), None)
            if row is None:
                errors.append(f"{n}: {x['key']} has no registered cell")
                continue
            if row["launch_id"] != bare(x["replacement"]["launch_id"]) and RANK[row["replacement_kind"]] <= RANK[kind_of(fam)]:
                errors.append(f"{n}: {x['key']} binds {x['replacement']['launch_id']} but the index resolves {row['launch_id']}")
            for side in ("original", "replacement"):
                if x[side]["run_key"] != f"{WORLD_ARM[x['arm']]}/{x['cell']}" or x["key"] != f"{x['arm']}|{x['cell']}":
                    errors.append(f"{n}: {x['key']} {side} run key {x[side]['run_key']} is not {WORLD_ARM[x['arm']]}/{x['cell']}")
            if row["launch_id"] == bare(x["replacement"]["launch_id"]):  # this record is the row's authority: run key and manifest must agree
                if row["run_key"] != x["replacement"]["run_key"]:
                    errors.append(f"{n}: {x['key']} run key {x['replacement']['run_key']} != index {row['run_key']}")
                rm = row["counted_manifest"].split("/")[-1].split("@")[0] if row["counted_manifest"] else None
                recm = x["replacement"].get("chain_manifest") or []
                same = recm == ([rm] if rm else [])
                # a record written before its manifest's v2 (marker <stem>.SUPERSEDED.md) still binds when its .eval sha256s are in the v2 entry
                via_v2 = (rm and len(recm) == 1 and recm[0][:-5] in superseded and rm == recm[0][:-5] + "_v2.json"
                          and set(x["replacement"].get("eval_sha256") or []) <= key_sha[(row["launch_id"], row["world"], row["cell"], rm)]
                          and x["replacement"].get("eval_sha256"))
                if not same and not via_v2:
                    errors.append(f"{n}: {x['key']} replacement manifest {recm} != index {rm}")
                elif via_v2:
                    row.setdefault("binding_notes", []).append(f"{n} binds {recm[0]} (superseded by marker); its .eval sha256s are in {rm}")
                staged_out = x["key"] in {y["key"] for y in r.get("staged_out_rows") or []}
                ex_row = row["status"].startswith(("staging blocker", "run_failed ("))
                if staged_out != ex_row:
                    errors.append(f"{n}: {x['key']} staged_out={staged_out} but the index status is {row['status']}")
            om = sorted(key_manifest.get((bare(x["original"]["launch_id"]), x["arm"], x["cell"]), []))
            if sorted(x["original"].get("chain_manifest") or []) != om:
                errors.append(f"{n}: {x['key']} original manifest {x['original'].get('chain_manifest')} != counted {om}")
    used_r = {k for k in RULINGS if any((r["launch_id"], r["world"], r["cell"]) == k for r in rows)}
    if used_r != set(RULINGS):
        errors.append(f"run_failed rulings not matched to a row: {sorted(set(RULINGS) - used_r)}")
    for name, table in (("ORIG_RULINGS", ORIG_RULINGS), ("NO_EVAL_RULINGS", NO_EVAL_RULINGS)):  # each entry actually applied (Codex r35)
        used = {k for k in table if any((r["launch_id"], r["world"], r["cell"]) == k and r["status_cite"] == table[k][1]
                                        and (table[k][0] is None or r["status"] == table[k][0]) for r in rows)}
        if used != set(table):
            errors.append(f"{name} entries not applied to their row: {sorted(set(table) - used)}")
    for r in rows:  # every scored row carries a label from its batch's label file (restored, Codex r35)
        if r["status"] == "scored" and r["attempt_label"] is None:
            errors.append(f"{r['launch_id']} {r['key']}: scored but no label in {r['label_file']}")
    if ended:  # every run key of the ended launch is designated by the per-arm map, and vice versa
        des = {(r["world"], r["cell"]) for r in rows if r["launch_id"] == ended["launch_id"]}
        lk = {(ARM_WORLD[k.split("/", 1)[0]], k.split("/", 1)[1]) for k in ended["keys"]}
        if des != lk:
            errors.append(f"{ended['launch_id']}: launch.json run keys != designated keys (only launch {sorted(lk - des)[:4]}, only map {sorted(des - lk)[:4]})")
    out = {"what": "IL resolution index (keys only; IL embargo intact; no outcome field read)", "chain_commit": CH, "run_tables": run_tables,
           "ended_unrouted": ({k: ended[k] for k in ("launch_id", "finished", "launch_json_sha256", "read")} if ended else None),
           "sources": {f"{C}/{k}": v for k, v in sorted(sources.items())}, "pending": PENDING,
           "counts": {"rows": len(rows), "by_status": dict(Counter(r["status"].split(":")[0] for r in rows)),
                      "by_kind": dict(Counter(r["replacement_kind"] for r in rows)), "rate_limit_cells": len(rlr_cells), "tier3_superseded_run3": tier3},
           "errors": errors, "pass": not errors, "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("counts", "pass")}), "errors:", len(errors))
    for e in errors[:25]:
        print("  ", e)
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
