"""Adapter (metadata only; no activation or score is read) for S_paper493 (RULE Addendum P): ONE message per d9 chain routing record
(one staged set, all its arms) plus an adapter record that score_step3_sp493.py accepts. Gates (all must pass):
- names: every folder passes sp493_rules.refuse_folder (09's regex, d9's names record, the _try rule);
- attestation: d9's pointer and its payload, failing {}, every arm's staged sha with every check passing;
- staging: the chain stage record whose out is the staged set (per-arm shas and epochs; exclusions equal the routing record's);
- extraction: every attempt of the root (root, staged set and shas, Qwen/Qwen3.5-27B, the record's Volume, NVIDIA H200 on every shard);
- repro-c: the routing record's compare (no failing shard, full coverage, min cosine at or above its bar);
- format checks: each arm's record (passes, same staged sha and arm);
- manifest: the one the routing record points to; one counted file (first or redraw; donor "reference" rows are not counted) per (arm, cell); sample_error on EVERY row and null on every admitted
  row (09; d9's revised manifests carry it, non-null only on staging-excluded rows);
- cells: every admitted epoch passes sp493_rules.admit_cell; t987 and the b_run2 t615_r2 key are LISTED as refused by name, never routed;
  any other refusal stops the adapter.

  python3 -I build_sp493_message.py --routing <d9 routing json> --attest <d9 attest pointer json> --release <4f's list> --release-sha256 <sha> --out-dir <dir>
"""

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CHAIN = "experiments/2026-10-14_q3_overnight_chain"
truthy = lambda v: v is True or v == "true" or v == "True"  # noqa: E731
_s = importlib.util.spec_from_file_location("sp493_rules", HERE / "sp493_rules.py")
R = importlib.util.module_from_spec(_s)
_s.loader.exec_module(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", required=True)
    ap.add_argument("--attest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--release", required=True, help="4f's release list (routing records by path and sha256)")
    ap.add_argument("--release-sha256", required=True)
    a = ap.parse_args()
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, check=True).stdout  # noqa: E731
    on_origin = lambda c: any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", c).decode().splitlines())  # noqa: E731
    P = R.load()
    pins = {}

    def show(name, c, p):
        if not on_origin(c):
            raise SystemExit(f"{name}: commit {c} is not on origin")
        b = git("show", f"{c}:{p}")
        pins[name] = {"commit": git("rev-parse", c).decode().strip(), "path": p, "sha256": hashlib.sha256(b).hexdigest()}
        return json.loads(b)

    def at(ref):
        p, c = (x.strip() for x in ref.split(" @ "))
        return c, p

    rb, ab, lb = Path(a.routing).read_bytes(), Path(a.attest).read_bytes(), Path(a.release).read_bytes()
    rec, ptr, rel = json.loads(rb), json.loads(ab), json.loads(lb)
    if hashlib.sha256(lb).hexdigest() != a.release_sha256:
        raise SystemExit("the release list's sha256 differs from the pin")
    hit = [x for x in rel["released"] if x["routing_file"] == Path(a.routing).name]  # 4f: pin by PATH AND SHA256; refused sets never build
    if any(x["routing_file"] == Path(a.routing).name for x in rel.get("refused", [])) or len(hit) != 1 or hit[0]["routing_sha256"] != hashlib.sha256(rb).hexdigest() \
            or (hit[0]["staged_set"], re.sub(r"\s*\(sha256 [0-9a-f]+\.*\)$", "", hit[0]["manifest"]), hit[0]["batch"]) != (rec["staged_set"], rec["manifest"], rec["batch"]):
        raise SystemExit(f"{Path(a.routing).name}: not released, refused, or its sha256 / staged set / manifest differs from 4f's release list")
    pins["routing"] = {"file": Path(a.routing).name, "sha256": hashlib.sha256(rb).hexdigest()}
    pins["attest_pointer"] = {"file": Path(a.attest).name, "sha256": hashlib.sha256(ab).hexdigest()}
    pins["release"] = {"file": Path(a.release).name, "sha256": a.release_sha256}
    arms = {x["meta_arm"]: x for x in rec["arms"]}
    tok = {m: m.split("@")[0] for m in arms}
    folder = {m: f"{rec['staged_set']}__{t}" for m, t in tok.items()}
    gate = {}
    why = {fo: R.refuse_folder(fo, P) for fo in folder.values()}
    gate["names_and_try_rule"] = {"refused": {f: w for f, w in why.items() if w}, "pass": not any(why.values()) and ptr["batch"] == rec["batch"]}
    att = show("attestation", ptr["commit"], ptr["path"])
    per = att.get("attestations") or {}
    gate["attestation"] = {"pass": ptr.get("failing") == {} and att.get("failing", {}) == {} and all(per.get(x["staged_sha"]) and all(truthy(c["pass"]) for c in per[x["staged_sha"]])
                                                                                                  for x in arms.values())}
    tree = git("ls-tree", "-r", "--name-only", ptr["commit"], "--", f"{CHAIN}/results").decode().split()
    st = None
    for p in sorted(x for x in tree if f"/chain_stage_{rec['target']}_20" in x):
        d = json.loads(git("show", f"{ptr['commit']}:{p}"))
        sets = [k for k in d if k != "git"]
        if len(sets) == 1 and d[sets[0]]["out"] == rec["staged_set"]:
            if st is not None:
                raise SystemExit("two stage records name this staged set")
            st, pins["stage_record"] = d[sets[0]], {"commit": ptr["commit"], "path": p, "sha256": hashlib.sha256(git("show", f"{ptr['commit']}:{p}")).hexdigest()}
    if st is None:
        raise SystemExit(f"no stage record for {rec['staged_set']} at the attestation commit")
    excluded = st["c_gate"]["excluded"]
    gate["staging"] = {"pass": st["assembled"]["staged_inputs_sha256"] == {f"{m}.jsonl": x["staged_sha"] for m, x in arms.items()}
                       and st["assembled"]["epochs"] == {m: x["epochs"] for m, x in arms.items()} and set(excluded) == set(rec["staging_exclusions"])
                       and st["c_gate"]["manifest_keys"] == st["c_gate"]["staged"] + len(excluded)}
    ex_ok = []
    for i, p in enumerate(sorted(x for x in tree if f"/extract_chain_{rec['extraction_root']}_a" in x)):
        dv = show(f"extraction_{i + 1}", ptr["commit"], p)
        ex_ok.append(dv["config"]["run"] == rec["extraction_root"] and dv["config"]["inputs"] == rec["staged_set"] and dv["volume"] == rec["volume"]
                     and all(dv["config"]["inputs_sha256"].get(f"{m}.jsonl") == x["staged_sha"] for m, x in arms.items()) and dv["config"]["model"] == "Qwen/Qwen3.5-27B"
                     and len(dv["shards"]) == dv["config"]["shards"] and all(s.get("gpu_name") == "NVIDIA H200" for s in dv["shards"]))
    gate["extraction_h200"] = {"attempts": len(ex_ok), "pass": bool(ex_ok) and all(ex_ok)}
    rp = show("repro", *at(rec["repro_gate"]))
    gate["repro_c"] = {"pass": rp["run"] == rec["extraction_root"] and all(rp["staged_sha"].get(f"{m}.jsonl") == x["staged_sha"] for m, x in arms.items())
                       and not rp["failing_shards"] and rp["passing"] == rp["epochs"] and rp["coverage"]["every_shard_one_of_each"] is True and rp["min_cosine"] >= rp["bar"]["cosine"]}
    fc = {m: show(f"format_{tok[m]}", *at(x["format_check"])) if x.get("format_check") else None for m, x in arms.items()}
    gate["format_checks"] = {"pass": all(f is not None and truthy(f["passes"]) and f["staged_sha"] == arms[m]["staged_sha"] and f["arm"] == m for m, f in fc.items())}
    man = show("manifest", *at(rec["manifest"]))
    msha = re.search(r"\(sha256 ([0-9a-f]+)\.*\)$", hit[0]["manifest"])  # a release may name the manifest's sha256 prefix: it must match
    if msha and not pins["manifest"]["sha256"].startswith(msha.group(1)):
        raise SystemExit("the manifest's sha256 differs from the release list's")
    files_all = [f for f in man["files"] if f["arm_dir"] in arms]  # every row, donors included, for the sample_error field rule
    files = [f for f in files_all if f["attempt_kind"] in ("first", "redraw")]  # counted attempts; "reference" rows are the sims' donors
    if any(f["attempt_kind"] not in ("first", "redraw", "reference") for f in files_all):
        raise SystemExit("a manifest row of an unknown attempt kind")
    keys = {f"{f['launch']}/{f['rel']}#e0": f for f in files}
    per_arm = {m: [f for f in files if f["arm_dir"] == m] for m in arms}
    staged = {k: f for k, f in keys.items() if k not in excluded}
    gate["manifest"] = {"files": {m: len(v) for m, v in per_arm.items()}, "pass": len(keys) == len(files) == st["c_gate"]["manifest_keys"]
                        and all(len({f["task"] for f in v}) == len(v) for v in per_arm.values()) and set(excluded) <= set(keys)
                        and all(sum(1 for f in per_arm[m] if f"{f['launch']}/{f['rel']}#e0" not in excluded) == arms[m]["epochs"] for m in arms)}
    gate["sample_error_null"] = {"pass": R.sample_error_ok(files_all, staged)}
    epochs, refused, other = [], [], []
    for k, f in sorted(staged.items()):
        m = f["arm_dir"]
        w = R.world_of(folder[m], arms[m]["world"][0])
        main, fold, reason = R.admit_cell(w, f["task"], P)
        if reason:
            (refused if (f["task"].startswith("t987_") or f"{w}|{f['task']}" in R.EXCLUDED_KEYS) else other).append({"key": k, "world": w, "cell": f["task"], "reason": reason})
            continue
        epochs.append({"key": k, "cell": f["task"], "world": w, "d9_world": arms[m]["world"][0], "stratum": R.STRATUM[R.kind_of(folder[m])], "run": rec["extraction_root"],
                       "arm": m, "staged": f"{rec['staged_set']}/inputs/{m}.jsonl", "staged_sha256": arms[m]["staged_sha"], "folder": folder[m],
                       "launch_task": f.get("launch_task")})
    gate["cells"] = {"refused_by_name": len(refused), "other_refusals": other[:5], "pass": not other and bool(epochs)}
    if not all(g["pass"] for g in gate.values()):
        raise SystemExit(f"adapter gates failed: {json.dumps({k: g for k, g in gate.items() if not g['pass']})}")
    exc = [{"key": k, "cell": keys[k]["task"], "arm": keys[k]["arm_dir"], "folder": folder[keys[k]["arm_dir"]], "reason": excluded[k]} for k in sorted(excluded)]
    msg = {"d9_arm": rec["target"], "batch": rec["batch"], "volume": rec["volume"], "refused": refused,
           "provenance": [{"kind": "sp493", "run": rec["extraction_root"], "staged_set": rec["staged_set"], "excluded_by_staging": exc, "rows": len(epochs)}], "epochs": epochs}
    mb = json.dumps(msg, indent=1, sort_keys=True).encode()
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    mp = od / f"msg_{rec['staged_set']}.json"
    mp.open("xb").write(mb)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"label": "adapter record: d9's chain routing record -> message for one S_paper493 staged set (metadata only); SEALED", "stamp (PT)": stamp,
           "git": git("rev-parse", "HEAD").decode().strip(), "target": rec["target"], "folders": folder, "message_path": str(mp.relative_to(REPO)) if mp.is_relative_to(REPO) else str(mp),
           "message_sha256": hashlib.sha256(mb).hexdigest(), "n_epochs": len(epochs), "refused_by_name": refused, "excluded_by_staging": exc, "pins": pins, "gate": gate}
    rp2 = od / f"adapter_{rec['staged_set']}_{stamp}.json"
    rp2.open("x").write(json.dumps(out, indent=1))
    print(rp2.name, out["message_sha256"][:12], len(epochs), len(refused), len(exc), {k: g["pass"] for k, g in gate.items()})


if __name__ == "__main__":
    main()
