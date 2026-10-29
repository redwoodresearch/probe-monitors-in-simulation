"""4f's per-root gates, BY KEY, before a pass-2 message is built (d9's routing files are read-only inputs):
manifest identity at the records commit; stage record (staged shas, exact exclusions); device records (all H200; attempts and
done_before reported, 21g if > 1 / > 0); the writer attestation (every check passes, covers this root's staged shas); repro-c
bit-exact; format checks pass. Prints one JSON verdict per root; never builds or launches anything.

  python verify_roots.py <routing.json>:<records commit>:<attest pointer.json> [...]
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = "experiments/2026-10-14_q3_overnight_chain/results"


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout


def show(c, p):
    return json.loads(git("show", f"{c}:{p}"))


def check(routing: str, rec: str, attest: str) -> dict:
    r, at_ptr = json.loads(Path(routing).read_text()), json.loads(Path(attest).read_text())
    root, target, staged = r["extraction_root"], r["target"], r["staged_set"]
    files = git("ls-tree", "-r", "--name-only", rec, "--", RES).split()
    out, fails = {"root": root, "target": target, "records": rec}, []
    for nm_, d_ in (("routing", r), ("attestation pointer", at_ptr)):  # d9's pointer convention (4f, 10-15): never read a superseded file as current
        if d_.get("superseded") is True or d_.get("authoritative") is False or d_.get("revisions"):
            fails.append(f"{nm_} file is superseded / non-authoritative or has revisions: read revisions[-1] or the _v2 pointer instead")
    mpath, mc = [x.strip() for x in r["manifest"].rsplit("@", 1)]
    if subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", mc, rec, "--", mpath]).returncode:
        fails.append("manifest differs between its commit and the records commit")
    st = [f for f in files if re.fullmatch(rf"{RES}/chain_stage_{re.escape(target)}_\d{{4}}-\d\d-\d\d_\d{{6}}\.json", f)]
    if len(st) > 1:  # a re-stage leaves the non-authoritative record beside it: keep only records that stage THIS staged set
        st = [f for f in st if any(isinstance(v, dict) and v.get("out") == staged for v in show(rec, f).values())]
    if len(st) != 1:
        fails.append(f"stage records for {target}: {st}")
    else:
        sr = show(rec, st[0])
        sets = {k: v for k, v in sr.items() if isinstance(v, dict) and v.get("out") == staged}
        if len(sets) != 1:
            fails.append(f"stage record: {len(sets)} sets with out == {staged}")
        else:
            (sk, sv), = sets.items()
            out |= {"stage_record": st[0], "set_key": sk}
            ss = sv["assembled"]["staged_inputs_sha256"]
            for arm in r["arms"]:
                f_ = arm["staged_inputs_file"].split("/")[-1]
                if ss.get(f_) != arm["staged_sha"]:
                    fails.append(f"staged sha {f_}: stage {ss.get(f_)} vs routing {arm['staged_sha']}")
            if (sv.get("c_gate") or {}).get("excluded") != r.get("staging_exclusions", {}):
                fails.append("exclusions: stage record c_gate.excluded != routing staging_exclusions")
    dev = sorted(f for f in files if re.fullmatch(rf"{RES}/extract_chain_{re.escape(root)}_a\d+_\d{{4}}-\d\d-\d\d_\d{{6}}\.json", f))
    shards = [s for f in dev for s in show(rec, f).get("shards", [])]
    gpus = sorted({str(s.get("gpu_name")) for s in shards})
    attempts = sorted({re.search(r"_a(\d+)_", f).group(1) for f in dev})
    done_before = max((s.get("done_before") or 0 for s in shards), default=0)
    out |= {"device_records": dev, "gpus": gpus, "shards": len(shards), "attempts": attempts, "max_done_before": done_before,
            "needs_21g": len(attempts) > 1 or done_before > 0}
    if not dev or gpus != ["NVIDIA H200"]:
        fails.append(f"device records: {gpus}")
    att = show(at_ptr["commit"], at_ptr["path"])["attestations"]
    bad = {}
    for arm in r["arms"]:
        chk = att.get(arm["staged_sha"])
        if not chk or any(c.get("pass") is not True for c in chk) or any((c.get("covers") or {}).get("extraction_root") != root for c in chk):
            bad[arm["meta_arm"]] = "missing, failing, or covering another root"
    out["attestation"] = {"commit": at_ptr["commit"], "path": at_ptr["path"], "failing": bad, "checks": sorted({c["check"] for v in att.values() for c in v})}
    if bad or at_ptr.get("failing") != {}:
        fails.append(f"attestation failing: {bad or at_ptr.get('failing')}")
    rp_path, rp_c = [x.strip() for x in r["repro_gate"].rsplit("@", 1)]
    rp = show(rp_c, rp_path)
    out["repro_c"] = f"{rp['bit_exact']}/{rp['epochs']} bit-exact, failing shards {rp['failing_shards']}"
    if rp["run"] != root or rp["bit_exact"] != rp["epochs"] or rp["failing_shards"]:
        fails.append("repro-c not bit-exact for this root")
    for arm in r["arms"]:
        fp, fc = [x.strip() for x in arm["format_check"].rsplit("@", 1)]
        fm = show(fc, fp)
        if fm.get("passes") is not True or fm.get("staged_sha") != arm["staged_sha"]:
            fails.append(f"format check {arm['meta_arm']}")
    out["arms"] = {a_["meta_arm"]: {"world": a_["world"], "epochs": a_["epochs"], "staged_sha": a_["staged_sha"]} for a_ in r["arms"]}
    out["exclusions"] = len(r.get("staging_exclusions", {}))
    out["pass"] = not fails
    out["fails"] = fails
    return out


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        print(json.dumps(check(*spec.split(":")), indent=1))
