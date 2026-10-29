"""d9's CUT fork: chain-format routing records for the 6 S_old (b) cutfull roots (09 @ 2d61af835: batch `sold`, one record per ROOT;
09's correction via d9 10-16 ~02:55: the cutfull routing dir's names carry _r<k>, mirroring the attestation revision), from committed records only. Each record points at its passing S_old
attestation (the highest revision) by path, commit and sha256, and copies the staged set, root, folders and per-arm staged shas and
keys from it and its extraction attestation (asserted equal). Gates (09): per-folder epochs sum to 2,266; each extraction root is in
exactly ONE active routing file. Write-once; nothing is written if any gate fails.

  python3 experiments/2026-10-14_q3_overnight_chain/cutfull/make_sold_routing.py
"""
import glob
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
REPO = CHAIN.parents[1]
REL = "experiments/2026-10-14_q3_overnight_chain"
OUT_R, OUT_A = CHAIN / "results/cutfull/routing", CHAIN / "results/cutfull/attest"
BATCH = "sold"  # 09 @ 2d61af835: the one constant
VOLUME = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"
TARGETS = ["q4f_main", "q4f_run2", "q4f_c", "rh1e4_stage", "simh_q4f_v1_full_b1", "simh_q4f_v1_full_b2"]
TOTAL = 2266


def git(*a, text=True):
    return subprocess.run(["git", *a], capture_output=True, text=text, check=True, cwd=REPO).stdout


def committed(path: Path) -> bytes:
    rel = str(path.resolve().relative_to(REPO))
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, cwd=REPO).stdout
    if not blob or blob != path.read_bytes():
        raise SystemExit(f"{rel}: not committed as read")
    return blob


def at(path: Path) -> str:
    rel = str(path.resolve().relative_to(REPO))
    return f"{rel} @ {git('log', '-1', '--format=%H', '--', rel).strip()[:12]}"


def token(arm):
    return arm.split("@")[0]


def main():
    writes, total = [], 0
    for t in TARGETS:
        have = {}
        for f in OUT_A.glob(f"attestations_sold__{t}_cutfull*.json"):
            m = re.fullmatch(rf"attestations_sold__{re.escape(t)}_cutfull(?:_r(\d+))?\.json", f.name)
            if m:
                have[int(m.group(1) or 1)] = f
        k = max(have)
        af = have[k]
        raw = committed(af)
        att = json.loads(raw)
        if att.get("failing") != {} or att.get("revision") != k or att.get("label") != "(b) descriptive" or not att.get("format_check"):
            raise SystemExit(f"{t}: attestation r{k} does not pass or lacks its format check; nothing written")
        cf = att["cutfull"]
        ext = [f.split(" @ ") for f in att["extraction_attestation"]] if isinstance(att["extraction_attestation"], list) else [att["extraction_attestation"].split(" @ ")]
        rows = [r for p, c in ext for r in json.loads(git("show", f"{c}:{p}"))["roots"] if r.get("root") == cf["extraction_root"] and r.get("uncut_set") == cf["staged_set"]]
        if len(rows) != 1 or rows[0].get("failing") != {}:
            raise SystemExit(f"{t}: {len(rows)} passing extraction-attestation rows for {cf['extraction_root']}; nothing written")
        x = rows[0]["arms"]
        if att.get("original_target") != t or cf.get("volume") != VOLUME or set(att["format_check"]) != set(x) or set(att["original_folders"]) != set(x) \
                or set(att["units"]) != set(x) or set(att["template_class"]) != set(x):
            raise SystemExit(f"{t}: target, volume or per-arm coverage (format_check, folders, units, template_class) disagree; nothing written")
        arms = []
        for tk, v in sorted(x.items()):
            m = v["meta_arm"]
            if token(m) != tk or tk not in cf["folders"] or tk not in att["original_folders"] or len(v["keys"]) != v["epochs"] or len(set(v["keys"])) != v["epochs"] \
                    or set(v["keys"]) != set(att["template_class"].get(tk, {})) or att["units"][tk]["epochs"] != v["epochs"]:
                raise SystemExit(f"{t}/{tk}: the attestation and its extraction record disagree; nothing written")
            fp, fc = att["format_check"][tk].split(" @ ")
            fr = json.loads(git("show", f"{fc}:{fp}"))  # the cited format record, at its commit: passing, same staged sha, every epoch
            if fr.get("passes") is not True or fr.get("failing_keys") != [] or fr.get("staged_sha") != [v["staged_sha256"]] or fr.get("epochs_checked") != v["epochs"] \
                    or sorted(r_["key"] for r_ in fr.get("rows", [])) != sorted(v["keys"]):
                raise SystemExit(f"{t}/{tk}: the cited format record does not pass or does not bind to this arm's staged sha and keys; nothing written")
            arms.append({"meta_arm": m, "staged_inputs_file": f"{cf['staged_set']}/inputs/{m}.jsonl", "staged_sha": v["staged_sha256"], "epochs": v["epochs"],
                         "keys": sorted(v["keys"]), "format_check": att["format_check"].get(tk)})
        if set(cf["folders"]) != {a_["meta_arm"].split("@")[0] for a_ in arms}:
            raise SystemExit(f"{t}: folders and arms differ; nothing written")
        total += sum(a_["epochs"] for a_ in arms)
        rec = {"batch": BATCH, "target": f"{t}_cutfull", "label": "(b) descriptive", "staged_set": cf["staged_set"], "extraction_root": cf["extraction_root"],
               "volume": cf["volume"], "cut": "full", "folders": cf["folders"], "original_folders": att["original_folders"],
               "attestation": {"path": at(af), "sha256": hashlib.sha256(raw).hexdigest(), "revision": k},
               "authority": att["authority"]["provenance"], "dropped_arms": att.get("dropped_arms", []), "dropped_note": att.get("dropped_note"),
               "arms": arms, "repro_gate": att["repro_gate"], "routing_spec": f"{REL}/cutfull/cutfull_routing_v2.json",
               "naming": f"{REL}/cutfull/cutfull_naming_v1_addendum_2.json", "names_ruling": "09 @ 2d61af835 (batch sold; one record per root), corrected via d9 ~02:55 (names carry the attestation's _r<k>)"}
        rec["revision"] = k  # mirrors the attestation revision (09: the cutfull routing dir carries _r<k>)
        writes.append((OUT_R / f"{BATCH}__{t}_cutfull{'' if k == 1 else f'_r{k}'}.json", json.dumps(rec, indent=1), cf["extraction_root"]))
    if total != TOTAL:
        raise SystemExit(f"per-folder epochs sum to {total}, not {TOTAL}; nothing written")
    # each extraction root in exactly ONE active routing file: no other routing record (any revision) names these roots
    others = {}
    for f in OUT_R.glob("*.json"):
        if f.name.endswith("_attest.json"):
            continue
        r_ = json.loads(f.read_bytes())
        if isinstance(r_, dict) and r_.get("extraction_root"):
            others.setdefault(r_["extraction_root"], []).append(f.name)
    clash = {root: others[root] for _, _, root in writes if root in others}
    if clash or len({root for _, _, root in writes}) != len(writes):
        raise SystemExit(f"an extraction root would be in more than one routing file: {clash}; nothing written")
    if any(d.exists() for d, _, _ in writes):
        raise SystemExit("output exists; nothing written")
    tmp = []  # publish the complete set or nothing: write temporaries, then rename all
    try:
        for d, body, _ in writes:
            t_ = d.with_name(f".{d.name}.tmp")
            t_.open("x").write(body)
            tmp.append((t_, d))
    except Exception:
        for t_, _ in tmp:
            t_.unlink()
        raise
    import os

    for t_, d in tmp:  # os.link fails if the destination exists (write-once); readers read the ONE git commit holding all six (the publication)
        os.link(t_, d)
        t_.unlink()
    print(len(writes), "written; epochs", total)


if __name__ == "__main__":
    main()
