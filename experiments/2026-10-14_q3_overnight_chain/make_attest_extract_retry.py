"""A new attestation version after an extraction retry of the SAME staged set (09's conditions via d9 10-15 ~07:45; ilnm_s5_sim2's
_extract_v1_try2): the target's longest_epoch_h200_gate is recomputed from the retry root alone, as chain.step_attest computes it,
and must also show that every chunk the root holds was written by a recorded H200 shard (chain.writer_coverage), never by a later
attempt that wrote nothing. The record must name the same staged set and staged sha. Every other entry is copied unchanged except
that its covers names the authoritative extraction root. Writes attestations_<pool>_<batch>_v<N>.json (open 'x'; earlier versions
kept); exits 2 without writing if any entry fails. Keys and hashes only.

  python make_attest_extract_retry.py <chain state dir> <batch> <target> <retry root> <previous attestation file> <new version tag, e.g. _v3>
"""
import copy
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import chain as C


def main():
    st_dir, bid, t, root, prev_name, tag = sys.argv[1:7]
    b = json.loads((Path(st_dir) / "state.json").read_text())["batches"][bid]
    hyg = C.HERE / "results" / "hygiene"
    prev_p = hyg / prev_name
    prev = json.loads(prev_p.read_text())
    tv = C.targets_for(b)[t]
    arm = tv["stage_arms"][0]
    ssha = b["stage"][t]["staged_inputs_sha256"][f"{arm}.jsonl"]
    recs = sorted((p for p in (C.HERE / "results").glob(f"extract_chain_{root}_a*_*.json") if re.match(rf"extract_chain_{re.escape(root)}_a\d+_20", p.name)
                   and C.valid_record(p) is not None), key=lambda p: (int(re.match(r"_a(\d+)_", p.name[len(f"extract_chain_{root}"):]).group(1)), p.name))
    assert recs, f"no committed record of {root}"
    for p in recs:  # the same staged set and staged sha (09)
        cfg = json.loads(p.read_text())["config"]
        assert cfg["inputs"] == b["stage"][t]["out"] and cfg["inputs_sha256"][f"{arm}.jsonl"] == ssha, f"{p.name}: another staged set or sha"
    shards = [s for p in recs for s in json.loads(p.read_text())["shards"]]
    last_ok = all("error" not in s for s in json.loads(recs[-1].read_text())["shards"])
    h200 = bool(shards) and all("error" in s or (s.get("gpu_name") == "NVIDIA H200" and s.get("gpu_count") == 1) for s in shards)
    wcov = C.writer_coverage(root, recs)
    cmps = sorted((C.HERE / "results").glob(f"compare_repro_{root}_repro-c_*.json"))
    assert cmps, f"no repro-c compare of {root}"
    cmp_p = cmps[-1]
    cmp = json.loads(cmp_p.read_text())
    cmp_ok = not cmp["failing_shards"] and all(cmp["shards"].values()) and cmp["coverage"]["every_shard_one_of_each"] and \
        cmp["coverage"]["shards_seen"] == cmp["coverage"]["shards_expected"] and cmp["passing"] == cmp["epochs"]
    stats_p = C.HERE / "results" / b["stats"][t]  # the staged set's token counts: unchanged, same staged set
    sx = {x["arm"]: x for x in json.loads(stats_p.read_text())["sets"]}.get(arm, {})
    long_ok = sx.get("staged_sha") == ssha and sx.get("n_epochs_over_32768") == 0
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=C.HERE).stdout.strip()
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new = copy.deepcopy(prev)
    old_root = None
    for e in new["attestations"][ssha]:
        old_root = old_root or e["covers"].get("extraction_root")
        e["covers"]["extraction_root"] = root
        if e["check"] != "longest_epoch_h200_gate":
            continue
        ev = recs[-1]
        has = ssha in ev.read_text()
        e["pass"] = h200 and last_ok and cmp_ok and long_ok and wcov["pass"] and has
        e["evidence"], e["evidence_extra"] = C.evidence(ev), [C.evidence(x) for x in (cmp_p, stats_p, *recs[:-1])]
        e["checker"], e["computed_utc"] = f"{C.REL}/make_attest_extract_retry.py@{head}", utc
        e["detail"] = {"every_shard_h200": h200, "last_attempt_no_errors": last_ok, "repro_c_compare_pass": cmp_ok, "max_staged_tokens": sx.get("max_staged_tokens"),
                       "n_epochs_over_32768": sx.get("n_epochs_over_32768"), "chunk_writers_recorded_h200": wcov["pass"], "volume_chunks": wcov["chunks"],
                       "unrecorded_chunks": wcov["unrecorded"][:20], "extraction_retry_of": old_root, "staged_sha_equal": True,
                       "evidence_contains_staged_sha": has, "reference": "repro-c: batch C's 32,768-token longest staged epoch + one seeded, per shard"}
    failing = {s: [e["check"] for e in es if not e["pass"]] for s, es in new["attestations"].items()}
    failing = {s: c for s, c in failing.items() if c}
    if failing:
        print(f"HARD STOP: {failing}", file=sys.stderr)
        sys.exit(2)
    new |= {"supersedes": prev_name, "failing": failing, "extraction_retry": {"target": t, "from_root": old_root, "to_root": root, "staged_sha": ssha}}
    out = hyg / f"attestations_{b['pool']}_{b['id']}{tag}.json"
    with out.open("x") as f:
        f.write(json.dumps(new, indent=1) + "\n")
    print(json.dumps({"out": str(out.relative_to(C.HERE)), "failing": failing, "h200": h200, "last_ok": last_ok, "cmp_ok": cmp_ok, "long_ok": long_ok,
                      "writer_coverage": wcov}))


if __name__ == "__main__":
    main()
