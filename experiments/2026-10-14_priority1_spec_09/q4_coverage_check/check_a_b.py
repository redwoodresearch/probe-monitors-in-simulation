"""Check (a) key sets and (b) per-epoch bit-exactness of the Q4 v1-vs-v2 compare files. Keys/counts only."""
import json
import sys
from collections import Counter
from pathlib import Path

R = Path(sys.argv[1])
ARMS = {"main": ["attack", "honest_a", "honest_b"], "run2": ["attack"]}

for root in ("main", "run2"):
    man = json.loads((R / f"manifest_su311_q4_{root}.json").read_text())
    stage = json.loads((R / f"stage_{root}.json").read_text())[root]
    cmp_ = json.loads((R / f"cmp_{root}.json").read_text())
    rep = {v: json.loads((R / f"repro_{v}_{root}.json").read_text()) for v in ("v1", "v2")}
    excluded = set(stage["c_gate"]["excluded"])
    want = {}
    for f in man["files"]:
        if f["arm_dir"] in ARMS[root]:
            want.setdefault(f["arm_dir"], []).append(f"{f['launch']}/{f['rel']}#e0")
    staged = {a: sorted(set(ks) - excluded) for a, ks in want.items()}
    dup_man = {a: [k for k, n in Counter(ks).items() if n > 1] for a, ks in want.items()}
    # repro-c expected: the sources named in the repro gate files, each copied ~ref000..~ref{N-1}, N = shards (1 copy per shard)
    shards = rep["v2"]["coverage"]["shards_expected"]
    srcs = sorted({e["c_key"] for e in rep["v2"]["per_epoch"]})
    srcs_v1 = sorted({e["c_key"] for e in rep["v1"]["per_epoch"]})
    exp_repro = sorted(f"{s}~ref{j:03d}" for s in srcs for j in range(shards))
    staged["repro-c"] = exp_repro
    rep_keys = {v: sorted(e["key"] for e in rep[v]["per_epoch"]) for v in rep}
    print(f"===== {root}: manifest files {len(man['files'])}, excluded {len(excluded)}, excluded-in-manifest {len(excluded & set(k for ks in want.values() for k in ks))}")
    print(f"  manifest dups per arm: { {a: len(d) for a, d in dup_man.items()} }")
    print(f"  repro sources v2 {len(srcs)} == v1 {srcs == srcs_v1}; shards {shards}; repro gate keys v1==expected {rep_keys['v1'] == exp_repro}, v2==expected {rep_keys['v2'] == exp_repro}")
    pe = cmp_["per_epoch"]
    total = 0
    for arm in sorted(set(staged) | set(pe)):
        got = [k for k in pe.get(arm, {}) if k != "__v2_only__"]
        exp = staged.get(arm, [])
        missing, extra = sorted(set(exp) - set(got)), sorted(set(got) - set(exp))
        dups = len(got) - len(set(got))
        v2only = pe.get(arm, {}).get("__v2_only__", [])
        ents = [pe[arm][k] for k in got]
        bad = [k for k in got if not (pe[arm][k].get("bit_exact") is True and pe[arm][k].get("max_abs_diff") == 0.0
                                      and pe[arm][k].get("layers") == 64 and pe[arm][k].get("problems") == [])]
        total += len(got)
        print(f"  {arm}: staged {len(exp)} compared {len(got)} missing {len(missing)} extra {len(extra)} dup {dups} v2_only {len(v2only)} | "
              f"bit_exact {sum(e.get('bit_exact') is True for e in ents)} max|d|=0 {sum(e.get('max_abs_diff') == 0.0 for e in ents)} "
              f"layers==64 {sum(e.get('layers') == 64 for e in ents)} problems==[] {sum(e.get('problems') == [] for e in ents)} "
              f"max(max_abs_diff) {max((e.get('max_abs_diff', float('nan')) for e in ents), default=None)} rows_sum {sum(e.get('rows', 0) for e in ents)} "
              f"rows==0 {sum(e.get('rows', 0) == 0 for e in ents)} fail {len(bad)}")
        for k in missing[:5] + extra[:5] + bad[:5]:
            print("     MISMATCH", k)
    print(f"  total compared {total}; summary {json.dumps(cmp_['summary'])}; all_identical {cmp_['all_identical']}")
