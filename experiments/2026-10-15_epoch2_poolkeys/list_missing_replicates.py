"""Per pool of a committed epoch-2 pool-key artifact: epoch-1 members, admitted epoch-2 replicates, and the members WITHOUT an epoch-2 replicate
(by key, with the reason the artifact records). "A -> B" in the artifact's summary means A epoch-1 members and B admitted epoch-2 replicates; the
member list itself does not shrink. Reads the artifact only (keys and reasons; no score values).
  python3 experiments/2026-10-15_epoch2_poolkeys/list_missing_replicates.py <commit> <artifact path>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout


def main():
    commit, path = sys.argv[1], sys.argv[2]
    b = git("show", f"{commit}:{path}")
    d = json.loads(b)
    out = {"artifact": f"{path} @ {commit} (sha256 {hashlib.sha256(b).hexdigest()})",
           "reading": "'A -> B' = A epoch-1 members, B admitted epoch-2 replicates; members_without_epoch2 = A - B, listed by key", "pools": {}}
    for o, od in d["origins"].items():
        for cfg, v in od.items():
            ps = [(p["pool"], p) for p in v["pools"]] if isinstance(v, dict) and "pools" in v else (
                list(v["families"].items()) if isinstance(v, dict) and "families" in v else [])
            for name, p in ps:
                mem, adm = set(p["epoch1_member_keys"]), {a["key"] for a in p["admitted_epoch2"]}
                arms = set(p["epoch2_arms_admitted"])
                by_rule = sorted(k for k in mem if k.split("|", 1)[0] not in arms)  # e.g. S_old honest_c: no epoch 2 by the spec
                miss = sorted(mem - adm - set(by_rule))
                assert adm <= mem and miss == sorted(m["key"] for m in p["members_without_epoch2"]), (o, cfg, name)
                assert (len(mem), len(adm), len(miss)) == (p["counts"]["epoch1_members"], p["counts"]["epoch2_admitted"], p["counts"]["members_without_epoch2"])
                out["pools"][f"{o} | {cfg} | {name}"] = {"epoch1_members": len(mem), "epoch2_admitted": len(adm), "epoch2_arms": sorted(arms),
                                                         "members_of_arms_without_epoch2_by_rule": len(by_rule),
                                                         "members_without_epoch2": [{"key": m["key"], "reason": m["reason"]} for m in p["members_without_epoch2"]]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out["meta"] = {"git": git("rev-parse", "HEAD").decode().strip(), "stamp_pt": stamp}
    dest = HERE / "results" / f"missing_replicates_{stamp}.json"
    dest.open("x").write(json.dumps(out, indent=1) + "\n")
    for k, v in out["pools"].items():
        print(f"{k}: {v['epoch1_members']} -> {v['epoch2_admitted']} (no epoch 2 by rule: {v['members_of_arms_without_epoch2_by_rule']}); "
              f"without epoch 2: {[m['key'] for m in v['members_without_epoch2']]}")
    print(dest)


if __name__ == "__main__":
    main()
