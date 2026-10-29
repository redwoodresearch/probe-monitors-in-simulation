"""Fix-full F9 (PLAN.md Amendment 4): the full + F9 mapping for d9, one JSON per reader. full + F9 = fix-full's `full` arm
with the F9 run's `fullF9` runs replacing it by key (36 attack, 45 honest; every other run is byte-identical to fix-full's
`full` run, by message sha256 in the subset record). Before writing, every override key must exist as a meta in BOTH the
base and the override arms on the Volume (listed with `modal volume ls`), and the base arms must hold 939 and 437 runs.
Local, $0 (Volume listings only).

  uv run --no-project --python 3.12 python experiments/2026-10-13_q2_fixfull/make_f9_mapping.py --reader qwen3|seed
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
SUBSET = HERE / "results" / "stage_f9r_2026-10-13_150031.json"
READERS = {"qwen3": {"volume": "probe-shift_aswin-surya_sub-la-activations", "base run": "2026-10-13_fixfull_qwen3-32b_l48-63_v2",
                     "override run": "2026-10-13_fixfull_f9_qwen3-32b_l48-63_v1", "layers": [48, 63]},
           "seed": {"volume": "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12", "base run": "2026-10-13_fixfull_seed-oss-36b_v1",
                    "override run": "2026-10-13_fixfull_f9_seed-oss-36b_v1", "layers": [16, 24, 32, 40, 48, 56, 63]}}
ARMS = {"attack": ("sim-attack-fx-full", "sim-attack-fx-fullF9", 939), "honest": ("sim-honest-fx-full", "sim-honest-fx-fullF9", 437)}


def filename(key: str) -> str:  # the extractors' file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def stems(volume, run, arm):
    out = subprocess.run(["uv", "run", "--no-project", "--with", "modal==1.6.1", "modal", "volume", "ls", volume, f"{run}/meta/{arm}"],
                         capture_output=True, text=True, check=True, timeout=600).stdout.split()
    return {Path(x).name.removesuffix(".json") for x in out if x.endswith(".json")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reader", required=True, choices=sorted(READERS)); a = ap.parse_args()
    R = READERS[a.reader]
    sub = json.loads(SUBSET.read_text())
    out = {"reader": a.reader, "volume": R["volume"], "layers": R["layers"], "made from": {SUBSET.name: hashlib.sha256(SUBSET.read_bytes()).hexdigest()},
           "rule": "full + F9 = the base arm's runs, with the override arm's run replacing the base's for each override key", "arms": {}}
    for grp, (base_arm, ov_arm, n) in ARMS.items():
        keys = sorted(sub["runs by arm"][ov_arm])
        base, ov = stems(R["volume"], R["base run"], base_arm), stems(R["volume"], R["override run"], ov_arm)
        if len(base) != n:
            raise SystemExit(f"{a.reader} {base_arm}: {len(base)} metas, not {n}")
        if {filename(k) for k in keys} != ov:
            raise SystemExit(f"{a.reader} {ov_arm}: the override metas differ from the subset's {len(keys)} keys")
        if not {filename(k) for k in keys} <= base:
            raise SystemExit(f"{a.reader}: an override key is missing from the base arm {base_arm}")
        out["arms"][f"full + F9, {grp}"] = {"base run": R["base run"], "base arm": base_arm, "override run": R["override run"], "override arm": ov_arm,
                                            "override keys": keys, "runs": n, "overrides": len(keys)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"f9_mapping_{a.reader}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(dest, hashlib.sha256(dest.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
