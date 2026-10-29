"""Fix-full (PLAN.md Amendment 6), $0, local: d9's as-is simulated-honest Seed dump (2026-10-13_g3_simhonest_seed-oss-36b_v1,
arm sim-honest) against our fx honest arms (2026-10-13_fixfull_seed-oss-36b_v1, sim-honest-fx-full and -NL): the same 437
run keys, and per run the same readout positions and keep flags (the C2 set) and row counts. Differences are counted, split
by whether the fix changed the run (stage record), and reported; the paired analysis needs zero differences outside runs the
fix changed. Inputs are local copies of the three meta folders (small JSONs, `modal volume get`) and the fx stage record.

  python3 experiments/2026-10-13_q2_fixfull/check_simh_alignment.py --simh <dir> --full <dir> --nl <dir> --stage-record <json>
"""
import argparse
import collections
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent


def load(d):
    out = {}
    for f in sorted(Path(d).rglob("*.json")):
        m = json.loads(f.read_text())
        if m["key"] in out:
            raise SystemExit(f"{d}: a key twice")
        out[m["key"]] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    for x in ("--simh", "--full", "--nl", "--stage-record"):
        ap.add_argument(x, required=True)
    a = ap.parse_args()
    A, st = load(a.simh), json.loads(Path(a.stage_record).read_text())
    rec = {"as-is dump": "2026-10-13_g3_simhonest_seed-oss-36b_v1 / sim-honest", "fx dump": "2026-10-13_fixfull_seed-oss-36b_v1",
           "stage record sha256": hashlib.sha256(Path(a.stage_record).read_bytes()).hexdigest(), "as-is runs": len(A),
           "as-is positions by (kind, kept)": {f"{k} {v}": n for (k, v), n in sorted(collections.Counter((p["kind"], p["kept"]) for m in A.values() for p in m["positions"]).items())},
           "arms": {}}
    for lab, d in (("sim-honest-fx-full", a.full), ("sim-honest-fx-NL", a.nl)):
        B = load(d)
        changed = {r["key"]: r["first_changed_message"] is not None for r in st["runs"] if r["arm"] == lab}
        r = {"runs": len(B), "same keys": set(A) == set(B), "runs the fix changed": sum(changed.values()), "position lists differ": 0,
             "keep sets differ": 0, "keep sets differ in runs the fix left unchanged": 0, "row counts differ": 0, "differing positions by kind": collections.Counter()}
        for k in sorted(set(A) & set(B)):
            pos = lambda m: [(p["kind"], p["step"], p["action"]) for p in m["positions"]]  # noqa: E731
            kept = lambda m: {(p["kind"], p["step"], p["action"]) for p in m["positions"] if p["kept"]}  # noqa: E731
            r["position lists differ"] += pos(A[k]) != pos(B[k])
            r["row counts differ"] += A[k]["n_rows"] != B[k]["n_rows"]
            if kept(A[k]) != kept(B[k]):
                r["keep sets differ"] += 1
                r["keep sets differ in runs the fix left unchanged"] += not changed[k]
                for x in kept(A[k]) ^ kept(B[k]):
                    r["differing positions by kind"][f"{x[0]}, {'as-is only' if x in kept(A[k]) else 'fx only'}"] += 1
        r["differing positions by kind"] = dict(r["differing positions by kind"])
        r["aligned"] = r["same keys"] and not r["position lists differ"] and not r["keep sets differ in runs the fix left unchanged"]
        rec["arms"][lab] = r
    rec["aligned"] = all(r["aligned"] for r in rec["arms"].values())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"simh_alignment_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(dest)
    if not rec["aligned"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
