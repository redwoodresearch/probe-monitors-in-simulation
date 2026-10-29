"""G7b on route C, the A-split equivalence positive control (09 / 2e 10-20): A run as 2 level parts (1/2, 2/2) against the committed UNSPLIT A
result, by key. Keys, verdicts and counts only (no value is read). Pre-stated (09): by-key identity on every compared key, and n_compared equal to
the unsplit A's (rows 116248, pooled 16776, aggregates 720, boot aggregate 2016). Level-independent keys recur in both parts, so rows and pooled are
compared as the union of (row, origin or combo, key) over the parts; the aggregate and boot-aggregate keys carry their level, so their counts are
summed over the parts. Writes results/split_equivalence_<stamp>.json in every case; exits non-zero on a STOP.
    uv run python experiments/2026-10-15_e2e_sim_pipeline/compare_split_equivalence.py <unsplit results file @ commit:path> <part results file> <part results file>"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent


def keysets(r_):
    rows = {(q["row"], q["origin"], str(x[0])): str(x[-1]) for q in r_.get("rows") or [] for x in q.get("ledger") or []}
    pooled = {(q["row"], q["combo"], str(x[0])): str(x[-1]) for q in r_.get("pooled") or [] for x in q.get("ledger") or []}
    agg = {str(x[0]): str(x[-1]) for x in (r_.get("aggregates") or {}).get("ledger") or []}
    return rows, pooled, agg


def main():
    spec_, parts_ = sys.argv[1], sys.argv[2:]
    c_, p_ = spec_.split(":", 1)
    un_ = json.loads(subprocess.run(["git", "show", f"{c_}:{p_}"], capture_output=True, check=True).stdout)
    ps_ = [json.loads((HERE / "results" / Path(f_).name).read_bytes()) for f_ in parts_]
    ur_, up_, ua_ = keysets(un_)
    pr_, pp_, pa_ = {}, {}, {}
    vio = []
    for r_ in ps_:
        a_, b_, g_ = keysets(r_)
        for d_, src_, nm_ in ((pr_, a_, "rows"), (pp_, b_, "pooled"), (pa_, g_, "aggregates")):
            for k_, v_ in src_.items():
                if k_ in d_ and d_[k_] != v_:
                    vio.append(f"{nm_}: a key with different verdicts across parts: {k_[-1] if isinstance(k_, tuple) else k_}")
                d_[k_] = v_
    for nm_, u_, s_ in (("rows", ur_, pr_), ("pooled", up_, pp_), ("aggregates", ua_, pa_)):
        if set(u_) != set(s_):
            vio.append(f"{nm_}: keys differ (only unsplit {len(set(u_) - set(s_))}, only split {len(set(s_) - set(u_))})")
        if any(v_ != "MATCH" for v_ in list(u_.values()) + list(s_.values())):
            vio.append(f"{nm_}: a verdict other than MATCH")
    sums_ = {"aggregates_n_compared": sum((r_.get("aggregates") or {}).get("n_compared") or 0 for r_ in ps_),
             "boot_aggregate_n_compared": sum((((r_.get("aggregates") or {}).get("value") or {}).get("boot_aggregate") or {}).get("n_compared") or 0 for r_ in ps_)}
    want_ = {"rows": 116248, "pooled": 16776, "aggregates_n_compared": 720, "boot_aggregate_n_compared": 2016}
    got_ = {"rows": len(pr_), "pooled": len(pp_)} | sums_
    unsplit_ = {"rows": len(ur_), "pooled": len(up_), "aggregates_n_compared": (un_.get("aggregates") or {}).get("n_compared"),
                "boot_aggregate_n_compared": (((un_.get("aggregates") or {}).get("value") or {}).get("boot_aggregate") or {}).get("n_compared")}
    if got_ != want_ or unsplit_ != want_:
        vio.append(f"counts: split {got_}, unsplit {unsplit_}, pre-stated {want_}")
    lp_ = sorted(str(r_.get("level_part")) for r_ in ps_)
    if lp_ != ["1/2", "2/2"] or any(r_.get("record_sha256") != un_.get("record_sha256") or r_.get("inputs_sha256") != un_.get("inputs_sha256") for r_ in ps_):
        vio.append(f"parts: not 1/2 and 2/2 of the unsplit record and inputs ({lp_})")
    if not all(r_.get("all_equal") is True for r_ in ps_):
        vio.append("a part is not all_equal")
    out = {"unsplit": spec_, "parts": [Path(f_).name for f_ in parts_], "level_parts": lp_, "counts_split": got_, "counts_unsplit": unsplit_,
           "counts_prestated": want_, "per_part_n_compared": [r_.get("n_compared") for r_ in ps_], "violations": vio,
           "verdict": "PASS" if not vio else "STOP",
           "rule": "09 / 2e 10-20 (pre-stated): by-key identity on every compared key, and n_compared equal to the unsplit A's (116248 / 16776 / 720 / 2016)"}
    dest = HERE / "results" / f"split_equivalence_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
    dest.open("x").write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"verdict": out["verdict"], "counts_split": got_, "violations": vio[:10], "file": str(dest)}, indent=1))
    if vio:
        raise SystemExit("STOP: the split run is not equivalent to the unsplit A by key")


if __name__ == "__main__":
    main()
