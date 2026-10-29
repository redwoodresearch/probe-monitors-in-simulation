"""G7b on route C, B's level split (GATES A22; 2e / 09 10-20): the committed MERGE RECORD of B's level parts, by key. Keys, shas, counts and
verdicts only (no value is read). PASS needs every part to PASS on its own and the parts' level subsets to partition the full level set exactly
(no overlap, no gap), all on one record, one full input set, one code and one checker. Writes results/merge_levelparts_<record stem>_<stamp>.json
in every case, then exits non-zero on a STOP.
    uv run python experiments/2026-10-15_e2e_sim_pipeline/merge_level_parts_routec.py <record sha256> <part results file> [<part results file> ...]"""
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
BOOT_SEED, BOOT_N = 20261016, 20


def part_summary(p_):
    r_ = json.loads(p_.read_bytes())
    ag_ = r_.get("aggregates") or {}
    val_ = ag_.get("value") or {}
    ba_ = val_.get("boot_aggregate") or {}
    rows_boot = sum(1 for q in r_.get("rows") or [] for x in q.get("ledger") or [] if str(x[0]).startswith("boot|"))
    pooled_boot = sum(1 for q in r_.get("pooled") or [] for x in q.get("ledger") or [] if str(x[0]).startswith("boot|"))
    pl_ = val_.get("plants") or []
    return {"file": p_.name, "level_part": r_.get("level_part"), "levels": r_.get("levels"), "levels_all": r_.get("levels_all"),
            "record_sha256": r_.get("record_sha256"), "inputs_sha256": r_.get("inputs_sha256"), "git": r_.get("git"), "script_sha256": r_.get("script_sha256"),
            "app_id": r_.get("app_id"), "all_equal": r_.get("all_equal"), "labels_all_equal": r_.get("labels_all_equal"), "n_compared": r_.get("n_compared"),
            "aggregates_all_equal": ag_.get("all_equal"), "aggregates_n_compared": ag_.get("n_compared"), "aggregates_mismatches": len(ag_.get("mismatches") or []),
            "value_pass": val_.get("pass"), "domain_missing": len(val_.get("domain_missing") or []), "boot_aggregate_n_compared": ba_.get("n_compared"),
            "boot_aggregate_mismatches": len(ba_.get("mismatches") or []), "rows_boot": rows_boot, "pooled_boot": pooled_boot,
            "boot_seed": (r_.get("boot") or {}).get("seed"), "boot_n": (r_.get("boot") or {}).get("n"), "boot_ids": len((r_.get("boot") or {}).get("ids") or []),
            "plants": [[x_.get("pass"), x_.get("fired")] for x_ in pl_], "missing_shards": r_.get("missing_shards")}


def merge_viol(rec_sha, parts):
    """-> violations by key (STOP when non-empty)."""
    v = []
    one = lambda k_: {json.dumps(s_[k_], sort_keys=True) for s_ in parts}  # noqa: E731
    for k_ in ("record_sha256", "inputs_sha256", "git", "script_sha256", "levels_all"):
        if len(one(k_)) != 1:
            v.append(f"parts differ in {k_}")
    if any(s_["record_sha256"] != rec_sha for s_ in parts):
        v.append("a part on another record")
    kn_ = [tuple(int(x_) for x_ in str(s_["level_part"]).split("/")) if s_["level_part"] else None for s_ in parts]
    n_ = {x_[1] for x_ in kn_ if x_}
    if None in kn_ or len(n_) != 1 or sorted(x_[0] for x_ in kn_) != list(range(1, len(parts) + 1)) or n_ != {len(parts)}:
        v.append(f"part indices are not exactly 1..n of one n: {sorted(str(s_['level_part']) for s_ in parts)}")
    full_ = sorted(float(x_) for x_ in (parts[0]["levels_all"] or []))
    subs_ = [sorted(float(x_) for x_ in (s_["levels"] or [])) for s_ in parts]
    flat_ = sorted(x_ for s_ in subs_ for x_ in s_)
    if not full_ or len(set(flat_)) != len(flat_):
        v.append("the parts' level subsets overlap (or the full set is empty)")
    if flat_ != full_:
        v.append(f"the parts' level subsets are not the full set: missing {sorted(set(full_) - set(flat_))}, extra {sorted(set(flat_) - set(full_))}")
    for s_ in parts:
        w_ = s_["level_part"]
        if not (s_["all_equal"] is True and s_["labels_all_equal"] is True and s_["aggregates_all_equal"] is True and s_["value_pass"] is True):
            v.append(f"part {w_}: not all equal / value not pass")
        if not (isinstance(s_["n_compared"], int) and s_["n_compared"] > 0 and (s_["aggregates_n_compared"] or 0) > 0 and (s_["boot_aggregate_n_compared"] or 0) > 0
                and s_["rows_boot"] > 0 and s_["pooled_boot"] > 0):
            v.append(f"part {w_}: a zero count (n_compared, aggregates, boot aggregate, boot ledgers)")
        if s_["aggregates_mismatches"] or s_["boot_aggregate_mismatches"] or s_["domain_missing"] or s_["missing_shards"]:
            v.append(f"part {w_}: mismatches, a missing value domain entry or a missing shard")
        if (s_["boot_seed"], s_["boot_n"], s_["boot_ids"]) != (BOOT_SEED, BOOT_N, BOOT_N + 1):
            v.append(f"part {w_}: the seeded draw sample is not seed {BOOT_SEED}, {BOOT_N} draws + draw 0")
        if len(s_["plants"]) != 2 or not all(x_ == [True, True] for x_ in s_["plants"]):
            v.append(f"part {w_}: a plant did not pass and fire")
    return v


def selftest():
    """Fixtures: a clean 4-part partition of 15 levels passes; an overlap, a gap, a repeated part index, a failing part, a part on another record,
    and parts on different full input sets each STOP."""
    import copy

    l15_ = [float(i_) for i_ in range(1, 16)]
    base_ = {"record_sha256": "r" * 64, "inputs_sha256": "i", "git": "g", "script_sha256": "s", "levels_all": l15_, "all_equal": True, "labels_all_equal": True,
             "n_compared": 5, "aggregates_all_equal": True, "aggregates_n_compared": 3, "aggregates_mismatches": 0, "value_pass": True, "domain_missing": 0,
             "boot_aggregate_n_compared": 2, "boot_aggregate_mismatches": 0, "rows_boot": 1, "pooled_boot": 1, "boot_seed": BOOT_SEED, "boot_n": BOOT_N,
             "boot_ids": BOOT_N + 1, "plants": [[True, True], [True, True]], "missing_shards": None}
    ok_ = [base_ | {"level_part": f"{k_}/4", "levels": l15_[k_ - 1::4]} for k_ in range(1, 5)]
    assert merge_viol("r" * 64, ok_) == [], merge_viol("r" * 64, ok_)
    bad_ = {"overlap": lambda p_: p_[1].update(levels=p_[1]["levels"] + [1.0]), "gap": lambda p_: p_[3].update(levels=p_[3]["levels"][:-1]),
            "index": lambda p_: p_[3].update(level_part="3/4"), "fail": lambda p_: p_[2].update(all_equal=False),
            "record": lambda p_: [x_.update(record_sha256="x" * 64) for x_ in p_], "inputs": lambda p_: p_[0].update(inputs_sha256="j"),
            "zero": lambda p_: p_[0].update(rows_boot=0)}
    for nm_, f_ in bad_.items():
        p_ = copy.deepcopy(ok_)
        f_(p_)
        assert merge_viol("r" * 64, p_), f"NEGATIVE FIXTURE {nm_} did not STOP"
    print("merge selftest ok: a clean partition passes; overlap, gap, part index, failing part, record, inputs and a zero count each STOP")


def main():
    if sys.argv[1:] == ["--selftest"]:
        return selftest()
    rec_sha, files = sys.argv[1], [Path(f_) for f_ in sys.argv[2:]]
    assert len(rec_sha) == 64 and files, "usage: merge_level_parts_routec.py <record sha256> <part results file> ..."
    parts = sorted((part_summary(HERE / "results" / f_.name) for f_ in files), key=lambda s_: str(s_["level_part"]))
    v = merge_viol(rec_sha, parts)
    out = {"record_sha256": rec_sha, "rule": "GATES A22 @ f5441bb2c: B's PASS needs every level part to PASS and the parts to partition the full level set "
           "exactly (no overlap, no gap) on one record, one full input set, one code and one checker", "parts": parts,
           "levels_all": parts[0]["levels_all"], "n_parts": len(parts), "violations": v, "verdict": "PASS" if not v else "STOP"}
    stem_ = Path(parts[0]["file"]).stem.split("_levelpart")[0]
    dest = HERE / "results" / f"merge_levelparts_{stem_}_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
    dest.open("x").write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"verdict": out["verdict"], "n_parts": len(parts), "violations": v, "file": str(dest)}, indent=1))
    if v:
        raise SystemExit("STOP: the level parts do not merge to a PASS")


if __name__ == "__main__":
    main()
