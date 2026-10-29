"""G7b VALUE: the SCOPE block of a value-reproduction results file, by script (2e 10-17 00:5x): which labels x readouts x combos x levels x modes were
reproduced and compared (the VALUE rules' counts and the ledger keys), fold-model vs single-model labels, the seeded draws and the keys per draw, the
tolerance, and the record's pooled labels NOT covered. Keys, counts and MATCH / MISMATCH only (no value is read or printed).
Run from the repo root: uv run python experiments/2026-10-15_e2e_sim_pipeline/analyze_g7b_value_scope.py <results file> <record git commit> <record path>"""
import collections
import gzip
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
HEADLINE_SIX = ("sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48")  # full_override.HEADLINE_SIX


def main(res_path, rec_commit, rec_path):
    rb_ = Path(res_path).read_bytes()
    R = json.loads(rb_)
    gb_ = subprocess.run(["git", "show", f"{rec_commit}:{rec_path}"], capture_output=True, check=True).stdout
    rec_ = json.loads(gzip.decompress(gb_) if rec_path.endswith(".gz") else gb_)
    labels = sorted({r_["row"] for r_ in R["rows"]})
    fold_ = sorted({q["row"] for q in R["rules"] if q["rule"] == "fold-model check: not covered (stated)"})
    vr_ = {q["row"]: q for q in R["rules"] if q["rule"].startswith("VALUE:")}
    cells_, ok_ = [], True
    for lb in labels:
        for k_, c_ in sorted((vr_[lb].get("counts") or {}).items()):
            cb, lv, md = k_.split(" | ")
            n_rows = sum(c_["rows"].values())
            nb_ = c_.get("boot_pooled", 0) + sum((c_.get("boot_rows") or {}).values())
            cells_.append({"label": lb, "family": lb.rsplit(" ", 1)[0], "readout": lb.rsplit(" ", 1)[1], "combo": cb, "level": lv, "mode": md,
                           "draw0_pooled": c_["pooled"], "draw0_member_rows": n_rows, "members": len(c_["rows"]), "seeded_draw_comparisons": nb_})
            ok_ = ok_ and c_["pooled"] > 0 and all(v_ > 0 for v_ in c_["rows"].values())
    verd_ = collections.Counter(v_ for sec in ("rows", "pooled") for r_ in R[sec] for _, v_ in r_.get("ledger") or [])
    verd_ |= collections.Counter()
    agg_ = R.get("aggregates") or {}
    av_ = collections.Counter(v_ for _, v_ in agg_.get("ledger") or [])
    pat_row = sorted({re.sub(r"^boot\|(joint|fixed)\|[0-9.e-]+\|draw [0-9]+\|", "", k_) for r_ in R["rows"] for k_, _ in r_.get("ledger") or []
                      if k_.startswith("boot|") and "self-check" not in k_ and "draws_sha256" not in k_})
    pat_pool = sorted({re.sub(r"^boot\|(joint|fixed)\|[0-9.e-]+\|draw [0-9]+\|", "", k_) for r_ in R["pooled"] for k_, _ in r_.get("ledger") or []
                       if k_.startswith("boot|") and "self-check" not in k_})
    v0_ = next(iter(vr_.values()))
    fams_ = sorted({lb.rsplit(" ", 1)[0] for lb in labels})
    rec_labels = sorted((rec_["meta"].get("pooled_world") or {}).get("labels") or rec_["rows"])
    out = {"results_file": str(Path(res_path).name), "results_sha256": hashlib.sha256(rb_).hexdigest(), "record": f"{rec_path} @ {rec_commit}",
           "record_decompressed_sha256": hashlib.sha256(gzip.decompress(gb_) if rec_path.endswith(".gz") else gb_).hexdigest(),
           "scope_line": R.get("label"), "all_equal": R["all_equal"],
           "labels": labels, "families": fams_, "fold_model_labels": fold_, "single_model_labels": sorted(set(labels) - set(fold_)),
           "headline_six_covered": {f_: all(f"{f_} {ro}" in labels for ro in "CR") for f_ in HEADLINE_SIX},
           "record_pooled_labels_not_covered": sorted(set(rec_labels) - set(labels)),
           "combos": sorted({c_["combo"] for c_ in cells_}), "levels": sorted({c_["level"] for c_ in cells_}), "modes": sorted({c_["mode"] for c_ in cells_}),
           "cells": len(cells_), "cells_expected": len(labels) * len({c_["combo"] for c_ in cells_}) * len({c_["level"] for c_ in cells_}) * 2,
           "every_cell_compared": ok_, "cells_detail": cells_,
           "seeded_draws": {"draw_ids": (v0_.get("boot") or {}).get("draw_ids"), "seed": (v0_.get("boot") or {}).get("seed"),
                            "n_sampled": len([b_ for b_ in (v0_.get("boot") or {}).get("draw_ids") or [] if b_ != 0]),
                            "draws_sha256": (v0_.get("boot") or {}).get("draws_sha256"),
                            "keys_per_member_row_per_draw_mode_level": pat_row, "keys_per_pooled_block_per_draw_mode_level": pat_pool,
                            "aggregate_groups": (agg_.get("value") or {}).get("boot_aggregate", {}).get("groups"),
                            "aggregate_keys_per_group_readout_draw": ["S", "Y"]},
           "tolerance": v0_.get("tolerance"), "not_compared": v0_.get("not_compared"),
           "verdicts": {"rows_and_pooled": dict(verd_), "aggregate": dict(av_)},
           "plants": [{k_: x_[k_] for k_ in ("plant", "pass", "fired", "expected_keys", "flagged_keys")} for x_ in (agg_.get("value") or {}).get("plants") or []],
           "domain_missing": len((agg_.get("value") or {}).get("domain_missing") or []),
           "run": {"read_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z"),
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    dest = HERE / "results" / f"g7b_value_scope_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
    dest.open("x").write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k_: v_ for k_, v_ in out.items() if k_ != "cells_detail"}, indent=1))
    print("written", dest, hashlib.sha256(dest.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:4])
