"""Checks on the stage-1 labelling lists (PLAN.md 4): IDs, schemas, hashes, counts against the rule, inclusion
probabilities, and whether the labels file's order or ids could reveal the world or the source. IDs and numbers only.

  uv run --no-project --python 3.12 --with pandas --with numpy python check_lists.py --lists <data dir> --results <lists json>
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ID = re.compile(r"^sab1-[0-9a-f]{8}$")


def main(lists: str, results: str):
    d, R = Path(lists), json.loads(Path(results).read_text())
    out, files = {}, {n: pd.read_csv(d / n, dtype=str, keep_default_na=False) for n in ("labels_stage1.csv", "locator_stage1.csv", "key_stage1.csv")}
    out["sha256_match"] = {n: hashlib.sha256((d / n).read_bytes()).hexdigest() == R["files"][n]["sha256"] for n in files}
    out["columns"] = {n: list(f.columns) for n, f in files.items()}
    L, Lo, K = files["labels_stage1.csv"], files["locator_stage1.csv"], files["key_stage1.csv"]
    ids = [set(f.opaque_id) for f in files.values()]
    out["ids"] = {"rows": [len(f) for f in files.values()], "unique": [f.opaque_id.is_unique for f in files.values()],
                  "same_set": ids[0] == ids[1] == ids[2], "format_ok": bool(L.opaque_id.str.match(ID).all())}
    out["labels_file_empty_fields"] = bool((L.label == "").all() and (L.note == "").all())
    out["locator_fields_numeric"] = {c: bool(Lo[c].str.fullmatch(r"\d+").all()) for c in ("epoch", "turn", "call_index")}
    out["run_hash_format"] = bool(Lo.run_hash.str.fullmatch(r"[0-9a-f]{16}").all())
    K = K.set_index("opaque_id").loc[L.opaque_id].reset_index()  # labels-file order
    Lo = Lo.set_index("opaque_id").loc[L.opaque_id].reset_index()
    out["run_hash_matches_run_id"] = bool((K.run_id.map(lambda r: hashlib.sha256(r.encode()).hexdigest()[:16]) == Lo.run_hash).all())
    out["locator_matches_key"] = bool((Lo[["epoch", "turn", "call_index"]].values == K[["epoch", "turn", "call_index"]].values).all())
    # order and id leakage: world and source shares by position decile; id order vs world
    dec = np.arange(len(K)) * 10 // len(K)
    out["world_share_by_decile"] = {w: [round(float(((K.world == w) & (dec == i)).sum() / (dec == i).sum()), 3) for i in range(10)] for w in sorted(K.world.unique())}
    out["source_share_by_decile"] = {s: [round(float(((K.source == s) & (dec == i)).sum() / (dec == i).sum()), 3) for i in range(10)] for s in sorted(K.source.unique())}
    sim = (K.world == "sim").astype(int).values
    runs = 1 + int((sim[1:] != sim[:-1]).sum())
    n1, n0 = sim.sum(), len(sim) - sim.sum()
    mu = 1 + 2 * n1 * n0 / len(sim)
    var = 2 * n1 * n0 * (2 * n1 * n0 - len(sim)) / (len(sim) ** 2 * (len(sim) - 1))
    out["runs_test_sim_vs_real_in_order"] = {"runs": runs, "expected": round(mu, 1), "z": round((runs - mu) / var**0.5, 2)}
    rank = pd.Series(L.opaque_id).rank().values
    out["id_rank_vs_sim_corr"] = round(float(np.corrcoef(rank, sim)[0, 1]), 4)
    out["id_rank_vs_position_corr"] = round(float(np.corrcoef(rank, np.arange(len(L)))[0, 1]), 4)
    # what the locator alone shows: rows per run_hash by world
    per = K.groupby(["world", "run_id"]).size()
    out["rows_per_run_by_world"] = {w: {"runs": int(len(per[w])), "mean": round(float(per[w].mean()), 2), "min": int(per[w].min()), "max": int(per[w].max())} for w in sorted(K.world.unique())}
    # counts against the rule and the results json
    got = K.groupby(["side", "world", "source"]).size()
    out["counts_match_results"] = {f"{a} / {b} / {c}": int(v) for (a, b, c), v in got.items()} == R["files"]["rows_by_side_world_source"]
    k = R["counts"]["k_flagged_per_run"]
    fl = K[K.source == "flagged_subset"].groupby(["world", "run_id"]).size()
    out["flagged_per_run_at_most_k"] = {w: bool((fl[w] <= k[w]).all()) for w in fl.index.get_level_values(0).unique()}
    out["flagged_per_run_equals_k_share"] = {w: round(float((fl[w] == k[w]).mean()), 3) for w in fl.index.get_level_values(0).unique()}
    p = K.p_incl.astype(float)
    out["p_incl"] = {s: [round(float(p[K.source == s].min()), 4), round(float(p[K.source == s].max()), 4)] for s in sorted(K.source.unique())}
    out["audit_rows"] = {s: int((K.source == s).sum()) for s in K.source.unique() if s.startswith("audit")}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", required=True)
    ap.add_argument("--results", required=True)
    main(**vars(ap.parse_args()))
