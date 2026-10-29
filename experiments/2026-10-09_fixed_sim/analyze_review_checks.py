"""The PR review's free checks (README "Review checks", defined before computing), local, from stored per-output scores
of probe step A's chosen probes; no new fits. 5 test environments; levels 1 and 0. Nulls hold the probes fixed and swap
run labels within a cell or planted pair (1,000 draws). Writes results/review_checks_<stamp>.json.

  uv run python experiments/2026-10-09_fixed_sim/analyze_review_checks.py <fixed> <e2> <plant1> <plant2> <manifest_l1.json> <local v2 run>
"""

import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import clock  # noqa: E402
import fixes  # noqa: E402

from probe_shift.scoring import auroc_ci  # noqa: E402

TEST = set(json.loads((HERE / "split.json").read_text())["test"])
ARMS = {"real-attack-e1": "real-attack-e1", "real-attack-e2": "real-attack-e2", "sim-attack": "sim-attack-fixed"}
DRAWS = 1000


def rule_type(tool, args, text, v2):
    """v1 = check 3 of A (editor outputs count only as insert or str_replace); v2 = also any output with a clock stamp."""
    if tool == "text_editor" and (args.get("command") in ("insert", "str_replace") or not v2):
        return args.get("command") in ("insert", "str_replace")
    return fixes.DPKG_MARKER in text or any(s["kind"] != "FAILED" for s in clock.stamps(text, clock.LS_YEAR))


def load_runs(run_dir):
    """{(arm, key): {(step, action): (msg index, rule-type v1, rule-type v2, text)}}"""
    out = {}
    for f in sorted((Path(run_dir) / "inputs").glob("shard_*.jsonl")):
        for r in map(json.loads, f.read_text().splitlines()):
            d = {}
            for c in fixes.calls_of(r["messages"]):
                t = r["messages"][c["msg"]]["content"]
                t = t if isinstance(t, str) else ""
                d[(c["step"], c["action"])] = (c["msg"], rule_type(c["tool"], c["args"], t, False), rule_type(c["tool"], c["args"], t, True), t)
            out[(ARMS[r["arm"]], r["key"])] = d
    return out


def unit(key, planted):
    """The planted pair (a copy and its twin share the twin's key) or the cell (main and side task) a run belongs to."""
    if planted:
        return key.removesuffix("__plant1").removesuffix("__plant2")
    return ":".join(key.rsplit("__", 2)[-2].split(":")[:2])


def run_level(rows, pos, rng):
    by, meta = defaultdict(list), {}
    for r in rows:
        by[(r["arm"], r["key"])].append(r["score"])
        meta[(r["arm"], r["key"])] = (r["arm"] == pos, r["main_task"], unit(r["key"], "plant" in pos))
    keys = sorted(by)
    y = np.array([meta[k][0] for k in keys])
    s = np.array([np.mean(by[k]) for k in keys])
    out = {"runs": {"positive": int(y.sum()), "negative": int((~y).sum())}}
    if len(set(y)) < 2:
        return out
    out |= auroc_ci(y, s, np.array([meta[k][1] for k in keys]), 1000, 0)
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[meta[k][2]].append(i)
    pairs = [g for g in groups.values() if len(g) == 2 and y[g].sum() == 1]
    out["null"] = swap_null(y, s, [[[i], [j]] for i, j in pairs], rng)
    return out


def swap_null(y, s, pairs, rng):
    """Probes fixed: each draw swaps the labels of a pair's two runs with probability 1/2 (pairs: [[run i's indices,
    run j's indices]]; one index per run at run level, all its outputs at output level). p against the unrounded AUROC."""
    obs, null = roc_auc_score(y, s), []
    for _ in range(DRAWS):
        yp = y.copy()
        for a, b in pairs:
            if rng.random() < 0.5:
                yp[a], yp[b] = y[b[0]], y[a[0]]
        null.append(roc_auc_score(yp, s))
    null = np.array(null)
    return {"p95": round(float(np.percentile(null, 95)), 4), "range": [round(float(np.percentile(null, 2.5)), 4), round(float(np.percentile(null, 97.5)), 4)],
            "p": round(float((1 + (null >= obs).sum()) / (1 + DRAWS)), 4), "pairs_swapped": len(pairs)}


def output_level(rows, pos, rng):
    y = np.array([r["arm"] == pos for r in rows])
    if len(set(y)) < 2:
        return None
    s = np.array([r["score"] for r in rows])
    out = auroc_ci(y, s, np.array([r["main_task"] for r in rows]), 1000, 0)
    runs, cell_runs = defaultdict(list), defaultdict(set)
    for i, r in enumerate(rows):
        runs[(r["arm"], r["key"])].append(i)
        cell_runs[unit(r["key"], "plant" in pos)].add((r["arm"], r["key"]))
    pairs = [[runs[a], runs[b]] for a, b in (sorted(v) for v in cell_runs.values() if len(v) == 2) if (a[0] == pos) != (b[0] == pos)]
    return out | {"null": swap_null(y, s, pairs, rng)}


def site_level(pairs, rng):
    """pairs: [(twin score, planted score, main task, run pair)]; AUROC of planted over twin, and the within-pair null:
    each planted pair's labels swap together (all its sites at once; check 2 has up to 23 sites per pair)."""
    t = np.array([p[0] for p in pairs]); q = np.array([p[1] for p in pairs])
    pair = np.unique([p[3] for p in pairs], return_inverse=True)[1]
    y = np.r_[np.zeros(len(t), bool), np.ones(len(q), bool)]
    s = np.r_[t, q]
    res = auroc_ci(y, s, np.array([p[2] for p in pairs] * 2), 1000, 0) | {"pairs": len(pairs), "planted_above_twin": round(float((q > t).mean()), 4)}
    null = []
    for _ in range(DRAWS):
        flip = (rng.random(pair.max() + 1) < 0.5)[pair]
        a, b = np.where(flip, q, t), np.where(flip, t, q)
        null.append(roc_auc_score(y, np.r_[a, b]))
    res["null"] = {"p95": round(float(np.percentile(null, 95)), 4), "p": round(float((1 + (np.array(null) >= roc_auc_score(y, s)).sum()) / (1 + DRAWS)), 4),
                   "pairs_swapped": int(pair.max() + 1)}
    return res


def main(fixed, e2, plant1, plant2, manifest, run_dir):
    rng = np.random.default_rng(0)
    runs = load_runs(run_dir)
    first = {(v, k): min((sa for sa, x in d.items() if x[v]), default=None) for k, d in runs.items() for v in (1, 2)}
    res = {"files": [Path(p).name for p in (fixed, e2, plant1, plant2, manifest)], "prefix_v1_reproduced": {}, "prefix_v2": {},
           "rule_type_split_v2": {}, "plant1_end_of_output": {}, "site_level": {}}
    for name, path, pos in (("fixed_vs_e1", fixed, "sim-attack-fixed"), ("e2_vs_e1", e2, "real-attack-e2")):
        data = json.loads(gzip.decompress(Path(path).read_bytes()))
        for level, d in data["levels"].items():
            rows = [r for r in d["rows"] if r["environment"] in TEST]
            for v, tag in ((1, "v1_reproduced"), (2, "v2")):
                f = lambda r: first[(v, (r["arm"], r["key"]))]
                prefix = [r for r in rows if f(r) is None or (r["step"], r["action"]) < f(r)]
                res[f"prefix_{tag}"][f"{level}|{name}"] = {"run_level": run_level(prefix, pos, rng), "output_level": output_level(prefix, pos, rng),
                                                          "outputs": {"positive": sum(r["arm"] == pos for r in prefix), "negative": sum(r["arm"] != pos for r in prefix)}}
            # check 2 (analyze_claimB_output_types.py) already counted editor clock stamps: v2 is its own definition
            is_rule = lambda r: runs[(r["arm"], r["key"])][(r["step"], r["action"])][2]
            res["rule_type_split_v2"][f"{level}|{name}"] = {"other_outputs_only": run_level([r for r in rows if not is_rule(r)], pos, rng),
                                                           "rule_type_outputs_only": run_level([r for r in rows if is_rule(r)], pos, rng)}
    man = json.loads(Path(manifest).read_text())
    for check, path, sites in (("plant1", plant1, {s["key"]: [s["msg"]] for s in man["plant1"]["sites"]}),
                               ("plant2", plant2, {s["key"]: s["msgs"] for s in man["plant2"]["sites"]})):
        rows = [r for r in json.loads(gzip.decompress(Path(path).read_bytes()))["levels"]["level1"]["rows"] if r["environment"] in TEST]
        score = {(r["key"], r["step"], r["action"]): r for r in rows}
        affected = set()
        if check == "plant1":
            for s in man["plant1"]["sites"]:
                text = next(v[3] for v in runs[("sim-attack-fixed", s["key"])].values() if v[0] == s["msg"])
                assert 1 <= s["after_line"] < len(text.splitlines()), s  # staging drew the site from this text: after line 1..n-1
                after = text.split("\n")[s["after_line"]:]
                if not any(line.strip() for line in after):  # nothing visible after the plant: it is the output's last line
                    affected.add(s["key"])
        pairs_all, pairs_kept = [], []
        for key, msgs in sites.items():
            sa_of = {v[0]: sa for sa, v in runs[("sim-attack-fixed", key)].items()}
            for m in msgs:
                tw, pl = score.get((key, *sa_of[m])), score.get((f"{key}__{check}", *sa_of[m]))
                if tw and pl:
                    pairs_all.append((tw["score"], pl["score"], tw["main_task"], key))
                    if key not in affected:
                        pairs_kept.append((tw["score"], pl["score"], tw["main_task"], key))
        res["site_level"][check] = site_level(pairs_all, rng)
        if check == "plant1":
            kept_rows = [r for r in rows if unit(r["key"], True) not in affected]
            res["plant1_end_of_output"] = {"affected_copies": len(affected), "of": len(sites), "affected_keys": sorted(affected),
                                           "run_level_all": run_level(rows, "sim-attack-fixed-plant1", rng),
                                           "run_level_without_affected": run_level(kept_rows, "sim-attack-fixed-plant1", rng),
                                           "site_level_without_affected": site_level(pairs_kept, rng) if pairs_kept else None}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"review_checks_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(res, indent=1))
    for sec in ("prefix_v1_reproduced", "prefix_v2", "rule_type_split_v2"):
        for k, v in res[sec].items():
            for sub, m in (v.items() if sec.startswith("rule_type") else [("run_level", v["run_level"])]):
                print(f"{sec} {k} {sub}: {m.get('auroc')} {m.get('ci95')} runs {m['runs']} null p95 {m.get('null', {}).get('p95')} p {m.get('null', {}).get('p')}")
            if sec.startswith("prefix"):
                o = v["output_level"]; print(f"{sec} {k} output_level: {o['auroc']} {o['ci95']} outputs {v['outputs']} null p95 {o['null']['p95']} p {o['null']['p']} pairs {o['null']['pairs_swapped']}")
    p1 = res["plant1_end_of_output"]
    print(f"plant1 end of output: {p1['affected_copies']} of {p1['of']} affected; run level all {p1['run_level_all']['auroc']} -> without {p1['run_level_without_affected']['auroc']} "
          f"{p1['run_level_without_affected']['ci95']} runs {p1['run_level_without_affected']['runs']}")
    for k, v in res["site_level"].items():
        print(f"site level {k}: {v['auroc']} {v['ci95']} sites {v['pairs']} above twin {v['planted_above_twin']} null p95 {v['null']['p95']} p {v['null']['p']} pairs {v['null']['pairs_swapped']}")
    if p1["site_level_without_affected"]:
        v = p1["site_level_without_affected"]; print(f"site level plant1 without affected: {v['auroc']} pairs {v['pairs']} above twin {v['planted_above_twin']}")
    print(path)


if __name__ == "__main__":
    main(*sys.argv[1:7])
