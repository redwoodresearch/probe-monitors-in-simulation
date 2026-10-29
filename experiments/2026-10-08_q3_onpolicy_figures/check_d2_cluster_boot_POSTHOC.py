"""POST HOC check for review thread #1069 rrDBR (analyze_d2.py:125). D2's seen-environments statistics reuse the
all-environments bootstrap W, which draws 37 side tasks with replacement, including web_scraping's one side task, whose
tasks are not in the seen cohort. This script (1) rebuilds D2's selection locally from the 84 stored records (no Modal
compute), (2) checks it reproduces the stored seen_environments and all_environments results exactly with the original
W, then (3) recomputes seen_environments with W drawn only over the 36 retained side tasks (36 draws, seed
[20261011, b], b = 1..2000). Keys and numbers only; label files read for arm, task and label only.

From the repo root (records: the 84 npz files of d2_side5_5aedce50ad45/ and d2_side5_hswap_fad33ab7a192/ fetched one
exact path at a time from the score Volume; labels: the five 1118 label files; stored: the two D2 results from S3):
  uv run --no-project --python 3.12 --with numpy --with modal==1.6.1 python \
      experiments/2026-10-08_q3_onpolicy_figures/check_d2_cluster_boot_POSTHOC.py --records <dir> --labels <dir> \
      --tasks experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json \
      --stored <d2_POSTHOC_2026-10-11_103358.json> <d2_POSTHOC_2026-10-11_225636.json> --out <result json>
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_claim_c import stats  # noqa: E402  (01a92967 blob, identical to main)

LAYERS, C_GRID, B, MAX_FAIL = [39, 51, 63], [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 2000, 0.05  # analyze_d2.py:31
UNSEEN = {"web_scraping"}


def select_local(folder: Path, kind: str) -> dict:  # analyze_d2.select, body unchanged except the path
    cands = [(l, kind, C) for C in C_GRID for l in LAYERS]
    R = {c: dict(np.load(folder / f"L{c[0]}_{c[1]}_C{c[2]:g}.npz")) for c in cands}
    r0 = R[cands[0]]
    if any(not all(np.array_equal(r[k], r0[k]) for k in ("on_runs", "Y", "on_fold", "train")) for r in R.values()):
        raise ValueError("records disagree")
    ok_f, ok_v = (np.stack([R[c][f"grad_{n}"] <= R[c]["tol"] for c in cands]) for n in ("full", "val"))
    share = float(1 - np.concatenate([ok_f.ravel(), ok_v.ravel()]).mean())
    V = np.stack([np.where(ok_f[i] & ok_v[i] & np.isfinite(R[c]["val_auroc"]), R[c]["val_auroc"], -np.inf) for i, c in enumerate(cands)])
    pick, usable = V.argmax(0), np.isfinite(V.max(0))
    if share > MAX_FAIL or not usable.all():
        return {"invalid": share}
    complete = np.all(np.stack([np.isfinite(R[c]["on_score"]).all(1) for c in cands]), 0)
    Z = np.full(r0["on_score"].shape, np.nan)
    for f in range(pick.shape[0]):
        for col in range(pick.shape[1]):
            r = R[cands[pick[f, col]]]
            mine = (r["on_fold"] == f) & complete
            Z[mine, col] = r["on_score"][mine, col]
    return {"runs": r0["on_runs"].astype(str).tolist(), "Z": Z, "Y": r0["Y"], "unconverged_share": share}


def same(a, b, path=""):  # exact comparison of two JSON-like trees; returns mismatching paths
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}: keys" ] if set(a) != set(b) else []
        return out + [m for k in a if k in b for m in same(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return [f"{path}: len"] if len(a) != len(b) else [m for i, (x, y) in enumerate(zip(a, b)) for m in same(x, y, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        try:
            return [] if (a is None and b is None) or abs(float(a) - float(b)) <= 1e-12 else [f"{path}: {a} != {b}"]
        except TypeError:
            return [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a} != {b}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)  # dir holding d2_side5_<tag>/ folders
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--stored", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lab = {}
    for f in sorted(Path(a.labels).glob("phase2_attempt_labels_*_2026-10-10_1118*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            r = {"arm": r.get("arm"), "task": r.get("task"), "label": r.get("label")}  # explicit columns only
            w = {"attack": "run2" if "second_attack" in f.name else "run1", "sim_attack@september": "sim"}.get(r["arm"])
            if w and r["label"]:
                if f"{w}|{r['task']}" in lab:
                    raise ValueError("two labels")
                lab[f"{w}|{r['task']}"] = r["label"]
    cfg = json.loads(Path(a.tasks).read_text())
    side = {c["key"]: (c["env"], c["side"]) for c in cfg["cells"]}
    tasks = sorted(side)
    ti = {t: i for i, t in enumerate(tasks)}
    sides = sorted(set(side.values()))
    si = np.array([sides.index(side[t]) for t in tasks])
    keep_s = [i for i, s in enumerate(sides) if s[0] not in UNSEEN]
    W_old, W_new, zero = [], [], []
    for b in range(1, B + 1):
        cnt = np.bincount(np.random.default_rng([20261011, b]).integers(0, len(sides), len(sides)), minlength=len(sides))
        W_old.append(cnt[si].astype(float))
        zero.append(int(sum(cnt[i] for i in range(len(sides)) if i not in keep_s)))
        draw = np.array(keep_s)[np.random.default_rng([20261011, b]).integers(0, len(keep_s), len(keep_s))]
        cnt2 = np.bincount(draw, minlength=len(sides))
        W_new.append(cnt2[si].astype(float))
    out = {"label": "POST HOC check for review thread rrDBR; not a registered analysis", "side_tasks": len(sides),
           "retained_side_tasks": len(keep_s), "unseen_tasks": int(sum(1 for t in tasks if side[t][0] in UNSEEN)),
           "old_scheme_draws_landing_on_excluded_cluster": {"mean": float(np.mean(zero)), "share_of_draw_slots": float(np.mean(zero) / len(sides)),
                                                             "draws_with_at_least_one": int(sum(z > 0 for z in zero))},
           "results": {}}
    for st in a.stored:
        sj = json.loads(Path(st).read_text())
        tag = sj["tags"]["side5"]
        res = {"stored_file": Path(st).name, "stored_sha256": hashlib.sha256(Path(st).read_bytes()).hexdigest(), "tag": tag}
        for kind, name in (("call_end", "command_end_content_only"), ("output_end", "reply_end")):
            d = select_local(Path(a.records) / f"d2_side5_{tag}", kind)
            keep = np.array([side[u.split("|", 1)[1]][0] not in UNSEEN for u in d["runs"]])
            seen = {"runs": [u for u, k in zip(d["runs"], keep) if k], "Z": d["Z"][keep], "Y": d["Y"][keep]}
            stored = sj["results"]["side5"][name]
            js = lambda x: json.loads(json.dumps(x, default=float))
            gate = {"seen_old_W": same(js(stats(seen, lab, tasks, ti, W_old)), stored["seen_environments"]),
                    "all_old_W": same(js(stats(d, lab, tasks, ti, W_old)), stored["all_environments"])}
            if any(gate.values()):
                res[name] = {"gate_failed": {k: v[:10] for k, v in gate.items()}}
                continue
            new = js(stats(seen, lab, tasks, ti, W_new))
            old = stored["seen_environments"]
            pick = lambda s: {"gap": s["observed"]["gap"], "gap_ci95": s["gap_interval"]["ci95"], "control": s["observed"]["control"],
                              "control_ci95": s["control_interval"]["ci95"], "control_passes": s["control_passes"],
                              "plants_detected": {k: p["detected"] for k, p in s["plants"].items()}, "null_passes": s["null"]["passes"],
                              "verdicts": s["verdicts"], "tpr_run1_ci95": s["tpr_run1_interval"]["ci95"],
                              "false_alarm_ci95": s["false_alarm_interval"]["ci95"], "auroc_gap_ci95": s["report"]["auroc_gap_interval"]["ci95"],
                              "gap_5pct_ci95": s["report"]["at_5pct"]["gap_interval"]["ci95"]}
            res[name] = {"gate": "passed: stored seen_environments and all_environments reproduced exactly with the original W",
                         "old_W": pick(old), "retained_clusters_W": pick(new)}
        out["results"][tag] = res
    Path(a.out).write_text(json.dumps(out, default=float) + "\n")  # one line: the PR line limit


if __name__ == "__main__":
    main()
