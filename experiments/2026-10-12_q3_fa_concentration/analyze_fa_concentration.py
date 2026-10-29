"""Do a few tasks drive the simulated false-alarm gap? (RULE.md; POST HOC, $0, local; keys, labels, scores, flags and
stored extraction metadata only, no command text.)

  PYTHONPATH=experiments/2026-10-08_q3_onpolicy_figures uv run --no-project --with numpy python \
      experiments/2026-10-12_q3_fa_concentration/analyze_fa_concentration.py <draws0 json> <sim metas root>

<sim metas root> holds <extraction>/**/<hash>.json, copied from <extraction>/meta/sim_attack@september on the Volume
probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08.
"""

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_scaleup_plants import threshold

HERE = Path(__file__).parent
REPO = HERE.parent.parent
DRAWS_SHA = "c2656658baaf3aa150f83f4d105e7572bfd854f98032831621419e1f8e8f36bf"
PERTASK = REPO / "experiments/2026-10-08_q3_onpolicy_figures/results/per_task_false_alarms_POSTHOC_2026-10-11_122057.json"
PERTASK_SHA = "7893a99899c889bd80e587fdb75f39aa62779abbbcb2154fa59b1b633e8c2548"
CELLS = {"real": "real->real", "sim": "real->sim", "run2": "real->run2"}
EXTRACTIONS = ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1",
               "2026-10-10_scaleup_l2_segends_v1", "2026-10-10_scaleup_topup_segends_v1"]
STEP_LIMITS = {"configured (sim.yaml BA_MAX_STEPS, run.yaml max_steps)": 200, "September's earlier (sim.yaml comment)": 150}
TOPS = (1, 3, 5)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def frac(k, n):
    return {"flagged": int(k), "total": int(n), "rate": (float(k) / n) if n else None}


def main(draws_path: str, metas_root: str):
    dirty = subprocess.run(["git", "status", "--porcelain", str(HERE)], capture_output=True, text=True, cwd=REPO).stdout.strip()
    if dirty:
        raise SystemExit(f"commit RULE.md and this script first:\n{dirty}")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    raw = Path(draws_path).read_bytes()
    if sha(raw) != DRAWS_SHA:
        raise SystemExit("draw-0 file sha256 differs")
    praw = PERTASK.read_bytes()
    if sha(praw) != PERTASK_SHA:
        raise SystemExit("per-task result sha256 differs")
    M, P = json.loads(raw), json.loads(praw)
    folds, meta = M["folds"], M["meta"]
    tasks_all = sorted({m["task"] for m in meta})

    # gate 1: the frozen thresholds rebuilt from the real->real calibration rows reproduce flag_frozen (as in the per-task script)
    cal = M["draws"][0]["cells"]["real->real"]["per_call"]["calibration"]
    cs = [(s, k) for i, s, k in zip(cal["row"], cal["score"], cal["model_fold"]) if meta[i]["world"] == "real" and meta[i]["y"] == 0]
    c_s, c_k = (np.array(x) for x in zip(*cs))
    th0 = np.array([threshold(c_s[c_k == k], np.ones(int((c_k == k).sum()))) for k in range(5)])
    gates, D, flag_of = {}, {}, {}
    gates["thresholds_equal_per_task_result"] = bool(np.max(np.abs(th0 - np.array(P["thresholds_point"]))) <= 1e-9)
    for w, cell in CELLS.items():
        pc = M["draws"][0]["cells"][cell]["per_call"]
        rows, s, fl = pc["row"], np.array(pc["score"]), np.array(pc["flag_frozen"], dtype=bool)
        if not len(rows) == len(s) == len(fl):
            raise SystemExit(f"{cell}: misaligned per-call arrays")
        if any(meta[r]["world"] != w for r in rows) or len(set(rows)) != len(rows):
            raise SystemExit(f"{cell}: rows from another world or repeated")
        f = np.array([folds[meta[r]["task"]] for r in rows])
        y = np.array([meta[r]["y"] for r in rows])
        gates[f"{w}_flags_reproduce"] = bool(np.array_equal(s > th0[f], fl))
        ro = y == 0
        pooled = float(fl[ro].mean())
        gates[f"{w}_pooled_equals_file"] = abs(pooled - M["draws"][0]["cells"][cell]["values"]["false_alarm_frozen"]) <= 1e-9
        n, k = defaultdict(int), defaultdict(int)
        for r, flag, yy in zip(rows, fl, y):
            if yy == 0:
                n[meta[r]["task"]] += 1
                k[meta[r]["task"]] += int(flag)
            if w == "sim":
                flag_of[(meta[r]["irid"], meta[r]["call"])] = (bool(flag), int(yy), meta[r]["task"])
        per_task_mean = float(np.mean([k[t] / n[t] for t in n if n[t] > 0]))
        gates[f"{w}_pooled_equals_per_task_result"] = abs(pooled - P["worlds"][w]["pooled"]) <= 1e-9
        gates[f"{w}_per_task_mean_equals_per_task_result"] = abs(per_task_mean - P["worlds"][w]["per_task_mean"]) <= 1e-9
        D[w] = {"n": dict(n), "k": dict(k), "pooled": pooled, "per_task_mean": per_task_mean}

    # the sim metas: per-file sha256s; join each draw-0 sim row (irid, call) -> command row -> physical call (irid, call_index)
    files = sorted(p for e in EXTRACTIONS for p in (Path(metas_root) / e).rglob("*.json"))
    file_shas = [sha(p.read_bytes()) for p in files]
    metas_digest = sha("\n".join(f"{p.relative_to(metas_root)} {h}" for p, h in zip(files, file_shas)).encode())
    by_run = defaultdict(list)
    cmd, call_end = defaultdict(list), defaultdict(list)
    for p in files:
        j = json.loads(p.read_text())
        by_run[j["run"]].append(j)
        for c in j["calls"]:
            if c.get("kind") == "command":
                cmd[(c["irid"], c["call"])].append((c["call_index"], c["label"], j["task"]))
            elif c.get("kind") == "call_end":
                call_end[(c["irid"], c["call_index"])].append(c)
    phys_flag, bad = {}, 0
    for (irid, call), (flag, yy, task) in flag_of.items():
        m = cmd.get((irid, call), [])
        ends = call_end.get((irid, m[0][0]), []) if len(m) == 1 else []
        if (len(m) != 1 or len(ends) != 1 or m[0][2] != task or (m[0][1] == "read_only") != (yy == 0)
                or ends[0]["label"] != m[0][1]):
            bad += 1
            continue
        phys_flag[(irid, m[0][0])] = (flag, yy)
    gates["sim_rows_join_once_with_agreeing_labels"] = bad == 0 and len(phys_flag) == len(flag_of)
    if not all(v is True for v in gates.values()):
        raise SystemExit(f"a gate failed: {gates}")

    def ranked(w):
        n, k = D[w]["n"], D[w]["k"]
        return sorted(n, key=lambda t: (-k.get(t, 0), -(k.get(t, 0) / n[t]), t))

    out = {"label": "POST HOC, exploratory (RULE.md); from raw data; keys, labels, scores, flags and stored metadata only",
           "git": git, "inputs": {Path(draws_path).name: DRAWS_SHA, PERTASK.name: PERTASK_SHA,
                                  "sim_metas": {"files": len(files), "digest_of_sorted_file_sha256s": metas_digest}},
           "gates": gates, "thresholds": th0.tolist(), "worlds": {}, "gap": {}, "sim_top5": []}
    for w in CELLS:
        n, k = D[w]["n"], D[w]["k"]
        N, K = sum(n.values()), sum(k.values())
        order = ranked(w)
        row = {"all": frac(K, N), "per_task_mean": D[w]["per_task_mean"], "top_tasks": order[:5], "top": {}}
        for j in TOPS:
            top = order[:j]
            kt, nt = sum(k.get(t, 0) for t in top), sum(n[t] for t in top)
            row["top"][str(j)] = {"share_of_flagged": frac(kt, K), "pooled_without": frac(K - kt, N - nt)}
        out["worlds"][w] = row

    # relative reduction of the sim - real per-call gap after excluding sim's top j tasks: paired (both worlds) and sim only
    sn, sk, rn, rk = D["sim"]["n"], D["sim"]["k"], D["real"]["n"], D["real"]["k"]
    full = sum(sk.values()) / sum(sn.values()) - sum(rk.values()) / sum(rn.values())
    out["gap"]["full_sim_minus_real"] = full
    for j in TOPS:
        top = set(ranked("sim")[:j])
        ns, nr = sum(v for t, v in sn.items() if t not in top), sum(v for t, v in rn.items() if t not in top)
        if full == 0 or ns == 0 or nr == 0:
            out["gap"][str(j)] = None
            continue
        s_rate = sum(v for t, v in sk.items() if t not in top) / ns
        r_pair = sum(v for t, v in rk.items() if t not in top) / nr
        r_all = sum(rk.values()) / sum(rn.values())
        out["gap"][str(j)] = {"paired": {"gap_after": s_rate - r_pair, "relative_reduction": 1 - (s_rate - r_pair) / full},
                              "sim_only": {"gap_after": s_rate - r_all, "relative_reduction": 1 - (s_rate - r_all) / full}}

    # (c), (d): sim's top 5 runs from the metas
    sim_run_of_task = defaultdict(set)
    for run, js in by_run.items():
        ts = {j["task"] for j in js}
        if len(ts) != 1:
            raise SystemExit(f"{run}: metas span tasks")
        sim_run_of_task[ts.pop()].add(run)
    def owner(d):  # an output_end row belongs to its source call (src_irid); command and call_end rows to their own request
        return (d["src_irid"], d.get("call_index")) if d.get("kind") == "output_end" else (d["irid"], d.get("call_index"))

    for t in ranked("sim")[:5]:
        rs = sim_run_of_task.get(t, set())
        if len(rs) != 1:
            raise SystemExit(f"{t}: {len(rs)} sim runs in the metas")
        js = sorted(by_run[next(iter(rs))], key=lambda j: (j["epoch"], j["attempt_index"]))
        draw_calls = sum(1 for m in meta if m["world"] == "sim" and m["task"] == t)
        seq, max_turn, no_kind = [], -1, 0
        for j in js:
            pos = [c for c in j["calls"] if c.get("readout") is not None]
            ends = sorted((c for c in j["calls"] if c.get("kind") == "call_end"), key=lambda c: (c["turn"], c["call_index"]))
            for c in j["calls"]:
                max_turn = max(max_turn, c.get("turn", -1))
            entries = []
            for i, c in enumerate(ends):
                own = (c["irid"], c["call_index"])
                prev = [d["readout"] for d in pos if d["readout"] < (c["readout"] if c["readout"] is not None else -1)
                        and owner(d) != own]
                seg = None if i == 0 or c["readout"] is None or not prev else c["readout"] - max(prev)
                fl = phys_flag.get(own)
                entries.append(((c["turn"], c["call_index"]), {"ro": c["label"] == "read_only", "tool": (c["name"], c["param"]),
                                "seg": seg, "flag": bool(fl and fl[1] == 0 and fl[0]), "epoch": j["epoch"]}))
            for c in j["calls"]:
                if c.get("kind") is None:
                    no_kind += 1
                    entries.append(((c.get("turn", -1), c.get("call_index", -1)), {"ro": False, "boundary": True, "epoch": j["epoch"]}))
            seq += [x for _, x in sorted(entries, key=lambda e: e[0])]

        def longest(member, same):
            best, cur = (0, 0, None), []
            for x in seq:
                if member(x) and cur and same(cur[-1], x):
                    cur.append(x)
                elif member(x):
                    cur = [x]
                else:
                    cur = []
                if cur and len(cur) > best[0]:  # strict: ties keep the earliest stretch
                    best = (len(cur), sum(z["flag"] for z in cur), list(cur[0]["tool"]))
            return {"length": best[0], "flagged_in_stretch": best[1], "tool_param": best[2]}

        L = longest(lambda x: x["ro"] and x["seg"] is not None,
                    lambda a, b: a["tool"] == b["tool"] and a["seg"] == b["seg"] and a["epoch"] == b["epoch"])
        T = longest(lambda x: x["ro"], lambda a, b: a["tool"] == b["tool"])
        out["sim_top5"].append({
            "task": t, "read_only": frac(sk.get(t, 0), sn[t]), "scored_calls_in_draw0": draw_calls,
            "physical_calls_in_metas": sum(1 for x in seq if "tool" in x), "rows_without_kind": no_kind, "epochs": len(js),
            "max_turn_plus_1": max_turn + 1, "step_limits": STEP_LIMITS,
            "reached_step_limit_inferred": {k: max_turn + 1 >= v for k, v in STEP_LIMITS.items()},
            "repeats_one_command": "not computable without a text pass, which the rule forbids",
            "proxy_L_same_tool_param_and_segment_length": L, "proxy_T_same_tool_param": T})

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"fa_concentration_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({"gates": gates, "gap": out["gap"]}, indent=1))
    for w, v in out["worlds"].items():
        print(w, v["all"], {j: (v["top"][j]["share_of_flagged"], v["top"][j]["pooled_without"]) for j in v["top"]})
    for r in out["sim_top5"]:
        print(r)
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("draws_path")
    ap.add_argument("metas_root")
    a = ap.parse_args()
    main(a.draws_path, a.metas_root)
