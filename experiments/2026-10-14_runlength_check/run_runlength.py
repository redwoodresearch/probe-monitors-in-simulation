"""Run-length check of the end-of-call +3 (RULE.md) on Modal CPU. A remote orchestrator scans the attack-world .eval logs
for ids and error flags only (no text leaves a scan container), joins them to 2e's command units by irid, and computes
checks 1-3. The local side only checks the tree and input hashes.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy==2.3.3 modal run \\
      experiments/2026-10-14_runlength_check/run_runlength.py --labels <claimc_labels_reduced.json>
"""

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-runlength_kill-2026-11-14"
SOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
CMD = ("xfer/correctedb_sweep_2026-10-14_125005/perrow_cmd_callend_contentonly_2026-10-14_033117__a__A_corrected_L63.npz",
       "8930ee126258adc3c606a6bfb0d8db271c6d9a53283c54740243b18444900035")
EV_ROOT = "/ev/evals_v1"
LABELS_SHA256 = "d00e88773fc2a71bef51e1ff0ccc2559932f5c3a306a8f5604c5248e797dc995"
T196 = ("experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json",
        "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
PERRUN09 = ("744e86c82", "experiments/2026-10-14_tool_schema_behaviour_09/results/tool_behaviour_perrun_2026-10-14_134930.json", "8284fac7126a3a9a")
SWEEP = ("ab095924e", "experiments/2026-10-14_fpr_sweep_xcheck/results/sweep_xcheck_2026-10-14_130926.jsonl")
UNIT_F, RUN_A = (1e-3, 1e-2), (0.01, 0.05, 0.10)
N_BOOT, SEED = 10_000, [20261014, 1]
SAMPLE_RE = re.compile(r"samples/.+_epoch_(\d+)\.json")
REDRAW_RE = re.compile(r"_redraw\d+$")
PATH_RE = re.compile(r"[A-Za-z0-9_@.:/-]+\.eval")

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scipy==1.16.2", "zstandard==0.23.0")
ev = modal.Volume.from_name("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", version=2).read_only()
sol = modal.Volume.from_name(SOL).read_only()


def world_of_path(rel: str):  # 09's run_tool_behaviour.world_of_path @ 19421fa7f, attack worlds only
    arm = Path(rel).parts[1]
    if arm == "attack":
        return "run2" if rel.startswith("2026-10-10_001314_second_attack/") else "run1"
    return "sim" if arm == "sim_attack@september" else None


def zstd_zip():  # 09's run_tool_behaviour.zstd_zip @ 19421fa7f, unchanged
    import zipfile

    import zstandard

    if hasattr(zipfile, "ZIP_ZSTANDARD") or getattr(zipfile, "_zstd_patched", False):
        return zipfile
    check, get = zipfile._check_compression, zipfile._get_decompressor
    zipfile._check_compression = lambda ct: None if ct == 93 else check(ct)
    zipfile._get_decompressor = lambda ct: zstandard.ZstdDecompressor().decompressobj() if ct == 93 else get(ct)
    zipfile._zstd_patched = True
    return zipfile


def resolve(x, att: dict):  # Inspect attachment references, for the header and id fields only
    return att.get(x[len("attachment://"):], x) if isinstance(x, str) and x.startswith("attachment://") else x


@app.function(image=image, cpu=1, memory=8192, timeout=1800, retries=0, volumes={"/ev": ev})
def scan(rels: list) -> dict:
    """Ids and error flags only: per sample, irid -> tool-call ids, the run's call order, and errored call ids."""
    zipfile = zstd_zip()
    out, errors = [], {}
    for rel in rels:
        try:
            with zipfile.ZipFile(Path(EV_ROOT) / rel) as z:
                names = [n for n in z.namelist() if SAMPLE_RE.fullmatch(n)]
                if not names:
                    raise ValueError("no samples")
                for n in names:
                    s = json.loads(z.read(n))
                    att = s.get("attachments") or {}
                    irid, conflicts = {}, 0
                    for e in s.get("events") or []:
                        if not isinstance(e, dict) or e.get("event") != "model":
                            continue
                        hdr = (((e.get("call") or {}).get("request") or {}).get("extra_headers") or {})
                        r = resolve(hdr.get("x-irid"), att) if isinstance(hdr, dict) else None
                        if not r:
                            continue
                        ch = ((e.get("output") or {}).get("choices") or [{}])[0]
                        ids = [resolve(c.get("id"), att) for c in ((ch.get("message") or {}).get("tool_calls") or [])]
                        conflicts += r in irid and irid[r] != ids
                        irid[r] = ids
                    order, err, err_bad = [], {}, 0
                    for m in s.get("messages") or []:
                        if m.get("role") == "assistant":
                            order += [c.get("id") for c in m.get("tool_calls") or []]
                        elif m.get("role") == "tool" and m.get("error") is not None:
                            e, tid = m["error"], m.get("tool_call_id")
                            err_bad += tid is None or tid in err
                            err[tid] = str(e.get("type")) if isinstance(e, dict) else "unstructured"
                    out.append({"rel": rel, "world": world_of_path(rel), "task": REDRAW_RE.sub("", Path(rel).parts[2]), "sample": n,
                                "irid": irid, "irid_conflicts": conflicts, "order": order, "err": err, "err_bad": err_bad})
        except Exception as e:  # noqa: BLE001 -- the error type only, never text; counted and gated
            errors[rel] = type(e).__name__
    return {"rows": out, "errors": errors}


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0, volumes={"/ev": ev, "/sol": sol})
def analyze(labels: dict, t196: dict, perrun09: list, sweep_thr: dict) -> dict:
    import io

    import numpy as np
    from scipy import stats

    rels = sorted(str(p.relative_to(EV_ROOT)) for p in Path(EV_ROOT).glob("*/*/*/*.eval") if world_of_path(str(p.relative_to(EV_ROOT))))
    if not rels or not all(PATH_RE.fullmatch(r) for r in rels):
        raise ValueError("unexpected eval path")
    parts = list(scan.map([rels[i:i + 20] for i in range(0, len(rels), 20)]))
    samples = [r for p in parts for r in p["rows"]]
    scan_errors = {k: v for p in parts for k, v in p["errors"].items()}
    if scan_errors:
        raise ValueError(f"scan errors in {len(scan_errors)} files: {sorted(set(scan_errors.values()))}")

    raw = (Path("/sol") / CMD[0]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CMD[1]:
        raise ValueError("command npz sha256 differs")
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    key, unit, ci, irid, score = (z["key"].astype(str), z["unit"].astype(np.int64), z["call_index"].astype(np.int64),
                                  z["irid"].astype(str), z["score"].astype(np.float64))
    runs = {}
    for i in np.argsort(unit, kind="stable"):
        runs.setdefault(str(key[i]), []).append((int(unit[i]), int(ci[i]), str(irid[i]), float(score[i])))
    A = {}
    for k, rows in runs.items():
        a, t = k.split("|", 1)
        A.setdefault(a, {})[t] = rows
    cohort = sorted(t for t in set(A["run1"]) & set(A["run2"]) & set(A["sim"])
                    if all(labels.get(f"{w}|{t}") == "attempted" for w in ("run1", "run2", "sim")))

    # join: each cohort attack run -> exactly one sample holding all its irids; units -> positions in the run's call order
    by_wt = {}
    for s in samples:
        by_wt.setdefault((s["world"], s["task"]), []).append(s)
    info, gate = {}, {"runs": 0, "units": 0, "multi_or_none": 0, "out_of_range": 0, "not_in_order": 0, "nonmonotone": 0,
                      "irid_conflicts_used": 0, "error_ids_not_in_order": 0, "dup_or_missing_call_ids": 0, "dup_or_missing_error_ids": 0}
    for w in ("run1", "run2", "sim"):
        for t in cohort:
            rows = A[w][t]
            ir = {r[2] for r in rows}
            hit = [s for s in by_wt.get((w, t), []) if ir <= set(s["irid"])]
            gate["runs"] += 1
            if len(hit) != 1:
                gate["multi_or_none"] += 1
                continue
            s = hit[0]
            gate["irid_conflicts_used"] += s["irid_conflicts"]
            gate["dup_or_missing_error_ids"] += s["err_bad"]
            if None in s["order"] or len(set(s["order"])) != len(s["order"]):
                gate["dup_or_missing_call_ids"] += 1
                continue
            pos_of = {c: i for i, c in enumerate(s["order"])}
            pos = []
            for _, c, r, _ in rows:
                ids = s["irid"][r]
                if not 0 <= c < len(ids):
                    gate["out_of_range"] += 1
                    continue
                if ids[c] not in pos_of:
                    gate["not_in_order"] += 1
                    continue
                pos.append(pos_of[ids[c]])
            gate["units"] += len(rows)
            gate["nonmonotone"] += int(any(b <= a for a, b in zip(pos, pos[1:])))
            gate["error_ids_not_in_order"] += sum(c not in pos_of for c in s["err"])
            ep = sorted(pos_of[c] for c in s["err"] if c in pos_of)
            info[(w, t)] = {"pos": pos, "n_calls": len(s["order"]), "n_err": len(ep), "first_err": ep[0] if ep else None, "rel": s["rel"]}
    bad = {k: v for k, v in gate.items() if k not in ("runs", "units") and v}
    if bad:
        raise ValueError(f"join gate failed: {bad}")

    # 09's per-run table (uncorrected B map), descriptive agreement
    t09 = {(r["world"], r["task"]): r for r in perrun09}
    agree = {"compared": 0, "calls_equal": 0, "errors_equal": 0, "first_error_equal": 0, "missing_in_09": 0}
    for (w, t), v in info.items():
        r = t09.get((w, t))
        if r is None:
            agree["missing_in_09"] += 1
            continue
        agree["compared"] += 1
        agree["calls_equal"] += r["calls"] == v["n_calls"]
        agree["errors_equal"] += r["tool_errors"] == v["n_err"]
        agree["first_error_equal"] += ((r.get("first_error") or {}).get("call") if r.get("first_error") else None) == v["first_err"]

    # thresholds (Append 20), recomputed and gated against my sweep's stored values
    def thr(pool, r):
        s = np.sort(np.asarray(pool, np.float64), kind="stable")[::-1]
        return float(s[int(np.floor(r * len(s) + 1e-9))])

    units = {a: {t: np.array([r[3] for r in rows]) for t, rows in A[a].items()} for a in A}
    pools = {"BC": ("honest_b", "honest_c"), "B": ("honest_b",)}
    T = {}
    for p, arms in pools.items():
        for f in UNIT_F:
            T[f"unit|{f:g}|{p}"] = thr(np.concatenate([v for a in arms for v in units[a].values()]), f)
        for al in RUN_A:
            T[f"run_max|{al:g}|{p}"] = thr([v.max() for a in arms for v in units[a].values()], al)
    if set(sweep_thr) != set(T) or not all(np.isfinite(v) for v in sweep_thr.values()) or any(T[k] != v for k, v in sweep_thr.items()):
        raise ValueError(f"thresholds differ from the sweep's: keys {sorted(set(T) ^ set(sweep_thr))}, "
                         f"values {[k for k in set(T) & set(sweep_thr) if T[k] != sweep_thr[k]]}")
    primary = {k: (k.startswith("unit") and k.endswith("BC")) or (k.startswith("run_max") and k.endswith("|B")) for k in T}

    # bootstrap weights (environment strata; universe = tasks with a unit in the command file)
    universe = sorted({t for a in A for t in A[a]})
    posu = {t: i for i, t in enumerate(universe)}
    env = {c["key"]: c["env"] for c in t196["cells"]}
    rng = np.random.default_rng(SEED)
    groups = [np.array([posu[t] for t in universe if env[t] == g]) for g in sorted({env[t] for t in universe})]
    W = np.zeros((N_BOOT + 1, len(universe)))
    W[0] = 1.0
    for b in range(1, N_BOOT + 1):
        for ii in groups:
            W[b, ii] = rng.multinomial(len(ii), np.full(len(ii), 1.0 / len(ii)))

    def draws(task_list, vals):
        idx = np.array([posu[t] for t in task_list], dtype=np.int64)
        Wc = W[:, idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (Wc @ np.asarray(vals, float)) / Wc.sum(1)

    def summ(m, n):
        ok = m[1:][~np.isnan(m[1:])]
        ci = [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if len(ok) else [None, None]
        return {"point": None if np.isnan(m[0]) else float(m[0]), "ci95": ci, "n": n, "nan_draws": int(len(m) - 1 - len(ok))}

    def wmean(task_list, vals):
        return summ(draws(task_list, vals), len(task_list))

    out = {"gate": gate, "agree_09": agree, "thresholds": T, "primary": primary, "cohort_n": len(cohort), "universe_n": len(universe)}
    # check 1
    nu = {w: {t: len(A[w][t]) for t in cohort} for w in ("run1", "run2", "sim")}
    nc = {w: {t: info[(w, t)]["n_calls"] for t in cohort} for w in ("run1", "run2", "sim")}
    c1 = {}
    for name, n in (("units", nu), ("calls", nc)):
        c1[name] = {"mean": {w: wmean(cohort, [n[w][t] for t in cohort]) for w in n},
                    "sim_minus_real": wmean(cohort, [n["sim"][t] - (n["run1"][t] + n["run2"][t]) / 2 for t in cohort]),
                    "run2_minus_run1": wmean(cohort, [n["run2"][t] - n["run1"][t] for t in cohort])}
    pairs = [(t, k) for t in cohort for k in ("run1", "run2") if info[(k, t)]["n_err"] > 0]
    after = {"units_real": [], "units_sim": [], "calls_real": [], "calls_sim": []}
    for t, k in pairs:
        p = info[(k, t)]["first_err"]
        after["units_real"].append(sum(x > p for x in info[(k, t)]["pos"]))
        after["units_sim"].append(sum(x > p for x in info[("sim", t)]["pos"]))
        after["calls_real"].append(info[(k, t)]["n_calls"] - p - 1)
        after["calls_sim"].append(max(info[("sim", t)]["n_calls"] - p - 1, 0))
    pt = [t for t, _ in pairs]
    c1["after_first_error"] = {"pairs": len(pairs), "pairs_by_run": {k: sum(kk == k for _, kk in pairs) for k in ("run1", "run2")},
                               "runs_with_error": {w: sum(info[(w, t)]["n_err"] > 0 for t in cohort) for w in ("run1", "run2", "sim")},
                               **{f"{m}_sim_minus_real": wmean(pt, np.array(after[f"{m}_sim"]) - np.array(after[f"{m}_real"])) for m in ("units", "calls")},
                               **{f"{m}_{s}": wmean(pt, after[f"{m}_{s}"]) for m in ("units", "calls") for s in ("real", "sim")}}
    out["check1"] = c1
    # checks 2 and 3
    out["check2"], out["check3"] = {}, {}
    nerr = {(w, t): info[(w, t)]["n_err"] for w in ("run1", "run2") for t in cohort}
    clean = [t for t in cohort if nerr[("run1", t)] == 0 and nerr[("run2", t)] == 0]
    for name, Tv in T.items():
        share = {w: {t: float((units[w][t] > Tv).mean()) for t in cohort} for w in ("run1", "run2", "sim")}
        hit = {w: {t: float(units[w][t].max() > Tv) for t in cohort} for w in ("run1", "run2", "sim")}
        out["check2"][name] = {
            "share_mean": {w: wmean(cohort, list(share[w].values())) for w in share},
            "share_sim_minus_real": wmean(cohort, [share["sim"][t] - (share["run1"][t] + share["run2"][t]) / 2 for t in cohort]),
            "share_run2_minus_run1": wmean(cohort, [share["run2"][t] - share["run1"][t] for t in cohort]),
            "catch": {w: wmean(cohort, list(hit[w].values())) for w in hit},
            "catch_sim_minus_real": wmean(cohort, [hit["sim"][t] - (hit["run1"][t] + hit["run2"][t]) / 2 for t in cohort]),
            "catch_run2_minus_run1": wmean(cohort, [hit["run2"][t] - hit["run1"][t] for t in cohort])}
        table = {}
        for t in cohort:
            for k in ("run1", "run2"):
                cls = {(1, 0): "sim_only", (0, 1): "real_only", (1, 1): "both", (0, 0): "neither"}[(int(hit["sim"][t]), int(hit[k][t]))]
                e = nerr[(k, t)]
                b = "0" if e == 0 else "1" if e == 1 else "2-3" if e <= 3 else "4+"
                table.setdefault(cls, {}).setdefault(b, 0)
                table[cls][b] += 1
        cls_pairs = {c: [] for c in ("sim_only", "real_only", "both", "neither")}
        for t in cohort:
            for k in ("run1", "run2"):
                c = {(1, 0): "sim_only", (0, 1): "real_only", (1, 1): "both", (0, 0): "neither"}[(int(hit["sim"][t]), int(hit[k][t]))]
                cls_pairs[c].append((t, float(nerr[(k, t)] > 0)))
        dr = {c: draws([t for t, _ in v], [e for _, e in v]) for c, v in cls_pairs.items()}
        err_share = {c: summ(dr[c], len(cls_pairs[c])) for c in dr}  # an empty class reports n 0, point None, all draws NaN
        err_share["sim_only_minus_real_only"] = summ(dr["sim_only"] - dr["real_only"], None)
        so, ro = table.get("sim_only", {}), table.get("real_only", {})
        fisher = stats.fisher_exact([[sum(so.values()) - so.get("0", 0), so.get("0", 0)], [sum(ro.values()) - ro.get("0", 0), ro.get("0", 0)]])
        out["check3"][name] = {"table": table, "error_share_by_class": err_share,
                               "error_share_all_pairs": wmean([t for t in cohort for _ in (0, 1)], [float(nerr[(k, t)] > 0) for t in cohort for k in ("run1", "run2")]),
                               "fisher_p_simonly_vs_realonly": float(fisher.pvalue),
                               "error_free_tasks_catch_sim_minus_real": wmean(clean, [hit["sim"][t] - (hit["run1"][t] + hit["run2"][t]) / 2 for t in clean])}
    return out


@app.local_entrypoint()
def main(labels: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    show = lambda c, p: subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=REPO).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    lab_raw, t_raw, p_raw = Path(labels).read_bytes(), (REPO / T196[0]).read_bytes(), show(*PERRUN09[:2])
    if (hashlib.sha256(lab_raw).hexdigest() != LABELS_SHA256 or hashlib.sha256(t_raw).hexdigest() != T196[1]
            or not hashlib.sha256(p_raw).hexdigest().startswith(PERRUN09[2])):
        raise SystemExit("an input's sha256 differs from RULE.md")
    perrun = [{k: r.get(k) for k in ("world", "task", "calls", "tool_errors", "first_error")}
              for r in json.loads(p_raw)["result"]["per_run"] if r.get("world") in ("run1", "run2", "sim")]
    sweep_thr = {}
    for line in show(*SWEEP).decode().splitlines():
        r = json.loads(line)
        if r.get("readout") == "command" and r.get("strata") == "env" and r["pool"] in ("BC", "B"):
            for lv, rates in (("unit", (1e-3, 1e-2)), ("run_max", (0.01, 0.05, 0.10))):
                for x in rates:
                    if r["level"] == lv and abs(r["rate"] - x) < 1e-12:
                        k = f"{lv}|{x:g}|{r['pool']}"
                        if k in sweep_thr:
                            raise SystemExit(f"duplicate sweep row for {k}")
                        sweep_thr[k] = r["threshold"]
    res = analyze.remote(json.loads(lab_raw)["labels"], json.loads(t_raw), perrun, sweep_thr)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    head = {"git": git("rev-parse", "HEAD"), "command_npz": f"{SOL}:{CMD[0]} sha256 {CMD[1]}", "labels_sha256": LABELS_SHA256,
            "t196_sha256": T196[1], "perrun09": f"{PERRUN09[1]}@{PERRUN09[0]} sha256 {hashlib.sha256(p_raw).hexdigest()}",
            "sweep_thresholds_from": f"{SWEEP[1]}@{SWEEP[0]}", "n_boot": N_BOOT, "seed": SEED}
    p = HERE / "results" / f"runlength_{stamp}.json"
    p.parent.mkdir(exist_ok=True)
    with p.open("x") as fh:
        json.dump(head | res, fh, indent=1)
    num = lambda x, s=1, fm="+.2f": "n/a" if x is None else format(s * x, fm)  # noqa: E731
    f = lambda d, s=1: f"{num(d['point'], s)} [{num(d['ci95'][0], s)}, {num(d['ci95'][1], s)}] n{d['n']}"  # noqa: E731
    print(f"wrote {p.name} sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}")
    print("join gate:", json.dumps(res["gate"]), "| 09 agreement:", json.dumps(res["agree_09"]), "| cohort", res["cohort_n"], "universe", res["universe_n"])
    c1 = res["check1"]
    for m in ("units", "calls"):
        print(f"check1 {m:<5} per run: " + " ".join(f"{w} {num(c1[m]['mean'][w]['point'], 1, '.2f')}" for w in ("run1", "run2", "sim"))
              + f" | sim-real {f(c1[m]['sim_minus_real'])} | r2-r1 {f(c1[m]['run2_minus_run1'])}")
    a = c1["after_first_error"]
    print(f"check1 after first real error: pairs {a['pairs']} {a['pairs_by_run']} runs with error {a['runs_with_error']} | units real "
          f"{num(a['units_real']['point'], 1, '.2f')} sim {num(a['units_sim']['point'], 1, '.2f')} sim-real {f(a['units_sim_minus_real'])} | calls sim-real {f(a['calls_sim_minus_real'])}")
    for name, v in res["check2"].items():
        tag = "PRIMARY" if res["primary"][name] else "sens"
        print(f"check2 {name:<18}{tag:<8} share% " + " ".join(f"{w} {num(v['share_mean'][w]['point'], 100, '.2f')}" for w in ("run1", "run2", "sim"))
              + f" | share sim-real {f(v['share_sim_minus_real'], 100)} r2-r1 {f(v['share_run2_minus_run1'], 100)}"
              + f" | catch sim-real {f(v['catch_sim_minus_real'], 100)} r2-r1 {f(v['catch_run2_minus_run1'], 100)}")
    for name, v in res["check3"].items():
        tag = "PRIMARY" if res["primary"][name] else "sens"
        es = {c: f(x, 100) for c, x in v["error_share_by_class"].items()}
        print(f"check3 {name:<18}{tag:<8} table {json.dumps(v['table'], sort_keys=True)} | real-error share % by class {es} all "
              f"{f(v['error_share_all_pairs'], 100)} | Fisher p {v['fisher_p_simonly_vs_realonly']:.3g} | error-free tasks catch sim-real "
              f"{f(v['error_free_tasks_catch_sim_minus_real'], 100)}")
