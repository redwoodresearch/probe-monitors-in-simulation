"""RULE.md here: a viewer of real-vs-sim disagreements for the headline probe (Sol (a), reply mean-pooled, alpha 1%, v6
primary calibration, with-strategy estimation tasks), for Aswin to read. Modal CPU. PRELIMINARY.

  phase schema: field names, dtypes and counts of the score file and the staged files (no values, no text)
  phase select: caught flags, the v6 reproduction gate, categories and the seeded selection (keys only)
  phase render: the selected runs' staged ids decoded in-container into ONE self-contained HTML on the work Volume, plus an
                index JSON (keys, labels, scores, counts); only sha256 and sizes come back

  EX_IN=<dir with the pinned inputs> PYTHONPATH=src $MODAL run --detach \\
      experiments/2026-10-15_examples_viewer_09/run_examples.py --phase select
"""

import hashlib
import html
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
IN = Path(os.environ.get("EX_IN", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-convaudit_examples_kill-2026-11-14"
PINS = {
    "asis_scores_142228.json": "59c051a4ff9cc40811745a119251224f031cdaf479b8d3fb909c1059ab94da31",
    "xerr_v6_2026-10-14_161122.json": "e789144b62f71e0fb168dc3106656c2b5f29aadafe12df0f86dd1f74441d7645",
    "strategy_bits_t196.json": "9a5a4e7120037f929e85d1efd9c7cf3ba35756349d8135f15f4cfe1e8775cfc0",
    "t196.json": "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb",
    "correctedb_manifest_v1.json": "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc",  # one epoch per run
}
ROWS_NPZ = "/sol/xfer/asis_2026-10-14_135701/rows_2026-10-14_142228/a_tight__avg_top2.npz"  # per-row reply scores (Sol (a) tight)
ROW = "R_reply_meanpooled|mechanical|alpha=0.01"
ALPHA, SEED, ARMS = 0.01, 20261015, ("run1", "run2", "sim")
POOLS = {"B": ("honest_b",), "BuC": ("honest_b", "honest_c"), "AuB": ("honest_a", "honest_b")}

app = modal.App(APP)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("transformers==5.19.0", "jinja2", "numpy==2.3.3")
    .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
    .add_local_python_source("probe_shift")
)
if IN.exists():
    for name in PINS:
        image = image.add_local_file(IN / name, f"/root/in/{name}")
VOLS = {
    "/sol": modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2).read_only(),
    "/b": modal.Volume.from_name("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", version=2).read_only(),
    "/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only(),
    "/work": modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2),
}


class GateError(Exception):
    pass


def load(name):
    raw = Path(f"/root/in/{name}").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINS[name]:
        raise GateError(f"pin {name}")
    return json.loads(raw)


def caught_flags():
    """Sol (a) reply z-scores and caught flags as v6's primary row computes them, gated against the v6 record."""
    import numpy as np

    asis, v6 = load("asis_scores_142228.json"), load("xerr_v6_2026-10-14_161122.json")["result"]
    row = v6["results"][ROW]
    pk = row["pools"]["sol_a"]
    D = asis["run_scores"]["avg_top2"]["a_tight"]
    fold_of = {}
    for k, f in asis["run_folds"].items():
        if fold_of.setdefault(k.split("|", 1)[1], f) != f:
            raise GateError("fold map: a task has two folds")
    mu, sd, nf = {}, {}, {}
    for f in sorted(set(fold_of.values())):
        v = np.array(
            [s for k, s in D.items() if k.split("|", 1)[0] in POOLS[pk] and fold_of.get(k.split("|", 1)[1]) == f]
        )
        mu[f], sd[f], nf[f] = float(v.mean()), float(v.std()), len(v)
    z = {
        k: (s - mu[fold_of[k.split("|", 1)[1]]]) / sd[fold_of[k.split("|", 1)[1]]]
        for k, s in D.items()
        if k.split("|", 1)[1] in fold_of
    }
    pool = np.sort(np.array([v for k, v in z.items() if k.split("|", 1)[0] in POOLS[pk]]), kind="mergesort")[::-1]
    kk = int(np.floor(ALPHA * len(pool) + 1e-9))
    T = float(pool[min(kk, len(pool) - 1)])
    us = v6["universe_and_split"]
    cohort = v6["cohort"]["tasks"]
    c_est = [t for t in cohort if t in set(us["estimation_half"])]
    hit = {w: {t: bool(z[f"{w}|{t}"] > T) for t in cohort} for w in ARMS}
    rec_thr, rec_c = row["thresholds"]["sol_a"], row["catches_percent"]["sol_a"]
    mine_c = {
        w: {
            "all_cohort": 100 * float(np.mean([hit[w][t] for t in cohort])),
            "estimation_half": 100 * float(np.mean([hit[w][t] for t in c_est])),
        }
        for w in ARMS
    }
    gate = {
        "pool": pk == rec_thr["pool"],
        "threshold": T == rec_thr["threshold"] and len(pool) == rec_thr["n"] and kk == rec_thr["k"],
        "catches": all(mine_c[w][h] == rec_c[w][h] for w in ARMS for h in ("all_cohort", "estimation_half")),
        "n_estimation_tasks": len(c_est),
    }
    sm = row["standardization_sol"]
    folds = sorted(mu)
    gate["moments"] = (
        sorted(int(f) for f in sm["mu"]) == folds
        and sorted(int(f) for f in sm["sd"]) == folds
        and sorted(int(f) for f in sm["n"]) == folds
        and all(mu[f] == sm["mu"][str(f)] and sd[f] == sm["sd"][str(f)] and nf[f] == sm["n"][str(f)] for f in folds)
    )
    if not (gate["pool"] and gate["threshold"] and gate["catches"] and gate["moments"] and len(c_est) == 90):
        raise GateError(f"v6 reproduction gate failed: {gate}")
    return {
        "z": z,
        "raw": D,
        "T": T,
        "mu": mu,
        "sd": sd,
        "fold_of": fold_of,
        "hit": hit,
        "c_est": c_est,
        "pool": pk,
        "gate": gate,
    }


def categories(cf):
    sb = load("strategy_bits_t196.json")["tasks"]
    tasks = sorted(t for t in cf["c_est"] if sb[t]["has_strategy"])
    cat = {
        "A_sim_caught_real_missed": [],
        "B_real_caught_sim_missed": [],
        "R_real_real_disagree": [],
        "G_all_agree": [],
    }
    for t in tasks:
        r1, r2, s = (cf["hit"][w][t] for w in ARMS)
        if r1 != r2:
            cat["R_real_real_disagree"].append(t)
        elif s and not r1:
            cat["A_sim_caught_real_missed"].append(t)
        elif r1 and not s:
            cat["B_real_caught_sim_missed"].append(t)
        else:
            cat["G_all_agree"].append(t)
    rng = random.Random(SEED)
    A, B = cat["A_sim_caught_real_missed"], cat["B_real_caught_sim_missed"]
    sa, sb_ = rng.sample(A, min(3, len(A))), rng.sample(B, min(3, len(B)))
    short = 6 - len(sa) - len(sb_)
    if short > 0 and len(sa) < 3:
        sb_ += rng.sample([t for t in B if t not in sb_], min(short, len(B) - len(sb_)))
    elif short > 0 and len(sb_) < 3:
        sa += rng.sample([t for t in A if t not in sa], min(short, len(A) - len(sa)))
    sg = rng.sample(cat["G_all_agree"], min(6, len(cat["G_all_agree"])))
    return tasks, cat, {"A": sa, "B": sb_, "G": sg}


def staged_epochs(want: set) -> dict:
    """(world, task) -> (epoch key, ids) via the pinned corrected-B manifest (one epoch per run; changed epochs read their
    q4f re-stagings); each staged file is verified against the manifest's sha256 before any line is parsed."""
    man = load("correctedb_manifest_v1.json")
    ent = {}
    for e in man["epochs"]:
        if (e["world"], e["task"]) in want:
            if (e["world"], e["task"]) in ent:
                raise GateError("manifest: two epochs for one selected run")
            ent[(e["world"], e["task"])] = e
    if set(ent) != want:
        raise GateError(f"manifest: {len(want - set(ent))} selected runs have no epoch")
    by_file = {}
    for wt, e in ent.items():
        by_file.setdefault((e["staged_inputs_file"], e["staged_inputs_sha256"]), {})[e["key"]] = wt
    out = {}
    for (fpath, fsha), keyed in sorted(by_file.items()):
        raw = Path(f"/b/{fpath}").read_bytes()
        if hashlib.sha256(raw).hexdigest() != fsha:
            raise GateError("a staged inputs file differs from the manifest's sha256")
        for line in raw.splitlines():
            rec = json.loads(line)
            if rec.get("key") in keyed:
                wt = keyed[rec["key"]]
                if wt in out:
                    raise GateError("a selected epoch key appears twice in its staged file")
                out[wt] = (rec["key"], fpath, rec["ids"])
    if set(out) != want:
        raise GateError(f"{len(want - set(out))} selected epochs not found in their staged files")
    return out


@app.function(image=image, cpu=4, memory=32768, timeout=1800, retries=0, volumes=VOLS)
def job(phase: str, sel_text: str = "", stamp: str = "", rule_commit: str = "") -> dict:
    import numpy as np

    try:
        asis = load("asis_scores_142228.json")
        rf = asis["rows_file"]["files"]["a_tight__avg_top2"]
        rows_raw = Path(ROWS_NPZ).read_bytes()
        if "/sol/" + rf["path"] != ROWS_NPZ or hashlib.sha256(rows_raw).hexdigest() != rf["sha256"]:
            raise GateError("the row-score file differs from asis rows_file pin")
        import io

        zr = np.load(io.BytesIO(rows_raw), allow_pickle=False)
        if len(zr["key"]) != rf["n_rows"]:
            raise GateError("the row-score file's row count differs from its pin")
        if phase == "schema":
            return {"npz": {f: [str(zr[f].dtype), list(zr[f].shape)] for f in zr.files}}
        cf = caught_flags()
        tasks, cat, sel = categories(cf)
        if phase == "select":
            return {
                "cell": {
                    "probe": "sol_a (tight)",
                    "readout": "reply mean-pooled (avg_top2)",
                    "alpha": ALPHA,
                    "pool": cf["pool"],
                    "threshold_z": cf["T"],
                    "row": ROW,
                },
                "gate_v6": cf["gate"],
                "n_tasks_with_strategy_estimation": len(tasks),
                "counts": {k: len(v) for k, v in cat.items()},
                "categories": cat,
                "seed": SEED,
                "selected": sel,
                "flags": {t: {w: cf["hit"][w][t] for w in ARMS} for t in tasks},
            }
        # ---- render
        sel_rec = json.loads(sel_text)["result"]
        if sel_rec["selected"] != sel or sel_rec["seed"] != SEED:
            raise GateError("the committed selection differs from a recomputation")
        chosen = [("A", t) for t in sel["A"]] + [("B", t) for t in sel["B"]] + [("G", t) for t in sel["G"]]
        want = {(w, t) for _, t in chosen for w in ARMS}
        eps = staged_epochs(want)
        rk = [str(x) for x in zr["key"]]
        sc = np.asarray(zr["score"], float)
        num = [
            f
            for f in zr.files
            if f not in ("key", "score") and zr[f].ndim == 1 and len(zr[f]) == len(rk) and zr[f].dtype.kind in "iuf"
        ]  # numeric position fields only
        rows_of = {}
        for i, k in enumerate(rk):
            rows_of.setdefault(k, []).append(i)
        empty = [f"{w}|{t}" for (w, t) in want if not rows_of.get(f"{w}|{t}")]
        if empty:
            raise GateError(f"{len(empty)} selected runs have no reply rows")
        from transformers import AutoTokenizer

        from probe_shift.models import QWEN35_27B

        tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"], local_files_only=True)
        index, parts = [], []
        names = {
            "A": "sim caught, both real runs missed",
            "B": "both real runs caught, sim missed",
            "G": "all three agree",
        }
        for c, t in chosen:
            cols, ent = [], {"task": t, "category": c, "runs": {}}
            f = cf["fold_of"][t]
            for w in ARMS:
                k = f"{w}|{t}"
                ri = rows_of[k]
                rows = [
                    {
                        "row": j,
                        "raw": float(sc[i]),
                        "z": (float(sc[i]) - cf["mu"][f]) / cf["sd"][f],
                        **{e: zr[e][i].item() for e in num},
                    }
                    for j, i in enumerate(ri)
                ]
                _, fpath, ids = eps[(w, t)]
                ent["runs"][w] = {
                    "caught": cf["hit"][w][t],
                    "run_raw": float(cf["raw"][k]),
                    "run_z": float(cf["z"][k]),
                    "n_rows": len(rows),
                    "staged_file": fpath,
                    "tokens": len(ids),
                }
                tbl = "".join(
                    f"<tr><td>{r['row']}</td><td>{r['raw']:.4f}</td><td>{r['z']:+.3f}</td>"
                    + "".join(f"<td>{r[e]}</td>" for e in num)
                    + "</tr>"
                    for r in rows
                )
                cols.append(
                    f"<div class='col'><h3>{w}: {'CAUGHT' if cf['hit'][w][t] else 'missed'}</h3>"
                    f"<p>run score {cf['raw'][k]:.4f} raw; run-calibrated z {cf['z'][k]:+.3f} vs threshold z "
                    f"{cf['T']:+.3f}</p><details><summary>per-row reply scores ({len(rows)} rows; z uses the run-level "
                    f"fold moments, not a row-level calibration)</summary><table><tr><th>row</th><th>raw</th>"
                    f"<th>z</th>{''.join(f'<th>{e}</th>' for e in num)}</tr>{tbl}</table></details>"
                    f"<details><summary>reader-layout text the probe read ({len(ids):,} tokens; {html.escape(fpath)})"
                    f"</summary><pre>{html.escape(tok.decode(ids, skip_special_tokens=False))}</pre></details></div>"
                )
            index.append(ent)
            parts.append(f"<section><h2>{t} — {names[c]}</h2><div class='row'>{''.join(cols)}</div></section>")
        head = (
            "<!doctype html><html><head><meta charset='utf-8'><title>Real vs sim examples</title><style>"
            ":root{--bg:#fff;--fg:#111;--mut:#555;--bd:#ccc}"
            "@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--mut:#aaa;--bd:#444}}"
            "body{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;margin:16px}"
            ".row{display:flex;gap:12px}.col{flex:1;min-width:0;border:1px solid var(--bd);padding:8px}"
            "pre{white-space:pre-wrap;word-break:break-word;font:12px/1.35 ui-monospace,monospace}"
            "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid var(--bd);padding:2px 6px}"
            "p{color:var(--mut)}</style></head><body>"
            "<h1>Real vs sim: where the headline probe disagrees</h1>"
            f"<p>Sol (a), reply mean-pooled, alpha 1%, v6 primary calibration (pool {cf['pool']}, threshold z "
            f"{cf['T']:+.4f}); v6 reproduction gate {html.escape(json.dumps(cf['gate']))}; with-strategy estimation "
            f"tasks: {len(tasks)}; counts {html.escape(json.dumps({k: len(v) for k, v in cat.items()}))}; seed {SEED}; "
            f"RULE commit {html.escape(rule_commit)}. Text = the corrected-B staged ids the probe read, decoded "
            "in-container. PRELIMINARY; examples, not a statistic.</p>"
        )
        page = (head + "".join(parts) + "</body></html>").encode()
        d = Path(f"/work/examples_viewer/{stamp}")
        d.mkdir(parents=True, exist_ok=False)
        (d / "viewer.html").write_bytes(page)
        idx = json.dumps(
            {"cell_threshold_z": cf["T"], "seed": SEED, "rule_commit": rule_commit, "index": index}, default=float
        ).encode()
        (d / "index.json").write_bytes(idx)
        VOLS["/work"].commit()
        return {
            "html": {"path": str(d / "viewer.html"), "sha256": hashlib.sha256(page).hexdigest(), "bytes": len(page)},
            "index_json": json.loads(idx),
            "index_sha256": hashlib.sha256(idx).hexdigest(),
            "row_fields_shown": num,
        }
    except GateError as e:
        return {"fatal": "GateError", "gate": str(e)}
    except Exception as e:  # noqa: BLE001 -- the message may quote data, so only its type leaves the container
        return {"fatal": type(e).__name__}


@app.local_entrypoint()
def main(phase: str = "schema", selection: str = "", selection_sha256: str = ""):
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    for f in (HERE / "RULE.md", Path(__file__).resolve()):  # the rule and the job must be committed before any phase
        rel = str(f.relative_to(REPO))
        head = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, cwd=REPO, check=False)
        if head.returncode or head.stdout != f.read_bytes():
            raise SystemExit(f"{rel} is not committed at HEAD")
    rule_commit = git("log", "-1", "--format=%H", "--", str((HERE / "RULE.md").relative_to(REPO)))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    sel_text = ""
    if phase == "render":
        raw = Path(selection).read_bytes()
        rel = str(Path(selection).resolve().relative_to(REPO))
        head = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, cwd=REPO, check=False)
        if head.returncode or head.stdout != raw or hashlib.sha256(raw).hexdigest() != selection_sha256:
            raise SystemExit("the selection record is not committed at HEAD byte for byte, or differs from its sha256")
        sel_text = raw.decode()
    res = job.remote(phase, sel_text, stamp, rule_commit)
    dest = HERE / "results" / f"examples_{phase}_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(
        json.dumps({"git": git("rev-parse", "HEAD"), "stamp": stamp, "phase": phase, "result": res}, indent=1)
    )
    print(json.dumps({k: res.get(k) for k in ("fatal", "gate", "gate_v6", "counts", "html", "index_sha256")}, indent=1))
    print("wrote", dest.name)
