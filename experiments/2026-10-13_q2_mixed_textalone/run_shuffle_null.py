"""The grouped label-shuffle null for the text share (PLAN.md Amendment 4; POST HOC). One Modal CPU function.

It rebuilds each record's run scores from its stored score arrays exactly as analyze_mixed.analyze() does, checks them
against the record (the gates), then permutes the simulated/e2 labels within each main task (2,000 permutations, seed
20261014; one permutation relabels both readings and every cell and view) and reads the text share against the null's
center instead of 0.5. "Text" means the output plus its own call, no history (NOTE_labels_2026-10-14.md).

Inputs are the two result records (S3 per results/s3_manifest_v1.json; also in git at their as-run commits), passed by
path and checked against their pinned sha256. Their score arrays are read from the job's Volume and checked against the
records' sha256. The laptop only launches the job and writes the record it returns.

    PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.2' --with numpy modal run \
      experiments/2026-10-13_q2_mixed_textalone/run_shuffle_null.py --asis <as-is record> --fx <fix-full record>
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy  # noqa: F401  (imported so a client without numpy fails before launch)

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q2-mixta_null_kill-2026-11-13"
MYW = "probe-shift_aswin-surya_sub-q2-mixta_work_kill-2026-11-13"
PIN = {"asis": "347950cbdef8", "fx": "3240220a31cd"}  # sha256 prefixes; the full digests come from results/s3_manifest_v1.json
FX = "sim-attack-fx-full"
NPERM, PSEED, TOL = 2000, 20261014, 1e-9
PRIMARY = {"asis": "c | g3 layer per fold", "fx": "c_ff | g3 layer per fold"}

app = modal.App(APP)
myw = modal.Volume.from_name(MYW, version=2).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy").add_local_python_source("probe_shift")
         .add_local_file(str(HERE / "analyze_mixed.py"), "/root/analyze_mixed.py"))


@app.function(image=image, volumes={"/my": myw}, cpu=8, memory=16384, timeout=900, retries=0)
def null_all(recs: dict) -> str:
    import sys

    import numpy as np
    sys.path.insert(0, "/root")
    import analyze_mixed as am
    out = {}
    for name, rec in recs.items():
        sc = rec["scores"]; p = Path("/my") / sc["path"]
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != sc["sha256"]:
            return json.dumps({"gate failed": f"{name}: scores sha256 differs from the record"})
        z = dict(np.load(p))
        U = [u | {"arm": am.SIM} if u["arm"] == FX else u for u in rec["units"]]
        n = len(U); arm = np.array([u["arm"] for u in U]); fold = np.array([u["fold"] for u in U]); ru = z["row_unit"]
        # the analysis's task draws, verbatim
        cells = sorted({U[j]["main_task"] for j in range(n) if arm[j] == am.E2})
        g = np.random.default_rng(am.SEED); dc = np.array([np.bincount(g.integers(0, len(cells), len(cells)), minlength=len(cells)) for _ in range(am.B)])
        W = np.zeros((am.B + 1, n)); ci = {t: i for i, t in enumerate(cells)}
        for j in range(n):
            W[0, j] = 1.0; W[1:, j] = dc[:, ci[U[j]["main_task"]]]
        nrow = np.bincount(ru, minlength=n)
        ip, iq = np.where(arm == am.SIM)[0], np.where(arm == am.E2)[0]
        pool = np.concatenate([ip, iq]); lab = np.concatenate([np.ones(len(ip), bool), np.zeros(len(iq), bool)])
        gid = np.array([ci[U[j]["main_task"]] for j in pool])
        sims = np.bincount(gid[lab], minlength=len(cells)); e2s = np.bincount(gid[~lab], minlength=len(cells))
        groups = {"main tasks": len(cells), "min simulated per task": int(sims.min()), "min e2 per task": int(e2s.min()),
                  "tasks with equal simulated and e2 counts": int((sims == e2s).sum()), "max runs per task": int((sims + e2s).max())}
        if sims.min() < 1 or e2s.min() < 1:
            return json.dumps({"gate failed": f"{name}: a main task lacks a simulated or an e2 run", "groups": groups})
        # one set of within-task permutations per record, shared by both readings and every cell and view
        pr = np.random.default_rng(PSEED); order0 = np.argsort(gid, kind="stable")
        perms = np.empty((NPERM, len(pool)), bool)
        for k in range(NPERM):
            o = order0[np.lexsort((pr.random(len(pool))[order0], gid[order0]))]
            perms[k, order0] = lab[o]
        kept = all((np.bincount(gid[perms[k]], minlength=len(cells)) == sims).all() for k in range(NPERM))
        moved = float((perms != lab).mean())
        groups |= {"every permutation keeps each task's simulated count": bool(kept), "share of labels moved, mean": moved}
        if not kept or moved == 0.0:
            return json.dumps({"gate failed": f"{name}: the permutations are not within-task shuffles", "groups": groups})
        cv = {c: r for c, r in rec["analysis"]["cells"].items()}
        res_c = {}
        for c, recorded in cv.items():
            if recorded["primary"] != "run AUROC share":
                return json.dumps({"gate failed": f"{name} {c}: recorded primary is {recorded['primary']}"})
            c0, vname = c.split(" | ")
            view = rec["convention"]["views"][vname]
            runl = {}
            for reading in ("history", "text"):  # analyze_mixed.analyze(), verbatim
                L = np.full((len(ru), 5, 5), np.nan)
                for f in range(5):
                    L[:, f, :] = z[f"{reading}__{c0}__L{view[f]}"][:, f, :]
                Sfv = np.stack([np.bincount(ru, np.nan_to_num(L[:, k // 5, k % 5]), minlength=n) / nrow for k in range(25)], 1).reshape(n, 5, 5)
                runl[reading] = np.array([np.mean([Sfv[j, fold[j], v] for v in range(5) if v != fold[j]]) for j in range(n)])
            A = {r: np.array([am.wauc(runl[r][ip], W[b, ip], runl[r][iq], W[b, iq]) for b in range(am.B + 1)]) for r in runl}
            ok = np.abs(A["history"] - 0.5) >= am.MIN_EXCESS
            share = am.pt_ci(np.where(ok, (A["text"] - 0.5) / np.where(ok, A["history"] - 0.5, 1), np.nan), need=ok)
            rs = recorded["text share (primary)"]
            gate = {"A_text point |diff|": abs(share["point"] - rs["point"]),
                    "A_text 95% |diff|": max(abs(a - b) for a, b in zip(share["95%"], rs["95%"])),
                    "run AUROC history |diff|": abs(A["history"][0] - recorded["history"]["AUROC per run: sim vs e2"]["point"]),
                    "run AUROC text |diff|": abs(A["text"][0] - recorded["text"]["AUROC per run: sim vs e2"]["point"])}
            if max(gate.values()) > TOL:
                return json.dumps({"gate failed": f"{name} {c}", "gate": gate})
            ones = np.ones(len(pool))
            nulls = {r: np.array([am.wauc(runl[r][pool][perms[k]], ones[perms[k]], runl[r][pool][~perms[k]], ones[~perms[k]]) for k in range(NPERM)]) for r in runl}
            cen = {r: float(nulls[r].mean()) for r in runl}
            okn = np.abs(A["history"] - cen["history"]) >= am.MIN_EXCESS
            star = am.pt_ci(np.where(okn, (A["text"] - cen["text"]) / np.where(okn, A["history"] - cen["history"], 1), np.nan), need=okn)
            p_ = star["point"]; lo, hi = star["95%"]
            rd = None if p_ is None else "text" if p_ > 0.6 else "history" if p_ < 0.3 else "mixed"
            firm = lo is not None and ((rd == "text" and lo > 0.6) or (rd == "history" and hi < 0.3) or (rd == "mixed" and lo >= 0.3 and hi <= 0.6))
            reading_star = f"{rd}{' (firm)' if firm else ''}"
            res_c[c] = {"role": "primary" if c == PRIMARY[name] else "beside",
                        "recorded": {"text share (vs 0.5)": rs, "reading": recorded["reading"]}, "gate": gate,
                        "null": {r: {"observed run AUROC": float(A[r][0]), "null center": cen[r],
                                     "null 95%": [float(np.percentile(nulls[r], 2.5)), float(np.percentile(nulls[r], 97.5))],
                                     "null sd": float(nulls[r].std()),
                                     "p (null >= observed)": float((1 + (nulls[r] >= A[r][0]).sum()) / (NPERM + 1))} for r in runl},
                        "text share vs the null": star, "reading vs the null": reading_star,
                        "share change (point)": float(p_ - rs["point"]), "reading changed": reading_star != recorded["reading"]}
        out[name] = {"groups": groups, "cells": res_c}
    return json.dumps(out)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a, check=True):
    return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True, check=check)


@app.local_entrypoint()
def main(asis: str, fx: str):
    for f in (Path(__file__), HERE / "PLAN.md", HERE / "analyze_mixed.py"):
        if git("ls-files", "--error-unmatch", str(f), check=False).returncode or git("diff", "--quiet", "HEAD", "--", str(f), check=False).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    man = json.loads((HERE / "results" / "s3_manifest_v1.json").read_text())["files"]
    paths = {"asis": asis, "fx": fx}
    keys = {"asis": "results/mixed_textalone_POSTHOC_2026-10-13_173607.json", "fx": "results/mixed_textalone_fx_POSTHOC_2026-10-13_182320.json"}
    recs = {}
    for k, p in paths.items():
        want = man[keys[k]]["sha256"]
        if not want.startswith(PIN[k]) or sha(p) != want:
            raise SystemExit(f"{k}: the record differs from the manifest's sha256")
        recs[k] = json.loads(Path(p).read_text())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = json.loads(null_all.remote(recs))
    if "gate failed" in res:
        print(json.dumps(res, indent=1))
        raise SystemExit("gate failed: no result written")
    rec = {"POST HOC": True, "status": "preliminary until Aswin reviews", "plan": "PLAN.md Amendment 4", "app": APP,
           "git": git("rev-parse", "HEAD").stdout.strip(), "permutations": NPERM, "permutation seed": PSEED,
           "inputs": {k: {"path": keys[k], "sha256": man[keys[k]]["sha256"], "scores": recs[k]["scores"]} for k in keys},
           "results": res}
    with open(dest := HERE / "results" / f"shuffle_null_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    for k, r in res.items():
        for c, x in r["cells"].items():
            print(f"{k:4s} {c:24s} {x['role']:7s} recorded {x['recorded']['text share (vs 0.5)']['point']:+.3f} {x['recorded']['reading']!r:16s} "
                  f"vs null {x['text share vs the null']['point']:+.3f} {x['text share vs the null']['95%']} {x['reading vs the null']!r}"
                  f"{'  READING CHANGED' if x['reading changed'] else ''}")
    print(dest)
    if any(x["reading changed"] for r in res.values() for x in r["cells"].values()):
        raise SystemExit("a reading changed: report to 0d before any reply")
