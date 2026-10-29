"""N1 Part 1, stages 3 and 4 (README sections 1–2): the prerequisites and the comparison with the concept probe.

Modal CPU scores every dialogue in the extracted runs that is not train or dev, with four probes:
- N1 content (per token at layer 24) and N1 turn start (layer 24), from N1's fits;
- the concept probe's content probe (fit_2026-10-11_015755, per token at layer 24) and turn-start probe
  (fit_2026-10-11_020707, layer 16).
A dialogue's score is its rows' mean logit. Only ids, labels, splits, mechanisms, tasks and scores return. Then,
locally, it computes:
- stage 3, from N1's content fit file: the held-out AUROC above its fixed-layer null p95, the median held-out pair
  change's task-bootstrap 2.5th percentile above 0, and convergence;
- stage 4: N1 minus the concept probe, in AUROC, on identical pairs (2,000 task-bootstrap draws, seed 20261011). It
  passes if, on the honest held-out implied pairs, the difference is at least +0.05 with its interval above 0, and on
  the attack panel's implied pairs its interval is above 0;
- reported, deciding nothing: per family, the phrasings, the synthetic 2×2 and turn start.

  PYTHONPATH=src uv run modal deploy experiments/2026-10-11_la_format_sim_probe/analyze_la_compare.py
  PYTHONPATH=src uv run --with scikit-learn python experiments/2026-10-11_la_format_sim_probe/analyze_la_compare.py launch results/extract_<stamp>.json ../2026-10-10_eval_awareness_probe/results/fit_<N1 C>.json ../2026-10-10_eval_awareness_probe/results/fit_<N1 T>.json
  ... collect <launch stamp>
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-n1_compare_kill-2026-11-10"
VOL = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09"
FITS = HERE.parent / "2026-10-10_eval_awareness_probe" / "results"
CONCEPT = {"C": ("conceptC_2026-10-11_015307/fits/fit_2026-10-11_015755", "fit_2026-10-11_015755.json"),
           "T": ("conceptT_2026-10-11_015307/fits/fit_2026-10-11_020707", "fit_2026-10-11_020707.json")}
B, SEED, LAYER, FLOOR, ACCEPT_GRAD = 2000, 20261011, 24, 0.05, 1e-5
CAVEAT = "Reading an inserted cue isn't the same as the model noticing it's in a simulation."

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, volumes={"/acts": vol}, cpu=4, memory=32768, timeout=1800, retries=0)
def score(runs: dict, probes: dict, expect_sha256: str) -> list[dict]:
    """runs: {"C": run dir, "T": run dir}; probes: name -> (readout C or T, probes.json path, sha256, readout key)."""
    import numpy as np

    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != expect_sha256:  # the deployed scorer is the launching commit's
        raise ValueError("the deployed analyze_la_compare.py is not the launching commit's: redeploy")
    vol.reload()
    P = {}
    for name, (v, path, sha, key) in probes.items():
        blob = (Path("/acts") / path).read_bytes()
        if hashlib.sha256(blob).hexdigest() != sha:
            raise ValueError(f"{name}: probes.json does not match its sha256")
        P[name] = (v, json.loads(blob)["probes"][key])
    meta = {v: json.loads((Path("/acts") / runs[v] / "meta.json").read_text()) for v in ("C", "T")}
    if [m["id"] for m in meta["C"]] != [m["id"] for m in meta["T"]]:
        raise ValueError("the content and turn-start runs list different dialogues")
    keep = [i for i, m in enumerate(meta["C"]) if m["split"] not in ("train", "dev")]
    out = [{k: meta["C"][i][k] for k in ("id", "pair", "label", "split", "family", "mechanism", "position", "task")} for i in keep]
    for name, (v, p) in P.items():
        X = np.load(Path("/acts") / runs[v] / f"L{p['layer']}.npy", mmap_mode=None)
        w, mu, sd = (np.array(p[k], dtype=np.float64) for k in ("w", "mu", "sd"))
        for o, i in zip(out, keep):
            m = meta[v][i]
            o[name] = float((((X[m["row_start"]:m["row_start"] + m["n_rows"]].astype(np.float64) - mu) / sd) @ w).mean())
    return out


def auc_boot(rows, a, b):
    """AUROC of score a and b, their difference, and its task-bootstrap interval, on the same dialogues."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y = np.array([r["label"] for r in rows])
    s = {k: np.array([r[k] for r in rows]) for k in (a, b)}
    tasks = sorted({r["task"] for r in rows})
    idx = {t: np.flatnonzero([r["task"] == t for r in rows]) for t in tasks}
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(B):
        ii = np.concatenate([idx[t] for t in rng.choice(tasks, len(tasks))])
        d.append(roc_auc_score(y[ii], s[a][ii]) - roc_auc_score(y[ii], s[b][ii]) if len(set(y[ii])) == 2 else np.nan)
    d = np.array(d)
    if np.isnan(d).any():
        raise ValueError("a bootstrap draw had one label only")
    A, Bv = roc_auc_score(y, s[a]), roc_auc_score(y, s[b])
    lo = float(np.percentile(d, 2.5))
    return {"unrounded (gates use these)": {"difference": A - Bv, "lower 95%": lo}, "pairs": len(rows) // 2, "tasks": len(tasks), f"AUROC {a}": round(A, 4), f"AUROC {b}": round(Bv, 4),
            "difference": round(A - Bv, 4), "difference 95%": [round(float(q), 4) for q in np.percentile(d, [2.5, 97.5])]}


def analyze(rows, fit_c, ext):
    import numpy as np

    out = {"caveat": CAVEAT, "extract": ext, "fit (N1 content)": fit_c["fit_id"],
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()}
    s = fit_c["summary"]["per_token"]
    held = fit_c["heldout_scores_at_chosen_layer"]["per_token"]
    task = {r["id"]: r["task"] for r in rows}
    ch = {}
    for x in held:
        ch.setdefault(x["id"].rsplit("_", 1)[0], {})[x["label"]] = x["score"]
    pid = sorted(ch)
    change = np.array([ch[p][1] - ch[p][0] for p in pid])
    tasks = sorted({task[f"{p}_1"] for p in pid})
    groups = [change[[task[f"{p}_1"] == t for p in pid]] for t in tasks]
    boot = [np.median(np.concatenate([groups[t] for t in d])) for d in np.random.default_rng(SEED).integers(0, len(tasks), (B, len(tasks)))]
    out["stage 3"] = {"layer": s["layer"], "held-out AUROC": round(s["heldout_auroc"], 4), "fixed-layer null p95": round(s["null_heldout_p95"], 4),
                      "reference (median held-out pair change)": round(float(np.median(change)), 4),
                      "reference 95%": [round(float(q), 4) for q in np.percentile(boot, [2.5, 97.5])],
                      "max gradient": fit_c["max_grad (all fits)"]}
    out["stage 3 passes"] = bool(s["layer"] == LAYER and s["heldout_auroc"] > s["null_heldout_p95"] and np.percentile(boot, 2.5) > 0
                                 and fit_c["max_grad (all fits)"] <= ACCEPT_GRAD)
    sel = lambda split, fam=None: [r for r in rows if r["split"] == split and (fam is None or r["mechanism"] == fam)]
    out["honest held-out implied"] = auc_boot(sel("heldout"), "N1_C", "concept_C")
    out["attack panel implied"] = auc_boot(sel("panel"), "N1_C", "concept_C")
    h, p = out["honest held-out implied"]["unrounded (gates use these)"], out["attack panel implied"]["unrounded (gates use these)"]
    out["stage 4 passes"] = bool(h["difference"] >= FLOOR and h["lower 95%"] > 0 and p["lower 95%"] > 0)
    out["context passes (stages 3 and 4)"] = bool(out["stage 3 passes"] and out["stage 4 passes"])
    fams = sorted({r["mechanism"] for r in sel("heldout")})
    out["by family (honest held-out)"] = {f: auc_boot(sel("heldout", f), "N1_C", "concept_C") for f in fams}
    out["by family (attack panel)"] = {f: auc_boot(sel("panel", f), "N1_C", "concept_C") for f in fams}
    out["held-out phrasings"] = auc_boot(sel("heldout_phrasing"), "N1_C", "concept_C")
    out["attack panel phrasings"] = auc_boot(sel("panel_phrasing"), "N1_C", "concept_C")
    out["2x2 synthetic test (cleaned wording)"] = auc_boot(sel("synthetic"), "N1_C", "concept_C")
    out["turn start (N1 at 24, concept at 16), honest held-out implied"] = auc_boot(sel("heldout"), "N1_T", "concept_T")
    out["turn start, attack panel implied"] = auc_boot(sel("panel"), "N1_T", "concept_T")
    return out


if __name__ == "__main__":
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
    if sys.argv[1] == "launch":
        st = git("status", "--porcelain", "--untracked-files=no")
        if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
            raise SystemExit("commit first: the result must record the commit of the code that made it")
        ext = json.loads((HERE / sys.argv[2]).read_text())
        fits = {v: json.loads((HERE / f).read_text()) for v, f in zip("CT", sys.argv[3:5])}
        for v, f in fits.items():
            if f["run"] != ext["runs"][v] or f.get("fixed_layer") != LAYER or f.get("null_unit") != "task":
                raise ValueError(f"{v}: the fit is not this extraction's fixed-layer-24 task-null fit")
        probes = {f"N1_{v}": (v, f"{fits[v]['run']}/fits/{fits[v]['fit_id']}/probes.json", fits[v]["probes_sha256"], "per_token") for v in "CT"}
        probes |= {f"concept_{v}": (v, f"{d}/probes.json", json.loads((FITS / r).read_text())["probes_sha256"],
                                    json.loads((FITS / r).read_text())["chosen_readout"] if v == "T" else "per_token") for v, (d, r) in CONCEPT.items()}
        sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (ROOT / ".run_cache" / "la_format_sim_probe").mkdir(parents=True, exist_ok=True)  # before the spawn
        call = modal.Function.from_name(APP, "score").spawn(ext["runs"], probes, sha)
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        rec = {"call": call.object_id, "stamp": stamp, "git": git("rev-parse", "HEAD").stdout.strip(), "compare_sha256": sha,
               "extract": sys.argv[2], "fits": sys.argv[3:5], "probes": probes}
        (ROOT / ".run_cache" / "la_format_sim_probe" / f"compare_launch_{stamp}.json").write_text(json.dumps(rec))
        print(json.dumps(rec, indent=1))
    elif sys.argv[1] == "collect":
        st = git("status", "--porcelain", "--untracked-files=no")
        if st.returncode or st.stdout.strip():
            raise SystemExit("collect from a clean tree: the analysis records the commit and hash of the code that ran it")
        rec = json.loads((ROOT / ".run_cache" / "la_format_sim_probe" / f"compare_launch_{sys.argv[2]}.json").read_text())
        paths = [HERE / "results" / f"compare_scores_{rec['stamp']}.json", HERE / "results" / f"compare_{rec['stamp']}.json"]
        if paths[1].exists():  # results/ is append-only: one analysis per launch
            raise FileExistsError(f"{paths[1]} exists")
        if paths[0].exists():  # a collect that died after saving the scores: reuse them if they are this launch's, whole
            saved = json.loads(paths[0].read_text())
            if saved["launch"] != rec or hashlib.sha256(json.dumps(saved["rows"]).encode()).hexdigest() != saved["rows_sha256"]:
                raise ValueError(f"{paths[0]} is not this launch's complete scores")
            rows = saved["rows"]
        else:
            rows = modal.FunctionCall.from_id(rec["call"]).get()
            tmp = paths[0].with_name("." + paths[0].name + ".tmp")
            tmp.write_text(json.dumps({"launch": rec, "rows": rows, "rows_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest()}) + "\n")
            tmp.rename(paths[0])
        out = {"launch git": rec["git"], "compare_sha256 (scorer, at launch)": rec["compare_sha256"],
               "collect git": git("rev-parse", "HEAD").stdout.strip(),
               "analyze sha256 (at collect)": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()} | analyze(rows, json.loads((HERE / rec["fits"][0]).read_text()), rec["extract"])
        path, tmp = paths[1], paths[1].with_name("." + paths[1].name + ".tmp")
        tmp.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
        tmp.rename(path)
        print(json.dumps(out, indent=1), f"\n-> {path}")
