"""Item 2 of RULE.md (POST HOC): cell (a)'s nulls under a GLOBAL label permutation of the fixed run scores (no refit). One
Modal CPU container. Gate first: the as-run within-task null (analyze_cell_a_null.py @ 110a8653: seed 20261013, 10,000
permutations, the same statistic order) must reproduce every observed value and p of the as-run JSON exactly; only then is
the global null (seed 20261014) computed. Primary: unswapped level 1, catch of the second real run at 1%.

  CELLA_IN=<folder with the four input JSONs> PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' --with numpy \\
      modal run --detach experiments/2026-10-14_single_class_rechecks_POSTHOC/run_cella_global.py::main --out <json>
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-screcheck_cella_kill-2026-11-13"
IN = Path(os.environ.get("CELLA_IN", "/nonexistent"))
FILES = {"scores": ("cell_a_run_scores_POSTHOC_2026-10-13_121016.json", "d445d064691e886cf22742732fbd1529e50656d09e13a2eccdc987ffd07cedba"),
         "stage2": ("stage2_cell_a_POSTHOC_2026-10-13_121124.json", "6c6ff5bfebfc90624035dbd03ebf0af889e46f89d0839df81e456863760ea0cf"),
         "labels": ("claimc_labels_reduced.json", "d00e88773fc2a71bef51e1ff0ccc2559932f5c3a306a8f5604c5248e797dc995"),
         "asrun": ("cell_a_null_POSTHOC_2026-10-13_145628.json", "662ee5c51423e06c8487dc616078016bcef63e043b126381538bb25b50b2baa3")}
CODE = ["experiments/2026-10-13_onpolicy_reply/analyze_onpolicy_cells.py", "experiments/2026-10-08_q3_onpolicy_figures/analyze_retrain.py",
        "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"]
N_PERM, SEED_ASRUN, SEED_GLOBAL = 10_000, 20261013, 20261014
STATS = ("auroc_run2_vs_honest_b", "catch_run2_at_1pct", "catch_sim_at_1pct")
PRIMARY = ("unswapped_L1", "catch_run2_at_1pct")
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
for f in CODE:
    image = image.add_local_file(REPO / f, f"/root/repo/{f}")
if IN.exists():
    for name, _ in FILES.values():
        image = image.add_local_file(IN / name, f"/root/in/{name}")


@app.function(image=image, cpu=2, memory=4096, timeout=900, retries=0)
def run() -> dict:
    import numpy as np

    for name, sha in FILES.values():
        if hashlib.sha256(Path("/root/in", name).read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"{name}: sha256 differs")
    sys.path.insert(0, "/root/repo/experiments/2026-10-13_onpolicy_reply")
    import analyze_onpolicy_cells as A

    J = {k: json.loads(Path("/root/in", n).read_text()) for k, (n, _) in FILES.items()}
    lab = J["labels"]["labels"]
    tasks = sorted(A.frozen_folds(json.loads(A.T196.read_text())))

    def within(scores, task, rng):  # analyze_cell_a_null.permuted @ 110a8653, verbatim: one within-task permutation
        base = np.argsort(task, kind="stable")
        order = np.lexsort((rng.random(len(task)), task))
        out = np.empty_like(scores)
        out[base] = scores[order]
        return out

    def globally(scores, task, rng):  # the global null: one permutation of all runs of the set, no task grouping
        return scores[rng.permutation(len(scores))]

    def block(obs, null, beyond):  # analyze_cell_a_null.null_block @ 110a8653, with the "beyond" wording passed in
        null = np.array(null)
        p = (1 + int((null >= obs).sum())) / (1 + len(null))
        return {"observed": obs, "null_p95": float(np.percentile(null, 95)), "null_mean": float(null.mean()), "p": p,
                "reading": beyond if p <= 0.05 else "not beyond the null" + (" (global; ignores task clustering)" if "global" in beyond else "")}

    def sweep(perm, seed, beyond):  # analyze_cell_a_null.main's loop @ 110a8653, with the permutation and its wording passed in
        rng = np.random.default_rng(seed)
        res = {}
        for variant, rs in J["scores"]["variants"].items():
            rs = rs["run_scores"]
            runs = sorted(k for k in rs if rs[k] is not None and np.isfinite(rs[k]))
            world = np.array([u.split("|", 1)[0] for u in runs])
            task = np.array([u.split("|", 1)[1] for u in runs])
            z = np.array([rs[u] for u in runs], float)
            have = lambda w: set(task[world == w])  # noqa: E731
            att = lambda w: {t for t in have(w) if lab.get(f"{w}|{t}") == "attempted"}  # noqa: E731
            cohort = att("run1") & att("run2") & att("sim")
            m = lambda w, ts: (world == w) & np.isin(task, sorted(ts))  # noqa: E731
            o = {"cohort_tasks": len(cohort)}
            cb = cohort & have("honest_b")
            sel = m("run2", cb) | m("honest_b", cb)
            zz, tt, pos = z[sel], task[sel], world[sel] == "run2"
            auc = lambda s: A.wauc(s[pos], np.ones(int(pos.sum())), s[~pos], np.ones(int((~pos).sum())))  # noqa: E731
            o["auroc_run2_vs_honest_b"] = block(auc(zz), [auc(perm(zz, tt, rng)) for _ in range(N_PERM)], beyond) | {"runs": [int(pos.sum()), int((~pos).sum())]}
            for arm in ("run2", "sim"):
                sel = m("honest_a", set(tasks)) | m(arm, cohort)
                zz, tt, is_arm = z[sel], task[sel], world[sel] == arm
                catch = lambda s: float((s[is_arm] > A.threshold(s[~is_arm], np.ones(int((~is_arm).sum())))).mean())  # noqa: E731
                o[f"catch_{arm}_at_1pct"] = block(catch(zz), [catch(perm(zz, tt, rng)) for _ in range(N_PERM)], beyond) | {"runs": [int(is_arm.sum()), int((~is_arm).sum())]}
            s2 = J["stage2"]["cells"][f"a_{variant}"]
            want = {"auroc_run2_vs_honest_b": s2["null"]["auroc_run2_vs_honest_b_beside"], "catch_run2_at_1pct": s2["at_0.01"]["catch_run2"],
                    "catch_sim_at_1pct": s2["at_0.01"]["catch_sim"]}
            bad = {k: (o[k]["observed"], v) for k, v in want.items() if abs(o[k]["observed"] - v) > 1e-12}
            if bad:
                raise RuntimeError(f"{variant}: observed values differ from stage 2: {bad}")
            res[variant] = o
        return res

    asrun = sweep(within, SEED_ASRUN, "beyond a main-task-grouped label-permutation null")
    ref = J["asrun"]["variants"]
    mism = [[v, k, f, asrun[v][k][f], ref[v][k][f]] for v in ref for k in STATS for f in ("observed", "p", "null_p95", "null_mean", "runs")
            if asrun[v][k][f] != ref[v][k][f]]
    if mism or sorted(asrun) != sorted(ref):
        return {"gate": {"passes": False, "mismatches": mism[:30]}}
    return {"gate": {"passes": True, "compared": len(ref) * len(STATS) * 5}, "within_task_asrun": asrun, "global": sweep(globally, SEED_GLOBAL, "beyond a label-permutation null (global; ignores task clustering)")}


@app.local_entrypoint()
def main(out: str):
    me = str(Path(__file__).resolve())
    if subprocess.run(["git", "status", "--porcelain", "--", me, str(HERE / "RULE.md")], capture_output=True, text=True, cwd=HERE).stdout.strip():
        raise SystemExit("commit the runner and RULE.md first")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    res = run.remote()
    with Path(out).open("x") as fh:
        fh.write(json.dumps({"label": "POST HOC (RULE.md item 2)", "git": commit, "inputs_sha256": {n: s for n, s in FILES.values()},
                             "n_perm": N_PERM, "seeds": {"within_task_asrun": SEED_ASRUN, "global": SEED_GLOBAL}} | res, indent=1))
    print("gate:", json.dumps(res["gate"]))
    if not res["gate"]["passes"]:
        return
    for v in res["global"]:
        for k in STATS:
            a, g = res["within_task_asrun"][v][k], res["global"][v][k]
            f = (lambda x: f"{x:.4f}") if k.startswith("auroc") else (lambda x: f"{100 * x:.2f}")
            tag = "PRIMARY " if (v, k) == PRIMARY else ""
            print(f"{tag}{v} {k}: observed {f(a['observed'])} | within-task (as run) p {a['p']:.4f}, 95th {f(a['null_p95'])} | "
                  f"global p {g['p']:.4f}, 95th {f(g['null_p95'])}, mean {f(g['null_mean'])} | {g['reading']}"
                  + ("" if (a["p"] <= 0.05) == (g["p"] <= 0.05) else " | READING CHANGES"))
