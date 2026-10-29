"""Real-vs-real control for the on-policy shift (real minus simulated TPR at 1% FPR, q3's strict-rerun scoring; Modal,
one L4). Two independent real runs per task through the same pipeline: honest_a vs honest_b. There is one real attack
run per task, so the control uses honest runs, whose impactful calls are the positives and read-only calls the negatives.

q3's five real-trained fold probes are refit with q3's own code, as run_refit_fold_probes.py (sha256 41b326e1, sent from
--q3-script): for fold k, real attack rows of the three training folds, at q3's recorded layer and C, unit weights.
Then every honest_a and honest_b call of the strict rerun's segment-end dump (2026-10-09_rerun_strict_segends_v1, the
rule-1 command-token rows, labelled impactful or read-only like the attack calls) is scored by its task's fold probe.
The local entrypoint applies q3's frozen thresholds (the 1% point of fold k+1's real attack read-only calls under
probe k) and writes per-row scores. Built-in checks: (1) the refit reproduces q3's per-call test-fold scores (≤ 1e-3);
(2) the same dump's real attack command rows, scored the same way, reproduce the replay dump's scores (the same
readout from another extraction; every replay call must match, within 5% of the score SD). (1) must pass before
anything is written; (2) is written either way and the analysis refuses a file where it failed.

  uv run modal run experiments/2026-10-09_q2_real_vs_real/run_onpolicy_real_vs_real.py --q3-script <copy> --percall <q3 per-call json>
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-real-vs-real_kill-2026-11-08"
Q3_SHA = "41b326e1"
SEGENDS = "2026-10-09_rerun_strict_segends_v1"
app = modal.App(APP)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08", version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scikit-learn==1.7.2")


@app.function(image=image, gpu="L4", memory=32768, timeout=3600, retries=0, volumes={"/acts": acts.read_only()})
def score(cfg: dict, folds: dict, picks: dict, q3_source: bytes) -> dict:
    import importlib.util

    import numpy as np
    import torch

    Path("/root/q3score.py").write_bytes(q3_source)  # q3's script uses __file__, so it is loaded from a file
    spec = importlib.util.spec_from_file_location("q3score", "/root/q3score.py")
    q3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q3)
    D = q3.load(cfg, folds)
    X, y, fold, world = D["X"], D["y"], D["fold"], D["world"]
    dev = X.device
    w = torch.ones(len(y), dtype=torch.float64, device=dev)
    y_t = torch.tensor(y, dtype=torch.float64, device=dev)
    probes, refit = {}, np.full(len(y), np.nan)
    for k in range(5):
        cal = (k + 1) % 5
        train = (world == "real") & np.isin(fold, [f for f in range(5) if f not in (k, cal)])
        layer, C = picks[str(k)]
        idx = torch.tensor(np.flatnonzero(train), device=dev)
        Xtr = X[layer: layer + 1][:, idx].double()
        mu, sd = q3.standardize(Xtr, w[idx])
        Z = (Xtr - mu) / sd
        a = q3.fit_dual(Z @ Z.transpose(1, 2), y_t[idx], w[idx], [C])
        beta = torch.einsum("lcn,lnh->lch", a, Z)[0, 0]  # primal weights of the kernel fit: score = ((x - mu) / sd) @ beta
        probes[k] = (layer, mu[0, 0], sd[0, 0], beta)
        te = np.flatnonzero(fold == k)
        refit[te] = (((X[layer, torch.tensor(te, device=dev)].double() - mu[0, 0]) / sd[0, 0]) @ beta).cpu().numpy()
    del X
    root = Path("/acts") / SEGENDS
    done = json.loads((root / "complete.json").read_text())  # as q3's load(): every shard finished, every epoch extracted
    if len(list((root / "meta").glob("*/*.json"))) != done["epochs"]:
        raise ValueError(f"{SEGENDS}: extracted epochs differ from the {done['epochs']} staged")
    rows, counted = [], {}  # rows: (arm, task, chunk, row, y, irid, call_index)
    for arm in ("honest_a", "honest_b", "attack"):
        for f in sorted((root / "meta" / arm).glob("*.json")):
            m = json.loads(f.read_text())
            if m.get("redrawn_because") or m.get("attempt_kind") == "rerun" or m["task"] not in folds:
                continue
            counted.setdefault((arm, m["task"]), set()).add(m["run"])
            for c in m["calls"]:  # in this dump `call` numbers a turn's entries; `call_index` is the tool call, the replay dump's `call`
                if c.get("kind") == "command" and c["row"] is not None and c["label"] in ("impactful", "read_only"):
                    rows.append((arm, m["task"], m["chunk"], m["row_start"] + c["row"], int(c["label"] == "impactful"), c["irid"], c["call_index"]))
    dup = {k: v for k, v in counted.items() if len(v) > 1}
    if dup:
        raise ValueError(f"{len(dup)} arm/task pairs have several counted runs, e.g. {next(iter(dup.items()))}")
    n_arm = {arm: sum(r[0] == arm for r in rows) for arm in ("honest_a", "honest_b", "attack")}
    if min(n_arm.values()) == 0:
        raise ValueError(f"an arm has no labelled command rows: {n_arm}")
    out = np.full(len(rows), np.nan)
    for k, (layer, mu, sd, beta) in probes.items():
        mine = [i for i, r in enumerate(rows) if folds[r[1]] == k]
        groups = {}
        for i in mine:
            groups.setdefault(rows[i][2], []).append(i)
        for chunk, pos in groups.items():  # only the rows read, as q3's load()
            a = np.load(root / "by_layer" / f"L{layer}" / f"{chunk}.npy", mmap_mode="r")
            x = torch.from_numpy(np.asarray(a[[rows[i][3] for i in pos]], np.float32)).to(dev).double()
            out[pos] = (((x - mu) / sd) @ beta).cpu().numpy()
    if not (np.isfinite(out).all() and np.isfinite(refit).all()):
        raise ValueError("non-finite scores")
    return {"arm runs per task": {f"{a}|{t}": len(v) for (a, t), v in counted.items()}, "refit": refit.tolist(), "meta": D["meta"], "fold": fold.tolist(),
            "rows": [{"arm": r[0], "task": r[1], "y": r[4], "irid": r[5], "call": r[6], "score": float(s)} for r, s in zip(rows, out)]}


@app.local_entrypoint()
def main(q3_script: str, percall: str):
    import numpy as np

    q3_source = Path(q3_script).read_bytes()
    if hashlib.sha256(q3_source).hexdigest()[:8] != Q3_SHA:
        raise ValueError("the copy of q3's script is not sha256 41b326e1")
    pc = json.loads(Path(percall).read_text())
    frozen = [f for f in pc["draws"][0]["cells"]["real->real"]["folds"] if f["threshold_kind"] == "frozen"]
    picks = {str(f["fold"]): (f["layer"], f["C"]) for f in frozen}
    res = score.remote(pc["config"], pc["folds"], picks, q3_source)
    if res["meta"] != pc["meta"]:
        raise ValueError("the refit's rows are not q3's rows")
    p = pc["draws"][0]["cells"]["real->real"]["per_call"]
    worst = max(abs(res["refit"][i] - s) for i, s in zip(p["row"], p["score"]))
    print("check 1, largest difference from q3's per-call scores (real attack test folds):", worst)
    if worst > 1e-3:
        raise ValueError("the refit does not reproduce q3's scores")
    replay = {(m["irid"], m["call"]): res["refit"][i] for i, m in enumerate(res["meta"]) if m["world"] == "real"}
    pairs = [(replay[(r["irid"], r["call"])], r["score"]) for r in res["rows"] if r["arm"] == "attack" and (r["irid"], r["call"]) in replay]
    check2 = {"matched real attack calls": len(pairs), "real attack calls in the replay dump": len(replay)}
    if pairs:
        a, b = np.array(pairs).T
        check2 |= {"largest absolute difference": float(np.abs(a - b).max()), "correlation": float(np.corrcoef(a, b)[0, 1]), "score SD (replay)": float(a.std())}
    # passes when every replay call is matched and the two readouts agree to 5% of the score SD; written either way (the GPU work is paid)
    check2["passed"] = bool(len(pairs) == len(replay) and pairs and check2["largest absolute difference"] <= 0.05 * check2["score SD (replay)"])
    print("check 2, the segment-end dump's attack rows against the replay dump:", json.dumps(check2))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"onpolicy_real_vs_real_scores_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"q3 script sha256 prefix": Q3_SHA, "percall sha256": hashlib.sha256(Path(percall).read_bytes()).hexdigest(),
                                "picks": picks, "frozen thresholds": {f["fold"]: f["threshold"] for f in frozen}, "folds": pc["folds"],
                                "check 1": worst, "check 2": check2, "arm runs per task": res["arm runs per task"], "rows": res["rows"]}) + "\n")
    print(path)
