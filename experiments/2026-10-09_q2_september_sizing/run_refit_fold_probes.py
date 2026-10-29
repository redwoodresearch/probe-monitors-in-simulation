"""Every on-policy attack call scored under each of the five real-trained fold probes (Modal, one L4), for the sizing
fix: a simulated study then sets each fold's threshold and scores its test calls with that fold's own probe.

q3 did not save the fold probes of the September per-call scoring (2026-10-09_000523_rerun_strict, observed draw), so
they are refit with q3's code itself: run_score_onpolicy_launch1144.py (sha256 41b326e1, branch aswin/q3-onpolicy-
figures at 19c8bc2), sent unchanged from --q3-script and written to a file in the container. load() builds the rows
from q3's acts Volume (read-only); for fold k the probe is fit with q3's standardize and fit_dual on the real rows of
the three training folds (k and its calibration fold k+1 left out), at the layer and C q3 recorded for that fold, with
unit task weights (draw 0); then every row is scored. The local entrypoint checks that the refit reproduces q3's per-call scores on each fold's test
calls before writing anything.

  uv run modal run experiments/2026-10-09_q2_september_sizing/run_refit_fold_probes.py --q3-script <copy of q3's script> --percall <q3 per-call json>
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-sizing-refit_kill-2026-11-08"
Q3_SHA = "41b326e1"
app = modal.App(APP)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08", version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scikit-learn==1.7.2")


@app.function(image=image, gpu="L4", memory=32768, timeout=3600, retries=0, volumes={"/acts": acts.read_only()})
def refit(cfg: dict, folds: dict, picks: dict, q3_source: bytes) -> dict:
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
    w = torch.ones(len(y), dtype=torch.float64, device=dev)  # draw 0: every task weight is 1
    y_t = torch.tensor(y, dtype=torch.float64, device=dev)
    scores = np.full((5, len(y)), np.nan)
    for k in range(5):
        cal = (k + 1) % 5
        train = (world == "real") & np.isin(fold, [f for f in range(5) if f not in (k, cal)])
        layer, C = picks[str(k)]
        Xl = X[layer: layer + 1]
        idx = torch.tensor(np.flatnonzero(train), device=dev)
        Xtr = Xl[:, idx].double()
        mu, sd = q3.standardize(Xtr, w[idx])
        Z = (Xtr - mu) / sd
        a = q3.fit_dual(Z @ Z.transpose(1, 2), y_t[idx], w[idx], [C])
        scores[k] = torch.einsum("lhn,lcn->lch", ((Xl.double() - mu) / sd) @ Z.transpose(1, 2), a)[0, 0].cpu().numpy()
    return {"scores": scores.tolist(), "meta": D["meta"], "fold": fold.tolist(), "picks": picks, "summary": D["summary"]}


@app.local_entrypoint()
def main(q3_script: str, percall: str):
    q3_source = Path(q3_script).read_bytes()
    if hashlib.sha256(q3_source).hexdigest()[:8] != Q3_SHA:
        raise ValueError("the copy of q3's script is not sha256 41b326e1")
    pc = json.loads(Path(percall).read_text())
    picks = {str(f["fold"]): (f["layer"], f["C"]) for f in pc["draws"][0]["cells"]["real->real"]["folds"] if f["threshold_kind"] == "frozen"}
    res = refit.remote(pc["config"], pc["folds"], picks, q3_source)
    if res["meta"] != pc["meta"]:
        raise ValueError("the refit's rows are not q3's rows")
    worst = 0.0
    for cell in ("real->real", "real->sim"):  # each fold's test calls: the refit must give q3's scores
        p = pc["draws"][0]["cells"][cell]["per_call"]
        for i, s in zip(p["row"], p["score"]):
            worst = max(worst, abs(res["scores"][res["fold"][i]][i] - s))
    print("largest difference from q3's per-call scores on the test folds:", worst)
    if worst > 1e-3:
        raise ValueError("the refit does not reproduce q3's scores")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"fold_probe_scores_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res | {"q3 script sha256 prefix": Q3_SHA, "largest difference from q3": worst}) + "\n")
    print(out)
