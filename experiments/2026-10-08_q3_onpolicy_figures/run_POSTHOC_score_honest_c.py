"""A2 (PLAN_A2_honest_batch_C_2026-10-11.md): score honest batch C's read-only calls with draw 0's saved real-trained
models, as run_score_onpolicy.honest_scores scored A and B. POST HOC, CPU on Modal, deploy and spawn.

run_score_onpolicy.py is imported unchanged (editing it would move every scoring folder's key). Its honest_scores runs
gate b first and writes honest_scores.json, so this file repeats that function with two differences only: C's arm is
read as `honB: honest_b@<C extraction>` (load() takes no new world names) and relabelled honC, and the output is
honest_scores_C.json, written once. Gate b is the same code: the saved models must reproduce the main per-call file's
real->real scores before any C activation is read.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml --with numpy modal deploy run_POSTHOC_score_honest_c.py
  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml --with numpy python run_POSTHOC_score_honest_c.py \
      --c-extraction <name> --honest-key scaleup196_probes_14b33da3d97a_percall --main-percall <file>
"""

import argparse
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE) if modal.is_local() else "/root")
import run_score_onpolicy as S  # noqa: E402  (unchanged)

APP = "probe-shift_aswin-surya_sub-a2-honest-c-score_kill-2026-11-10"
app = modal.App(APP)
image = S.image.add_local_file(HERE / "run_score_onpolicy.py", "/root/run_score_onpolicy.py")


def c_config(ext: dict, cx: str) -> dict:
    """score_scaleup196_simhon.yaml with its honest worlds replaced by C's arm of extraction `cx` (or, for the
    equivalence check, B's honest_b arm of one of the existing extractions)."""
    worlds = {w: v for w, v in ext["worlds"].items() if w not in ("honA", "honB", "simH")} | {"honB": f"honest_b@{cx}"}
    if cx in ext["extractions"]:  # the check: B's runs of an existing extraction, read through C's path
        return ext | {"label": "scaleup196_honc_check", "worlds": worlds}
    return ext | {"label": "scaleup196_honc", "extractions": ext["extractions"] + [cx], "worlds": worlds}


@app.function(image=image, cpu=8, memory=32768, timeout=3600, retries=0, volumes={"/acts": S.acts_vol.read_only(), "/score": S.score_vol})
def honest_c(cfg: dict, cfg_c: dict, folds: dict, key: str, main: dict, out_name: str = "honest_scores_C.json", provenance: dict | None = None) -> dict:
    import hashlib

    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    S.score_vol.reload()
    d = Path("/score") / key
    dst = d / out_name
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    draw = json.loads((d / "draw_00000.json").read_text())
    if draw["choices"] != main["choices"]:
        raise ValueError("draw-0 choices differ from the main run's")
    pz = np.load(d / "probes_draw0.npz")
    real = sorted((k, layer, C) for W, k, layer, C in json.loads(str(pz["meta"])) if W == "real")
    if [k for k, _, _ in real] != list(range(5)) or any(draw["choices"][f"real|{k}"] != [layer, C] for k, layer, C in real):
        raise ValueError("the saved models are not the draw's five real-trained models")
    P = {k: [pz[f"real|{k}|{n}"] for n in ("mu", "sd", "beta")] for k, _, _ in real}
    for k, arrs in P.items():
        if any(a.dtype != np.float64 or a.shape != (cfg["hidden"],) or not np.isfinite(a).all() for a in arrs) or (arrs[1] <= 0).any():
            raise ValueError(f"saved model {k}: wrong dtype or shape, non-finite values, or a non-positive scale")
    layers = sorted({layer for _, layer, _ in real})

    def scored(X):
        out = np.stack([((X[layer].astype(np.float64) - P[k][0]) / P[k][1]) @ P[k][2] for k, layer, _ in real], 1)
        if not np.isfinite(out).all():
            raise ValueError("non-finite scores")
        return out

    # gate b, as honest_scores: the saved models reproduce the main per-call file's real->real scores
    rows = S.load(cfg | {"_rows_only": True}, folds)["rows"]
    c = Path("/score") / cfg["x_cache"]
    if json.loads((c / "rows.json").read_text())["rows_sha256"] != S.rows_sha(rows):
        raise ValueError(f"{cfg['x_cache']} holds other rows")
    if [[r[0], r[1], r[5]] for r in rows] != main["meta"]:
        raise ValueError("the main config's rows differ from the main per-call file's meta")
    ref, idx = np.array(main["scores"], dtype=np.float64), np.array(main["rows"])
    Xc = np.load(c / "X.npy", mmap_mode="r")
    s = scored({layer: np.asarray(Xc[layer])[idx] for layer in layers})
    diff = float(np.abs(np.round(s[np.arange(len(idx)), [folds[rows[i][1]] for i in idx]], 6) - ref).max())
    if not diff <= 1.000001e-6:
        raise ValueError(f"the saved models do not reproduce the main per-call scores: max difference {diff}")
    del Xc

    # only now: C's read-only calls, at the five models' layers
    ext = [r for r in S.load(cfg_c | {"_rows_only": True}, folds)["rows"] if r[0] == "honB" and r[5] == 0]
    cx = cfg_c["worlds"]["honB"].split("@", 1)[1]
    if not ext or {r[2][1] for r in ext} != {cx}:
        raise ValueError("the rows must all come from the named extraction")
    groups = {}
    for i, r in enumerate(ext):
        g = groups.setdefault((r[2], r[3]), ([], []))
        g[0].append(i), g[1].append(r[4])
    X = {layer: np.empty((len(ext), cfg["hidden"]), np.float16) for layer in layers}

    def one(job):
        layer, ((vol, run), chunk), (pos, sel) = job
        X[layer][pos] = np.load(Path("/" + vol) / run / "by_layer" / f"L{layer}" / f"{chunk}.npy")[sel]

    with ThreadPoolExecutor(32) as ex:
        list(ex.map(one, [(layer, k, v) for layer in layers for k, v in groups.items()]))
    sx = scored(X)
    out = {"gate": {"choices_equal": True, "rows_reproduced": len(idx), "max_abs_difference_6dp": diff}, "key": key,
           "choices": draw["choices"], "c_extraction": cx, "rows_sha256": S.rows_sha(ext),  # the ordered activation rows scored
           # the launcher's git commit and local file hashes, beside the hashes of the code this container ran
           "provenance": {"launcher": provenance, "container_sha256": {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest()
                                                                       for f in (__file__, "/root/run_score_onpolicy.py")}},
           "rows": [{"world": "honC", "task": r[1], "fold": folds[r[1]], "y": r[5], "s": [round(float(v), 6) for v in v5]} for r, v5 in zip(ext, sx)]}
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    S.score_vol.commit()
    return {k: v for k, v in out.items() if k != "rows"} | {"n_rows": len(out["rows"]), "tasks": len({r[1] for r in ext})}


if __name__ == "__main__":
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--c-extraction", required=True, help="C's extraction; or, with --check, an existing extraction holding B's runs")
    ap.add_argument("--check", action="store_true", help="score B's runs of --c-extraction through this path, to honest_scores_Ccheck_<x>_<git>.json")
    ap.add_argument("--honest-key", required=True, help="the save_probes draw's folder (A and B's honest_scores key)")
    ap.add_argument("--main-percall", required=True)
    a = ap.parse_args()
    import hashlib
    import subprocess

    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes: commit (and deploy) them first")
    files = ("run_POSTHOC_score_honest_c.py", "run_score_onpolicy.py", "config/score_scaleup196_probes.yaml", "config/score_scaleup196_simhon.yaml")
    provenance = {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
                  "sha256": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in files}}
    if a.honest_key != "scaleup196_probes_14b33da3d97a_percall":  # A and B's honest_scores.json key
        raise SystemExit("C must be scored with A and B's saved models")
    cfg = yaml.safe_load((HERE / "config" / "score_scaleup196_probes.yaml").read_text())  # the save_probes config A and B were scored under
    ext = yaml.safe_load((HERE / "config" / "score_scaleup196_simhon.yaml").read_text())
    folds = S.frozen_folds(HERE.parents[1] / cfg["pilot_tasks"])
    f = json.loads(Path(a.main_percall).read_text())
    pc = f["draws"][0]["cells"]["real->real"]["per_call"]
    main = {"choices": f["draws"][0]["choices"], "meta": [[m["world"], m["task"], m["y"]] for m in f["meta"]], "rows": pc["row"], "scores": pc["score"]}
    if a.check != (a.c_extraction in ext["extractions"]):
        raise SystemExit("--check takes an existing extraction; C's extraction is scored without it")
    out_name = f"honest_scores_Ccheck_{a.c_extraction}_{provenance['git'][:8]}.json" if a.check else "honest_scores_C.json"
    call = modal.Function.from_name(APP, "honest_c").spawn(cfg, c_config(ext, a.c_extraction), folds, a.honest_key, main, out_name, provenance)
    print(json.dumps({"call": call.object_id, "c_extraction": a.c_extraction, "honest_key": a.honest_key, "out": out_name}))
