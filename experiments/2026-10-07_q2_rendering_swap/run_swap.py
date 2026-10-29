"""Rendering swap: does re-rendering the simulator's insert outputs in the real tool's style close pre_call's real->sim gap?

Steps (Modal CPU here; the GPU extraction is the all-layers extractor with config/alllayers_q2_swap.yaml there):
  --step stage   Read level 0's staged reader messages for the runs in config/swap_runs.json. Write two copies of each
                 to a new staged run: as is (arm sim-attack), and with every insert output re-rendered (arm from config).
                 Writes a per-insert category report, and returns 3 transcript pairs for the by-eye check.
  --step score   Score both copies with the frozen level-0 probes (the refit with weights, layers 9/16/17). Each call
                 gets the probe of its own outer fold, at that fold's headline layer. Also returns the refit's saved
                 test scores for every Q2 call, for the real side of the gap and as a reproduction check.
  --step fetch   Read a saved result.

  uv run modal run experiments/2026-10-07_q2_rendering_swap/run_swap.py --step stage
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-swap_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
FEATS = "probe-shift_aswin-surya_sub-q2_probe-features_kill-2026-11-06"

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
feats_vol = modal.Volume.from_name(FEATS, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "pyyaml")
         .add_local_file(HERE / "rerender.py", "/root/rerender.py")
         .add_local_file(HERE / "str_replace_tells.py", "/root/str_replace_tells.py")
         .add_local_python_source("probe_shift"))  # rerender's view check uses the Q2 call labeler


def positions_of(meta: dict, step: int, action: int) -> dict:
    """The three readouts of a call in an all-layer meta file, as run_q2_alllayers.py reads them."""
    at = {(q["kind"], q["step"], q["action"]): q for q in meta["positions"]}
    last_output = {}
    for q in meta["positions"]:
        if q["kind"] == "output_end":
            last_output[q["step"]] = max(last_output.get(q["step"], -1), q["action"])
    return {"call_end": at.get(("call_end", step, action)), "pre_call": at.get(("pre_call", step, action)),
            "prev_content": at.get(("call_end", step, action - 1)) if action > 0 else
            (at.get(("output_end", step - 1, last_output[step - 1])) if step - 1 in last_output else None)}


def filename(key: str) -> str:
    """The extractor's file stem for a key."""
    import hashlib

    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=1, memory=16384, timeout=3600)  # one Python loop
def stage(cfg: dict, keys: list[str]) -> dict:
    import sys

    sys.path.insert(0, "/root")
    import rerender as rr
    import str_replace_tells as st

    src, dst = Path("/acts") / cfg["inputs_from"] / "inputs", Path("/acts") / cfg["staged_run"]
    if dst.exists():
        raise FileExistsError(f"{dst} exists: a staged run is never overwritten")
    want = set(keys)
    items = [it for p in sorted(src.glob("shard_*.jsonl")) for it in map(json.loads, p.read_text().splitlines())
             if it["arm"] == "sim-attack" and it["key"] in want]
    if len(items) != len(want):
        raise ValueError(f"found {len(items)} of {len(want)} runs in {src}")
    out, report, pairs = [], {}, []
    variants = cfg.get("variants") or {cfg["swap_arm"]: "full"}  # arm -> rerender variant
    for it in sorted(items, key=lambda x: x["key"]):
        if cfg.get("include_original", True):
            out.append(it)
        for arm, variant in variants.items():
            new, rendered = it["messages"], set()
            for part in variant.split("+"):  # a combined variant ("a+b") applies its parts in turn, each to its own outputs
                if part in st.VARIANTS:  # the str_replace tells: only str_replace outputs may differ
                    new, records = st.swap_messages(new, part)
                    rendered |= {r["msg"] for r in records if r["changed"]}
                    report.setdefault(it["key"], {})[f"str_replaces {arm}"] = records
                else:  # the insert re-render: only the re-rendered insert outputs may differ
                    new, records = rr.rerender_messages(new, part)
                    rendered |= {r["msg"] for r in records if r["category"] in rr.RENDERED}
                    report.setdefault(it["key"], {}).setdefault("inserts", records)
            rendered = sorted(rendered)
            changed = [i for i, (a, b) in enumerate(zip(it["messages"], new)) if a != b]
            if changed != rendered or len(new) != len(it["messages"]):
                raise ValueError(f"{it['key']} {arm}: messages {changed} changed, expected exactly {rendered} ({variant})")
            out.append(it | {"arm": arm, "messages": new})
            report[it["key"]][f"changed_messages {arm}"] = changed
            if len(pairs) < 3 * len(variants) and changed:
                pairs.append({"key": it["key"], "arm": arm, "original": it["messages"], "rerendered": new})
    (dst / "inputs").mkdir(parents=True)
    (dst / "inputs" / "shard_0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    (dst / "rerender_report.json").write_text(json.dumps(report))
    acts_vol.commit()
    return {"items": len(out), "report": report, "pairs": pairs}


@app.function(image=image, volumes={"/acts": acts_vol, "/feats": feats_vol}, cpu=2, memory=32768, timeout=3600)  # one Python loop
def score(cfg: dict, keys: list[str], tolerance: float) -> dict:
    import numpy as np

    acts_vol.reload()
    feats_vol.reload()
    run, feats = Path("/acts") / cfg["extraction_run"], Path("/feats") / cfg["features_run"]
    wdir = feats / cfg["weights_out_sub"]
    manifest = json.loads((feats / "manifest.json").read_text())
    test = [m for m in manifest if m["arm"] in ("real-attack-e1", "sim-attack")]  # the fits' row order
    logs = [wdir / "fits_index.jsonl"] + sorted((wdir / "fits_index").glob("*.jsonl"))
    fits = {(f["position"], f["train_set"], f["layer"]): f for log in logs if log.exists()
            for f in map(json.loads, log.read_text().splitlines()) if f["seed"] is None and f.get("view", "") == ""}
    t, layers = cfg["train_set"], cfg["layers"]
    base = json.loads((feats / "alllayers" / "result.json").read_text())  # the 64-layer level-0 run (read only)
    head, npz, source = {}, {}, {}
    for pos in cfg["positions"]:
        for layer in layers:
            npz[(pos, layer)] = dict(np.load(wdir / "fits" / fits[(pos, t, layer)]["file"]))
        # The 64-layer level-0 run's own per-fold picks where they all lie among these layers (pre_call: the headline
        # probe as reported), else the same rule over these layers only (the controls, as in the cleaned-text test).
        picks = {h["fold"]: h["layer"] for h in base["positions"][pos]["headline_layers"][t]}
        if set(picks.values()) <= set(layers):
            head |= {(pos, f): l for f, l in picks.items()}
            source[pos] = "64-layer headline picks"
        else:
            for f in sorted({fo["fold"] for fo in fits[(pos, t, layers[0])]["folds"]}):
                val = {l: next(fo["val_auroc"] for fo in fits[(pos, t, l)]["folds"] if fo["fold"] == f) for l in layers}
                head[(pos, f)] = min(l for l in layers if val[l] >= max(val.values()) - tolerance)
            source[pos] = f"rule over layers {layers}"
    saved = {pos: {"score": np.zeros(len(test)), "threshold": np.zeros(len(test))} for pos in cfg["positions"]}
    for pos in cfg["positions"]:
        for i, m in enumerate(test):
            a = npz[(pos, head[(pos, m["fold"])])]
            saved[pos]["score"][i], saved[pos]["threshold"][i] = a["score"][i], a["threshold"][i]
    row_of = {(m["arm"], m["key"], m["step"], m["action"]): i for i, m in enumerate(test)}
    chunks = {}

    def acts(run_dir: Path, chunk: str, layer: int):
        if (run_dir, chunk, layer) not in chunks:
            chunks[(run_dir, chunk, layer)] = np.load(run_dir / "by_layer" / f"L{layer}" / f"{chunk}.npy", mmap_mode="r")
        return chunks[(run_dir, chunk, layer)]

    # The as-simulated copy (arm sim-attack) comes from this run, or from the reference run when this run holds variants only.
    ref = Path("/acts") / cfg["reference_extraction_run"] if cfg.get("reference_extraction_run") else run
    arm_runs = {"sim-attack": ref} | {arm: run for arm in (cfg.get("variants") or {cfg["swap_arm"]: "full"})}
    swap, skipped = [], []
    want = set(keys)
    for key in sorted(want):
        metas = {arm: json.loads((d / "meta" / arm / f"{filename(key)}.json").read_text()) for arm, d in arm_runs.items()}
        for m in (m for m in test if m["arm"] == "sim-attack" and m["key"] == key):
            qs = {arm: positions_of(meta, m["step"], m["action"]) for arm, meta in metas.items()}
            for pos in cfg["positions"]:
                at = {arm: q[pos] for arm, q in qs.items()}
                a = at["sim-attack"]
                if not all(b and b["kept"] and (b["kind"], b["step"], b["action"]) == (a["kind"], a["step"], a["action"]) for b in at.values()):
                    skipped.append({"key": key, "step": m["step"], "action": m["action"], "position": pos})
                    continue
                layer = head[(pos, m["fold"])]
                w, mu, sd = (npz[(pos, layer)][k][m["fold"]] for k in ("w", "mu", "sd"))
                val = lambda arm, q: float(((np.asarray(acts(arm_runs[arm], metas[arm]["chunk"], layer)[metas[arm]["row_start"] + q["row"]],
                                                        np.float32) - mu) / sd) @ w[:-1] + w[-1])
                i = row_of[("sim-attack", key, m["step"], m["action"])]
                vals = {arm: val(arm, q) for arm, q in at.items()}
                rec = {"key": key, "step": m["step"], "action": m["action"], "y": m["y"], "fold": m["fold"], "position": pos,
                       "layer": layer, "original": vals["sim-attack"], "scores": vals,
                       "saved": float(saved[pos]["score"][i]), "threshold": float(saved[pos]["threshold"][i])}
                if not cfg.get("variants"):
                    rec["rerendered"] = vals[cfg["swap_arm"]]
                swap.append(rec)
    out = {"headline_layers": {f"{p} fold {f}": l for (p, f), l in head.items()}, "layer_source": source, "swap": swap, "skipped": skipped,
           "saved": [{"arm": m["arm"], "key": m["key"], "step": m["step"], "action": m["action"], "y": m["y"], "fold": m["fold"],
                      **{f"{pos}_score": float(saved[pos]["score"][i]) for pos in cfg["positions"]},
                      **{f"{pos}_threshold": float(saved[pos]["threshold"][i]) for pos in cfg["positions"]}} for i, m in enumerate(test)]}
    (run / "scores.json").write_text(json.dumps(out))
    acts_vol.commit()
    return {"swap": len(swap), "skipped": len(skipped), "saved": len(test)}  # the scores themselves: read scores.json


@app.local_entrypoint()
def main(step: str = "stage", config: str = "swap.yaml"):
    import yaml

    cfg = yaml.safe_load((HERE / "config" / config).read_text())
    keys = sorted(json.loads((HERE / "config" / cfg.get("runs_file", "swap_runs.json")).read_text())["runs"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    tag = "" if config == "swap.yaml" else Path(config).stem + "_"  # local files say which run they belong to
    cache = HERE.parents[1] / ".run_cache" / "q2_swap"
    cache.mkdir(parents=True, exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    if step == "stage":
        r = stage.remote(cfg, keys)
        (HERE / "results" / f"stage_report_{tag}{stamp}.json").write_text(json.dumps({"items": r["items"], "report": r["report"]}, indent=1))
        (cache / f"check_pairs_{tag}{stamp}.json").write_text(json.dumps(r["pairs"]))  # transcript text: not in git
        print(f"staged {r['items']} items; report results/stage_report_{stamp}.json; pairs for the check in {cache}")
    elif step == "score":
        probes = yaml.safe_load((HERE.parent / "2026-10-06_q2_impactful_call_transfer" / "config" / "q2_probes.yaml").read_text())
        n = score.remote(cfg, keys, probes["layer_tolerance"])
        r = json.loads(b"".join(acts_vol.read_file(f"{cfg['extraction_run']}/scores.json")))
        (cache / f"scores_{tag}{stamp}.json").write_text(json.dumps(r))
        print(f"{n['swap']} swap scores, {n['skipped']} skipped; saved scores for {n['saved']} calls")
    elif step == "fetch":
        r = json.loads(b"".join(acts_vol.read_file(f"{cfg['extraction_run']}/scores.json")))  # the run named by --config
        (cache / f"scores_{tag}{stamp}.json").write_text(json.dumps(r))
        print("fetched", len(r["swap"]), "swap scores")
    else:
        raise ValueError(step)
