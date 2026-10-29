"""Off-policy 2x2 for the impactful-command probe with a bootstrap that refits (config/refit.yaml; README.md).

Steps (each a Modal call; the local client exits after launching `run`):
  prepare  (CPU)  per reader: the matched attack calls' rows of the Q2 features manifest, and those rows of the 16
                  candidate layers' call_end packs, written once to the output Volume (<out_run>/<reader>/).
  pilot    (GPU)  replicates 0-2 of one reader, timed, for the cost estimate.
  run      (CPU driver + GPU workers)  replicates 0..n_boot of both readers (replicate 0 is the point estimate) and
                  n_null label-shuffle draws. Each worker writes one file per item into replicates/<fingerprint>/ and
                  skips items already written there, so a relaunch with the same code and config resumes, and one with
                  other code or config starts a new directory instead of mixing.
  aggregate (CPU) collects the current fingerprint's files; the local client writes results/refit_<stamp>.json with
                  percentile intervals for every cell, the domain gaps and the transfer losses, and each cell against
                  its shuffle null.

  uv run modal run experiments/2026-10-08_impactful_refit_bootstrap/run_refit_bootstrap.py --step prepare
  uv run modal run --detach experiments/2026-10-08_impactful_refit_bootstrap/run_refit_bootstrap.py --step run --readers qwen3.5
"""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import yaml

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # in a Modal container the script sits at /root
CFG = yaml.safe_load((HERE / "config" / "refit.yaml").read_text())
APP = "probe-shift_aswin-surya_sub-q2-refit_boot_kill-2026-11-08"
WORKERS = 12  # GPU containers at most; the watcher stops the app above WORKERS + 1 (the driver)
app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn", "pyyaml", "torch")
         .add_local_file(HERE / "config" / "refit.yaml", "/root/config/refit.yaml")  # the module reads it at import
         .add_local_python_source("probe_shift"))
feats = {r: modal.Volume.from_name(c["volume"], version=2) for r, c in CFG["readers"].items()}
out_vol = modal.Volume.from_name(CFG["out_volume"], version=2, create_if_missing=True)
mounts = {f"/feats_{r.replace('.', '')}": v for r, v in feats.items()} | {"/out": out_vol}


def fingerprint(rows_sha: str, packs_sha: str) -> str:
    """What a replicate file depends on: the config (less the replicate counts), the rows it was fit on, every
    probe_shift module the fitting, thresholds and resampling run, the bytes of the activation packs it read, the source of `write_rows` (which writes the packs)
    and `run_items` (which loads them), and the numeric packages' versions. Not the rest of this file, so editing the
    summary code keeps finished replicates readable."""
    import inspect

    import numpy
    import sklearn
    import torch

    import probe_shift.eligibility as e
    import probe_shift.probes as p
    import probe_shift.refit_bootstrap as rb
    import probe_shift.transfer as tr

    cfg = {k: v for k, v in CFG.items() if k not in ("n_boot", "n_null")}
    parts = [json.dumps(cfg, sort_keys=True), rows_sha, packs_sha] + [Path(m.__file__).read_text() for m in (e, p, rb, tr)] + \
            [inspect.getsource(f) for f in (write_rows, run_items)] + [m.__version__ for m in (numpy, sklearn, torch)]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def packs_sha(d: Path) -> str:
    """sha256 over the copied activation packs' bytes, in the config's layer order (streamed)."""
    h = hashlib.sha256()
    for layer in CFG["layers"]:
        with open(d / f"L{layer}.npy", "rb") as fh:
            while chunk := fh.read(1 << 26):
                h.update(chunk)
    return h.hexdigest()


def rep_dir(reader: str) -> tuple[Path, str, str]:
    """This code, config, row set and pack contents' replicate directory for a reader, its fingerprint and the packs'
    sha256."""
    d = Path("/out") / CFG["out_run"] / reader
    packs = packs_sha(d)
    fp = fingerprint(hashlib.sha256((d / "rows.json").read_bytes()).hexdigest(), packs)
    return d / "replicates" / fp, fp, packs


def feats_dir(reader):
    return Path(f"/feats_{reader.replace('.', '')}") / CFG["readers"][reader]["features_run"]


@app.function(image=image, volumes=mounts, cpu=4, memory=16384, timeout=3600, retries=0)
def prepare(reader: str, keys: list) -> dict:
    return write_rows(reader, keys)


def write_rows(reader: str, keys: list) -> dict:
    """keys: [arm, key, step, action, y] for every matched attack call. Writes rows.json and L<l>.npy (rows x hidden)."""
    import numpy as np

    d = feats_dir(reader)
    manifest = json.loads((d / "manifest.json").read_text())
    at = {(m["arm"], m["key"], m["step"], m["action"]): i for i, m in enumerate(manifest)}
    rows, missing, y_mismatch = [], 0, 0
    for arm, key, step, action, y in keys:
        i = at.get((arm, key, step, action))
        if i is None:
            missing += 1
            continue
        y_mismatch += int(manifest[i]["y"] != y)
        rows.append(i)
    if y_mismatch:
        raise ValueError(f"{reader}: {y_mismatch} labels differ between the labels file and the manifest")
    rows = np.array(sorted(rows))
    out = Path("/out") / CFG["out_run"] / reader
    out.mkdir(parents=True, exist_ok=True)
    meta = [{k: manifest[i][k] for k in ("arm", "key", "step", "action", "fold", "y", "main_task")} for i in rows]
    (out / "rows.json").write_text(json.dumps(meta))
    for layer in CFG["layers"]:
        pack = np.load(d / "alllayers" / f"L{layer}" / f"{CFG['readout']}.npy", mmap_mode="r")
        if pack.shape[0] != len(manifest):
            raise ValueError(f"{reader} L{layer}: pack has {pack.shape[0]} rows, manifest {len(manifest)}")
        x = np.asarray(pack[rows])
        if not np.isfinite(x.astype(np.float32)).all():
            raise ValueError(f"{reader} L{layer}: non-finite values")
        tmp = out / f".L{layer}.npy.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, x)
        tmp.rename(out / f"L{layer}.npy")
    out_vol.commit()
    return {"reader": reader, "rows": len(rows), "keys": len(keys), "not_in_manifest": missing,
            "per_world": {a: sum(m["arm"] == a for m in meta) for a in CFG["arms"].values()}}


@app.function(image=image, volumes={"/out": out_vol}, gpu=CFG["gpu"], memory=32768, timeout=8 * 3600, retries=0,
              max_containers=WORKERS)
def worker(reader: str, items: list[list]) -> dict:
    return run_items(reader, items)


def run_items(reader: str, items: list[list]) -> dict:
    """items: [kind, i] pairs, kind "boot" (replicate i) or "null" (shuffle draw i); see refit_bootstrap.run_item."""
    import numpy as np
    import torch

    from probe_shift.refit_bootstrap import run_item

    out_vol.reload()
    d = Path("/out") / CFG["out_run"] / reader
    meta = json.loads((d / "rows.json").read_text())
    reps, fp, packs = rep_dir(reader)
    rows_sha = hashlib.sha256((d / "rows.json").read_bytes()).hexdigest()
    tasks = np.unique([m["main_task"] for m in meta])  # the order of task_counts
    X = torch.stack([torch.from_numpy(np.load(d / f"L{l}.npy")) for l in CFG["layers"]]).to("cuda")
    reps.mkdir(parents=True, exist_ok=True)
    done, seconds = [], []
    for kind, i in items:
        f = reps / f"{kind}_{i:04d}.json"
        if f.exists():
            if json.loads(f.read_text()).get("fingerprint") != fp:
                raise ValueError(f"{f}: its fingerprint differs from its directory's")
            continue
        t0 = time.time()
        r = run_item(X, meta, kind, i, CFG)
        r |= {"kind": kind, "replicate": i, "seconds": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(), "fingerprint": fp,
              "packs_sha256": packs}
        # Every row's score from each fold's probe, the task weights and the labels used (shuffled for a null draw), in
        # rows.json's order (arm, key = the run, step, action): any subset of runs can be rescored without refitting.
        arrays = {k: r.pop(k) for k in ("scores", "task_counts", "labels") if k in r}
        tmp = reps / f".scores_{kind}_{i:04d}.npz.tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays, tasks=tasks, rows_sha256=rows_sha, packs_sha256=packs)
        tmp.rename(reps / f"scores_{kind}_{i:04d}.npz")
        r["scores_file"] = f"scores_{kind}_{i:04d}.npz"
        if "probes" in r:  # the point estimate's 10 probes (5 folds x 2 training worlds), before its JSON marks it done
            pr = r.pop("probes")
            arrays = {k: np.stack([p[k] for p in pr]) for k in ("w", "mu", "sd")}
            table = [{k: v for k, v in p.items() if k not in arrays} for p in pr]
            tmp = reps / f".probes_{kind}_{i:04d}.npz.tmp"
            with open(tmp, "wb") as fh:
                np.savez(fh, **arrays, table=json.dumps(table), rows_sha256=rows_sha, packs_sha256=packs,
                         score="((x.float() - mu) / sd) @ w + b at the probe's layer; flags are score > threshold_<test world>")
            tmp.rename(reps / f"probes_{kind}_{i:04d}.npz")
            r["probes_file"] = f"probes_{kind}_{i:04d}.npz"
        tmp = reps / f".{kind}_{i:04d}.json.tmp"
        tmp.write_text(json.dumps(r))
        tmp.rename(f)
        out_vol.commit()
        done.append(f"{kind} {i}"), seconds.append(r["seconds"])
    return {"reader": reader, "fingerprint": fp, "done": done, "seconds": seconds}


@app.function(image=image, volumes={"/out": out_vol}, cpu=1, memory=2048, timeout=12 * 3600, retries=0)
def driver(n_boot: int, n_null: int, readers: list[str], done_file: str) -> dict:
    """Each reader's replicates 0..n_boot and shuffle draws 1..n_null, dealt round-robin into WORKERS chunks per reader."""
    t0 = time.time()
    items = [["boot", i] for i in range(n_boot + 1)] + [["null", i] for i in range(1, n_null + 1)]
    jobs = [(r, items[k::WORKERS]) for r in readers for k in range(WORKERS)]
    res = list(worker.starmap(jobs, return_exceptions=True))
    errors = [repr(r)[:300] for r in res if isinstance(r, Exception)]
    summary = {"wall_seconds": round(time.time() - t0), "n_ok": len(res) - len(errors), "n_crashed": len(errors), "n_unreachable": 0,
               "errors": errors}
    stats = Path("/out") / CFG["out_run"] / "stats"  # the watcher's done file, one per launch
    stats.mkdir(parents=True, exist_ok=True)
    if (stats / done_file).exists():
        raise FileExistsError(f"{stats / done_file}: a launch with this name already finished")
    (stats / done_file).write_text(json.dumps(summary | {"readers": readers, "n_boot": n_boot, "n_null": n_null}))
    out_vol.commit()
    if errors:
        raise RuntimeError(f"{len(errors)} of {len(jobs)} workers failed: {errors[:2]}")
    return summary


@app.function(image=image, volumes={"/out": out_vol}, cpu=1, memory=4096, timeout=1800, retries=0)
def aggregate(to_volume: bool = False) -> dict:
    """Every finished item per reader. to_volume: also write them to <out_run>/stats/aggregate_items.json, so a detached
    call survives the local client disconnecting (then `--step aggregate --from-volume` reads that file)."""
    out_vol.reload()
    res = {}
    for reader in CFG["readers"]:
        reps, fp, packs = rep_dir(reader)
        rows = json.loads((Path("/out") / CFG["out_run"] / reader / "rows.json").read_text())  # the calls this reader analyzed
        res[reader] = {"fingerprint": fp, "packs_sha256": packs, "items": [json.loads(f.read_text()) for f in sorted(reps.glob("*_[0-9]*.json"))],
                       "analyzed calls": {a: {"impactful": sum(r["arm"] == a and r["y"] == 1 for r in rows), "read-only": sum(r["arm"] == a and r["y"] == 0 for r in rows)}
                                          for a in CFG["arms"].values()}}
    if to_volume:
        (Path("/out") / CFG["out_run"] / "stats" / "aggregate_items.json").write_text(json.dumps(res))
        out_vol.commit()
    return res


@app.function(image=image, volumes={"/out": out_vol}, cpu=2, memory=4096, timeout=1800, retries=0)
def fingerprints() -> dict:
    """Each reader's fingerprint under the current code, config, rows and packs (hashes the packs: about a minute)."""
    out_vol.reload()
    return {r: rep_dir(r)[1] for r in CFG["readers"]}


def eligible_keys() -> list:
    """[arm, key, step, action, y] for each call of the two attack arms in the matched subset (eligibility.matched_call:
    the last argument is the non-empty command parameter, so call_end reads the on-policy readout's token)."""
    from probe_shift.eligibility import matched_call

    arms = set(CFG["arms"].values())
    keys = []
    with open(ROOT / CFG["labels_file"]) as f:
        for line in f:
            r = json.loads(line)
            if r["arm"] in arms and matched_call(r["tool"], r["arguments"]):
                keys.append([r["arm"], r["key"], r["step"], r["action"], int(r["label"] == "impactful")])
    return keys


def summarize(items: list[dict]) -> dict:
    """Point estimate (replicate 0) and 95% percentile intervals over replicates 1..n for every cell and contrast; each
    cell's catch and AUROC against the shuffle draws (null mean, max, and p = (1 + draws >= point) / (1 + draws))."""
    import numpy as np

    nulls = [r for r in items if r["kind"] == "null" and "undefined" not in r]
    reps = [r for r in items if r["kind"] == "boot"]
    ok = [r for r in reps if "undefined" not in r]
    point = next((r for r in ok if r["replicate"] == 0), None)
    boots = [r for r in ok if r["replicate"] != 0]
    cells = ["real->real", "real->sim", "sim->real", "sim->sim"]
    metrics = [k for k in point["real->real"] if not k.startswith("n_")]
    contrasts = {"domain gap, trained on real (real->real minus real->sim)": ("real->real", "real->sim"),
                 "domain gap, trained on sim (sim->real minus sim->sim)": ("sim->real", "sim->sim"),
                 "transfer loss on the simulated target (sim->sim minus real->sim)": ("sim->sim", "real->sim"),
                 "transfer loss on the real target (real->real minus sim->real)": ("real->real", "sim->real")}
    ci = lambda vals: [round(float(x), 4) for x in np.nanpercentile(vals, [2.5, 97.5])] if vals else None
    out = {"replicates": {"point": point is not None, "bootstrap": len(boots), "undefined": len(reps) - len(ok),
                          "null draws": len(nulls), "null undefined": sum(r["kind"] == "null" for r in items) - len(nulls)}}
    for c in cells:
        out[c] = {m: {"value": round(point[c][m], 4), "ci95": ci([b[c][m] for b in boots])} for m in metrics} | \
                 {"n_pos": point[c]["n_pos"], "n_neg": point[c]["n_neg"]}
        for m in (m for m in metrics if not m.startswith("false-alarm") and nulls):
            v = np.array([r[c][m] for r in nulls])
            out[c][m]["null"] = {"mean": round(float(v.mean()), 4), "max": round(float(v.max()), 4),
                                 "p": round(float((1 + (v >= point[c][m]).sum()) / (1 + len(v))), 4)}
    for name, (a, b) in contrasts.items():
        out[name] = {m: {"value": round(point[a][m] - point[b][m], 4), "ci95": ci([r[a][m] - r[b][m] for r in boots])} for m in metrics}
    picks = [p for r in ok for p in r["picks"]]
    out["picks"] = {"layers": dict(sorted({str(l): sum(p["layer"] == l for p in picks) for l in {p["layer"] for p in picks}}.items())),
                    "C": {str(c): sum(p["C"] == c for p in picks) for c in {p["C"] for p in picks}},
                    "max grad": max(p["grad_max"] for p in picks)}
    out["seconds per replicate (median)"] = float(np.median([r["seconds"] for r in reps]))
    return out


@app.local_entrypoint()
def main(step: str = "prepare", reader: str = "qwen3.5", n_boot: int = 0, readers: str = "", partial: bool = False,
         from_volume: bool = False):
    """--reader: the pilot's reader. --readers: comma-separated readers for `run`, required (no default). --partial:
    let `aggregate` write a result from an incomplete replicate set, labeled partial with the counts found."""
    if n_boot < 0 or n_boot > CFG["n_boot"]:  # more resamples is a separate spend decision: change the config, with its OK
        raise ValueError(f"--n-boot {n_boot} exceeds the configured n_boot {CFG['n_boot']}")
    n_boot, n_null = n_boot or CFG["n_boot"], CFG["n_null"]
    run_readers = readers.split(",") if readers else []
    if len(run_readers) != len(set(run_readers)):  # a repeated reader would run its jobs twice on the same files
        raise ValueError(f"--readers repeats a reader: {readers}")
    if step == "run" and not run_readers:  # each reader is a separate spend decision: name them explicitly
        raise ValueError(f"--step run needs --readers (comma-separated, from {list(CFG['readers'])})")
    if not set(run_readers) <= set(CFG["readers"]):
        raise ValueError(f"unknown readers: {set(run_readers) - set(CFG['readers'])}")
    if step == "prepare":
        keys = eligible_keys()
        for r in CFG["readers"]:
            print(prepare.remote(r, keys))
    elif step == "pilot":
        t0 = time.time()
        print(worker.remote(reader, [["boot", 0], ["boot", 1], ["boot", 2]]), f"wall {time.time() - t0:.0f} s")
    elif step == "run":
        done_file = f"summary_{'+'.join(run_readers)}_{datetime.now(ZoneInfo('America/Los_Angeles')):%Y-%m-%d_%H%M%S}.json"
        call = driver.spawn(n_boot, n_null, run_readers, done_file)
        print(f"driver spawned ({call.object_id}): {run_readers} x {WORKERS} workers, replicates 0..{n_boot}, "
              f"{n_null} shuffle draws; the watcher's done file: stats/{done_file}")
    elif step == "fingerprints":  # each reader's replicate directory under the current code, config and data
        print(fingerprints.remote())
    elif step == "aggregate_detached":  # with `modal run --detach`: the items land on the Volume even if this client drops
        print("aggregate spawned:", aggregate.spawn(True).object_id)
    elif step == "aggregate":
        if from_volume:
            reps = json.loads(b"".join(out_vol.read_file(f"{CFG['out_run']}/stats/aggregate_items.json")))
            now = fingerprints.remote()  # the cached items must come from this code, config and data
            if set(reps) != set(now):
                raise ValueError(f"stats/aggregate_items.json covers readers {sorted(reps)}, the config {sorted(now)}")
            stale = {r: [v["fingerprint"], now[r]] for r, v in reps.items() if v["fingerprint"] != now[r]}
            if stale:
                raise ValueError(f"stats/aggregate_items.json was written under other code, config or data ([file, now]): {stale}; "
                                 "rerun --step aggregate_detached")
        else:
            reps = aggregate.remote()
        want = {("boot", i) for i in range(CFG["n_boot"] + 1)} | {("null", i) for i in range(1, n_null + 1)}  # the config's counts
        # The fingerprint leaves out the replicate counts, so a directory can hold items beyond this config (a larger
        # earlier launch): summarize only the configured set, and say how many were left out.
        extra = {r: sorted(f"{it['kind']} {it['replicate']}" for it in v["items"] if (it["kind"], it["replicate"]) not in want) for r, v in reps.items()}
        for r, v in reps.items():
            v["items"] = [it for it in v["items"] if (it["kind"], it["replicate"]) in want]
        if any(extra.values()):
            print("left out, beyond the configured n_boot / n_null:", {r: f"{len(e)} ({e[:3]}...)" for r, e in extra.items() if e})
        missing = {r: len(want - {(it["kind"], it["replicate"]) for it in v["items"]}) for r, v in reps.items()}
        no_point = [r for r, v in reps.items() if v["items"] and not any(it["kind"] == "boot" and it["replicate"] == 0 and "undefined" not in it
                                                                           for it in v["items"])]
        if no_point:
            raise ValueError(f"no point estimate (boot 0) for {no_point}")
        if any(missing.values()) and not partial:  # a stopped or failed run must not read as complete
            raise ValueError(f"missing items per reader (of {len(want)}): {missing}; rerun `run` to finish, or pass --partial")
        res = {"config": CFG, "complete": not any(missing.values()), "missing items": missing,
               "left out (beyond the configured counts)": {r: len(e) for r, e in extra.items()},
               "readers": {r: {"fingerprint": v["fingerprint"], "analyzed calls": v["analyzed calls"]} | summarize(v["items"])
                           for r, v in reps.items() if v["items"]}}
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        out = HERE / "results" / f"refit_{stamp}{'' if res['complete'] else '_partial'}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
        print(out, {r: v["replicates"] for r, v in res["readers"].items()})
    else:
        raise ValueError(step)
