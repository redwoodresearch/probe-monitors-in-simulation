"""Long-epoch H200 re-reads of four q2 dumps' B200 shards (RULE.md in this folder).

For each dump: a CPU job picks runs from its B200 shards (exact token counts from the stored metas) and copies their stored
rows into small files; then an H200-pinned container imports the dump's own extractor (the file at its launch commit,
byte-identical, sha256-checked) and runs its extraction code unchanged on the picks only, into a new run; then a CPU job
compares the re-read rows with the pre-staged stored rows and writes one result file per dump to the work Volume.
  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run [--detach] \\
      experiments/2026-10-14_q2_b200_longrepro/run_b200_longrepro.py --git <commit> [--check-only] [--only <dump> --reuse <selection record>]
--only/--reuse (the full+F9 follow-up, 0d's (b)): one dump, its selection read from an earlier run's selection record on the
work Volume (same picks, stored rows already staged); the extractor resumes the same re-read run, skipping every pick
already written, so only missing picks are read; then the comparison covers every pick.
--check-only: the CPU selections, pre-staging, extractor import and item checks, and the cost basis; no GPU.
"""
import hashlib
import importlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-b200long_repro_kill-2026-11-14"
WORK = "probe-shift_aswin-surya_sub-q2-b200long_work_kill-2026-11-14"
G3 = "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
SEED, COS_MIN, PEAK_REL_MAX = 20261014, 0.9999, 2.0 ** -6
SEED_PIP = ["torch==2.8.0", "accelerate==1.10.1", "transformers==4.57.1", "jinja2==3.1.6", "huggingface_hub[hf_transfer]==0.35.3", "numpy"]
QWEN_PIP = ["torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy", "huggingface_hub[hf_transfer]"]
# name: extractor family, Volume, stored run, re-read run, picks rule (None = every run on the B200 shards, else the longest
# plus k long-weighted), the as-run extractor (commit, path, sha256) and its launch record (commit, path).
DUMPS = {
    "fresh_s7": {"kind": "seed", "vol": "g3", "run": "2026-10-13_q2_fresh_s7_seedcall_seed-oss-36b_v1",
                 "new": "2026-10-14_q2_b200long_fresh_s7_seedcall_seed-oss-36b_v1", "k": None,
                 "asrun": ("f8cc4598bafadb3ba2ab07f6bef0aedba5be90b0", "experiments/2026-10-13_q2_fresh_honest/run_extract_seedcall_fresh.py",
                           "e19e13f53e0832c18a8bb9cfa85455f81c63b12b63722fb05af89e4c98630104"),
                 "launch": ("d5dc48df", "experiments/2026-10-13_q2_fresh_honest/results/extract_launch_2026-10-13_q2_fresh_s7_seedcall_seed-oss-36b_v1_2026-10-14_075424.json")},
    "fixfull": {"kind": "seed", "vol": "g3", "run": "2026-10-13_fixfull_seed-oss-36b_v1",
                "new": "2026-10-14_q2_b200long_fixfull_seed-oss-36b_v1", "k": 7,
                "asrun": ("b3e59f6555cf1789c0bc67d79126fcba3ccf6427", "experiments/2026-10-13_q2_fixfull/run_extract_g3_fx.py",
                          "a8c915f630d675038d7908188c232ca15066d3d19c620cf59c479d3db7a929af"),
                "launch": ("51846989", "experiments/2026-10-13_q2_fixfull/results/extract_launch_2026-10-13_fixfull_seed-oss-36b_v1_2026-10-13_142924.json")},
    "fixfull_f9": {"kind": "seed", "vol": "g3", "run": "2026-10-13_fixfull_f9_seed-oss-36b_v1",
                   "new": "2026-10-14_q2_b200long_fixfull_f9_seed-oss-36b_v1", "k": None,
                   "asrun": ("b6530d33a95c82b46456f4348cee9652d75fbd21", "experiments/2026-10-13_q2_fixfull/run_extract_g3_f9.py",
                             "d938131193a4801b345bcea44ef0630bdf83c358f2a2285a8dd517fe117ec033"),
                   "launch": ("51846989", "experiments/2026-10-13_q2_fixfull/results/extract_launch_2026-10-13_fixfull_f9_seed-oss-36b_v1_2026-10-13_152337.json")},
    "fxavg": {"kind": "qwen", "vol": "acts", "run": "2026-10-13_q2_fxavg_qwen3-32b_l7_v1",
              "new": "2026-10-14_q2_b200long_fxavg_qwen3-32b_l7_v1", "k": 5, "inputs_new": "2026-10-14_q2_b200long_fxavg_inputs_v1",
              "asrun": ("486d3c6047865fca0d614ce8e6ee8d12bf0e5631", "experiments/2026-10-13_q2_fixfull/run_extract_qwen3_fxavg.py",
                        "127086ddce9ddcce3eb0881b5aaef832f42859afe6421253166e0890dcad2a41"), "launch": None},
}
ASRUN_DIR = HERE / "asrun"  # the as-run extractors, written here from git at launch (not committed: they are in git at their commits)
if modal.is_local():
    ASRUN_DIR.mkdir(exist_ok=True)
    for _n, _d in DUMPS.items():
        _c, _p, _sha = _d["asrun"]
        _b = subprocess.run(["git", "show", f"{_c}:{_p}"], cwd=HERE, capture_output=True, check=True).stdout
        if not hashlib.sha256(_b).hexdigest().startswith(_sha):
            raise SystemExit(f"{_n}: the as-run extractor at {_c} does not hash to {_sha}")
        (ASRUN_DIR / f"asrun_{_n}.py").write_bytes(_b)

app = modal.App(APP)
work = modal.Volume.from_name(WORK, create_if_missing=True, version=2)
g3_ro = modal.Volume.from_name(G3, version=2).read_only()
g3_rw = modal.Volume.from_name(G3, version=2)
acts_ro = modal.Volume.from_name(ACTS, version=2).read_only()
acts_rw = modal.Volume.from_name(ACTS, version=2)
hf = modal.Volume.from_name(HF_CACHE)
ENV = {"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}
seed_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*SEED_PIP).env(ENV).add_local_dir(ASRUN_DIR, "/root/asrun")
qwen_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*QWEN_PIP).env(ENV)
              .add_local_python_source("probe_shift").add_local_dir(ASRUN_DIR, "/root/asrun"))


def _asrun(name):
    sys.path.insert(0, "/root/asrun")
    b = Path(f"/root/asrun/asrun_{name}.py").read_bytes()
    if not hashlib.sha256(b).hexdigest().startswith(DUMPS[name]["asrun"][2]):
        raise RuntimeError(f"{name}: the as-run extractor in the image is not the pinned one")
    return importlib.import_module(f"asrun_{name}")


def _root(d):
    return Path("/g3") if d["vol"] == "g3" else Path("/acts")


def _stem(key):
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def _tokens(m, kind):
    return int(m.get("forward_tokens") or 0) if kind == "seed" else int(m.get("kept_tokens") or m.get("n_tokens") or 0)


def _select(name: str) -> dict:
    """Stored stats -> B200 shards; chunk files -> each shard's transcripts; metas -> tokens; picks; stored rows staged."""
    import numpy as np

    d = DUMPS[name]
    run_dir = _root(d) / d["run"]
    gpu_of = {int(p.stem.split("_")[1]): json.loads(p.read_text()).get("gpu") for p in sorted((run_dir / "stats").glob("shard_*.json"))}
    b200 = sorted(s for s, g in gpu_of.items() if g and "B200" in g)
    rows_seen, by_shard = {}, {s: [] for s in b200}
    for s in b200:
        for f in sorted((run_dir / "chunks").glob(f"shard{s:03d}_chunk*.json")):
            if not (f.with_suffix(".done")).exists():
                continue
            for t in json.loads(f.read_text())["transcripts"]:
                m = json.loads((run_dir / "meta" / t["arm"] / f"{t['filename']}.json").read_text())
                if m["chunk"] != f.stem:
                    continue  # superseded copy
                by_shard[s].append({"shard": s, "arm": t["arm"], "key": m["key"], "filename": t["filename"], "chunk": m["chunk"],
                                    "row_start": m["row_start"], "n_rows": m["n_rows"], "tokens": _tokens(m, d["kind"]),
                                    "n_tokens": m.get("n_tokens"), "original gpu": gpu_of[s]})
    rng, picks = np.random.default_rng(SEED), []
    for s in b200:
        ts = sorted(by_shard[s], key=lambda t: (-t["tokens"], t["arm"], t["key"]))
        if not ts:
            continue
        for t in ts:
            t["shard longest"] = t is ts[0]
        if d["k"] is None:
            chosen = ts
        else:
            rest = ts[1:]
            w = np.array([t["tokens"] for t in rest], dtype=float)
            k = min(d["k"], len(rest))
            idx = rng.choice(len(rest), k, replace=False, p=w / w.sum()).tolist() if k else []
            chosen = [ts[0]] + [rest[i] for i in sorted(idx)]
        picks += chosen
    layers = json.loads((run_dir / "chunks" / f"{picks[0]['chunk']}.json").read_text())["layers"] if picks else []
    out = Path("/work") / name / "stored"
    out.mkdir(parents=True, exist_ok=True)
    for p in picks:  # the stored rows of each pick, every stored layer, one small file per run
        arrs = {f"L{l}": np.asarray(np.load(run_dir / "by_layer" / f"L{l}" / f"{p['chunk']}.npy", mmap_mode="r")[p["row_start"]: p["row_start"] + p["n_rows"]])
                for l in layers}
        np.savez(out / f"{p['arm']}__{p['filename']}.npz", **arrs)
        p["stored rows file"] = f"{name}/stored/{p['arm']}__{p['filename']}.npz"
    stats_timing = {}
    for s, g in gpu_of.items():
        st = json.loads((run_dir / "stats" / f"shard_{s}.json").read_text())
        ft = st.get("forward [tokens, seconds]") or []
        stats_timing[s] = [sum(x[0] for x in ft), sum(x[1] for x in ft)]
    h200 = [v for s, v in stats_timing.items() if v[1] and "H200" in (gpu_of[s] or "")]
    rate = sum(v[0] for v in h200) / sum(v[1] for v in h200) if h200 else None
    return {"dump": name, "run": d["run"], "stored config": json.loads((run_dir / "config.json").read_text()), "gpu by shard": gpu_of,
            "B200 shards": b200, "transcripts on B200 shards": {s: len(v) for s, v in by_shard.items()},
            "longest tokens by B200 shard": {s: max((t["tokens"] for t in v), default=0) for s, v in by_shard.items()},
            "layers": layers, "picks": picks, "pick tokens": sum(p["tokens"] for p in picks),
            "H200 forward tokens per second (stored stats)": rate,
            "estimated forward seconds on H200": round(sum(p["tokens"] for p in picks) / rate, 1) if rate else None}


def _seed_items(name: str, sel: dict, launch_inputs: dict):
    """The module's own work_items on the ORIGINAL staged inputs (checked against the launch record), filtered to picks."""
    mod = _asrun(name)
    d, cfg = DUMPS[name], sel["stored config"]
    if mod.PIP_GPU != SEED_PIP:
        raise RuntimeError(f"{name}: pip pins differ from the extractor's")
    which = cfg["which"]
    if which == "fx":
        pref = f"{mod.FX_INPUTS}/inputs/"
        shas = {k[len(pref):]: v for k, v in launch_inputs.items() if k.startswith(pref)}
        files = [Path("/acts") / mod.FX_INPUTS / "inputs" / n for n in sorted(shas)]
    elif which == "f10":
        batch = next(b for b, v in mod.BATCHES.items() if v["run"] == d["run"])
        bt = mod.BATCHES[batch]
        pref = f"{bt['staged']}/inputs/"
        names = sorted(k[len(pref):] for k in launch_inputs if k.startswith(pref))
        files = [Path("/acts") / bt["staged"] / "inputs" / n for n in names]
        arm_of = {f.name: json.loads(f.read_text().splitlines()[0])["arm"] for f in files}
        shas = {"f10": {n: {"arm": arm_of[n], "sha256": launch_inputs[pref + n]} for n in names}, "staged": bt["staged"],
                "qwen_run": bt["qwen_run"], "arms": list(bt["arms"])}
    else:
        raise RuntimeError(f"{name}: unexpected which {which!r}")
    cohort = [{"arm": x["arm"], "key": x["key"], "filename": mod.filename(x["key"])}
              for f in files for x in map(json.loads, f.read_text().splitlines())]
    items = mod.work_items(which, cohort, shas)  # checks every staged file's sha256 against the launch record's
    want = {(p["arm"], p["key"]) for p in sel["picks"]}
    mine = [it for it in items if (it[0], it[2]["key"]) in want]
    if len(mine) != len(want):
        raise RuntimeError(f"{name}: {len(want) - len(mine)} picks are not among the extractor's items")
    return mod, mine, cohort, shas


@app.function(image=seed_image, cpu=8, memory=32768, timeout=1800, retries=0, volumes={"/g3": g3_ro, "/acts": acts_ro, "/work": work})
def select_seed(name: str, launch_inputs: dict) -> dict:
    sel = _select(name)
    mod, mine, _, _ = _seed_items(name, sel, launch_inputs)  # import and item checks before any GPU
    sel |= {"extractor items for the picks": len(mine), "raw extract_shard callable": callable(_raw(mod.extract_shard))}
    work.commit()
    return sel


@app.function(image=qwen_image, cpu=8, memory=32768, timeout=1800, retries=0, volumes={"/acts": acts_rw, "/work": work})
def select_fxavg(name: str) -> dict:
    sel = _select(name)
    d, cfg = DUMPS[name], sel["stored config"]
    mod = _asrun(name)
    if mod.PIP != QWEN_PIP:
        raise RuntimeError("fxavg: pip pins differ from the extractor's")
    want = {(p["arm"], p["key"]) for p in sel["picks"]}
    src = Path("/acts") / cfg["inputs_from"] / "inputs"
    lines = [l for f in sorted(src.glob("shard_*.jsonl")) for l in f.read_text().splitlines()
             if (lambda x: (x["arm"], x["key"]) in want)(json.loads(l))]
    if len(lines) != len(want):
        raise RuntimeError(f"fxavg: {len(want) - len(lines)} picks are not in the staged inputs")
    dst = Path("/acts") / d["inputs_new"] / "inputs" / "shard_0.jsonl"
    body = ("\n".join(lines) + "\n").encode()
    if dst.exists() and dst.read_bytes() != body:
        raise RuntimeError("fxavg: the subset inputs already exist with different content")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(body)
    acts_rw.commit()
    sel |= {"subset inputs": f"{d['inputs_new']}/inputs/shard_0.jsonl", "subset inputs sha256": hashlib.sha256(body).hexdigest(),
            "subset lines byte-identical to the stored run's staged lines": True}
    work.commit()
    return sel


def _raw(f):
    return f.get_raw_f() if hasattr(f, "get_raw_f") else f.local


@app.function(image=seed_image, gpu="H200", memory=65536, timeout=3600, retries=0, volumes={"/hf": hf, "/out": g3_rw, "/acts": acts_ro})
def reread_seed(name: str, sel: dict, launch_inputs: dict) -> dict:
    import torch

    device = torch.cuda.get_device_name(0)
    print(f"{name}: container GPU {device}; {len(sel['picks'])} picks", flush=True)
    sel = sel | {"stored config": json.loads((Path("/out") / DUMPS[name]["run"] / "config.json").read_text())}
    mod, mine, cohort, shas = _seed_items(name, sel, launch_inputs)
    mod.work_items = lambda which, c, s: list(mine)  # only the selection is narrowed; extraction code runs unchanged
    cfg = dict(sel["stored config"]) | {"run": DUMPS[name]["new"], "shards": 1, "chunk_size": 1, "limit_items": None,
                                        "ids sha256": mod.ids_sha((a, x["key"]) for a, _, x, _ in mine)}
    cfg.pop("assign", None)
    t0 = time.time()
    summary = _raw(mod.extract_shard)(cfg, 0, cohort, shas)
    return {"device": device, "seconds": round(time.time() - t0), "summary": summary}


@app.function(image=qwen_image, gpu="H200", memory=65536, timeout=3600, retries=0, volumes={"/hf": hf, "/out": acts_rw})
def reread_fxavg(name: str, sel: dict) -> dict:
    import torch

    device = torch.cuda.get_device_name(0)
    print(f"{name}: container GPU {device}; {len(sel['picks'])} picks", flush=True)
    mod = _asrun(name)
    d = DUMPS[name]
    cfg = json.loads((Path("/out") / d["run"] / "config.json").read_text())
    cfg |= {"run": d["new"], "inputs_from": d["inputs_new"], "shards": 1, "chunk_size": 1, "limit_items": None}
    t0 = time.time()
    summary = mod._extract_shard(cfg, 0)
    return {"device": device, "seconds": round(time.time() - t0), "summary": summary}


def _pos(m):
    return [(p.get("kind"), p.get("step"), p.get("action"), p.get("token"), p.get("kept"), p.get("row")) for p in m["positions"]]


@app.function(image=seed_image, cpu=4, memory=16384, timeout=1800, retries=0, volumes={"/g3": g3_ro, "/acts": acts_ro, "/work": work})
def compare(name: str, sel: dict, gpu: dict, git: str, stamp: str) -> str:
    import numpy as np

    d = DUMPS[name]
    root, layers = _root(d), sel["layers"]
    new_dir, old_dir = root / d["new"], root / d["run"]
    recs = []
    for p in sel["picks"]:
        r = {k: p[k] for k in ("shard", "arm", "key", "tokens", "n_tokens", "shard longest", "original gpu")} | {"re-read gpu": gpu.get("device")}
        nm, om = new_dir / "meta" / p["arm"] / f"{p['filename']}.json", old_dir / "meta" / p["arm"] / f"{p['filename']}.json"
        if not nm.exists():
            recs.append(r | {"verdict": "INCONCLUSIVE", "why": "no re-read"})
            continue
        new_m = json.loads(nm.read_text())
        r["gate P (positions identical)"] = _pos(new_m) == _pos(json.loads(om.read_text())) and new_m["n_rows"] == p["n_rows"]
        if not r["gate P (positions identical)"]:
            recs.append(r | {"verdict": "INCONCLUSIVE", "why": "positions differ"})
            continue
        stored = np.load(Path("/work") / p["stored rows file"])
        per, exact, rows, worst_cos, max_abs, max_rel, nonfin = {}, 0, 0, 1.0, 0.0, 0.0, 0
        for l in layers:
            s = stored[f"L{l}"].astype(np.float32)
            n = np.asarray(np.load(new_dir / "by_layer" / f"L{l}" / f"{new_m['chunk']}.npy", mmap_mode="r")[new_m["row_start"]: new_m["row_start"] + new_m["n_rows"]])
            nonfin += int((~np.isfinite(n)).sum())
            n = n.astype(np.float32)
            dif = np.abs(n - s)
            ns, nn = np.linalg.norm(s, axis=1), np.linalg.norm(n, axis=1)
            cos = np.where((ns > 0) & (nn > 0), (s * n).sum(1) / np.maximum(ns * nn, 1e-30), (ns == nn).astype(np.float32))
            rel = dif.max(1, initial=0) / np.maximum(np.abs(s).max(1, initial=0), 1e-30)
            e = int((dif.max(1, initial=0) == 0).sum())
            per[str(l)] = {"rows": len(s), "bit-exact rows": e, "worst cosine": float(cos.min(initial=1.0)), "max abs diff": float(dif.max(initial=0))}
            exact, rows = exact + e, rows + len(s)
            worst_cos, max_abs, max_rel = min(worst_cos, float(cos.min(initial=1.0))), max(max_abs, float(dif.max(initial=0))), max(max_rel, float(rel.max(initial=0)))
        v = "EXACT" if exact == rows and nonfin == 0 else ("PASS" if worst_cos >= COS_MIN and max_rel <= PEAK_REL_MAX and nonfin == 0 else "FAIL")
        recs.append(r | {"bit-exact rows / rows compared": [exact, rows], "worst cosine": worst_cos, "max abs diff": max_abs,
                         "max peak-relative diff": max_rel, "non-finite": nonfin, "per layer": per, "verdict": v})
    order = {"EXACT": 0, "PASS": 1, "INCONCLUSIVE": 2, "FAIL": 3}
    shards = {}
    for r in recs:
        shards.setdefault(r["shard"], []).append(r)
    by_shard = {str(s): {"original gpu": rs[0]["original gpu"], "re-read gpu": gpu.get("device"), "transcripts": len(rs),
                         "longest tokens (stored)": sel["longest tokens by B200 shard"].get(str(s), sel["longest tokens by B200 shard"].get(s)),
                         "longest re-read": any(r["shard longest"] for r in rs),
                         "worst cosine": min((r["worst cosine"] for r in rs if "worst cosine" in r), default=None),
                         "max abs diff": max((r["max abs diff"] for r in rs if "max abs diff" in r), default=None),
                         "bit-exact rows / rows": [sum(r.get("bit-exact rows / rows compared", [0, 0])[0] for r in rs),
                                                   sum(r.get("bit-exact rows / rows compared", [0, 0])[1] for r in rs)],
                         "verdict": max((r["verdict"] for r in rs), key=order.get)} for s, rs in sorted(shards.items())}
    counts = {k: sum(v["verdict"] == k for v in by_shard.values()) for k in order}
    out = {"POST HOC": True, "rule": "RULE.md (experiments/2026-10-14_q2_b200_longrepro)", "git": git, "stamp (PT)": stamp,
           "earlier re-read containers": gpu.get("earlier containers"),
           "dump": name, "stored run": d["run"], "re-read run": d["new"], "as-run extractor": d["asrun"],
           "bar": {"cosine >=": COS_MIN, "max peak-relative diff <=": PEAK_REL_MAX},
           "re-read container": {k: gpu.get(k) for k in ("device", "seconds")}, "re-read stats": gpu.get("summary"),
           "selection": {k: v for k, v in sel.items() if k not in ("picks", "stored config")},
           "per transcript": recs, "per shard": by_shard,
           "per dump": {"shards by verdict": counts, "transcripts by verdict": {k: sum(r["verdict"] == k for r in recs) for k in order},
                        "verdict": max((v["verdict"] for v in by_shard.values()), key=order.get, default="INCONCLUSIVE")},
           "status": "preliminary"}
    path = Path("/work") / "results" / f"b200long_{name}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    work.commit()
    print(f"{name}: wrote results/{path.name}: {out['per dump']['verdict']} {counts}", flush=True)
    return f"results/{path.name}"


@app.function(image=seed_image, cpu=1, memory=4096, timeout=3 * 3600, retries=0, volumes={"/work": work})
def orchestrate(git: str, stamp: str, check_only: bool, launch_inputs: dict, only: str = "", reuse: str = "", earlier: dict | None = None) -> dict:
    names = [only] if only else list(DUMPS)
    sels, failed = {}, {}
    if reuse:  # the earlier run's own selections: same picks, stored rows already staged
        prior = json.loads((Path("/work") / reuse).read_text())["selections"]
        sels = {n: prior[n] for n in names}
    else:
        calls = {n: (select_fxavg.spawn(n) if DUMPS[n]["kind"] == "qwen" else select_seed.spawn(n, launch_inputs[n])) for n in names}
        for n, c in calls.items():
            try:
                sels[n] = c.get()
            except Exception as e:
                failed[n] = type(e).__name__
    rec = {"POST HOC": True, "git": git, "stamp (PT)": stamp, "selections": sels, "failed": failed}
    (Path("/work") / "results").mkdir(parents=True, exist_ok=True)
    (Path("/work") / "results" / f"b200long_check_{stamp}.json").write_text(json.dumps(rec, indent=1))
    work.commit()
    if check_only or failed:
        return {"check": f"results/b200long_check_{stamp}.json", "failed": failed}
    gcalls = {n: (reread_fxavg.spawn(n, s) if DUMPS[n]["kind"] == "qwen" else reread_seed.spawn(n, s, launch_inputs[n])) for n, s in sels.items()}
    done = {}
    while gcalls:
        for n, c in list(gcalls.items()):
            try:
                g = c.get(timeout=20)
            except Exception as e:
                if "Timeout" in type(e).__name__:
                    continue
                g = {"error": type(e).__name__}
            del gcalls[n]
            if earlier and n in earlier:
                g = g | {"earlier containers": earlier[n]}
            done[n] = compare.remote(n, sels[n], g, git, stamp) if "error" not in g else f"re-read failed: {g['error']}"
            print(f"{n}: {done[n]}", flush=True)
    return {"check": f"results/b200long_check_{stamp}.json", "results": done}


@app.local_entrypoint()
def main(git: str, check_only: bool = False, only: str = "", reuse: str = ""):
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    launch_inputs = {}
    for n, d in DUMPS.items():
        if d["launch"]:
            c, p = d["launch"]
            launch_inputs[n] = json.loads(subprocess.run(["git", "show", f"{c}:{p}"], cwd=HERE, capture_output=True, check=True).stdout)["inputs sha256"]
    earlier = {"fixfull_f9": {"app": "ap-cwkoBqwNwPOSQtPTz2c6Bv", "device": "NVIDIA H200 (its log line 'fixfull_f9: container GPU NVIDIA H200; 58 picks')",
                              "re-reads written": "57 of 58 before the 14:34:16 PT watcher stop"}} if only == "fixfull_f9" and reuse else None
    print(json.dumps(orchestrate.remote(git, stamp, check_only, launch_inputs, only, reuse, earlier)), flush=True)
