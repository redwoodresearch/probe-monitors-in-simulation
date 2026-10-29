"""Q2 at all 64 layers: per-layer 2x2 curves for call_end, pre_call and prev_content, and the inner-CV headline layer.

The qwen35 track's copy of experiments/2026-10-06_q2_impactful_call_transfer/run_q2_alllayers.py (PR #31 at 4d01d90), for
Qwen3.5-27B as the reader. Settings come from q2's own configs (q2_probes.yaml, q2_alllayers.yaml, q2_threeway.yaml), with
this track's overlay (config/<--qwen35>.yaml) on top: the Qwen3.5 five-arm all-layer run, this track's Volumes, and the
base call set. `base_manifest` builds the attack-only manifest that q2 takes from its run_q2_probes.py features run:
  - matched: q2's eligible calls (its features run, read only) that Qwen3.5 can read, i.e. all three readouts have a
    Qwen3.5 token; for the views, the simulated honest calls whose readouts Qwen3-32B read inside its 32k cut
    (this track's cut index), as q2 keeps them;
  - whole: every labeled call eligible on whole transcripts (all three readouts read; both attack transcripts read).
Everything after that (index, pack, fit, summary, orchestrate) is q2's code, except `callend_patch_run` (2026-10-07,
the call_end fix): the call_end rows the all-layer run has no token for are read from that run (the fixed reader's
re-extraction of the affected transcripts), and every other row from the all-layer run as before (`load_metas`).
Without it every step is unchanged. The original docstring follows.

Uses the identical eligible calls (and labels) as a run_q2_probes.py features run, read from the main session's
all-layer segment-end run (one fp16 file per layer and chunk). Steps, all on Modal:
  1. index (CPU): map each eligible call's three positions to (chunk, row) in the all-layer files.
  2. pack (CPU, one container per layer): gather those rows into [call, hidden] arrays per layer and position.
  3. fit (GPU): per layer, probe_shift.transfer with C chosen on each outer fold's training-domain validation; for
     the real labels and for within-transcript label shuffles. Saves per-layer test scores and fold validation AUROCs.
  4. summary (CPU): per-layer 2x2 with main-task bootstrap intervals; the headline layer chosen inside each outer
     fold (earliest within layer_tolerance of the best validation AUROC across all 64 layers), with the same choice
     repeated per shuffle draw for its null.
An orchestrator runs the steps on Modal (launch with modal run --detach); --fetch reads its saved result.

--config q2_threeway adds the simulated honest arm and fits three views (attack only, honest only, pooled), each
mapping arms to a real and a simulated domain. Its extra arm is indexed and packed on its own and stacked after the
attack-only rows, which are reused as packed, so the attack-only view reproduces the attack-only run.

  uv run modal run --detach experiments/2026-10-06_qwen35_replication/run_q2_alllayers_qwen35.py --qwen35 q2_alllayers_qwen35_matched [--draws 1-5] [--config q2_threeway]
"""

import gzip
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
Q2 = ROOT / "experiments" / "2026-10-06_q2_impactful_call_transfer"  # q2's folder: its configs are the settings
APP = "probe-shift_aswin-surya_sub-qwen35_q2all_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06"  # this track's Qwen3.5 extractions; read only
FEATS = "probe-shift_aswin-surya_sub-qwen35_q2feats_kill-2026-11-06"
Q2FEATS = "probe-shift_aswin-surya_sub-q2_probe-features_kill-2026-11-06"  # q2's features Volume; read only
POSITIONS = ["call_end", "pre_call", "prev_content"]
FIT_KEYS = ["labeler_sha", "label_rule", "alllayers_run", "extra_alllayers_run", "C_grid", "max_iter", "fpr", "seed",
            "train_sets", "test_domains", "home", "views", "honest_tasks", "primary_real_honest"]

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2).read_only()
feats_vol = modal.Volume.from_name(FEATS, create_if_missing=True, version=2)
q2feats_vol = modal.Volume.from_name(Q2FEATS, version=2).read_only()
cpu_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2", "joblib")
             .add_local_python_source("probe_shift"))
gpu_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2", "torch==2.8.0")
             .add_local_python_source("probe_shift"))


def out_dir(cfg, base: bool = False) -> Path:
    return Path("/feats") / cfg["features_run"] / ("alllayers" if base else cfg.get("out_sub", "alllayers"))


# q2's #62 (a075b10, 31210cd), ported 2026-10-07 for the cleaned-text test: fit reads its own run's packs, packs record
# the activations run they hold, and `reference_alllayers_run` keeps only the calls whose readouts sit where they do in
# that read. Packed before packs recorded their source: (pack directory, the activations run it holds).
LEGACY_PACKS = {("alllayers", "2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1"),
                ("threeway", "2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1")}


def pack_dirs(cfg) -> list[tuple[Path, str]]:
    """The pack directories fit stacks, in row order, each with the activations run its rows must come from: the run's
    own out_sub (alllayers_run); with views, the attack-only pack of alllayers_run, then the extra arm's pack."""
    if cfg.get("views"):
        return [(out_dir(cfg, base=True), cfg["alllayers_run"]), (out_dir(cfg), cfg["extra_alllayers_run"])]
    return [(out_dir(cfg), cfg["alllayers_run"])]


def check_pack_source(layer_dir: Path, run: str) -> None:
    """Raise unless the pack in layer_dir was made from activations run `run`. A pack without pack_source.json is
    accepted only where LEGACY_PACKS names it."""
    f = layer_dir / "pack_source.json"
    if f.exists():
        got = json.loads(f.read_text())["activations_run"]
        if got != run:
            raise ValueError(f"{layer_dir}: packed from {got}, but this config reads {run}")
    elif (layer_dir.parent.name, run) not in LEGACY_PACKS:
        raise ValueError(f"{layer_dir}: no pack_source.json, so its activations run cannot be checked against {run}")


def views_of(cfg) -> dict:
    """View name -> {domains: arm -> domain, one_run}. Without `views`: one unnamed view, each arm its own domain."""
    return cfg.get("views") or {"": {"domains": None}}


def view_domain(cfg, manifest: list[dict], view: str):
    """Each row's domain in the view; "-" leaves the row out (never trained on, never tested). Honest rows outside the
    view's tasks, and real honest runs other than each task's chosen one when the view has one_run, are left out."""
    import numpy as np

    v = views_of(cfg)[view]
    if v["domains"] is None:
        return np.array([m["arm"] for m in manifest])
    keep = lambda m: m.get("in_honest_tasks", True) and (m.get("primary_run", True) or not v.get("one_run"))
    return np.array([v["domains"].get(m["arm"], "-") if keep(m) else "-" for m in manifest])


def fit_name(view: str, position: str, train_set: str, layer: int, seed) -> str:
    return f"{view + '__' if view else ''}{position}__{train_set}__L{layer}__{'real' if seed is None else f'null{seed}'}"


def positions_of(meta: dict, step: int, action: int) -> dict:
    """The three readouts of a call in an all-layer meta file (None where missing)."""
    at = {(q["kind"], q["step"], q["action"]): q for q in meta["positions"]}
    last_output = {}
    for q in meta["positions"]:
        if q["kind"] == "output_end":
            last_output[q["step"]] = max(last_output.get(q["step"], -1), q["action"])
    return {"call_end": at.get(("call_end", step, action)), "pre_call": at.get(("pre_call", step, action)),
            "prev_content": at.get(("call_end", step, action - 1)) if action > 0 else
            (at.get(("output_end", step - 1, last_output[step - 1])) if step - 1 in last_output else None)}


def same_readouts(meta: dict, ref: dict, step: int, action: int) -> dict | None:
    """The call's three readouts in `meta` if each is kept and sits on the same (kind, step, action) as in the
    reference read `ref`, else None. A scrubbed text can empty an output (level 4 drops stderr-looking lines), which
    removes its output_end and would leave prev_content missing or moved to an earlier output."""
    qs, rs = positions_of(meta, step, action), positions_of(ref, step, action)
    at = lambda q: q and (q["kind"], q["step"], q["action"])
    return qs if all(qs[p] is not None and qs[p]["kept"] and at(qs[p]) == at(rs[p]) for p in POSITIONS) else None


def load_metas(cfg: dict, run: str, want: set) -> tuple[dict, int]:
    """(arm, key) -> the all-layer metadata of `run` for the wanted transcripts, and how many positions were patched.
    With `callend_patch_run`: each call_end the run has no token for takes the patch run's position, tagged with its
    chunk (`<patch run>::<chunk>`) and absolute row; a patched transcript's other positions must equal the run's."""
    from concurrent.futures import ThreadPoolExecutor

    def read(root):
        paths = [p for arm in sorted({a for a, _ in want}) for p in (root / "meta" / arm).glob("*.json")]
        with ThreadPoolExecutor(32) as pool:
            return {(m["arm"], m["key"]): m for m in pool.map(lambda p: json.loads(p.read_text()), paths)}

    metas = read(Path("/acts") / run)
    patch = cfg.get("callend_patch_run")
    if not patch:
        return metas, 0
    n, same = 0, lambda q: (q["kind"], q["step"], q["action"], q["tool"], q["token"], q["token_text"])
    for k, pm in read(Path("/acts") / patch).items():
        if k not in metas:
            continue
        m = metas[k]
        if len(pm["positions"]) != len(m["positions"]):
            raise ValueError(f"{k}: the patch run has {len(pm['positions'])} positions, {run} has {len(m['positions'])}")
        merged = []
        for q, pq in zip(m["positions"], pm["positions"]):
            if q["kind"] == "call_end" and q["token"] is None:
                if (pq["kind"], pq["step"], pq["action"]) != (q["kind"], q["step"], q["action"]) or pq["token"] is None or not pq["kept"]:
                    raise ValueError(f"{k} step {q['step']} action {q['action']}: the patch run has no call_end for it")
                merged.append(pq | {"src_chunk": f"{patch}::{pm['chunk']}", "src_row": pm["row_start"] + pq["row"]})
                n += 1
            elif same(q) != same(pq):
                raise ValueError(f"{k}: the patch run differs from {run} at {same(q)} vs {same(pq)}")
            else:
                merged.append(q)
        metas[k] = m | {"positions": merged}
    left = sorted(k for k in want if k in metas and any(q["kind"] == "call_end" and q["token"] is None for q in metas[k]["positions"]))
    if left:  # an incomplete or misnamed patch run would silently keep the old call set
        raise ValueError(f"{len(left)} wanted transcripts still have a call_end with no token after {patch}, e.g. {left[:3]}")
    return metas, n


def patch_guard(cfg: dict):
    """One features run, one set of inputs: its packs and fits (base and views) are all patched with the same
    `callend_patch_run`, or none is. A patched run records it in callend_patch.json when it is created; a run without
    that file was built unpatched."""
    feats_vol.reload()
    root, want = Path("/feats") / cfg["features_run"], cfg.get("callend_patch_run")
    marker = root / "callend_patch.json"
    have = json.loads(marker.read_text())["callend_patch_run"] if marker.exists() else None
    if have == want:
        return
    if have is None and not marker.exists() and not (root / "manifest.json").exists():  # a new run takes the patch
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"callend_patch_run": want}))
        feats_vol.commit()
        return
    raise ValueError(f"{cfg['features_run']} was built with callend_patch_run={have!r}, not {want!r}: use a new features_run")


# Sizes (2026-10-07): what each step uses. Reading metadata is one core of JSON parsing behind I/O threads, pack is one
# core gathering rows (I/O bound), a fit holds about 2 GB of packed rows on the host, and summary runs n_jobs (32)
# workers. The earlier sizes (16-64 cores, 64-256 GB) were mostly idle; memory was 35% of a pass's cost.
@app.function(image=cpu_image, volumes={"/acts": acts_vol, "/feats": feats_vol}, cpu=4, memory=32768, timeout=2 * 3600)
def index(cfg: dict, extra: list[dict] | None = None) -> dict:
    """Rows of each eligible call's three positions in the all-layer files, in manifest order. With `extra` (calls of
    the extra arm): index only those, keeping the calls whose three readouts are inside the reader cut, and write the
    view manifest (the attack-only manifest with the view fields, then the kept extra calls)."""
    import numpy as np

    patch_guard(cfg)
    base = json.loads((Path("/feats") / cfg["features_run"] / "manifest.json").read_text())
    manifest = base if extra is None else extra
    want = {(m["arm"], m["key"]) for m in manifest}
    metas, n_patched = load_metas(cfg, cfg["alllayers_run"] if extra is None else cfg["extra_alllayers_run"], want)
    ref_run = cfg.get("reference_alllayers_run") if extra is None else None
    if ref_run:  # keep the calls whose readouts sit where they do in the reference read, and write that manifest
        refs, _ = load_metas(cfg, ref_run, want)
        missing = sorted(k for k in want if k not in metas or k not in refs)
        if missing:
            raise ValueError(f"{len(missing)} transcripts missing from {cfg['alllayers_run']} or {ref_run}, e.g. {missing[:3]}")
        keep = [same_readouts(metas[(c["arm"], c["key"])], refs[(c["arm"], c["key"])], c["step"], c["action"]) is not None
                for c in manifest]
        dropped_by_arm = {a: sum(1 for c, k in zip(manifest, keep) if c["arm"] == a and not k) for a in sorted({a for a, _ in want})}
        manifest = [c for c, k in zip(manifest, keep) if k]
    chunks = sorted({m["chunk"] for m in metas.values()} | {q["src_chunk"] for m in metas.values() for q in m["positions"]
                                                             if "src_chunk" in q})
    cid = {c: i for i, c in enumerate(chunks)}
    if extra is not None:
        missing = sorted({k for k in want if k not in metas})
        if missing:
            raise ValueError(f"{len(missing)} extra-arm transcripts missing from the all-layer run, e.g. {missing[:3]}")
        qs = [positions_of(metas[(c["arm"], c["key"])], c["step"], c["action"]) for c in manifest]
        kept = [i for i, q in enumerate(qs) if all(q[p] is not None and q[p]["kept"] for p in POSITIONS)]
        if cfg["base"] == "matched":  # qwen35: and inside Qwen3-32B's 32k cut, as q2 keeps them (its reader's cut)
            cut = json.loads((Path("/acts") / cfg["cut_file"]).read_text())["read"]
            read = {k: {tuple(x) for x in v} for k, v in cut.items()}
            kept = [i for i in kept if all((qs[i][p]["kind"], qs[i][p]["step"], qs[i][p]["action"]) in
                                           read[f"{manifest[i]['arm']}|{manifest[i]['key']}"] for p in POSITIONS)]
        dropped = len(manifest) - len(kept)
        manifest = [manifest[i] for i in kept]
    chunk_of, row_of = np.full((3, len(manifest)), -1, np.int32), np.full((3, len(manifest)), -1, np.int64)
    for i, c in enumerate(manifest):
        m = metas[(c["arm"], c["key"])]
        qs = positions_of(m, c["step"], c["action"])
        for k, pos in enumerate(POSITIONS):
            q = qs[pos]
            if q is None or not q["kept"] or (pos != "prev_content" and q["tool"] != c["tool"]):
                raise ValueError(f"{c['arm']} {c['key']} step {c['step']} action {c['action']}: {pos} missing or mismatched in the all-layer run")
            chunk_of[k, i], row_of[k, i] = ((cid[q["src_chunk"]], q["src_row"]) if "src_chunk" in q else
                                            (cid[m["chunk"]], m["row_start"] + q["row"]))
    d = out_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "index.npz", chunk_of=chunk_of, row_of=row_of)
    (d / "chunks.json").write_text(json.dumps(chunks))
    report = {"calls": len(manifest), "chunks": len(chunks)} | ({"callend_patched": n_patched} if cfg.get("callend_patch_run") else {})
    if ref_run:
        (d / "manifest.json").write_text(json.dumps(manifest))
        report |= {"reference_alllayers_run": ref_run, "calls_offered": len(base), "dropped_by_arm": dropped_by_arm}
    if extra is not None:  # the view manifest: attack-only rows first, in their packed order, then the extra arm
        honest, primary = set(cfg["honest_tasks"]), set(cfg["primary_real_honest"])
        slices = {name: set(ts) for name, ts in cfg["slice_tasks"].items()}
        rows = []
        for m in base + manifest:
            honest_arm = m["arm"] in ("real-honest", cfg["extra_arm"])
            rows.append(m | {"in_honest_tasks": m["main_task"] in honest or not honest_arm,
                             "primary_run": m["arm"] != "real-honest" or m["key"] in primary,
                             **{name: m["main_task"] in ts for name, ts in slices.items()}})
        (d / "manifest.json").write_text(json.dumps(rows))
        report |= {"extra_calls_offered": len(extra), "extra_calls_outside_cut": dropped, "base_calls": len(base)}
    feats_vol.commit()
    return report


@app.function(image=cpu_image, volumes={"/acts": acts_vol, "/feats": feats_vol, "/q2feats": q2feats_vol}, cpu=4,
              memory=32768, timeout=2 * 3600)
def base_manifest(cfg: dict, calls_gz: bytes) -> dict:
    """qwen35: the attack-only call set, written as <features_run>/manifest.json (q2 takes it from its run_q2_probes.py
    features run). `base: matched`: q2's eligible calls (its features run) whose three readouts have a Qwen3.5 token.
    `base: whole`: every labeled call whose three readouts Qwen3.5 read, keeping a task's attack calls only if both
    attack arms have calls in it (q2's paired rule)."""
    from collections import Counter, defaultdict

    patch_guard(cfg)
    if cfg["base"] == "matched":
        calls = json.loads((Path("/q2feats") / cfg["q2_features_run"] / "manifest.json").read_text())
    elif cfg["base"] == "whole":
        calls = json.loads(gzip.decompress(calls_gz))
    else:
        raise ValueError(f"base must be matched or whole, not {cfg['base']!r}")
    want = {(c["arm"], c["key"]) for c in calls}
    metas, n_patched = load_metas(cfg, cfg["alllayers_run"], want)
    missing = sorted(want - set(metas))
    if missing:
        raise ValueError(f"{len(missing)} transcripts missing from {cfg['alllayers_run']}, e.g. {missing[:3]}")
    reasons, kept = Counter(), []
    for i, c in enumerate(calls):
        qs = positions_of(metas[(c["arm"], c["key"])], c["step"], c["action"])
        if qs["prev_content"] is None:
            reasons["no previous content token (first call, or after an output empty after the template's trim)"] += 1
        elif not all(q is not None and q["kept"] for q in qs.values()):
            reasons["a readout with no Qwen3.5 token"] += 1
        else:
            kept.append(i)
    if cfg["base"] == "whole":
        arms_of = defaultdict(set)
        for i in kept:
            arms_of[calls[i]["cell"]].add(calls[i]["arm"])
        both = set(cfg["test_domains"])
        dropped = [i for i in kept if calls[i]["arm"] in both and not both <= arms_of[calls[i]["cell"]]]
        reasons["task's other attack arm has no eligible call"] += len(dropped)
        kept = [i for i in kept if i not in set(dropped)]
    rows = [calls[i] | {"row": r} for r, i in enumerate(kept)]
    d = Path("/feats") / cfg["features_run"]
    d.mkdir(parents=True, exist_ok=True)
    if (d / "manifest.json").exists() and json.loads((d / "manifest.json").read_text()) != json.loads(json.dumps(rows)):
        raise ValueError(f"{cfg['features_run']} already has a different call set; its packs and fits match that one: "
                         "use a new features_run")  # never re-pair cached packs and fits with other calls
    (d / "manifest.json").write_text(json.dumps(rows))
    report = {"base": cfg["base"], "calls_offered": len(calls), "calls": len(rows), "excluded": dict(reasons),
              "by_arm": dict(Counter(r["arm"] for r in rows))} | ({"callend_patched": n_patched} if cfg.get("callend_patch_run") else {})
    (d / "base_report.json").write_text(json.dumps(report))
    feats_vol.commit()
    return report


def view_manifest(cfg) -> list[dict]:
    d = out_dir(cfg)
    own = cfg.get("views") or cfg.get("reference_alllayers_run")  # index wrote this run's own manifest
    return json.loads(((d if own else Path("/feats") / cfg["features_run"]) / "manifest.json").read_text())


@app.function(image=cpu_image, volumes={"/acts": acts_vol, "/feats": feats_vol}, cpu=2, memory=16384, timeout=2 * 3600)
def pack(cfg: dict, layer: int) -> int:
    import numpy as np

    feats_vol.reload()
    d = out_dir(cfg)
    idx = np.load(d / "index.npz")
    chunk_of, row_of = idx["chunk_of"], idx["row_of"]
    chunks = json.loads((d / "chunks.json").read_text())
    n = chunk_of.shape[1]
    out = {pos: np.zeros((n, 5120), np.float16) for pos in POSITIONS}
    src_run = cfg["extra_alllayers_run" if cfg.get("views") else "alllayers_run"]
    src = Path("/acts") / src_run / "by_layer" / f"L{layer}"
    for c, name in enumerate(chunks):
        sel = np.nonzero(chunk_of == c)
        if not len(sel[0]):
            continue
        run, _, stem = name.rpartition("::")  # a patched call_end's chunk names its run
        arr = np.load((Path("/acts") / run / "by_layer" / f"L{layer}" if run else src) / f"{stem}.npy", mmap_mode="r")
        for k, pos in enumerate(POSITIONS):
            rows = np.flatnonzero(chunk_of[k] == c)
            out[pos][rows] = arr[row_of[k, rows]]
    (d / f"L{layer}").mkdir(exist_ok=True)
    for pos in POSITIONS:
        if not np.isfinite(out[pos]).all():
            raise ValueError(f"non-finite activations at layer {layer}, {pos}")
        np.save(d / f"L{layer}" / f"{pos}.npy", out[pos])
    (d / f"L{layer}" / "pack_source.json").write_text(json.dumps({"activations_run": src_run, "positions": POSITIONS, "rows": int(n)}))
    feats_vol.commit()
    return layer


@app.function(image=gpu_image, gpu="A10G", volumes={"/feats": feats_vol}, memory=16384, timeout=6 * 3600)
def fit(cfg: dict, position: str, layers: list[int], combos: list) -> list[dict]:
    """Per layer, every (view, train_set, seed) in combos (seed None = real labels): transfer with C chosen per outer
    fold on validation AUROC. Grouped (2026-10-07): a layer's packed rows are loaded once and shared by all the combos,
    since they do not depend on the view, training set or seed; one job per position and layer group replaces one per
    view, training set and seed. Each fit's own computation is unchanged. Fits are written to cfg["fits_subdir"]
    (default fits/)."""
    import time

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from probe_shift.probes import fit_logreg
    from probe_shift.transfer import transfer

    import platform
    import subprocess

    # Where this job ran. Modal's A10G pool mixes NVIDIA A10 (72 SMs) and A10G (80 SMs): identical inputs give
    # bit-identical fits on one GPU model, and up to about 0.04 logit apart across the two (2026-10-07).
    props = torch.cuda.get_device_properties(0)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True)
    cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")),
               platform.processor())
    env = {"gpu": props.name, "sms": props.multi_processor_count, "torch_cuda": torch.version.cuda,
           "driver": smi.stdout.strip() or f"nvidia-smi failed: {smi.stderr.strip()[:80]}", "cpu": cpu}
    feats_vol.reload()
    d = out_dir(cfg)
    fits_dir = d / cfg.get("fits_subdir", "fits")
    manifest = view_manifest(cfg)
    y0 = np.array([m["y"] for m in manifest])
    fold = np.array([m["fold"] for m in manifest])
    tkey = np.array([f"{m['arm']}|{m['key']}" for m in manifest])
    domains = {v: view_domain(cfg, manifest, v) for v in sorted({v for v, _, _ in combos})}
    labels = {}

    def labels_for(seed):  # the same within-transcript permutation as run_q2_probes.py for this seed
        if seed not in labels:
            y = y0.copy()
            if seed is not None:
                rng = np.random.default_rng(seed)
                for k in np.unique(tkey):
                    rows = np.flatnonzero(tkey == k)
                    y[rows] = rng.permutation(y0[rows])
            labels[seed] = y
        return labels[seed]

    out, done, total = [], 0, len(layers) * len(combos)
    for layer in layers:
        t_load = time.time()
        parts = []  # this run's own pack; with views, the attack-only rows as packed, then the extra arm's rows
        for pack_dir, run in pack_dirs(cfg):
            check_pack_source(pack_dir / f"L{layer}", run)
            parts.append(np.load(pack_dir / f"L{layer}" / f"{position}.npy"))
        X = torch.from_numpy(np.concatenate(parts)).cuda().float()
        if len(X) != len(manifest):
            raise ValueError(f"layer {layer} {position}: {len(X)} packed rows for {len(manifest)} manifest rows")
        load_s = time.time() - t_load
        for view, train_set, seed in combos:
            t_fit = time.time()
            domain, y = domains[view], labels_for(seed)
            test = np.isin(domain, cfg["test_domains"])
            chosen = []

            def fit_score(train, score_on):
                val, tst = score_on
                mu, sd = X[train].mean(0), X[train].std(0).clamp_min(1e-6)
                Z = (X - mu) / sd
                ytr = torch.tensor(y[train], dtype=torch.float32, device="cuda")
                best = None
                for C in cfg["C_grid"]:
                    w, g = fit_logreg(Z[train], ytr, C, cfg["max_iter"])
                    s = (Z @ w[:-1] + w[-1]).cpu().numpy()
                    auc = float(roc_auc_score(y[val], s[val]))
                    if best is None or auc > best[0]:
                        best = (auc, C, g, s)
                chosen.append({"C": best[1], "val_auroc": best[0], "grad_max": best[2]})
                return [best[3][val], best[3][tst]]

            r = transfer(fit_score, y, domain, fold, set(cfg["train_sets"][train_set]), fpr=cfg["fpr"])
            name = fit_name(view, position, train_set, layer, seed)
            fits_dir.mkdir(exist_ok=True)
            np.savez(fits_dir / f"{name}.npz", score=r["score"][test].astype(np.float32 if seed is None else np.float16),
                     flag=r["flag"][test], threshold=r["threshold"][test].astype(np.float32),
                     **({"y": y[test].astype(np.int8)} if seed is not None else {}))
            out.append({"view": view, "position": position, "train_set": train_set, "layer": layer, "seed": seed, "file": f"{name}.npz",
                        "folds": [f | c for f, c in zip(r["folds"], chosen)],
                        "seconds": round(time.time() - t_fit, 1), "layer_load_seconds": round(load_s, 1), "env": env})
            done += 1  # progress in the container log: a grouped job can run many fits (Codex)
            print(f"{position} L{layer} {view or '-'} {train_set} seed {seed}: {out[-1]['seconds']} s, fit {done}/{total}", flush=True)
        del X
        torch.cuda.empty_cache()
    feats_vol.commit()
    return out

@app.function(image=cpu_image, volumes={"/feats": feats_vol}, cpu=32, memory=65536, timeout=4 * 3600)  # n_jobs: 32
def summary(cfg: dict, fits: list[dict]) -> dict:
    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score

    from probe_shift.transfer import two_by_two

    feats_vol.reload()
    d = out_dir(cfg)
    manifest = view_manifest(cfg)
    layers, tol = cfg["layers"], cfg["layer_tolerance"]
    by_all = {(f.get("view", ""), f["position"], f["train_set"], f["layer"], f["seed"]): f for f in fits}
    load = lambda f: dict(np.load(d / cfg.get("fits_subdir", "fits") / f["file"]))  # per_layer, in joblib workers, reads its own
    cache = {}

    def load_fit(f):  # the main process (headline, nulls, reproduction) reads each fit file once, not once per use
        if f["file"] not in cache:
            cache[f["file"]] = load(f)
        return cache[f["file"]]
    home = {t: cfg["home"][t] for t in cfg["train_sets"]}

    def one_view(view):
        by = {k[1:]: f for k, f in by_all.items() if k[0] == view}
        dom = view_domain(cfg, manifest, view)
        test = np.isin(dom, cfg["test_domains"])
        rows = [m for m, t in zip(manifest, test) if t]
        y = np.array([m["y"] for m in rows])
        domain = dom[test]
        groups = np.array([m["main_task"] for m in rows])
        fold = np.array([m["fold"] for m in rows])

        def per_layer(pos, layer):
            res = {t: load(by[(pos, t, layer, None)]) for t in cfg["train_sets"]}
            return layer, two_by_two(res, home, y, domain, groups, cfg["test_domains"], cfg["n_boot"], cfg["seed"])

        def headline(pos, t, seed):
            """Per outer fold, the earliest layer within tol of the best validation AUROC; its test rows from that layer."""
            score, flag, thr, picks = np.zeros(len(y)), np.zeros(len(y), bool), np.zeros(len(y)), []
            for f in sorted({fo["fold"] for fo in by[(pos, t, layers[0], seed)]["folds"]}):
                val = {l: next(fo["val_auroc"] for fo in by[(pos, t, l, seed)]["folds"] if fo["fold"] == f) for l in layers}
                top = max(val.values())
                pick = min(l for l in layers if val[l] >= top - tol)
                a = load_fit(by[(pos, t, pick, seed)])
                m = fold == f
                score[m], flag[m], thr[m] = a["score"][m], a["flag"][m], a["threshold"][m]
                picks.append({"fold": f, "layer": pick, "val_auroc": round(val[pick], 4)})
            return {"score": score, "flag": flag, "threshold": thr}, picks, (np.array(load_fit(by[(pos, t, layers[0], seed)])["y"]) if seed is not None else y)

        out = {"positions": {}}
        for pos in POSITIONS:
            curves = dict(Parallel(n_jobs=cfg["n_jobs"])(delayed(per_layer)(pos, l) for l in layers))
            heads, picks = {}, {}
            for t in cfg["train_sets"]:
                heads[t], picks[t], _ = headline(pos, t, None)
            head_rep = two_by_two(heads, home, y, domain, groups, cfg["test_domains"], cfg["n_boot"], cfg["seed"])
            # Sensitivity: the headline probes scored without a slice's main tasks, in every arm (no refit).
            slices = {}
            for name in cfg.get("slices", []):  # the slice's tasks are dropped before resampling tasks
                keep = ~np.array([m[name] for m in rows])
                sub = {t: {k: v[keep] for k, v in h.items()} for t, h in heads.items()}
                slices[f"without {name}"] = two_by_two(sub, home, y[keep], domain[keep], groups[keep], cfg["test_domains"],
                                                       cfg["n_boot"], cfg["seed"])
            seeds = sorted({s for (p, _, _, s) in by if p == pos and s is not None})
            nulls = {}
            for t in cfg["train_sets"]:
                if not any((pos, t, layers[0], s) in by for s in seeds):
                    continue
                for te in cfg["test_domains"]:
                    mte = domain == te
                    head_null = []
                    for s in seeds:
                        if all((pos, t, l, s) in by for l in layers):
                            h, _, ys = headline(pos, t, s)
                            head_null.append(float(roc_auc_score(ys[mte], h["score"][mte])))
                    layer_null = {f"L{l}": [round(float(roc_auc_score(load_fit(by[(pos, t, l, s)])["y"][mte],
                                                                      load_fit(by[(pos, t, l, s)])["score"][mte].astype(np.float32))), 4)
                                            for s in seeds if (pos, t, l, s) in by] for l in layers}
                    nulls[f"train {t} / test {te}"] = {"headline": {"n": len(head_null), "max": round(max(head_null), 4) if head_null else None,
                                                                    "mean": round(float(np.mean(head_null)), 4) if head_null else None},
                                                       "per_layer": layer_null}
            out["positions"][pos] = {"per_layer": {f"L{l}": curves[l] for l in layers}, "headline": head_rep,
                                     "headline_layers": picks, "null": nulls, **({"slices": slices} if slices else {})}
        arms = np.array([m["arm"] for m in rows])
        out["calls_by_arm"] = {te: {a: {"calls": int(((domain == te) & (arms == a)).sum()),
                                        "main_tasks": len(set(groups[(domain == te) & (arms == a)]))}
                                    for a in sorted(set(arms[domain == te]))} for te in cfg["test_domains"]}
        return out

    if not cfg.get("views"):
        return {"layers": layers} | one_view("")
    out = {"layers": layers, "views": {v: one_view(v) for v in views_of(cfg)}}
    rep = cfg.get("reproduces")  # the attack-only view against the attack-only run's own fits: same calls, same scores
    if rep:
        base, check = out_dir(cfg, base=True) / "fits", {}
        for pos in POSITIONS:
            for t, t0 in rep["train_sets"].items():
                diffs = [(np.abs(load_fit(by_all[(rep["view"], pos, t, l, None)])["score"] - np.load(base / f"{fit_name('', pos, t0, l, None)}.npz")["score"]).max(),
                          int((load_fit(by_all[(rep["view"], pos, t, l, None)])["flag"] != np.load(base / f"{fit_name('', pos, t0, l, None)}.npz")["flag"]).sum()))
                         for l in layers if (rep["view"], pos, t, l, None) in by_all and (base / f"{fit_name('', pos, t0, l, None)}.npz").exists()]
                check[f"{pos} {t} vs {t0}"] = {"layers": len(diffs), "max_score_diff": float(max((x for x, _ in diffs), default=np.nan)),
                                               "flags_differ": sum(n for _, n in diffs)}
        out["reproduction"] = check
    return out


@app.function(image=cpu_image, volumes={"/feats": feats_vol}, timeout=12 * 3600)
def orchestrate(cfg: dict, seeds: list, do_index: bool, extra: list[dict] | None = None) -> dict:
    patch_guard(cfg)
    d = out_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    fit_cfg = {k: cfg.get(k) for k in FIT_KEYS}  # later draws and summaries must reuse the fits' own procedure
    for k in ("callend_patch_run", "reference_alllayers_run"):  # and inputs, where set (earlier fit configs lack them)
        if cfg.get(k):
            fit_cfg[k] = cfg[k]
    saved = d / "fit_config.json"
    if saved.exists() and json.loads(saved.read_text()) != fit_cfg:
        raise ValueError(f"{d}: fits were made with {saved.read_text()}, not {json.dumps(fit_cfg)}")
    report = index.remote(cfg, extra) if do_index else None
    if do_index:
        list(pack.map([cfg] * len(cfg["layers"]), cfg["layers"]))
    saved.write_text(json.dumps(fit_cfg))  # recorded once the inputs exist, just before the first fits
    feats_vol.commit()
    chunk = cfg["layers_per_job"]
    groups = [cfg["layers"][i:i + chunk] for i in range(0, len(cfg["layers"]), chunk)]
    null_views = cfg.get("null_views", list(views_of(cfg)))
    combos = [(v, t, s) for v in views_of(cfg) for t in cfg["train_sets"] for s in seeds
              if s is None or (t in cfg["null_train_sets"] and v in null_views)]
    jobs = [(cfg, pos, g, combos) for pos in POSITIONS for g in groups] if combos else []  # one job per position and layer group
    fits = [f for part in fit.starmap(jobs) for f in part]
    feats_vol.reload()
    log = d / "fits_index.jsonl"  # every fit ever made for this features run, so later draws add to earlier ones
    old = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    keyed = {(f.get("view", ""), f["position"], f["train_set"], f["layer"], f["seed"]): f for f in old + fits}
    log.write_text("".join(json.dumps(f) + "\n" for f in keyed.values()))
    feats_vol.commit()
    out = summary.remote(cfg, list(keyed.values())) | {"index": report, "config": cfg}  # --fetch reports this config
    (d / "result.json").write_text(json.dumps(out))
    feats_vol.commit()
    return out


def view_inputs(cfg: dict, labels: Path) -> list[dict]:
    """For a config with views: the calls to index from the extra run, and, in cfg, the honest tasks, each task's
    chosen real honest run and the slices' main tasks. Honest tasks: every main task with both a real and a simulated
    honest run (the paired honest tasks of the shared splits). The extra calls are the simulated honest ones, plus the
    real honest calls of honest tasks without an attack pair, which the attack-only features run does not hold.
    Chosen run (fixed before any result): lowest epoch, then lowest sample ID, then lowest .eval file name."""
    if cfg["label_rule"] != "as_labeled":
        raise ValueError("views support label_rule as_labeled only")
    folds = json.loads((ROOT / "splits" / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    rows = [json.loads(l) for l in labels.read_text().splitlines()]
    paired = {r["main_task"] for r in rows if r[cfg["tasks"]]}  # tasks whose real honest calls are already packed
    honest = sorted({r["main_task"] for r in rows if r["arm"] == "real-honest"} & {r["main_task"] for r in rows if r["arm"] == cfg["extra_arm"]})
    runs = {}
    for r in rows:
        if r["arm"] == "real-honest" and r["main_task"] in honest:
            file, sample_id, epoch = r["key"].rsplit("__", 2)
            runs.setdefault(r["main_task"], set()).add((int(epoch), sample_id, file, r["key"]))
    meta = json.loads(labels.with_suffix(".meta.json").read_text())
    cfg |= {"honest_tasks": honest, "primary_real_honest": sorted(min(v)[3] for v in runs.values()),
            "slice_tasks": {name: meta["slices"][name] for name in cfg["slices"]}}
    return [{k: r[k] for k in ("arm", "key", "cell", "step", "action", "tool", "category", "main_task")} |
            {"fold": folds[r["main_task"]], "y": int(r["label"] == "impactful")}
            for r in rows if r["main_task"] in honest and (r["arm"] == cfg["extra_arm"] or (r["arm"] == "real-honest" and
                                                                                             r["main_task"] not in paired))]


@app.local_entrypoint()
def main(qwen35: str, draws: str = "", fetch: bool = False, skip_index: bool = False, layers: str = "",
         summary_only: bool = False, config: str = "", spawn: bool = False, inputs_only: bool = False):
    """--draws a-b: fit shuffle draws a..b (the real labels are fit on the first call, without --draws). --fetch: read
    the saved result instead of computing. --layers 16,40: a pilot on those layers only. --summary-only: recompute the
    summary from every fit saved so far (after more draws land). --config q2_threeway: merge that config on top (views).
    --qwen35 <name>: this track's overlay in config/ (the Qwen3.5 run, the base call set and the features run name).
    --spawn (with modal run --detach): start the orchestrator on Modal and exit; read its result later with --fetch.
    --inputs-only: build the call set and the row index only (CPU, no packs or fits), to compare with another features
    run's."""
    import yaml

    from probe_shift import provenance
    from probe_shift.labels import resolve_labels

    cfg = yaml.safe_load((Q2 / "config" / "q2_probes.yaml").read_text()) | \
        yaml.safe_load((Q2 / "config" / "q2_alllayers.yaml").read_text())
    if config:
        cfg |= yaml.safe_load((Q2 / "config" / f"{config}.yaml").read_text())
    q2_features_run = cfg["features_run"]  # q2's run_q2_probes.py features run: the matched base reads its calls
    cfg |= yaml.safe_load((HERE / "config" / f"{qwen35}.yaml").read_text())  # qwen35: last, over q2's settings
    labels, made_by = resolve_labels(ROOT, cfg["labels_file"])
    cfg["labels_file"], cfg["labeler_sha"] = str(labels.relative_to(ROOT)), made_by
    cfg["q2_features_run"] = f"{q2_features_run}_{made_by}_{cfg['label_rule']}"
    cfg["features_run"] += f"_{made_by}_{cfg['label_rule']}"  # this track's features run for these labels
    cfg["provenance"] = provenance.capture(ROOT, [Path(__file__)] + [ROOT / "src" / "probe_shift" / f for f in
                                                   ("probes.py", "transfer.py", "labels.py")])  # at launch
    if cfg.get("views") and not cfg.get("extra_alllayers_run", "").startswith("20"):
        raise ValueError(f"set extra_alllayers_run in config/{config}.yaml (now {cfg.get('extra_alllayers_run')!r})")
    extra = view_inputs(cfg, labels) if cfg.get("views") else None
    if not cfg.get("views") and not cfg.get("reference_alllayers_run") and not (fetch or summary_only or draws or skip_index):
        # qwen35: the base call set first (a run with a reference read reuses its features run's, filtered by index)
        folds = json.loads((ROOT / "splits" / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
        rows = [json.loads(l) for l in labels.read_text().splitlines()]
        if cfg["label_rule"] != "as_labeled":
            raise ValueError("the whole base supports label_rule as_labeled only")
        tasks = {r["main_task"] for r in rows if r[cfg["tasks"]]}
        arms = set(cfg["test_domains"]) | {a for t in cfg["train_sets"].values() for a in t}
        calls = [{k: r[k] for k in ("arm", "key", "cell", "step", "action", "tool", "category", "main_task")} |
                 {"fold": folds[r["main_task"]], "y": int(r["label"] == "impactful")}
                 for r in rows if r["arm"] in arms and r["main_task"] in tasks and (r[cfg["tasks"]] or r["arm"] not in cfg["test_domains"])]
        print("base call set:", json.dumps(base_manifest.remote(cfg, gzip.compress(json.dumps(calls).encode()))), flush=True)
    if layers:
        cfg["layers"] = [int(l) for l in layers.split(",")]
        if cfg.get("views"):
            cfg["out_sub"] += "_pilot"
    if inputs_only:
        print("index:", json.dumps(index.remote(cfg, extra)), flush=True)
        return
    seeds = [None] if not draws else []
    if draws:
        a, b = map(int, draws.split("-"))
        seeds = list(range(cfg["seed"] + a, cfg["seed"] + b + 1))
    if fetch:  # the saved result's own config and provenance, not this process's
        out = json.loads(b"".join(feats_vol.read_file(f"{cfg['features_run']}/{cfg.get('out_sub', 'alllayers')}/result.json")))
        cfg = out.get("config", cfg)
    elif summary_only:
        out = orchestrate.remote(cfg, [], False)
    elif spawn:  # qwen35: no long-lived laptop client; the orchestrator writes result.json to the Volume
        call = orchestrate.spawn(cfg, seeds, not skip_index and not draws, extra)
        print(f"orchestrating on Modal: {call.object_id}; then --fetch with the same --qwen35/--config", flush=True)
        return
    else:
        out = orchestrate.remote(cfg, seeds, not skip_index and not draws, extra)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q2_{config or 'alllayers'}_{qwen35}_{stamp}.json"
    cfg = out.pop("config", cfg)
    path.write_text(json.dumps({"config": cfg | {"modal": {"app": APP}}, "labeler_sha": cfg["labeler_sha"], **out}, indent=1))
    for view, v in (out["views"] if "views" in out else {"": out}).items():
        t0 = list(cfg["train_sets"])[0]
        for pos, e in v["positions"].items():
            h = e["headline"]
            print(f"\n=== {view} {pos}: headline layers {[p['layer'] for p in e['headline_layers'][t0]]}  " + "  ".join(
                f"{k.removeprefix('train ')}: AUROC {h[k]['auroc']['value']:.3f} recall {h[k]['recall']['value']:.3f}"
                for k in h if k.startswith("train ") and " minus " not in k))
    if "reproduction" in out:
        print("\nattack-only view vs the attack-only run:", json.dumps(out["reproduction"]))
    print(f"\nresults -> {path.relative_to(ROOT)}")
