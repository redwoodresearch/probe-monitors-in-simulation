"""RULE.md (this folder): a simulated-honest vs real-honest detector on Qwen3.5-27B's own runs, at the content-only end-of-call
(command) readout and the averaged reply readout (output_end_mean rows), layers 48 and 63; plus real honest C vs B as the negative
check. Stored activations only; no text is read.

Rows: real honest B (corrected-B manifest, world honest_b) and C (q4b_extract_c_v1) on the q4own_evals Volume, epoch list as
run_asis_fits.main / run_simh_score._main (aswin/cmd-readout-curve @ 66850d6bf); simulated honest from d9's dumps in
config/inputs_v1.json (q3onpol_work Volume), each passing run_simh_score.provenance first. skip_rows and provenance are copied
verbatim from run_simh_score.py @ 66850d6bf; the row selection is run_simh_score.score_tight's.
Fit: per (layer, readout) on one H100, torch_probe.fit (certified; C = 0.01; allow_uncertified=True, return_certificate=True so
every column's certificate is recorded), standardized on the training folds' rows; 5 folds = the #36 fold of the run's main task
(runs36.json only); run score = mean of its rows' logits. Null: 100 within-task label shuffles, each refit through the whole CV
(one batched 100-column fit per fold).

  PYTHONPATH=src <scratch>/modalenv/bin/modal run --detach experiments/2026-10-14_cmd_readout_simh_control/run_simh_control.py \\
      --r36 <runs36.json>
"""

import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-cmdsimdet_uncut_kill-2026-11-14"  # renamed for spend tracking (4f detU fork); the only change in this copy (2e detector fork renamed it before)
BVOL, DVOL, HFVOL = ("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13",
                     "probe-shift_aswin-surya_hf-cache")
CONFIG = "config/inputs_v1.json"  # default; --config selects a later version (e.g. with batch 2)
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
R36_SHA = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
C, LAYERS, READOUTS, N_NULL, SEED = 0.01, (48, 63), ("command", "reply"), 100, 20261014
PRIMARY = {"command": 63, "reply": 48}
FIXED_TOKENS = ["0", "n"]  # Amendment 12's fixed call tokens (cmd_readout_2026-10-14_033117.json fixed_tokens_dropped)
FIT_TIMEOUT, MARGIN_S = 600, 45  # expected about 8 min per container (+25%); a null fit starts only if 1.25 x its estimated time fits
# before FIT_TIMEOUT - MARGIN_S (estimate: the slowest null fit so far, else 3 x the slowest observed fit); a skipped fold = incomplete null
HDIM = 5120

app = modal.App(APP)
b_vol = modal.Volume.from_name(BVOL, version=2).read_only()
d_vol = modal.Volume.from_name(DVOL, version=2).read_only()
hf_vol = modal.Volume.from_name(HFVOL).read_only()
tok_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy==2.3.3")
             .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))
gpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3").add_local_python_source("probe_shift")


# ---- copied verbatim from run_simh_score.py @ 66850d6bf ----
def skip_rows(root: str, want: dict, shas: dict) -> dict:
    """run_asis_fits.b_skip (run_avg_transfer.cmd_prep's rule), for staged inputs under `root`: want = {staged run: {arm: [keys]}}."""
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN35_27B
    from probe_shift.q1_eval import fixed_call_tokens

    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"], local_files_only=True)
    fixed = fixed_call_tokens("q4b_main_v1")
    fixed_ids = set(tok.convert_tokens_to_ids(fixed))
    skip, counts, file_sha = {}, {}, {}
    for run, arms in want.items():
        for arm, keys in arms.items():
            rel = f"{run}/inputs/{arm}.jsonl"
            b = (Path(root) / rel).read_bytes()
            file_sha[rel] = hashlib.sha256(b).hexdigest()
            if rel in shas and shas[rel] != file_sha[rel]:
                raise ValueError(f"{rel}: staged inputs sha256 differs")
            ks, seen = set(keys), set()
            for line in b.decode().splitlines():
                ep = json.loads(line)
                if ep["key"] not in ks:
                    continue
                if ep["key"] in seen:
                    raise ValueError("a key twice in one staged file")
                seen.add(ep["key"])
                skip[ep["key"]] = [[r["irid"], r["call_index"]] for r in ep["calls"] if r["kind"] == "call_end" and r["readout"] is not None and ep["ids"][r["readout"]] in fixed_ids]
                counts[rel] = counts.get(rel, 0) + len(skip[ep["key"]])
            if seen != ks:
                raise ValueError(f"{rel}: {len(ks - seen)} requested epochs not staged there")
    return {"fixed_tokens": fixed, "skip": skip, "dropped_rows": counts, "staged_sha256": file_sha}


def kp(d, path: str):
    for k in path.split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def provenance(man: dict, extract_run: str, staged_run: str, arm: str, shards_key: str, gpu_key: str, longest_key: str, run_key: str, inputs_sha_key: str,
               gpu_pin: str = "", pin_line: str = "") -> dict:
    """The B200 rule, fail-closed, on fields named explicitly at launch: the manifest's extraction run is the one scored; its shards are
    unique and contiguous from 0. GPU: either every shard's ACTUAL device field (gpu_key) names H200, or (pin mode, 2e's ruling for d9's
    sim-honest dump, which does not log the device) the extraction code's function pin is the single string gpu="H200" at the recorded
    commit (pin_line, read with git show) AND the manifest's git is that commit, clean. Longest-epoch gate: a global path with passes
    True, or "shard:<field>" = every shard's batch-C references (all at the longest 32,768 tokens) with max_abs_diff exactly 0.0."""
    import re

    if not all((shards_key, longest_key, run_key, inputs_sha_key)) or bool(gpu_key) == bool(gpu_pin):
        raise SystemExit("set --prov-shards-key, --prov-longest-key, --prov-run-key, --prov-inputs-sha-key and exactly one of --prov-gpu-key / --prov-gpu-pin")
    if kp(man, run_key) != extract_run:
        raise SystemExit(f"the manifest's extraction run {kp(man, run_key)!r} is not the one to score ({extract_run!r})")
    shards = kp(man, shards_key)
    ids = [s_["shard"] for s_ in shards]
    if not shards or sorted(ids) != list(range(len(ids))):
        raise SystemExit(f"shard ids not unique and contiguous from 0: {ids}")
    gpus = {}
    if gpu_key:
        for s_ in shards:
            g = kp(s_, gpu_key)
            if not isinstance(g, str) or "H200" not in g:
                raise SystemExit(f"shard {s_['shard']}: actual GPU {g!r} is not H200 (B200 rule)")
            gpus[str(s_["shard"])] = g
        gpu_note = f"per-shard device field {gpu_key}"
    else:
        path_commit, line_no = gpu_pin.rsplit(":", 1)
        _path, commit = path_commit.rsplit("@", 1)
        if not re.fullmatch(r'@app\.function\(.*\bgpu="H200",.*', pin_line.strip()) or "gpu=[" in pin_line:
            raise SystemExit(f"the pinned line is not a single gpu=\"H200\" function pin: {pin_line.strip()[:160]!r}")
        if not longest_key.startswith("shard:"):  # pin mode accepts only the bit-exact per-shard gate
            raise SystemExit("pin mode needs the per-shard bit-exact longest-epoch gate (--prov-longest-key shard:<field>)")
        mg = man.get("git", {})
        if len(commit) != 40 or mg.get("commit") != commit or mg.get("dirty") is not False:
            raise SystemExit(f"the manifest's git {mg} is not the clean pinned commit {commit}")
        gpus = {str(i): "H200 (function pin)" for i in ids}
        gpu_note = f"gpu: function pin H200 (code @ {commit[:8]}:{line_no}) + bit-exact longest-epoch gate; device name not logged"
    if longest_key.startswith("shard:"):
        fld, lg = longest_key.split(":", 1)[1], {}
        for s_ in shards:
            refs = [r for r in s_.get(fld, []) if r.get("reference") == "batch C"]
            if not refs or any(r.get("max_abs_diff") != 0.0 or r.get("tokens") != 32768 for r in refs):
                raise SystemExit(f"shard {s_['shard']}: the longest-epoch gate is not bit-exact at 32,768 tokens ({refs})")
            lg[str(s_["shard"])] = s_[fld]
    else:
        lg = kp(man, longest_key)
        if not (lg is True or (isinstance(lg, dict) and lg.get("passes") is True)):
            raise SystemExit(f"the longest-epoch gate at {longest_key!r} is not a pass: {str(lg)[:200]}")
    ins = kp(man, inputs_sha_key)
    want = ins if isinstance(ins, str) else ins.get(f"{staged_run}/inputs/{arm}.jsonl", ins.get(f"{arm}.jsonl", ins.get(arm)))
    if not isinstance(want, str) or len(want) != 64:
        raise SystemExit(f"no staged-inputs sha256 for {arm} at {inputs_sha_key!r}")
    return {"gpu_per_shard": gpus, "gpu_note": gpu_note, "shard_ids": ids, "longest_epoch_gate": {"key": longest_key, "value": lg}, "staged_inputs_sha256": want,
            "fields": {"shards": shards_key, "gpu": gpu_key or gpu_pin, "longest": longest_key, "run": run_key, "inputs_sha": inputs_sha_key}}
# ---- end of the verbatim copy ----


@app.function(image=tok_image, cpu=(4, 4), memory=(16384, 16384), timeout=900, retries=0, volumes={"/b": b_vol, "/hf": hf_vol})
def skip_b(want: dict, shas: dict) -> dict:
    return skip_rows("/b", want, shas)


@app.function(image=tok_image, cpu=(4, 4), memory=(16384, 16384), timeout=900, retries=0, volumes={"/d": d_vol, "/hf": hf_vol})
def skip_d(extract_run: str, staged_run: str, arm: str, staged_sha: str) -> dict:
    """run_simh_score.skip_d, plus: the dump is complete (complete.json names these staged inputs). The dump's epochs (key, task)
    from its staged inputs (sha256 must equal the extraction record's) and their skip rows (the same rule)."""
    cp = Path("/d") / extract_run / "complete.json"
    if not cp.exists() or json.loads(cp.read_text()).get("inputs_sha256", {}).get(f"{arm}.jsonl") != staged_sha:
        raise ValueError(f"{extract_run}: no complete.json naming the staged inputs' sha256")
    rel = f"{staged_run}/inputs/{arm}.jsonl"
    if hashlib.sha256((Path("/d") / rel).read_bytes()).hexdigest() != staged_sha:
        raise ValueError(f"{rel}: sha256 differs from the dump manifest's")
    eps = [json.loads(x) for x in (Path("/d") / rel).read_text().splitlines()]
    pairs = [(e["key"], e["task"]) for e in eps]
    if len({k for k, _ in pairs}) != len(pairs):
        raise ValueError("a key twice in the dump's staged file")
    return {"pairs": pairs} | skip_rows("/d", {staged_run: {arm: [k for k, _ in pairs]}}, {})


def auroc(s, y):
    """Mann-Whitney AUROC of scores s for labels y (1 = positive), ties counted half."""
    import numpy as np

    s, y = np.asarray(s, np.float64), np.asarray(y).astype(bool)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def tpr_at(s, y, fold, a):
    """Per fold, the threshold on that fold's class-0 runs (run_asis_fits.catches' rule: the first score whose descending rank exceeds
    floor(a n + 1e-9)); a class-1 hit is strictly above it; TPR = pooled hits / pooled class-1 runs."""
    import numpy as np

    hits, npos, per = 0, 0, {}
    for f in range(5):
        neg = np.sort(s[(y == 0) & (fold == f)])[::-1]
        pos = s[(y == 1) & (fold == f)]
        if not len(neg):
            raise ValueError(f"fold {f}: no class-0 test runs")
        k = min(int(np.floor(a * len(neg) + 1e-9)), len(neg) - 1)
        h = int((pos > neg[k]).sum())
        hits, npos = hits + h, npos + len(pos)
        per[str(f)] = {"k": k, "n_class0": int(len(neg)), "threshold": float(neg[k]), "class0_above": int((neg > neg[k]).sum()), "hits": h,
                       "n_class1": int(len(pos))}
    return hits / npos, per


def _load(L: int, readout: str, epochs: list, skip: dict, allowed: dict):
    """run_simh_score.score_tight's row selection for one layer and readout: per epoch its meta (path from sha1(key)[:20], key checked),
    reply = output_end calls' row_mean, command = call_end rows minus the epoch's fixed-call-token rows; rows read from
    by_layer/L<L>/<chunk>.npy at row_start + row (float16, as stored). allowed = {extraction run: chunk names its verified extraction
    record lists} (batch C and every sim dump): a chunk outside it stops the job. B's runs come from the corrected-B manifest
    (Amendment 23's provenance; meta paths checked against it)."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    def meta(e):
        rt = Path("/" + e["vol"]) / e["extraction_run"]
        mf = rt / "meta" / e["arm"] / f"{hashlib.sha1(e['key'].encode()).hexdigest()[:20]}.json"
        if e.get("meta_file") and f"{e['extraction_run']}/meta/{e['arm']}/{mf.name}" != e["meta_file"]:
            raise ValueError("meta path differs from the manifest's")
        m = json.loads(mf.read_text())
        if m["key"] != e["key"]:
            raise ValueError("meta key mismatch")
        if readout == "reply":
            cs = [c for c in m["calls"] if c["kind"] == "output_end" and c.get("row_mean") is not None]
            rr = [c["row_mean"] for c in cs]
        else:
            sk = {tuple(x) for x in skip[e["key"]]}
            cs = [c for c in m["calls"] if c["kind"] == "call_end" and c.get("row") is not None and (c["irid"], c["call_index"]) not in sk]
            rr = [c["row"] for c in cs]
        us = [(str(c["src_irid"]), int(c["call_index"])) for c in cs]
        if len(set(us)) != len(us):
            raise ValueError("duplicate (src_irid, call_index) within a run")
        if rr and (max(rr) >= m["n_rows"] or min(rr) < 0):
            raise ValueError("a row outside the epoch's rows")
        if e["extraction_run"] in allowed and m["chunk"] not in allowed[e["extraction_run"]]:
            raise ValueError(f"{e['extraction_run']}: chunk {m['chunk']} is not in its extraction record")
        if e["vol"] == "d" and e["extraction_run"] not in allowed:
            raise ValueError("a sim dump without an extraction record")
        return str(rt), m["chunk"], m["row_start"], rr

    with ThreadPoolExecutor(32) as ex:
        metas = list(ex.map(meta, epochs))
    by = {}
    for i, (rt, ch, _, rr) in enumerate(metas):
        if rr:
            by.setdefault((rt, ch), []).append(i)

    def read(item):
        (rt, ch), idx = item
        A = np.load(Path(rt) / "by_layer" / f"L{L}" / f"{ch}.npy", mmap_mode="r")
        out = {}
        for i in idx:
            _, _, r0, rr = metas[i]
            q = [r0 + x for x in rr]
            if max(q) >= A.shape[0] or A.shape[1] != HDIM:
                raise ValueError("rows beyond the chunk or wrong width")
            out[i] = np.asarray(A[q])
        return out

    blocks = {}
    with ThreadPoolExecutor(16) as ex:
        for d in ex.map(read, list(by.items())):
            blocks.update(d)
    keep = [i for i in range(len(epochs)) if metas[i][3]]
    X = np.concatenate([blocks[i] for i in keep])
    if X.dtype != np.float16 or not np.isfinite(X).all():
        raise ValueError(f"L{L} {readout}: rows not float16 or non-finite")
    row_run = np.concatenate([[j] * len(metas[i][3]) for j, i in enumerate(keep)]).astype(np.int64)
    return X, row_run, keep, [epochs[i]["name"] for i in range(len(epochs)) if not metas[i][3]], sorted(f"{Path(rt).name}/{ch}" for rt, ch in by)


def within_task_perms(task, y, n: int, seed: int):
    """n label vectors: y permuted among the runs of each main task (runs in the given, sorted-name order)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    groups = {}
    for i, t in enumerate(task):
        groups.setdefault(t, []).append(i)
    out = np.empty((n, len(y)), np.int64)
    for d in range(n):
        yy = y.copy()
        for t in sorted(groups):
            idx = np.array(groups[t])
            yy[idx] = y[idx][rng.permutation(len(idx))]
        out[d] = yy
    single = sum(len(set(y[np.array(g)].tolist())) == 1 for g in groups.values())
    return out, {"n_tasks": len(groups), "n_tasks_single_class": int(single), "n_runs_in_single_class_tasks": int(sum(len(g) for g in groups.values() if len(set(y[np.array(g)].tolist())) == 1))}


@app.function(image=gpu_image, gpu="H100!", cpu=(8, 8), memory=(32768, 32768), timeout=FIT_TIMEOUT, retries=0, volumes={"/b": b_vol, "/d": d_vol})
def fit_cell(L: int, readout: str, epochs: list, skip: dict, allowed: dict) -> dict:
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp

    t0 = time.time()
    X16, row_run, keep, no_rows, chunks = _load(L, readout, epochs, skip, allowed)
    runs = [epochs[i] for i in keep]  # epochs arrive sorted by name
    t_load = time.time() - t0
    XT = torch.from_numpy(X16).cuda()
    del X16
    problems = {"sim_vs_real": ([j for j, r in enumerate(runs) if r["cls"] in ("sim", "real")], lambda r: int(r["cls"] == "sim")),
                "c_vs_b": ([j for j, r in enumerate(runs) if r["batch"] in ("B", "C")], lambda r: int(r["batch"] == "C"))}
    prep = {}
    for p, (ids, lab) in problems.items():
        ids = np.array(ids)
        y = np.array([lab(runs[j]) for j in ids], np.int64)
        perms, pinfo = within_task_perms([runs[j]["task"] for j in ids], y, N_NULL, SEED)
        Y = np.vstack([y[None, :], perms])  # [1 + N_NULL, runs]: row 0 observed
        pos = np.full(len(runs), -1)
        pos[ids] = np.arange(len(ids))
        fold = np.array([runs[j]["fold"] for j in ids])
        prep[p] = {"ids": ids, "Y": Y, "pos": pos, "fold": fold, "pinfo": pinfo, "perm_sha256": hashlib.sha256(perms.tobytes()).hexdigest(),
                   "oof": np.full(Y.shape, np.nan)}
    certs, null_done, took = [], {p: set() for p in problems}, {"observed": [], "null": []}
    for phase, cols in (("observed", [0]), ("null", list(range(1, N_NULL + 1)))):
        for p, d in prep.items():
            for f in range(5):
                if phase == "null":  # reserve: start only if 1.25 x the estimated fit time fits before the hard timeout
                    est = max(took["null"]) if took["null"] else 3 * max(took["observed"])
                    if time.time() - t0 + 1.25 * est > FIT_TIMEOUT - MARGIN_S:
                        certs.append({"phase": phase, "problem": p, "fold": f, "skipped": f"reserve: elapsed {time.time() - t0:.0f} s, estimate {est:.0f} s"})
                        continue
                tr_runs, te_runs = d["ids"][d["fold"] != f], d["ids"][d["fold"] == f]
                tr = np.flatnonzero(np.isin(row_run, tr_runs))
                te = np.flatnonzero(np.isin(row_run, te_runs))
                Xtr, Xte = tp.standardize(XT[torch.from_numpy(tr).cuda()].double(), XT[torch.from_numpy(te).cuda()].double())
                Ytr = torch.from_numpy(d["Y"][cols][:, d["pos"][row_run[tr]]].T.astype(np.float64)).cuda()
                ts = time.time()
                W, b, cert = tp.fit(Xtr, Ytr, C, allow_uncertified=True, return_certificate=True)
                S = (Xte @ W + b).cpu().numpy()  # [te rows, columns]
                loc = d["pos"][row_run[te]]
                tot = np.zeros((len(d["ids"]), len(cols)))
                np.add.at(tot, loc, S)
                cnt = np.bincount(loc, minlength=len(d["ids"]))
                tp_ = d["pos"][te_runs]
                if (cnt[tp_] == 0).any():
                    raise ValueError("a test run without rows")
                d["oof"][np.ix_(cols, tp_)] = (tot[tp_] / cnt[tp_, None]).T
                g = np.array(cert["max_grad_per_column"])
                took[phase].append(time.time() - ts)
                certs.append({"phase": phase, "problem": p, "fold": f, "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
                              "n_train_runs": int(len(tr_runs)), "n_test_runs": int(len(te_runs)), "n_evals": int(cert["n_evals"]),
                              "columns": [int(c) for c in cols], "max_grad_per_column": [float(x) for x in g],
                              "certified_per_column": [bool(x <= tp.CERT_BAR) for x in g], "max_grad": float(g.max()),
                              "uncertified_columns": [int(cols[i]) for i in np.flatnonzero(~(g <= tp.CERT_BAR))], "seconds": round(took[phase][-1], 2),
                              "columns_note": "column 0 = observed labels, column k = null draw k"})
                if phase == "null":
                    null_done[p].add(f)
                del Xtr, Xte, Ytr, W, b
                torch.cuda.empty_cache()
    out = {"layer": L, "readout": readout, "gpu": torch.cuda.get_device_name(0), "load_seconds": round(t_load), "seconds": round(time.time() - t0),
           "rows": int(len(row_run)), "chunks_read": chunks, "runs_without_rows": no_rows,
           "rows_per_run": {r["name"]: int((row_run == j).sum()) for j, r in enumerate(runs)}, "certificates": certs, "problems": {}}
    for p, d in prep.items():
        y, fold, oof = d["Y"][0], d["fold"], d["oof"]
        if not np.isfinite(oof[0]).all():
            raise ValueError(f"{p}: observed out-of-fold scores incomplete or non-finite")
        unc = sorted({(c["fold"], u) for c in certs if c.get("problem") == p for u in c.get("uncertified_columns", [])})
        obs = {"auroc": auroc(oof[0], y), "auroc_per_fold": {str(f): (auroc(oof[0][fold == f], y[fold == f]) if len(set(y[fold == f].tolist())) == 2 else None) for f in range(5)}}
        pf = [v for v in obs["auroc_per_fold"].values() if v is not None]
        obs["auroc_per_fold_mean"] = float(np.mean(pf)) if pf else None
        for a in (0.01, 0.05):
            obs[f"tpr_{a:g}"], obs[f"tpr_{a:g}_per_fold"] = tpr_at(oof[0], y, fold, a)
        res = {"n_runs": {"class1": int(y.sum()), "class0": int((y == 0).sum())},
               "n_rows": {"class1": int(np.isin(row_run, d["ids"][y == 1]).sum()), "class0": int(np.isin(row_run, d["ids"][y == 0]).sum())},
               "n_runs_per_fold": {str(f): {"class1": int(((fold == f) & (y == 1)).sum()), "class0": int(((fold == f) & (y == 0)).sum())} for f in range(5)},
               "shuffle": d["pinfo"], "perm_sha256": d["perm_sha256"], "observed": obs,
               "observed_certified": not any(c == 0 for _, c in unc), "uncertified_fold_columns": [list(x) for x in unc],
               "run_scores": {runs[j]["name"]: float(oof[0][i]) for i, j in enumerate(d["ids"])},
               "null_complete": null_done[p] == set(range(5)) and bool(np.isfinite(oof[1:]).all())}
        if res["null_complete"]:
            na = np.array([auroc(oof[k], d["Y"][k]) for k in range(1, N_NULL + 1)])
            bad = sorted({c for _, c in unc if c > 0})

            def summ(v, o):  # all 100 draws, always (an uncertified draw is labelled, never dropped)
                return {"n_draws": int(len(v)), "p95": float(np.quantile(v, 0.95)), "max": float(v.max()), "mean": float(v.mean()),
                        "p_add_one": float((1 + (v >= o).sum()) / (1 + len(v)))}

            res["null"] = {"auroc": na.tolist(), "summary": summ(na, obs["auroc"]), "uncertified_draws": bad, "certified": not bad,
                           "two_sided_abs_auroc_minus_half": summ(np.abs(na - 0.5), abs(obs["auroc"] - 0.5))}
            for a in (0.01, 0.05):
                nt = np.array([tpr_at(oof[k], d["Y"][k], fold, a)[0] for k in range(1, N_NULL + 1)])
                res["null"][f"tpr_{a:g}"] = nt.tolist()
                res["null"][f"tpr_{a:g}_summary"] = summ(nt, obs[f"tpr_{a:g}"])
        out["problems"][p] = res
    return out


def _git_bytes(spec: dict) -> bytes:
    b = subprocess.run(["git", "show", f"{spec['commit']}:{spec['path']}"], capture_output=True, check=True, cwd=REPO).stdout
    if hashlib.sha256(b).hexdigest() != spec["sha256"]:
        raise SystemExit(f"{spec['path']} @ {spec['commit'][:9]}: sha256 differs from the config's")
    return b


@app.local_entrypoint()
def main(r36: str, config: str = CONFIG):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no") or git("status", "--porcelain", "--", str(HERE)):
        raise SystemExit("uncommitted tracked changes, or uncommitted files in this experiment folder")
    git_at_launch = git("rev-parse", "HEAD")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    if sha(r36) != R36_SHA:
        raise SystemExit("runs36.json sha256 mismatch")
    cfg = json.loads((HERE / config).read_text())
    if git("ls-files", "--error-unmatch", str(HERE / config)) == "":
        raise SystemExit("the config is not tracked")
    # folds: r36 ONLY (never simulated-honest rows); every T196 cell is its own main task
    fold_of_main = {}
    for r in json.loads(Path(r36).read_text()):
        if fold_of_main.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("a main task in two folds")
    main_of = {c["key"]: c["main"] for c in json.loads((REPO / T196).read_text())["cells"]}
    # real honest: corrected B (world honest_b) + C, as run_asis_fits.main / run_simh_score._main
    rc = cfg["real"]
    mf = json.loads(_git_bytes(rc["manifest"]))
    up, st, ex = (json.loads(_git_bytes(rc[k])) for k in ("c_upload", "c_stage", "c_extract"))
    real = [{k: e[k] for k in ("key", "world", "task", "arm", "extraction_run", "staged_inputs_run", "meta_file")} | {"batch": "B"}
            for e in mf["epochs"] if e["world"] in rc["worlds_from_manifest"]]
    shas = {e["staged_inputs_file"]: e["staged_inputs_sha256"] for e in mf["epochs"] if e["world"] in rc["worlds_from_manifest"]}
    excl = set(st["c"]["c_gate"]["excluded"])
    ce = [{"key": f"{f_['path']}#e0", "world": "honest_c", "task": f_["task"], "arm": "honest_b", "extraction_run": ex["config"]["run"],
           "staged_inputs_run": ex["config"]["inputs"], "batch": "C"} for f_ in up["files"] if f"{f_['path']}#e0" not in excl]
    if len(ce) != st["c"]["c_gate"]["staged"]:
        raise SystemExit("batch C epochs differ from its staging gate's count")
    real += ce
    allowed = {ex["config"]["run"]: sorted(c["chunk"] for s_ in ex["shards"] for c in s_["chunks"])}  # batch C's record lists every chunk
    shas |= {f"{ex['config']['inputs']}/inputs/{a}": s for a, s in ex["config"]["inputs_sha256"].items()}
    want = {}
    for e in real:
        want.setdefault(e["staged_inputs_run"], {}).setdefault(e["arm"], []).append(e["key"])
    # simulated honest: each dump's provenance (B200 rule) and counted keys, before any spend
    sims, dump_rec = [], {}
    for dmp in cfg["sim_dumps"]:
        man = json.loads(_git_bytes(dmp["record"]))
        pv = dmp["prov"]
        pin_line = ""
        if pv["gpu_pin"]:
            (pc_path, pc_commit), pc_line = pv["gpu_pin"].rsplit(":", 1)[0].rsplit("@", 1), int(pv["gpu_pin"].rsplit(":", 1)[1])
            if git("rev-parse", "--verify", f"{pc_commit}^{{commit}}") != pc_commit:
                raise SystemExit("the gpu pin must name a full reachable commit")
            pin_line = git("show", f"{pc_commit}:{pc_path}").splitlines()[pc_line - 1]
        prov = provenance(man, dmp["extract_run"], dmp["staged_run"], dmp["arm"], pv["shards_key"], pv["gpu_key"], pv["longest_key"], pv["run_key"],
                          pv["inputs_sha_key"], gpu_pin=pv["gpu_pin"], pin_line=pin_line)
        ck = json.loads(_git_bytes(dmp["counted_keys"]))
        counted = {k["task"] for k in ck["keys"] if k.get("world") == dmp["world"] and k.get("usable", True) is True}
        bk = json.loads(_git_bytes(dmp["batch_keys"])) if dmp["batch_keys"] else None
        if any(s_["done_before"] for s_ in man["shards"]) or sum(c["epochs"] for s_ in man["shards"] for c in s_["chunks"]) != sum(s_["epochs"] for s_ in man["shards"]):
            raise SystemExit(f"{dmp['name']}: the extraction record does not list every chunk")
        allowed[dmp["extract_run"]] = sorted(c["chunk"] for s_ in man["shards"] for c in s_["chunks"])
        dump_rec[dmp["name"]] = {"provenance": prov, "pin_line": pin_line.strip(), "record": dmp["record"], "counted_keys": dmp["counted_keys"],
                                 "batch_keys": dmp["batch_keys"], "n_counted_usable": len(counted), "exclusions": ck.get("exclusions")}
        dump_rec[dmp["name"]]["_counted"], dump_rec[dmp["name"]]["_batch"] = counted, ({k.split("/", 1)[1] for k in bk["usable_keys"]} if bk else None)
    # remote: skip lists (tokenizer on CPU), then the four fits
    calls = {"B+C": skip_b.spawn(want, shas)} | {d["name"]: skip_d.spawn(d["extract_run"], d["staged_run"], d["arm"], dump_rec[d["name"]]["provenance"]["staged_inputs_sha256"])
                                                for d in cfg["sim_dumps"]}
    sk = {k: c.get() for k, c in calls.items()}
    if any(s_["fixed_tokens"] != FIXED_TOKENS for s_ in sk.values()):
        raise SystemExit("fixed call tokens differ from Amendment 12's")
    for dmp in cfg["sim_dumps"]:
        r_ = dump_rec[dmp["name"]]
        staged = {t for _, t in sk[dmp["name"]]["pairs"]}
        if not staged <= r_.pop("_counted") or (r_["_batch"] is not None and staged != r_["_batch"]):
            raise SystemExit(f"{dmp['name']}: staged tasks are not counted usable keys (or differ from the batch's usable keys)")
        r_.pop("_batch")
        r_ |= {"staged_sha256": sk[dmp["name"]]["staged_sha256"], "skip_dropped_rows": sk[dmp["name"]]["dropped_rows"], "n_staged": len(staged)}
        sims += [{"key": k, "world": dmp["world"], "task": t, "arm": dmp["arm"], "extraction_run": dmp["extract_run"], "batch": dmp["batch"], "vol": "d"}
                 for k, t in sk[dmp["name"]]["pairs"]]
    epochs = [e | {"cls": "real", "vol": "b"} for e in real] + [e | {"cls": "sim"} for e in sims]
    for e in epochs:
        e["name"] = f"{e['world']}|{e['task']}"
        e["fold"] = fold_of_main[main_of[e["task"]]]  # KeyError = a task outside T196 or without a #36 fold: stop
    if len({e["key"] for e in epochs}) != len(epochs) or len({e["name"] for e in epochs}) != len(epochs):
        raise SystemExit("duplicate epoch key or run name")
    if len({e["task"] for e in epochs if e["cls"] == "sim"}) != sum(e["cls"] == "sim" for e in epochs):
        raise SystemExit("two simulated-honest epochs share a task")
    epochs.sort(key=lambda e: e["name"])
    skip = sk["B+C"]["skip"] | {k: v for n, s_ in sk.items() if n != "B+C" for k, v in s_["skip"].items()}
    counts = {b_: sum(e["batch"] == b_ for e in epochs) for b_ in sorted({e["batch"] for e in epochs})}
    print(json.dumps({"stamp": stamp, "git": git_at_launch, "epochs_per_batch": counts, "skip_dropped_rows": {k: v["dropped_rows"] for k, v in sk.items()}}), flush=True)
    jobs = {(L, ro): fit_cell.spawn(L, ro, epochs, skip, allowed) for ro in READOUTS for L in LAYERS}
    cells, failed = {}, {}
    for (L, ro), c in jobs.items():
        try:
            cells[f"{ro}|L{L}"] = c.get()
        except Exception as e:  # noqa: BLE001 - recorded, then the run exits nonzero after the record is written
            failed[f"{ro}|L{L}"] = f"{type(e).__name__}: {str(e)[:500]}"
    same = {ro: len({json.dumps(cells[f"{ro}|L{L}"]["rows_per_run"], sort_keys=True) for L in LAYERS if f"{ro}|L{L}" in cells}) <= 1 for ro in READOUTS}
    rec = {"label": "RULE.md (2026-10-14_cmd_readout_simh_control); PRELIMINARY until 09 and Codex review", "git": git_at_launch, "stamp": stamp, "app": APP,
           "volumes": {"real": BVOL, "sim": DVOL}, "config": {"path": config, "sha256": sha(HERE / config)}, "r36": {"path": r36, "sha256": R36_SHA},
           "t196": {"path": T196, "sha256": sha(REPO / T196)}, "settings": {"C": C, "layers": LAYERS, "primary": PRIMARY, "n_null": N_NULL, "seed": SEED,
                                                                           "fitter": "probe_shift.torch_probe.fit defaults (float64, max_iter 5000, tol 1e-8), certificate bar 1e-6"},
           "epochs_per_batch": counts, "real_skip": {"dropped_rows": sk["B+C"]["dropped_rows"], "staged_sha256": sk["B+C"]["staged_sha256"]}, "sim_dumps": dump_rec,
           "run_folds": {e["name"]: e["fold"] for e in epochs}, "rows_per_run_equal_across_layers": same, "cells": cells, "failed": failed}
    dest = HERE / "results" / f"simh_control_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(rec, indent=1))
    for k, c in cells.items():
        for p, r in c["problems"].items():
            o, n = r["observed"], r.get("null", {})
            print(json.dumps({"cell": k, "problem": p, "runs": r["n_runs"], "auroc": round(o["auroc"], 4), "auroc_per_fold_mean": o["auroc_per_fold_mean"],
                              "null": n.get("summary"), "null_certified": n.get("certified"), "tpr_1pct": o["tpr_0.01"],
                              "tpr_5pct": o["tpr_0.05"], "k_of_n_1pct": {f: (v["k"], v["n_class0"]) for f, v in o["tpr_0.01_per_fold"].items()},
                              "observed_certified": r["observed_certified"], "uncertified": len(r["uncertified_fold_columns"]), "null_complete": r["null_complete"],
                              "seconds": c["seconds"], "load_seconds": c["load_seconds"]}), flush=True)
    print(dest.name)
    incomplete = sorted(f"{k}|{p}" for k, c in cells.items() for p, r in c["problems"].items() if not r["null_complete"])
    if failed or not all(same.values()) or incomplete:
        raise SystemExit(f"failed cells {sorted(failed)}, rows differ across layers {same}, or incomplete nulls {incomplete}")
