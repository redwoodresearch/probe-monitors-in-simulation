"""RULE.md here (the Sol decision, POSTHOC RULE.md Amendment 22): score Sol (a') on corrected B + batch C. Scores only: this job
computes no threshold, catch, difference, AUROC or comparison between probes; the parent session does the analysis.

(a') = Sol (a) refit without the 41 changed Sol runs (Amendment 21): as-run weights xfer/solex_avg_2026-10-14_110104
(averaged reply, layers 34/46/47/49/63) and xfer/solex_cmd_2026-10-14_110104 (content-only call_end, L63) on the q35solavg
Volume, made by run_avg_transfer.py --phase solex at 1c7100288 (record solex_2026-10-14_110104.json, committed 1eb26a0d1).

The scoring path is run_asis_fits.py's score() (51bf51ec4 / 74992a373) restricted to two models and two readouts: the same
corrected-B manifest + batch C epochs (1,154), the same staged-input skip rule (b_skip as skip_rule; hashes now required before
parsing, parse errors content-free), the same rows, fold routing (#36 fold of the run's main task), per-fold standardization
stored with each fold model, avg_top2 (row = mean of the fold's two chosen layers' logits, run = mean of rows) and cmd_A (L63
call_end rows minus the epoch's fixed-call-token rows). Rows are read once per readout instead of once per layer (same rows,
same order).

Gates, all before anything is written to the work Volume:
  G3 inputs (local, before any Modal call, then after the skip rule): pinned sha256 of every input; epochs, run_folds (both
     maps), layer choice, staged-input sha256, dropped fixed-token rows and fixed tokens equal the asis records' (corrected and
     current B). A G3 failure exits nonzero with its checks printed; no local record is written.
  G1 path: the TIGHT Sol (a) weights through this path reproduce asis_scores_2026-10-14_142228.json run_scores.{avg_top2,
     cmd_A}.a_tight on every run (honest_b reported separately), |diff| <= 1e-9, same run sets, same runs without rows.
  G2 rows: those a_tight rows equal the stored rows_2026-10-14_142228/a_tight__{avg_top2,cmd_A}.npz field by field (key,
     unit, call_index, src_irid, fold, layers/layer exact with the same dtype; score same dtype, <= 1e-9), stored files'
     sha256 checked first.
  G4 weights: (a') through this path on the current (unfixed) B map reproduces the solex record's a_solex at avg_top2 and
     A_content_only_call_end, same run sets, |diff| <= 1e-6 (float32 weights; the observed maximum is recorded). This is a
     functional reproduction, not byte identity: no earlier hash of these weight files exists; their sha256s are recorded now.
  G5 token cut: every scored row has an integer readout token in [0, 32,768).
A G1/G2/G4/G5 failure writes nothing to the Volume, writes results/aprime_<stamp>_GATE_FAILED.json (gates and weight hashes,
no (a') score) and exits nonzero. Any exception inside a Modal function is re-raised as a bounded message (type and at most 160
characters), so no exception object leaves the job.

Detach: the local entrypoint orchestrates (two skip-rule calls, then the job). Keep the client alive until `job` is submitted:
if it is lost earlier, nothing is scored or written. Once `job` runs it finishes and commits remotely even if the client is
lost; only the local results/ copy is then missing (pull it from /aprime/ on the work Volume).

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-14_sol_aprime_correctedB_09/run_aprime_score.py
"""

import hashlib
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-convaudit_aprime_kill-2026-11-14"
SOLVOL, BVOL, HFVOL = ("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13",
                       "probe-shift_aswin-surya_hf-cache")
WORKVOL, OUT = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", "aprime"
CUT, TOL_PATH, TOL_UNFIXED = 32768, 1e-9, 1e-6
SH = "sim-honest"
MODELS = {  # name -> readout -> (folder on the q35solavg Volume, file prefix); every file holds keys a__f{f}__{W,b,mean,std}
    "a_tight": {"avg": ("xfer/asis_2026-10-14_135701", "outer_tight"), "cmd_A": ("xfer/asis_2026-10-14_135701_cmd", "outer_tight")},
    "a_prime": {"avg": ("xfer/solex_avg_2026-10-14_110104", "outer"), "cmd_A": ("xfer/solex_cmd_2026-10-14_110104", "outer")},
}
STORED_ROWS = {"avg_top2": "xfer/asis_2026-10-14_135701/rows_2026-10-14_142228/a_tight__avg_top2.npz",
               "cmd_A": "xfer/asis_2026-10-14_135701/rows_2026-10-14_142228/a_tight__cmd_A.npz"}
B_INPUTS_CURRENT = {"q4b_extract_main_v1": "q4b_main_v1", "q4b_extract_run2_v1": "q4b_run2_v1"}  # run_asis_fits.B_INPUTS_CURRENT
P, Q = "experiments/2026-10-14_q35_sol_avg_POSTHOC/results", "experiments/2026-10-13_q4_why_weaker_POSTHOC/results"
PINNED = {  # name -> (commit, path, sha256): read with git show, never from another session's scratchpad
    "asis_record": ("74992a373dbeb9919f5affd758246fa698760212", f"{P}/asis_scores_2026-10-14_142228.json",
                    "59c051a4ff9cc40811745a119251224f031cdaf479b8d3fb909c1059ab94da31"),
    "asis_current_record": ("51bf51ec4eb13b511fd720a58876ccb0c98681f5", f"{P}/asis_scores_2026-10-14_142000.json",
                            "07457bb23e361dbc38ca8e5d4c93953c356511080af8138a3bb06295cadf932b"),
    "solex_record": ("1eb26a0d127f2f55b27899bd2ca91f981a9b805c", f"{P}/solex_2026-10-14_110104.json",
                     "d2039d0a76bbacac0b3dbcf2be02800506e18b0b1ec75e1223b1939623c33f05"),
    "choice_record": ("aa41408f65b07d0eb2a38655d2897b23be426422", f"{P}/avg_transfer_avg_v1_2026-10-14_011347.json",
                      "cc90ed914bdee4119e9648f7bc85f0e835f9fcd61e2ade3a6185b448b95fd4f4"),
    "manifest": ("46ca25fc48905bca51ec79822970abc655df7b55", f"{P}/correctedb_manifest_v1.json",
                 "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc"),
    "c_upload": ("50a7664eacccc0774e36d4dd891f7c6c96887474", f"{Q}/manifest_q4f_c_eval_upload_v1.json",
                 "98cbbbd52aeb73cfce884b08350130ddd7cc23bf637d1603896f3ae0137be157"),
    "c_stage": ("4c8c2f6a1e8f17579bd8ddeace2eba9119856354", f"{Q}/q4f_stage_c_2026-10-14_120510.json",
                "7864132d252fa225eb6019e0b9558669b121b9ea374c0d77b5de6b95424556a0"),
    "c_extract": ("34ed9fcd91a3969407596dfa9828220cc687026b", f"{Q}/extract_q4b_q4b_extract_c_v1_a1_2026-10-14_121640.json",
                  "d362ab2c23f0d05105901955aaa335eca98e70cb8422ce3285ae863116ac43f1"),
    "b_map": ("581ffb229404e717f6fcc1788d0c60de2ebf8172", f"{Q}/q4b_epoch_map_v1.json",
              "57495aed8870c34fd056b05464c600f5574f677c7ad8f4bca87eb90c57a2dc9a"),
    "r36": ("c24106ca0897b228980ad31d8e415a18dff2ff6c", "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json",
            "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"),
    "simh": ("573054a4d1f56025e9da0f18d6c8684d7693384a", "experiments/2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json",
             "f420ed2401dcc776cbbcfd414bbe9d6d1132a3bbd012b529a33aba7fa5ecd068"),
    "t196": ("debef0af286c2b7dab9eab9ecd6f4a5479079499", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json",
             "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"),
}

app = modal.App(APP)
sol_vol = modal.Volume.from_name(SOLVOL, version=2).read_only()
b_vol = modal.Volume.from_name(BVOL, version=2).read_only()
work_vol = modal.Volume.from_name(WORKVOL, version=2)
tok_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy==2.3.3")
             .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))
cpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


@app.function(image=tok_image, cpu=(4, 4), memory=(32768, 32768), timeout=600, retries=0,
              volumes={"/b": b_vol, "/hf": modal.Volume.from_name(HFVOL).read_only()})
def skip_rule(want: dict, shas: dict) -> dict:
    try:
        return _skip_rule(want, shas)
    except Exception as e:  # noqa: BLE001 - re-raised as a bounded message: no exception object (or its document) leaves the job
        raise RuntimeError(f"skip_rule failed: {type(e).__name__}: {str(e)[:160]}") from None


def _skip_rule(want: dict, shas: dict) -> dict:
    """run_asis_fits.b_skip with two changes that cannot change its output: every staged file's sha256 is REQUIRED and checked
    before it is parsed, and a decode or parse error is re-raised without content. Otherwise verbatim: run_avg_transfer.cmd_prep's
    skip rule, per epoch from its OWN staged inputs: want = {staged run: {arm: [keys]}}; each requested key must be found
    exactly once; shas = {staged run/inputs/arm.jsonl: sha256}."""
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
            b = (Path("/b") / rel).read_bytes()
            file_sha[rel] = hashlib.sha256(b).hexdigest()
            if shas.get(rel) != file_sha[rel]:
                raise ValueError(f"{rel}: staged inputs sha256 missing or different")
            ks, seen = set(keys), set()
            try:
                lines = b.decode().splitlines()
            except UnicodeDecodeError:
                raise ValueError(f"{rel}: not UTF-8") from None
            for i, line in enumerate(lines):
                try:
                    ep = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(f"{rel}: line {i} is not JSON") from None
                if ep["key"] not in ks:
                    continue
                if ep["key"] in seen:
                    raise ValueError("a key twice in one staged file")
                seen.add(ep["key"])
                skip[ep["key"]] = [[r["irid"], r["call_index"]] for r in ep["calls"] if r["kind"] == "call_end" and r["readout"] is not None and ep["ids"][r["readout"]] in fixed_ids]
                counts[rel] = counts.get(rel, 0) + len(skip[ep["key"]])
            if seen != ks:
                raise ValueError(f"{rel}: {len(ks - seen)} requested epochs not staged there")
    return {"fixed_tokens": fixed, "fixed_ids": sorted(fixed_ids), "skip": skip, "dropped_rows": counts, "staged_sha256": file_sha}


def _weights(cache: dict, rel: str) -> dict:
    """One weight file, loaded once; its sha256, and shape and finiteness checks (5 folds, 5,120 features, one bias)."""
    import numpy as np

    if rel not in cache:
        p = Path("/sol") / rel
        raw = p.read_bytes()
        Z = np.load(io.BytesIO(raw))
        shapes = {}
        for f in range(5):
            for q, n in (("W", 5120), ("b", 1), ("mean", 5120), ("std", 5120)):
                a = Z[f"a__f{f}__{q}"]
                if a.size != n or not np.isfinite(a).all():
                    raise ValueError(f"{rel}: a__f{f}__{q} has size {a.size} or non-finite values")
                shapes[f"a__f{f}__{q}"] = [list(a.shape), str(a.dtype)]
        cache[rel] = {"Z": Z, "sha256": hashlib.sha256(raw).hexdigest(), "shapes": shapes}
    return cache[rel]["Z"]


def _score_set(epochs, skip, fold_of_task, choice, names, cache):
    """run_asis_fits.score's rows and arithmetic for the models in `names` at avg_top2 and cmd_A. Returns per-row arrays, per-run
    scores, runs without rows and the token-cut count, per readout."""
    import numpy as np

    layers = sorted({L for c in choice for L in c})
    rows_out, runs_out, no_rows, cut = {}, {}, {}, {}
    for ro, kind, Ls in (("avg", "output_end_mean", layers), ("cmd_A", "call_end", [63])):
        sel, run_of_row, folds, unit, empty, over, no_tok = [], [], [], [], [], 0, 0
        for e in epochs:  # the rows: run_asis_fits.score's selection, read once (identical at every layer)
            root = Path("/b") / e["extraction_run"]
            mf = root / "meta" / e["arm"] / f"{hashlib.sha1(e['key'].encode()).hexdigest()[:20]}.json"
            if e.get("meta_file") and f"{e['extraction_run']}/meta/{e['arm']}/{mf.name}" != e["meta_file"]:
                raise ValueError("meta path differs from the manifest's")
            try:
                m = json.loads(mf.read_text())
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError(f"{e['extraction_run']}/meta/{e['arm']}/{mf.name}: unreadable meta") from None
            if m["key"] != e["key"]:
                raise ValueError("B meta key mismatch")
            if kind == "output_end_mean":
                cs = [c for c in m["calls"] if c["kind"] == "output_end" and c.get("row_mean") is not None]
                rr = [c["row_mean"] for c in cs]
            else:
                sk = {tuple(x) for x in skip[e["key"]]}
                cs = [c for c in m["calls"] if c["kind"] == "call_end" and c.get("row") is not None and (c["irid"], c["call_index"]) not in sk]
                rr = [c["row"] for c in cs]
            if not rr:
                empty.append(e["key"])
                continue
            for c in cs:  # G5: every scored row has an integer readout token in [0, 32,768) (B is extracted cut at 32,768)
                tk = c.get("readout")
                if not isinstance(tk, int) or isinstance(tk, bool):
                    no_tok += 1
                elif not 0 <= tk < CUT:
                    over += 1
            us = [(f"{e['world']}|{e['task']}", str(c["src_irid"]), int(c["call_index"])) for c in cs]
            if len(set(us)) != len(us):
                raise ValueError("duplicate (src_irid, call_index) within a run")
            sel.append((root, m["chunk"], m["row_start"], rr))
            run_of_row += [f"{e['world']}|{e['task']}"] * len(rr)
            folds += [fold_of_task[e["task"]]] * len(rr)
            unit += us
        folds = np.array(folds)
        no_rows[ro], cut[ro] = empty, {"rows": len(unit), "rows_outside_0_to_cut": over, "rows_without_integer_readout": no_tok}
        logit = {}
        for L in Ls:
            X = np.concatenate([np.asarray(np.load(root / "by_layer" / f"L{L}" / f"{ch}.npy", mmap_mode="r")[[s0 + q for q in rr]]).astype(np.float32)
                                for root, ch, s0, rr in sel])
            if not np.isfinite(X).all():
                raise ValueError(f"{ro} L{L}: non-finite B rows")
            for name in names:
                d, pre = MODELS[name][ro]
                Z = _weights(cache, f"{d}/{pre}_L{L}.npz")
                s = np.full(len(unit), np.nan)
                for f in range(5):
                    mk = folds == f
                    if mk.any():
                        W, b, mu, sd = (Z[f"a__f{f}__{p}"] for p in ("W", "b", "mean", "std"))
                        s[mk] = ((X[mk] - mu) / sd) @ W + float(b.reshape(-1)[0])
                if not np.isfinite(s).all():
                    raise ValueError(f"{ro} L{L} {name}: non-finite scores")
                logit[(name, L)] = s
            del X
        fr = np.array([fold_of_task[u[0].split("|")[1]] for u in unit])
        order, seen_ = [], {}
        for u in unit:  # unit = the 0-based order of the row within its run for this readout
            order.append(seen_.get(u[0], 0))
            seen_[u[0]] = order[-1] + 1
        v, Lfix = ("avg_top2", None) if ro == "avg" else ("cmd_A", 63)
        rows_out[v], runs_out[v] = {}, {}
        for name in names:
            if Lfix is None:
                s = np.array([np.mean([logit[(name, L)][i] for L in choice[fr[i]]]) for i in range(len(unit))])
            else:
                s = logit[(name, Lfix)]
            rows_out[v][name] = {"key": np.array([u[0] for u in unit]), "unit": np.array(order, np.int64), "call_index": np.array([u[2] for u in unit], np.int64),
                                 "src_irid": np.array([u[1] for u in unit]), "score": np.asarray(s, np.float64), "fold": fr.astype(np.int64)}
            rows_out[v][name] |= ({"layers": np.array([choice[f] for f in fr], np.int64)} if Lfix is None else {"layer": np.full(len(unit), Lfix, np.int64)})
            tot, cnt = {}, {}
            for u, x in zip(run_of_row, s):
                tot[u], cnt[u] = tot.get(u, 0.0) + float(x), cnt.get(u, 0) + 1
            runs_out[v][name] = {u: tot[u] / cnt[u] for u in tot}
    return rows_out, runs_out, no_rows, cut


def _runs_gate(mine: dict, ref: dict, tol: float, prefix: str = "") -> dict:
    """Same run set and every |diff| <= tol (restricted to runs whose key starts with prefix, when given)."""
    a = {u: x for u, x in mine.items() if u.startswith(prefix)}
    r = {u: x for u, x in ref.items() if u.startswith(prefix)}
    common = sorted(set(a) & set(r))
    d = [abs(a[u] - r[u]) for u in common]
    bad = sum(not (x == x and x <= tol) for x in d)
    return {"runs_compared": len(common), "only_mine": len(set(a) - set(r)), "only_stored": len(set(r) - set(a)), "max_abs_diff": max(d) if d else None,
            "failing": bad, "tolerance": tol, "passes": bool(common) and set(a) == set(r) and bad == 0}


@app.function(image=cpu_image, cpu=(16, 16), memory=(65536, 65536), timeout=900, retries=0,
              volumes={"/sol": sol_vol, "/b": b_vol, "/work": work_vol})
def job(cor: dict, unf: dict, choice: list, fold_of_task: dict, ref: dict, prov: dict, g3: dict) -> dict:
    try:
        return _job(cor, unf, choice, fold_of_task, ref, prov, g3)
    except Exception as e:  # noqa: BLE001 - re-raised as a bounded message: no exception object (or its document) leaves the job
        raise RuntimeError(f"job failed: {type(e).__name__}: {str(e)[:160]}") from None


def _job(cor: dict, unf: dict, choice: list, fold_of_task: dict, ref: dict, prov: dict, g3: dict) -> dict:
    """cor / unf = {"epochs", "skip"} for corrected B + C and for the current (unfixed) B map; ref = the stored values the gates
    compare to; prov = provenance for the outputs. Writes /work/aprime/ only if every gate passes."""
    import numpy as np

    work_vol.reload()
    od = Path("/work") / OUT
    if od.exists():
        raise FileExistsError(f"/{OUT} exists on the work Volume")
    cache = {}
    rows_c, runs_c, nr_c, cut_c = _score_set(cor["epochs"], cor["skip"], fold_of_task, choice, ["a_tight", "a_prime"], cache)
    rows_u, runs_u, nr_u, cut_u = _score_set(unf["epochs"], unf["skip"], fold_of_task, choice, ["a_prime"], cache)
    gates = {}
    for v in ("avg_top2", "cmd_A"):  # G1: the path, on every run and on honest_b alone
        gates[f"G1|{v}|all_runs"] = _runs_gate(runs_c[v]["a_tight"], ref["a_tight"][v], TOL_PATH)
        gates[f"G1|{v}|honest_b"] = _runs_gate(runs_c[v]["a_tight"], ref["a_tight"][v], TOL_PATH, "honest_b|")
    for ro in ("avg", "cmd_A"):
        gates[f"G1|runs_without_rows|{ro}"] = {"mine": len(nr_c[ro]), "stored": len(ref["no_rows"][ro]), "passes": nr_c[ro] == ref["no_rows"][ro]}
    for v, rel in STORED_ROWS.items():  # G2: the rows, field by field
        raw = (Path("/sol") / rel).read_bytes()
        S, M = np.load(io.BytesIO(raw)), rows_c[v]["a_tight"]
        g = {"sha256_ok": hashlib.sha256(raw).hexdigest() == ref["rows_sha256"][v], "fields": sorted(S.files) == sorted(M), "n_rows": [int(len(S["score"])), int(len(M["score"]))]}
        if g["sha256_ok"] and g["fields"] and len(S["score"]) == len(M["score"]):
            g |= {k: bool(np.array_equal(S[k], M[k]) and S[k].dtype == M[k].dtype) for k in M if k != "score"}  # exact, same dtype
            g["score_dtype"] = bool(S["score"].dtype == M["score"].dtype)  # score: same dtype, values within TOL_PATH
            dd = np.abs(S["score"] - M["score"])
            g["score_max_abs_diff"] = float(dd.max())
            g["passes"] = all(g[k] for k in M if k != "score") and g["score_dtype"] and bool(np.isfinite(dd).all() and dd.max() <= TOL_PATH)
        else:
            g["passes"] = False
        gates[f"G2|{v}"] = g
    for v, rv in (("avg_top2", "avg_top2"), ("cmd_A", "A_content_only_call_end")):  # G4: (a') weights reproduce solex on unfixed B
        gates[f"G4|{v}"] = _runs_gate(runs_u[v]["a_prime"], ref["solex"][rv], TOL_UNFIXED)
    for name, c_ in (("corrected", cut_c), ("unfixed", cut_u)):  # G5
        for ro, x in c_.items():
            gates[f"G5|{name}|{ro}"] = x | {"passes": x["rows"] > 0 and x["rows_outside_0_to_cut"] == 0 and x["rows_without_integer_readout"] == 0}
    ok = all(g["passes"] for g in gates.values())
    weights = {rel: {"sha256": c["sha256"], "shapes": c["shapes"]} for rel, c in cache.items()}
    if not ok:  # nothing written; no (a') score leaves
        return {"passes": False, "gates": gates, "weights": weights}
    run_scores = {v: {"a_prime": runs_c[v]["a_prime"]} for v in ("avg_top2", "cmd_A")}
    counts = {"rows": {v: int(len(rows_c[v]["a_prime"]["score"])) for v in rows_c}, "runs": {v: len(run_scores[v]["a_prime"]) for v in run_scores},
              "runs_by_world": {v: {w: sum(u.split("|", 1)[0] == w for u in run_scores[v]["a_prime"]) for w in sorted({u.split("|", 1)[0] for u in run_scores[v]["a_prime"]})}
                                for v in run_scores},
              "finite": {v: bool(np.isfinite(rows_c[v]["a_prime"]["score"]).all()) and all(x == x and abs(x) != float("inf") for x in run_scores[v]["a_prime"].values())
                         for v in run_scores}}
    od.mkdir(exist_ok=False)
    files = {}
    for v in ("avg_top2", "cmd_A"):
        buf = io.BytesIO()
        np.savez(buf, **rows_c[v]["a_prime"])
        with open(od / f"a_prime__{v}.npz", "xb") as fh:
            fh.write(buf.getvalue())
        files[f"a_prime__{v}"] = {"path": f"{OUT}/a_prime__{v}.npz", "sha256": hashlib.sha256(buf.getvalue()).hexdigest(), "n_rows": int(len(rows_c[v]["a_prime"]["score"]))}
    scores = prov | {"epochs": len(cor["epochs"]), "layer_choice_a": choice, "weights": weights, "rows_file": {"volume": WORKVOL, "files": files},
                     "runs_without_rows": nr_c, "run_scores": run_scores, "run_folds": {f"{e['world']}|{e['task']}": fold_of_task[e["task"]] for e in cor["epochs"]},
                     "run_folds_note": "the outer model that scored each run: the #36 (Sol) fold of the run's main task (never canonical_folds)"}
    sb = json.dumps(scores, indent=1).encode()
    with open(od / "aprime_scores.json", "xb") as fh:
        fh.write(sb)
    manifest = {"app": APP, "git": prov["git"], "stamp": prov["stamp"], "passes": True, "gates": gates, "gates_local_G3": g3, "counts": counts,
                "files": files | {"aprime_scores": {"path": f"{OUT}/aprime_scores.json", "sha256": hashlib.sha256(sb).hexdigest()}},
                "weights": {rel: w["sha256"] for rel, w in weights.items()}}
    with open(od / "manifest.json", "x") as fh:
        fh.write(json.dumps(manifest, indent=1))
    work_vol.commit()
    return {"passes": True, "manifest": manifest, "aprime_scores": scores}


def _pinned(name: str):
    commit, path, want = PINNED[name]
    raw = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=REPO).stdout
    if hashlib.sha256(raw).hexdigest() != want:
        raise SystemExit(f"{name}: sha256 differs from the pin")
    return json.loads(raw)


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no", "--", str(HERE.relative_to(REPO)), "src/probe_shift"):
        raise SystemExit("uncommitted tracked changes in this folder or src/probe_shift")
    head = git("rev-parse", "HEAD")
    me = f"{HERE.relative_to(REPO)}/{Path(__file__).name}"
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec, cur, sx, chrec = (_pinned(k) for k in ("asis_record", "asis_current_record", "solex_record", "choice_record"))
    choice = chrec["analysis"]["layer_choice"]["a"]
    g3 = {"choice_equal": choice == rec["layer_choice_a"] == cur["layer_choice_a"] == sx["layer_choice_a_as_run"],
          "asis_tag": rec["tag"] == cur["tag"] == "asis_2026-10-14_135701",
          "solex": sx["gate_passes"] is True and sx["git"] == "1c7100288932e4a1f3a272105dbcd8f4d8b21665" and sx["dropped_from_training"] == {"H": 5, "E1": 36},
          "manifest_sha": rec["inputs"]["manifest"]["sha256"] == PINNED["manifest"][2],
          "c_shas": sorted(v["sha256"] for v in rec["inputs"]["c"].values()) == sorted(PINNED[k][2] for k in ("c_upload", "c_stage", "c_extract")),
          "b_map_sha": cur["inputs"]["b_map"]["sha256"] == PINNED["b_map"][2] and cur["inputs"]["staged_inputs_of"] == B_INPUTS_CURRENT}
    base, simh, t196 = _pinned("r36"), _pinned("simh"), _pinned("t196")
    fold_of_main = {}
    for r in base + [r for r in simh if r["arm"] == SH]:  # run_asis_fits: base + sh
        if fold_of_main.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("a main task in two folds")
    fold_of_task = {c["key"]: fold_of_main[c["main"]] for c in t196["cells"]}
    mf = _pinned("manifest")  # corrected B, then batch C: run_asis_fits.main's construction
    epochs = [{k: e[k] for k in ("key", "world", "task", "arm", "extraction_run", "staged_inputs_run", "meta_file")} for e in mf["epochs"]]
    shas = {e["staged_inputs_file"]: e["staged_inputs_sha256"] for e in mf["epochs"]}
    up, st, ex = _pinned("c_upload"), _pinned("c_stage"), _pinned("c_extract")
    excluded = set(st["c"]["c_gate"]["excluded"])
    ce = [{"key": f"{f_['path']}#e0", "world": "honest_c", "task": f_["task"], "arm": "honest_b", "extraction_run": ex["config"]["run"],
           "staged_inputs_run": ex["config"]["inputs"]} for f_ in up["files"] if f"{f_['path']}#e0" not in excluded]
    if len(ce) != st["c"]["c_gate"]["staged"]:
        raise SystemExit("batch C epochs differ from its staging gate's count")
    epochs += ce
    shas |= {f"{ex['config']['inputs']}/inputs/{a}": s for a, s in ex["config"]["inputs_sha256"].items()}
    bm = _pinned("b_map")  # current (unfixed) B: run_asis_fits.main's --b-map construction
    ep_u = [{"key": k, "world": w, "task": t, "arm": bm["arm_of"][k], "extraction_run": bm["run_of"][k], "staged_inputs_run": B_INPUTS_CURRENT[bm["run_of"][k]]}
            for k, (w, t) in bm["map"].items()]
    for eps in (epochs, ep_u):
        if len({e["key"] for e in eps}) != len(eps) or len({(e["world"], e["task"]) for e in eps}) != len(eps):
            raise SystemExit("duplicate epoch key or (world, task)")
    g3 |= {"epochs_corrected": [len(epochs), rec["epochs"]], "epochs_unfixed": [len(ep_u), cur["epochs"]],
           "run_folds_equal": {f"{e['world']}|{e['task']}": fold_of_task[e["task"]] for e in epochs} == rec["run_folds"],
           "run_folds_equal_unfixed": {f"{e['world']}|{e['task']}": fold_of_task[e["task"]] for e in ep_u} == cur["run_folds"]}
    want_c, want_u = {}, {}
    for eps, w in ((epochs, want_c), (ep_u, want_u)):
        for e in eps:
            w.setdefault(e["staged_inputs_run"], {}).setdefault(e["arm"], []).append(e["key"])
    if not all(v is True for k, v in g3.items() if not k.startswith("epochs_")) or len(epochs) != rec["epochs"] or len(ep_u) != cur["epochs"]:
        raise SystemExit(f"gate G3 (inputs) FAILED before any Modal call: {json.dumps(g3)}")
    c1, c2 = skip_rule.spawn(want_c, shas), skip_rule.spawn(want_u, cur["staged_sha256"])  # every file hash-checked before parsing
    sk_c, sk_u = c1.get(), c2.get()
    g3 |= {"staged_sha256_corrected": sk_c["staged_sha256"] == rec["staged_sha256"], "dropped_rows_corrected": sk_c["dropped_rows"] == rec["skip_dropped_rows"],
           "staged_sha256_unfixed": sk_u["staged_sha256"] == cur["staged_sha256"], "dropped_rows_unfixed": sk_u["dropped_rows"] == cur["skip_dropped_rows"],
           "fixed_tokens": sk_c["fixed_tokens"] == sk_u["fixed_tokens"] == rec["fixed_tokens"] == cur["fixed_tokens"]}
    print(json.dumps({"stamp": stamp, "git": head, "G3": g3}), flush=True)
    if not all(v is True for k, v in g3.items() if not k.startswith("epochs_")):
        raise SystemExit("gate G3 (staged inputs) FAILED before scoring")
    ref = {"a_tight": {v: rec["run_scores"][v]["a_tight"] for v in ("avg_top2", "cmd_A")}, "no_rows": rec["runs_without_rows"],
           "rows_sha256": {v: rec["rows_file"]["files"][f"a_tight__{v}"]["sha256"] for v in STORED_ROWS},
           "solex": {v: sx["scores"][v]["a_solex"] for v in ("avg_top2", "A_content_only_call_end")}}
    if any(rec["rows_file"]["files"][f"a_tight__{v}"]["path"] != p for v, p in STORED_ROWS.items()):
        raise SystemExit("stored a_tight rows paths differ from the record's")
    prov = {"git": head, "script": me, "stamp": stamp, "label": "Spec Append 22's Sol decision (POSTHOC RULE.md Amendment 22): (a') on corrected B + batch C; scores only",
            "probe": "a_prime: Sol (a) refit without the 41 changed Sol runs (5 real honest, 36 first real attack), Amendment 21; AS-RUN fits "
                     "(torch_probe.fit, C 0.01, max_iter 1000; select_l4 at 1c7100288), not tight-refit",
            "inputs": {k: {"commit": c, "path": p, "sha256": s} for k, (c, p, s) in PINNED.items()},
            "staged_sha256": sk_c["staged_sha256"], "skip_dropped_rows": sk_c["dropped_rows"], "fixed_tokens": sk_c["fixed_tokens"]}
    res = job.remote({"epochs": epochs, "skip": sk_c["skip"]}, {"epochs": ep_u, "skip": sk_u["skip"]}, choice, fold_of_task, ref, prov, g3)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    dest = out / f"aprime_{stamp}{'' if res['passes'] else '_GATE_FAILED'}.json"
    with open(dest, "x") as fh:
        fh.write(json.dumps({"G3": g3} | res, indent=1))
    gates = res["gates"] if "gates" in res else res["manifest"]["gates"]
    keep = ("runs_compared", "max_abs_diff", "score_max_abs_diff", "rows_outside_0_to_cut", "rows_without_integer_readout", "passes")
    print(json.dumps({"passes": res["passes"], "gates": {k: {x: g[x] for x in keep if x in g} for k, g in gates.items()}}), flush=True)
    print(dest.name)
    if not res["passes"]:
        raise SystemExit("a gate FAILED: nothing was written to the work Volume")
