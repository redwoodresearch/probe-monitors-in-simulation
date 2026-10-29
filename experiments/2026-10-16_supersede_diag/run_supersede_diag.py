"""Why the S_paper493 supersession stopped (2e 03:0x; 09's GO, under $5): when d9 re-staged three attack sets to add one recovered cell
(b_main _try4 -> _try5, c_main _try3 -> _try4, c_run2 _try3 -> _try4), the shared rows' score bytes differed between the old and new Sol / own_a
folders (supersede_sp493.py @ 9b0a5ee8a, results/superseded_sp493_2026-10-16_030149.json: every attack pair stopped, every honest pair passed).
This job measures, keys and magnitudes only (no score, threshold or activation value is returned or written):

1. per pair and per-row file: the shared rows (aligned by key, unit, call_index), the rows whose bytes differ in each numeric field, overall and per
   run key; max and median |difference| over the differing rows; their folds; for score_by_model, whether the differing column is the row's own
   fold model (the one its score uses) or another;
2. decisions: a run's alarm at any threshold depends only on its run maximum of the score it is judged by (fold >= 0: its held-out model's score;
   fold -1: the five models, 21b). Counted per family and readout: runs whose run maximum changed, and the largest change. A run whose maximum is
   unchanged cannot flip at any threshold; the exact count at the registered lead threshold needs that threshold (from the frozen pools) only if
   some run maximum changed;
3. for two shared run keys (the one with the largest score difference on b_main Sol a_tight cmd_A, and the first shared key there whose rows are
   byte-identical, as a control): per layer, the max |difference| of the extracted activation rows between the two extraction roots, each epoch's
   shard, chunk, token count, padded length and readout positions in both roots, and the GPU of each shard;
4. both roots' extraction settings (config.json) and, over every shared attack epoch, whether token counts and readout positions agree.

Modal CPU only; both Volumes mounted read-only.
  PYTHONPATH=src <modal> run experiments/2026-10-16_supersede_diag/run_supersede_diag.py
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
SRC = ("9b0a5ee8a", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/superseded_sp493_2026-10-16_030149.json")  # the stopped supersession record
WORK = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
ACTS = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
ROOTS = {"sp493_b_main_v1_try4__attack": "sp493_b_main_extract_v1_try4", "sp493_b_main_v1_try5__attack": "sp493_b_main_extract_v1_try5",
         "sp493_c_main_v1_try3__attack": "sp493_c_main_extract_v1_try3", "sp493_c_main_v1_try4__attack": "sp493_c_main_extract_v1_try4",
         "sp493_c_run2_v1_try3__attack": "sp493_c_run2_extract_v1_try3", "sp493_c_run2_v1_try4__attack": "sp493_c_run2_extract_v1_try4"}
FILES = {"sol": ("manifest_sol.json", ("a_tight__avg_top2.npz", "a_tight__cmd_A.npz", "c_asis_tight__avg_top2.npz", "c_asis_tight__cmd_A.npz")),
         "own_a": ("manifest_owna.json", ("own_a_tight__avg_top2.npz", "own_a_tight__cmd_A.npz"))}
FAMILY = {"a_tight": "Sol (a)", "c_asis_tight": "c_asis", "own_a_tight": "own_a"}
ID = ("key", "unit", "call_index")
app = modal.App("probe-shift_aswin-surya_sub-supersedediag_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


@app.function(image=image, cpu=4, memory=32768, timeout=1800, retries=0,
              volumes={"/work": modal.Volume.from_name(WORK, version=2).read_only(), "/acts": modal.Volume.from_name(ACTS, version=2).read_only()})
def diag(pairs: list) -> dict:
    import numpy as np

    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    out = {"pairs": [], "activations": [], "settings": {}, "epoch_agreement": {}}
    pick, maxd_by_key = {}, {}
    for p in pairs:
        od, nd = Path("/work/scaleup") / p["old"] / p["sub"], Path("/work/scaleup") / p["new"] / p["sub"]
        mf, npzs = FILES[p["sub"]]
        mob, mnb = (od / mf).read_bytes(), (nd / mf).read_bytes()
        assert sha(mnb) == p["new_manifest_sha256"], (p["new"], "the replacement manifest is not the committed record's")
        mo, mn = json.loads(mob), json.loads(mnb)
        shared = set(mo["folds"]) & set(mn["folds"])
        assert all(mo["folds"][k] == mn["folds"][k] for k in shared), (p["old"], "a fold differs on a shared key")
        rp = {"old": p["old"], "new": p["new"], "sub": p["sub"], "old_manifest_sha256": sha(mob), "new_manifest_sha256": sha(mnb),
              "shared_keys": len(shared), "added": sorted(set(mn["folds"]) - set(mo["folds"])), "removed": sorted(set(mo["folds"]) - set(mn["folds"])), "files": {}}
        for fn in npzs:
            bo, bn = (od / fn).read_bytes(), (nd / fn).read_bytes()
            assert sha(bo) == mo["files"][fn] and sha(bn) == mn["files"][fn], (p["old"], fn, "a file is not its manifest's")
            zo, zn = np.load(io.BytesIO(bo)), np.load(io.BytesIO(bn))
            assert sorted(zo.files) == sorted(zn.files), (p["old"], fn, "fields differ")

            def rows(z):
                ids = list(zip(z["key"].astype(str).tolist(), z["unit"].tolist(), z["call_index"].tolist()))
                ix = {u: i for i, u in enumerate(ids) if u[0] in shared}
                assert len(set(ids)) == len(ids), "duplicate row identity"
                return ix
            xo, xn = rows(zo), rows(zn)
            assert set(xo) == set(xn), (p["old"], fn, "shared row identities differ")
            u_ = sorted(xo)
            io_, in_ = np.array([xo[u] for u in u_], int), np.array([xn[u] for u in u_], int)
            keys_ = np.array([u[0] for u in u_])
            fold_ = np.array([int(mo["folds"][k]) for k in keys_.tolist()])
            rf = {"shared_rows": len(u_), "fields": sorted(zo.files), "by_field": {}}
            for f in zo.files:
                a_, b_ = zo[f], zn[f]
                if f in ID or a_.dtype.kind not in "fiu":
                    continue
                a_, b_ = a_[io_], b_[in_]
                assert a_.dtype == b_.dtype and a_.shape == b_.shape, (fn, f, "dtype or shape differs")
                byte_ = np.array([x.tobytes() != y.tobytes() for x, y in zip(a_, b_)], bool)
                af, bf = a_.astype(np.float64), b_.astype(np.float64)
                nonf_ = ~(np.isfinite(af) & np.isfinite(bf))
                d_ = np.where(nonf_, 0.0, np.abs(af - bf))
                if d_.ndim == 2:  # score_by_model: one column per fold model
                    col_ = np.array([x.tobytes() != y.tobytes() for x, y in zip(a_.T, b_.T)], bool)
                    own_ = np.array([(fo >= 0) and (a_[i, fo].tobytes() != b_[i, fo].tobytes()) for i, fo in enumerate(fold_)], bool)
                    oth_ = np.array([any(a_[i, j].tobytes() != b_[i, j].tobytes() for j in range(a_.shape[1]) if j != fo) for i, fo in enumerate(fold_)], bool)
                    d_ = d_.max(1)
                    nonf_ = nonf_.any(1)
                    extra = {"columns_with_a_difference": [int(j) for j in np.flatnonzero(col_)], "rows_whose_own_fold_column_differs": int(own_.sum()),
                             "rows_where_another_column_differs": int(oth_.sum())}
                else:
                    extra = {}
                dk_, mk_ = {}, {}
                for k, dd in zip(keys_[byte_].tolist(), d_[byte_].tolist() if d_.ndim == 1 else d_[byte_].max(1).tolist()):
                    dk_[k], mk_[k] = dk_.get(k, 0) + 1, max(mk_.get(k, 0.0), dd)
                if f == "score":
                    maxd_by_key = mk_  # step 3's selection: the key with the largest score difference (Codex sdiag1)
                fd_ = {}
                for fo in fold_[byte_].tolist():
                    fd_[str(fo)] = fd_.get(str(fo), 0) + 1
                rf["by_field"][f] = {"rows_differing": int(byte_.sum()), "keys_with_a_differing_row": len(dk_), "differing_rows_by_key": dict(sorted(dk_.items())),
                                     "max_abs_diff": float(d_[byte_].max()) if byte_.any() else 0.0,
                                     "median_abs_diff": float(np.median(d_[byte_])) if byte_.any() else 0.0,
                                     "differing_rows_by_fold": fd_, "rows_non_finite_on_either_side": int(nonf_.sum()), **extra}
            # decisions: a fold >= 0 run is judged by the run maximum of its held-out model's score ('score', else score_by_model's own-fold column);
            # a fold -1 run by all five models (21b): counted apart, by any differing row
            n_, g0_ = len(u_), fold_ >= 0
            if "score" in zo.files:
                src_, A_, B_ = "score", zo["score"][io_].astype(np.float64), zn["score"][in_].astype(np.float64)
            else:
                cf_ = np.clip(fold_, 0, None)
                src_ = "score_by_model[own fold]"
                A_, B_ = zo["score_by_model"][io_][np.arange(n_), cf_].astype(np.float64), zn["score_by_model"][in_][np.arange(n_), cf_].astype(np.float64)
            assert np.isfinite(A_[g0_]).all() and np.isfinite(B_[g0_]).all(), (p["old"], fn, "a non-finite fold >= 0 score")
            ro_, rn_ = {}, {}
            for k, x, y in zip(keys_[g0_].tolist(), A_[g0_].tolist(), B_[g0_].tolist()):
                ro_[k], rn_[k] = max(ro_.get(k, -np.inf), x), max(rn_.get(k, -np.inf), y)
            ch_ = sorted(k for k in ro_ if ro_[k] != rn_[k])
            anyd_ = np.zeros(n_, bool)
            for f, v in rf["by_field"].items():
                if f in ("score", "score_by_model"):
                    anyd_ |= np.isin(keys_, list(v["differing_rows_by_key"]))
            rf["decision"] = {"judged_by": src_, "runs_fold_ge0": len(ro_), "runs_whose_run_max_changed": len(ch_), "keys": ch_,
                              "max_abs_change_of_run_max": max((abs(ro_[k] - rn_[k]) for k in ch_), default=0.0),
                              "runs_fold_minus1": len(set(keys_[~g0_].tolist())), "runs_fold_minus1_with_a_differing_score_row": len(set(keys_[~g0_ & anyd_].tolist())),
                              "reading": "a run whose maximum is unchanged cannot flip at any threshold"}
            rp["files"][fn] = rf
            if p["sub"] == "sol" and fn == "a_tight__cmd_A.npz" and p["old"].startswith("sp493_b_main"):  # step 3's two keys (deterministic)
                b_ = rf["by_field"]["score"]
                worst = max(sorted(maxd_by_key), key=lambda k: maxd_by_key[k]) if maxd_by_key else None  # ties: the first key in sort order
                same = sorted(k for k in set(keys_.tolist()) if k not in b_["differing_rows_by_key"])
                pick = {"most_differing_rows": worst, "control_byte_identical": same[0] if same else None}
        out["pairs"].append(rp)
    # ---- step 3: activations of the two keys between the b_main roots; step 4: settings and per-epoch agreement of every shared attack epoch
    for old_f, new_f in sorted({(p["old"], p["new"]) for p in pairs}):
        ro_, rn_ = Path("/acts") / ROOTS[old_f], Path("/acts") / ROOTS[new_f]
        for r_ in (ro_, rn_):
            out["settings"][r_.name] = json.loads((r_ / "config.json").read_bytes())
            gpu_ = {}
            for f in sorted((r_ / "shard_results").glob("shard*_a*.json")):
                g_ = json.loads(f.read_bytes()).get("gpu_name")
                gpu_[g_] = gpu_.get(g_, 0) + 1
            out["settings"][r_.name] = {k: v for k, v in out["settings"][r_.name].items() if k not in ("inputs_sha256",)} | {"shards_by_gpu": gpu_}

        fk_ = {k for p in pairs if p["old"] == old_f for k in json.loads((Path("/work/scaleup") / old_f / p["sub"] / FILES[p["sub"]][0]).read_bytes())["folds"]}
        wd_ = {k.split("|", 1)[0] for k in fk_}
        assert len(wd_) == 1, (old_f, "an attack folder with more than one world", sorted(wd_))
        wd_ = wd_.pop()

        def metas(r_):  # the staged epoch's run is "<launch>/<rel>" (stage_q4b.process; Codex sdiag1): the scorer's key is "<world>|<task>", one world per folder
            m_ = {}
            for f in (r_ / "meta" / "attack").glob("*.json"):
                x = json.loads(f.read_bytes())
                t_ = x["task"].removesuffix("_r2") if wd_ == "run2" else x["task"]  # the run-2 set stages <cell>_r2; the scorer's key drops the suffix (Codex sdiag2)
                m_.setdefault(f"{wd_}|{t_}", {})[(x["epoch"], x.get("attempt_index"))] = x
            return m_
        mo_, mn_ = metas(ro_), metas(rn_)
        assert set(mo_) & fk_, (old_f, "no extraction attack task matches the folder's keys", sorted(mo_)[:2], sorted(fk_)[:2])
        sk_ = sorted(set(mo_) & set(mn_) & fk_)
        eq_ = {"folder_keys_without_an_extraction_epoch": len(fk_ - set(mo_)), "extraction_tasks_outside_the_folder": len(set(mo_) - fk_), "shared_run_keys": len(sk_), "epoch_sets_differ": 0, "epochs_compared": 0, "n_tokens_differ": 0, "readouts_differ": 0, "padded_tokens_differ": 0}
        for k in sk_:
            if set(mo_[k]) != set(mn_[k]):
                eq_["epoch_sets_differ"] += 1
                continue
            for e in mo_[k]:
                a, b = mo_[k][e], mn_[k][e]
                eq_["epochs_compared"] += 1
                eq_["n_tokens_differ"] += a["n_tokens"] != b["n_tokens"]
                eq_["padded_tokens_differ"] += a.get("padded_tokens") != b.get("padded_tokens")
                eq_["readouts_differ"] += [c.get("readout") for c in a["calls"]] != [c.get("readout") for c in b["calls"]]
        out["epoch_agreement"][f"{ROOTS[old_f]} vs {ROOTS[new_f]}"] = eq_
        if old_f.startswith("sp493_b_main"):
            for role, k in pick.items():
                if k is None:
                    continue
                ent = {"role": role, "run_key": k, "roots": [ro_.name, rn_.name], "epochs": []}
                for e in sorted(set(mo_[k]) & set(mn_[k]), key=str):
                    a, b = mo_[k][e], mn_[k][e]
                    sh_ = {r_.name: int(m["chunk"].split("_", 1)[0].removeprefix("shard")) for r_, m in ((ro_, a), (rn_, b))}
                    gp_ = {r_.name: json.loads((r_ / "shard_results" / f"{'_'.join(m['chunk'].split('_')[:2])}.json").read_bytes()).get("gpu_name")
                           for r_, m in ((ro_, a), (rn_, b))}  # the shard and attempt that wrote the chunk
                    per = []
                    for ly in range(64):
                        xa = np.load(ro_ / "by_layer" / f"L{ly}" / f"{a['chunk']}.npy", mmap_mode="r")[a["row_start"]:a["row_start"] + a["n_rows"]].astype(np.float32)
                        xb = np.load(rn_ / "by_layer" / f"L{ly}" / f"{b['chunk']}.npy", mmap_mode="r")[b["row_start"]:b["row_start"] + b["n_rows"]].astype(np.float32)
                        assert xa.shape == xb.shape, (k, e, ly, "row counts differ")
                        per.append(float(np.abs(xa - xb).max(initial=0.0)))
                    ent["epochs"].append({"epoch": list(e) if isinstance(e, tuple) else e, "chunk": {ro_.name: a["chunk"], rn_.name: b["chunk"]}, "shard": sh_, "gpu": gp_,
                                          "n_tokens": [a["n_tokens"], b["n_tokens"]], "padded_tokens": [a.get("padded_tokens"), b.get("padded_tokens")],
                                          "readouts_equal": [c.get("readout") for c in a["calls"]] == [c.get("readout") for c in b["calls"]],
                                          "rows": a["n_rows"], "max_abs_diff_by_layer": per, "first_layer_with_a_difference": next((i for i, v in enumerate(per) if v > 0), None)})
                out["activations"].append(ent)
    return out


@app.function(image=image, cpu=4, memory=32768, timeout=1800, retries=0,
              volumes={"/work": modal.Volume.from_name(WORK, version=2).read_only(), "/acts": modal.Volume.from_name(ACTS, version=2).read_only()})
def l63_all(pairs: list) -> dict:
    """(b), 2e 03:2x: is every shared attack epoch's L63 block (cmd_A's layer) byte-identical between the old and new extraction roots?
    Epochs matched as in diag(): "<world>|<task>" (run-2 tasks without _r2), one world per folder; counts, and magnitudes only where not identical."""
    import numpy as np

    out = {}
    for old_f, new_f in sorted({(p["old"], p["new"]) for p in pairs}):
        ro_, rn_ = Path("/acts") / ROOTS[old_f], Path("/acts") / ROOTS[new_f]
        fk_ = {k for p in pairs if p["old"] == old_f for k in json.loads((Path("/work/scaleup") / old_f / p["sub"] / FILES[p["sub"]][0]).read_bytes())["folds"]}
        wd_ = {k.split("|", 1)[0] for k in fk_}
        assert len(wd_) == 1, (old_f, "an attack folder with more than one world")
        wd_ = wd_.pop()

        def metas(r_):
            m_ = {}
            for f in (r_ / "meta" / "attack").glob("*.json"):
                x = json.loads(f.read_bytes())
                t_ = x["task"].removesuffix("_r2") if wd_ == "run2" else x["task"]
                m_.setdefault(f"{wd_}|{t_}", {})[(x["epoch"], x.get("attempt_index"))] = x
            return m_
        mo_, mn_ = metas(ro_), metas(rn_)
        fkn_ = {k for p in pairs if p["new"] == new_f for k in json.loads((Path("/work/scaleup") / new_f / p["sub"] / FILES[p["sub"]][0]).read_bytes())["folds"]}
        sk_ = sorted(fk_ & fkn_)  # every key both folders score must have its epochs in both roots (Codex sdiag4)
        miss_ = {"old_root": sorted(set(sk_) - set(mo_)), "new_root": sorted(set(sk_) - set(mn_))}
        assert sk_ and not miss_["old_root"] and not miss_["new_root"], (old_f, "shared folder keys without an extraction epoch", {k: v[:3] for k, v in miss_.items()})
        cache = {}

        def rows(r_, m):  # one L63 chunk file read once per root
            if (r_.name, m["chunk"]) not in cache:
                cache[(r_.name, m["chunk"])] = np.load(r_ / "by_layer" / "L63" / f"{m['chunk']}.npy")
            c_ = cache[(r_.name, m["chunk"])]
            assert m["n_rows"] > 0 and 0 <= m["row_start"] and m["row_start"] + m["n_rows"] <= c_.shape[0], (r_.name, m["chunk"], "an epoch's rows outside its chunk")
            x_ = c_[m["row_start"]:m["row_start"] + m["n_rows"]]
            assert x_.shape[0] == m["n_rows"], (r_.name, m["chunk"], "row count")  # numpy clips out-of-range slices silently (Codex sdiag4)
            return x_
        res = {"shared_run_keys": len(sk_), "epochs_compared": 0, "epochs_byte_identical": 0, "epochs_differing": 0, "rows_compared": 0,
               "epochs_in_the_same_chunk_name": 0, "epochs_in_another_chunk_name": 0, "differing": [], "epoch_sets_differ": 0}
        for k in sk_:
            if set(mo_[k]) != set(mn_[k]):
                res["epoch_sets_differ"] += 1
                continue
            for e in sorted(mo_[k], key=str):
                a, b = mo_[k][e], mn_[k][e]
                xa, xb = rows(ro_, a), rows(rn_, b)
                assert xa.shape == xb.shape and xa.dtype == xb.dtype, (k, e, "L63 row block shape or dtype differs")
                res["epochs_compared"] += 1
                res["rows_compared"] += int(xa.shape[0])
                res["epochs_in_the_same_chunk_name" if a["chunk"] == b["chunk"] else "epochs_in_another_chunk_name"] += 1
                if xa.tobytes() == xb.tobytes():
                    res["epochs_byte_identical"] += 1
                else:
                    res["epochs_differing"] += 1
                    res["differing"].append({"run_key": k, "epoch": list(e), "max_abs_diff": float(np.abs(xa.astype(np.float32) - xb.astype(np.float32)).max())})
        res["chunk_files_read"] = len(cache)
        out[f"{ROOTS[old_f]} vs {ROOTS[new_f]}"] = res
    return out


@app.local_entrypoint()
def main(l63: bool = False):  # --l63: step (b) only (2e 03:2x), its own result file
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout  # noqa: E731
    head = git("rev-parse", "HEAD").strip()
    assert not git("status", "--porcelain", "--untracked-files=no").strip(), "uncommitted tracked changes"
    assert head == git("ls-remote", "origin", "aswin/supersede-diag-2e").split()[0], "HEAD is not pushed"
    src = json.loads(git("show", f"{SRC[0]}:{SRC[1]}"))
    pairs = [p for p in src["pairs"] if p["old"].endswith("__attack")]
    want = {("sp493_b_main_v1_try4__attack", "sp493_b_main_v1_try5__attack"), ("sp493_c_main_v1_try3__attack", "sp493_c_main_v1_try4__attack"),
            ("sp493_c_run2_v1_try3__attack", "sp493_c_run2_v1_try4__attack")}
    assert len(pairs) == 6 and {(p["old"], p["new"], p["sub"]) for p in pairs} == {(o, n, s) for o, n in want for s in ("sol", "own_a")}, "the six attack pairs"
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if l63:
        rec = {"what": "(b) every shared S_paper493 attack epoch's L63 rows (cmd_A's layer), old vs new extraction root: byte-identical or not (counts; "
                       "a magnitude only where not identical; no activation value)", "question": "2e 03:2x: turn 'two keys' into 'all keys'",
               "source": {"commit": SRC[0], "path": SRC[1]}, "git": head, "read_started_pt": stamp, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "by_root_pair": l63_all.remote(pairs)}
        dest = HERE / "results" / f"supersede_diag_l63_{stamp}.json"
        dest.open("x").write(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({"result": str(dest.relative_to(HERE)), "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}))
        return
    res = diag.remote(pairs)
    rec = {"what": "S_paper493 re-extraction: shared attack rows old vs new (keys, counts and difference magnitudes only; no score, threshold or "
                   "activation value)", "question": "2e 03:0x: did the shared rows' scores change enough to flip any alarm decision?",
           "source": {"commit": SRC[0], "path": SRC[1]}, "git": head, "read_started_pt": stamp,
           "extraction_code": {"rule": "one epoch per forward pass (batch size 1); padding after the last id to a multiple of pad_to (causal: no read "
                                       "position changes); weights bfloat16, attention sdpa, flash-linear-attention 0.5.2, torch 2.8.0, transformers "
                                       "5.19.0; rows stored float16 (experiments/2026-10-14_q3_overnight_chain/extract_chain.py, identical at the "
                                       "roots' commits 7ec9b2c41, 0cc33b9e0, c56a1d0d7, a01079d80)"},
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()} | res
    dest = HERE / "results" / f"supersede_diag_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"result": str(dest.relative_to(HERE)), "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}))
