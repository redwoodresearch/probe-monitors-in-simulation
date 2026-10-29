"""Why are sim - real differences detectable but catch gaps small? Descriptive add-on (RULE.md in this folder): the task-paired
activation shift (sim - run1; run2 - run1 as the real-vs-real reference) projected onto each probe's weights in its own
standardized space, in honest-unit score SDs, and its cosine with the weights. Stored activations and stored weights only; no
refit. The laptop reads the pinned git inputs, checks their sha256 and launches; everything else runs in ONE Modal CPU
function, which writes the record to the solavg Volume (xfer/projection_why/<name>/) and returns it.

  PYTHONPATH=src <modal 1.6.1> run --detach experiments/2026-10-15_projection_why_small_gaps/run_projection.py [--addendum]

--addendum (RULE.md Addendum 1, item 3): also sim - run2 for the four own-real readouts (their registered primary; run1 is their
training arm), per task and therefore paired in the task bootstrap; gate G6 (AMENDED after a failure, RULE.md): every contrast
block of the first record (projection_why_2026-10-15_023002.json @ dc9f2079d) reproduces with floats <= 1e-9 (list elements
included) and non-floats, structure and the bootstrap count-matrix sha256 exact.

--diagnostic (0d's OK, 10-15 03:1x via 2e, after G6 failed in projection_why_addendum_2026-10-15_025433): recompute the first
record's contrasts only (no addendum contrast) and write ONLY, per key (probe|contrast), each float field's |diff| from the first
record and their max, the non-float or missing fields that differ, and whether the bootstrap count-matrix sha256 is equal. No
result number is written.
"""

import hashlib
import io
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-projection_why_kill-2026-11-14"
SOLVOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
P2VOL = "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13"
BVOL = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
TIMEOUT = 1800  # one attempt (retries 0 and the exist_ok=False folder guard); 8 cores / 32 GiB, worst case about $0.32
A3 = ("run1", "run2", "sim")
CONTRASTS = (("sim", "run1"), ("run2", "run1"))
FIBS_LAYERS = (19, 27, 35, 43, 51, 58)
TOL_SCORE, TOL_SIG, TOL_MOM, TOL_LIN = 1e-3, 1e-4, 1e-9, 1e-3
PROBES = [  # id, label, role, readout (R = output_end row_mean, C = call_end row), kind
    ("sol_a", "Sol (a) tight", "monitor", "R", "avg_top2"),
    ("fibs", "FIBS", "monitor", "R", "fibs"),
    ("harmfulness", "harmfulness", "monitor", "R", "direction"),
    ("refusal", "refusal", "monitor", "R", "direction"),
    ("highstakes", "high-stakes", "monitor", "R", "direction"),
    ("own_a_avg", "own_a tight, avg_top2", "own-real", "R", "own_a_avg"),
    ("own_a_cmd", "own_a tight, command A (L63)", "own-real", "C", "own_a_cmd"),
    ("own_frac100_R", "own_frac100_L48, R", "own-real", "R", "famexp"),
    ("own_frac100_C", "own_frac100_L48, C", "own-real", "C", "famexp"),
    ("c_asis", "c_asis tight (positive control)", "positive control", "R", "avg_top2"),
]
FPR_KEY = {"sol_a": ("probes", "sol_a|avg_top2|corrected|none|tight"), "c_asis": ("probes", "c_asis|avg_top2|corrected|none"),
           "fibs": ("probes", "fibs|reply_mean|corrected|none"), "harmfulness": ("probes", "harmfulness|reply_mean|corrected|none"),
           "refusal": ("probes", "refusal|reply_mean|corrected|none"), "highstakes": ("probes", "highstakes|reply_mean|corrected|none"),
           "own_a_avg": ("own_real", "own_a_tight avg_top2"), "own_a_cmd": ("own_real", "own_a_tight cmd_A"),
           "own_frac100_R": ("own_real", "own_frac100_L48 R"), "own_frac100_C": ("own_real", "own_frac100_L48 C")}
DIRECTION = {"harmfulness": ("refharm_record", "harmfulness", 34), "refusal": ("refharm_record", "refusal", 47), "highstakes": ("highstakes_record", "highstakes", 43)}

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "safetensors==0.8.0")
sol_vol = modal.Volume.from_name(SOLVOL, version=2)
p2_vol = modal.Volume.from_name(P2VOL, version=2).read_only()
b_vol = modal.Volume.from_name(BVOL, version=2).read_only()


def _compute(name: str, cfg: dict, blobs: dict, meta: dict, od: Path) -> dict:  # noqa: C901
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import safetensors.numpy as stn

    t0 = time.time()
    log, gates, read_sha = [], {}, {}

    def note(msg):
        log.append(f"{time.time() - t0:7.1f} s  {msg}")
        print(log[-1], flush=True)

    def gate(ok, key, info):
        gates[key] = {"pass": bool(ok)} | info
        if not ok:
            diag = {"label": "GATE FAILED: stop and ask 2e (no result numbers)", "where": key, "meta": meta, "gates": gates, "log": log, "read_sha256": read_sha}
            (od / f"{name}_GATE_FAILED.json").write_text(json.dumps(diag, default=float, indent=1))
            sol_vol.commit()
            raise SystemExit(f"gate {key} failed: {json.dumps(info, default=float)[:800]}")

    root = {"sol": Path("/sol"), "p2": Path("/p2"), "b": Path("/b")}

    def read(vol, path, sha=None):
        b = (root[vol] / path.lstrip("/")).read_bytes()
        h = hashlib.sha256(b).hexdigest()
        read_sha[f"{vol}:{path}"] = h
        if sha is not None:
            gate(h == sha, f"G1 {vol}:{path}", {"got": h, "pinned": sha})
        return b

    def npz(vol, path, sha=None):
        return dict(np.load(io.BytesIO(read(vol, path, sha)), allow_pickle=False))

    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    J = {k: json.loads(blobs[k]) for k in ("fpr_record", "manifest", "own_a_record", "asis_scores", "fibs_record", "refharm_record", "highstakes_record")}
    fpr, own_rec = J["fpr_record"], J["own_a_record"]

    # ---------------- G1: the external weights are the ones their scoring records used ----------------
    g1 = {f"fibs_L{L}": sha(blobs[f"fibs_w_L{L}"]) == J["fibs_record"]["inputs"][f"weights_L{L}"] for L in FIBS_LAYERS}
    for pid, (rk, dk, L) in DIRECTION.items():
        d = J[rk]["directions"][dk]
        g1[pid] = sha(blobs[f"{pid}_w"]) == d["sha256"] and d["layer"] == L
    gate(all(g1.values()), "G1 external weights vs scoring records", {"checks": g1})

    # ---------------- cohort, runs, folds ----------------
    coh_lists = [p["cohort"] for p in fpr["probes"].values()] + [v["cohort"] for v in fpr["own_real"].values()]
    cohort = list(coh_lists[0])
    gate(len(cohort) == 183 and len(set(cohort)) == 183 and all(list(c) == cohort for c in coh_lists), "G3 cohort",
         {"n": len(cohort), "lists": len(coh_lists)})
    universe = set(fpr["universe"])
    runs = [f"{w}|{t}" for w in A3 for t in cohort]  # world-major: run index = world index * 183 + task index
    rix = {r: i for i, r in enumerate(runs)}
    ep = {}
    for e in J["manifest"]["epochs"]:
        if e["join"] in rix:
            ep.setdefault(e["join"], []).append(e)
    gate(all(len(ep.get(r, [])) == 1 for r in runs), "G3 manifest epochs", {"runs": len(runs), "bad": [r for r in runs if len(ep.get(r, [])) != 1][:5]})

    # content-only call end (run_asis_fits.b_skip / run_fibs_score.prep): drop call_end rows whose readout token, in the epoch's OWN
    # staged ids, is a fixed call token; staged token ids are read only for that membership test (no text is read or decoded)
    fixed = {tuple(J[k]["fixed_ids"]) for k in ("fibs_record", "refharm_record", "highstakes_record")}
    gate(len(fixed) == 1, "G3 fixed call-token ids agree across scoring records", {"fixed_ids": [list(x) for x in fixed]})
    fixed_ids = set(next(iter(fixed)))
    want = {}
    for r in runs:
        e = ep[r][0]
        want.setdefault((e["staged_inputs_file"], e["staged_inputs_sha256"]), set()).add(e["key"])
    skip, dropped = {}, {}
    for (rel, sh), keys in sorted(want.items()):
        seen = set()
        for line in read("b", rel, sh).decode().splitlines():
            st = json.loads(line)
            if st["key"] not in keys:
                continue
            if st["key"] in seen:
                raise SystemExit(f"G3: {rel}: a key twice")
            seen.add(st["key"])
            skip[st["key"]] = {(c["irid"], c["call_index"]) for c in st["calls"]
                               if c["kind"] == "call_end" and c["readout"] is not None and st["ids"][c["readout"]] in fixed_ids}
        gate(seen == keys, f"G3 staged epochs found in {rel}", {"missing": len(keys - seen)})
        dropped[rel] = sum(len(skip[k]) for k in keys)
    note(f"staged skip sets built: {dropped}")

    def meta_of(r):
        e = ep[r][0]
        m = json.loads((Path("/b") / e["meta_file"]).read_text())
        rt = e["meta_file"].split("/meta/")[0]
        if m["key"] != e["key"] or rt != e["extraction_run"]:
            raise SystemExit(f"G3: {r}: meta key or extraction run differs from the manifest")
        R = [(int(c["call_index"]), m["row_start"] + int(c["row_mean"])) for c in m["calls"] if c["kind"] == "output_end" and c.get("row_mean") is not None]
        ce = [c for c in m["calls"] if c["kind"] == "call_end" and c.get("row") is not None]
        if len({(c["irid"], c["call_index"]) for c in ce}) != len(ce):
            raise SystemExit(f"G3: {r}: (irid, call_index) repeats among call_end rows")
        sk = skip[e["key"]]
        C = [(int(c["call_index"]), m["row_start"] + int(c["row"])) for c in ce if (c["irid"], c["call_index"]) not in sk]
        return r, {"root": rt, "chunk": m["chunk"], "R": R, "C": C, "end": m["row_start"] + m["n_rows"]}

    with ThreadPoolExecutor(32) as ex:
        metas = dict(ex.map(meta_of, runs))
    note(f"read {len(metas)} metas")

    # ---------------- unit files ----------------
    choice_a = [sorted(c) for c in J["asis_scores"]["layer_choice_a"]]
    choice_own = [sorted(c) for c in own_rec["layer_choice_own_a"]]

    def keyarr(z):
        return np.array([str(k) for k in z["key"]])

    def load_units(pid, kind):
        """Unit table: key, unit, call_index, score (the thresholded unit score), lin (its linear surrogate), fold (-1 for one model),
        comp {layer: per-unit stored component logit} where the file stores one."""
        comp = {}
        if pid in ("sol_a", "c_asis"):
            s = cfg["unit_files"][pid]
            z = npz(s["vol"], s["path"], s["sha256"])
            fold = z["fold"].astype(np.int64)
            lay = np.stack([z["layer_a"], z["layer_b"]], 1) if pid == "sol_a" else z["layers"]
            ok = all(sorted(lay[i].tolist()) == choice_a[fold[i]] for i in range(len(fold)))
            gate(ok, f"G3 {pid} layers per row", {"rows": len(fold)})
            score = z["score"].astype(np.float64)
            lin = score
        elif pid == "fibs":
            s = cfg["unit_files"][pid]
            z = npz(s["vol"], s["path"], s["sha256"])
            gate(z["layers"].tolist() == list(FIBS_LAYERS), "G3 fibs layers", {"layers": z["layers"].tolist()})
            lg = z["logits"].astype(np.float64)
            score, lin, fold = z["score"].astype(np.float64), lg.mean(1), np.full(len(lg), -1, np.int64)
            comp = {L: lg[:, k] for k, L in enumerate(FIBS_LAYERS)}
        elif kind == "direction":
            s = cfg["unit_files"][pid]
            z = npz(s["vol"], s["path"], s["sha256"])
            gate(int(z["layer"]) == DIRECTION[pid][2], f"G3 {pid} layer", {"layer": int(z["layer"])})
            score = z["score"].astype(np.float64)
            lin, fold = score, np.full(len(score), -1, np.int64)
        elif pid == "own_a_avg":
            parts = []
            for f in range(5):
                La, Lb = choice_own[f]
                ea, eb = own_rec["saved"]["tight"]["avg"][f"{La}|{f}"], own_rec["saved"]["tight"]["avg"][f"{Lb}|{f}"]
                za, zb = npz("sol", ea["rows"], ea["rows_sha256"]), npz("sol", eb["rows"], eb["rows_sha256"])
                same = all(np.array_equal(za[x], zb[x]) for x in ("key", "unit", "call_index", "fold")) and set(za["fold"].tolist()) == {f}
                gate(same, f"G3 own_a_avg fold {f} layer files aligned", {"layers": [La, Lb]})
                parts.append((za, zb, La, Lb))
            z = {x: np.concatenate([p[0][x] for p in parts]) for x in ("key", "unit", "call_index", "fold")}
            score = np.concatenate([(p[0]["score"].astype(np.float64) + p[1]["score"].astype(np.float64)) / 2 for p in parts])
            lin, fold = score, z["fold"].astype(np.int64)
            n = [len(p[0]["score"]) for p in parts]
            for L in sorted({L for c in choice_own for L in c}):
                v = np.full(len(score), np.nan)
                off = 0
                for (za, zb, La, Lb), k in zip(parts, n):
                    if La == L:
                        v[off:off + k] = za["score"]
                    if Lb == L:
                        v[off:off + k] = zb["score"]
                    off += k
                comp[L] = v
        elif pid == "own_a_cmd":
            e = own_rec["saved"]["cmd"]["cmdA"]["rows|own_a_tight"]
            z = npz("sol", e["path"], e["sha256"])
            score = z["score"].astype(np.float64)
            lin, fold = score, z["fold"].astype(np.int64)
        else:  # own_frac100 R / C: the open pack and the (released) sim pack
            fs = cfg["unit_files"]["own_frac100_L48_" + pid[-1]]
            zo, zs = npz(fs[0]["vol"], fs[0]["path"], fs[0]["sha256"]), npz(fs[1]["vol"], fs[1]["path"], fs[1]["sha256"])
            wo, ws = {str(k).split("|", 1)[0] for k in zo["key"]}, {str(k).split("|", 1)[0] for k in zs["key"]}
            gate(ws == {"sim"} and "sim" not in wo, f"G3 {pid} open/sim worlds", {"open": sorted(wo), "sim": sorted(ws)})
            z = {x: np.concatenate([zo[x], zs[x]]) for x in ("key", "unit", "call_index", "fold")}
            score = np.concatenate([zo["score"], zs["score"]]).astype(np.float64)
            lin, fold = score, z["fold"].astype(np.int64)
        keys = keyarr(z)
        if not (len(keys) == len(score) == len(lin) == len(fold) and np.isfinite(score).all() and np.isfinite(lin).all()):
            raise SystemExit(f"G3: {pid}: misaligned or non-finite unit arrays")
        return {"key": keys, "unit": z["unit"].astype(np.int64), "call_index": z["call_index"].astype(np.int64), "score": score, "lin": lin,
                "fold": fold, "comp": comp}

    U = {}
    for pid, _, _, ro, kind in PROBES:
        U[pid] = load_units(pid, kind)
    note("unit files loaded")

    # ---------------- honest SD (G4) ----------------
    honest_sd, g4 = {}, {}
    for pid, _, _, _, kind in PROBES:
        u = U[pid]
        w = np.array([k.split("|", 1)[0] for k in u["key"]])
        t = np.array([k.split("|", 1)[1] for k in u["key"]])
        pool = np.isin(w, ["honest_b", "honest_c"]) & np.isin(t, sorted(universe))
        folds = [-1] if (u["fold"] == -1).all() else list(range(5))
        mom = {}
        for f in folds:
            sel = pool & (u["fold"] == f)
            v, vs = u["lin"][sel], u["score"][sel]
            mom[f] = {"mean": float(v.mean()), "sd": float(v.std(ddof=0)), "n": int(sel.sum()), "score_sd": float(vs.std(ddof=0))}
            if not mom[f]["sd"] > 0:
                gate(False, f"G4 {pid}", {"fold": f, "sd": mom[f]["sd"]})
        pool_runs = sorted(set(u["key"][pool].tolist()))
        sec, key = FPR_KEY[pid]
        ref = fpr[sec][key]
        chk = {"pool_runs": len(pool_runs), "pool_units": int(pool.sum())}
        if sec == "probes":
            chk["pool_runs_equal_fpr"] = pool_runs == sorted(ref["pool_runs"])
            refm = ref["z_moments_unit"]
        else:
            (blk,) = [ref[b] for b in ("primary", "sensitivity") if ref[b]["pool"] == "honest_b+honest_c"]
            chk["pool_counts_equal_fpr"] = (blk["pool_runs"], blk["n_pool_units"]) == (len(pool_runs), int(pool.sum()))
            refm = blk["z_moments_unit"]
        if refm is not None:
            d = max(max(abs(mom[f]["mean"] - refm[str(f)]["mean"]), abs(mom[f]["sd"] - refm[str(f)]["sd"])) for f in range(5))
            chk |= {"moments_max_abs_diff": d, "n_equal": all(mom[f]["n"] == refm[str(f)]["n"] for f in range(5))}
            ok = d <= TOL_MOM and chk["n_equal"]
        else:
            ok = True
        ok = ok and all(v for k, v in chk.items() if k.endswith("_fpr"))
        g4[pid] = chk
        gate(ok, f"G4 {pid} honest moments", chk)
        honest_sd[pid] = mom
    note("honest SDs computed (G4 passed)")

    # ---------------- unit -> row mapping for the cohort runs (G3) ----------------
    tables = {"R": {}, "C": {}}  # (root, chunk, row) -> table index

    def tindex(ro, ent):
        return tables[ro].setdefault(ent, len(tables[ro]))

    M = {}
    for pid, _, _, ro, _ in PROBES:
        u = U[pid]
        cohort_mask = np.isin(u["key"], runs)
        by = {}
        for i in np.flatnonzero(cohort_mask):
            by.setdefault(u["key"][i], []).append(i)
        bad = [r for r in runs if r not in by]
        gate(not bad, f"G3 {pid} every cohort run has units", {"missing": bad[:5], "n_missing": len(bad)})
        order, trow, run_of, fold_of_run, mism = [], [], [], {}, []
        for r in runs:
            ix = sorted(by[r], key=lambda i: u["unit"][i])
            fs = set(u["fold"][ix].tolist())
            if len(fs) != 1:
                mism.append((r, "folds"))
                continue
            fold_of_run[r] = fs.pop()
            m = metas[r]
            if [int(u["unit"][i]) for i in ix] != list(range(len(ix))):
                mism.append((r, "units are not 0..n-1"))
                continue
            # R: every output_end row_mean call; C: every content-only call_end row; both in meta order, equal count and call_index
            if len(ix) != len(m[ro]) or any(int(u["call_index"][i]) != m[ro][q][0] for q, i in enumerate(ix)):
                mism.append((r, f"{ro} count/call_index"))
                continue
            rows = [m[ro][q][1] for q in range(len(ix))]
            if max(rows) >= m["end"]:
                mism.append((r, "row past the epoch"))
                continue
            order += ix
            trow += [tindex(ro, (m["root"], m["chunk"], q)) for q in rows]
            run_of += [rix[r]] * len(ix)
        gate(not mism, f"G3 {pid} unit-to-row mapping", {"bad": [list(x) for x in mism[:5]], "n_bad": len(mism)})
        order, run_of = np.array(order), np.array(run_of)
        fr = np.array([fold_of_run[r] for r in runs])
        if pid != "fibs" and DIRECTION.get(pid) is None:
            gate(all(fr[w * 183 + j] == fr[j] for w in range(3) for j in range(183)), f"G3 {pid} one fold per task", {})
        sc = u["score"][order]
        starts = np.flatnonzero(np.r_[True, run_of[1:] != run_of[:-1]])
        gate(len(starts) == len(runs) and (run_of[starts] == np.arange(len(runs))).all(), f"G3 {pid} contiguous runs", {})
        mx = np.array([s + int(np.argmax(sc[s:e])) for s, e in zip(starts, np.r_[starts[1:], len(sc)])])  # first max on ties
        M[pid] = {"order": order, "trow": np.array(trow), "run_of": run_of, "starts": starts, "maxpos": mx, "fold_task": fr[:183].copy()}
    folds_task = {pid: M[pid]["fold_task"] for pid in M}
    five = [pid for pid, *_ in PROBES if not (U[pid]["fold"] == -1).all()]
    gate(all(np.array_equal(folds_task[p], folds_task[five[0]]) for p in five), "G3 folds agree across five-fold files", {"probes": five})
    note(f"rows mapped: R table {len(tables['R'])}, C table {len(tables['C'])}")

    # ---------------- weights: per probe, per fold group, components (layer, coef, W, b, mean, std) ----------------
    W8 = {}

    def comps_of(pid, kind):
        if kind == "avg_top2":
            rc = "a" if pid == "sol_a" else "c_asis"
            Z = {L: npz("sol", f"xfer/asis_2026-10-14_135701/outer_tight_L{L}.npz") for L in sorted({L for c in choice_a for L in c})}
            g = lambda L, f, p: Z[L][f"{rc}__f{f}__{p}"].astype(np.float64).reshape(-1)  # noqa: E731
            return {f: [(L, 0.5, g(L, f, "W"), float(g(L, f, "b")[0]), g(L, f, "mean"), g(L, f, "std")) for L in choice_a[f]] for f in range(5)}, {}
        if kind == "fibs":
            out, info = [], {}
            for L in FIBS_LAYERS:
                t = stn.load(blobs[f"fibs_w_L{L}"])
                if t["linear.weight"].shape != (1, 5120) or float(np.asarray(t["input_mean"]).reshape(-1)[0]) != 0.0:
                    raise SystemExit(f"G1: fibs L{L}: unexpected weight shape or nonzero input_mean")
                sc = float(np.asarray(t["input_scale"]).reshape(-1)[0])
                out.append((L, 1 / 6, t["linear.weight"][0].astype(np.float64), float(t["linear.bias"].reshape(-1)[0]), np.zeros(5120), np.full(5120, sc)))
                info[L] = {"input_scale": sc}
            return {-1: out}, info
        if kind == "direction":
            t = stn.load(blobs[f"{pid}_w"])
            r = t["direction"].astype(np.float64).reshape(-1)
            b = float(t["bias"].reshape(-1)[0]) if "bias" in t else 0.0
            if r.shape != (5120,):
                raise SystemExit(f"G1: {pid}: direction shape {r.shape}")
            return {-1: [(DIRECTION[pid][2], 1.0, r, b, np.zeros(5120), np.ones(5120))]}, {"bias": b, "norm": float(np.linalg.norm(r))}
        if kind == "own_a_avg":
            out = {}
            for f in range(5):
                out[f] = []
                for L in choice_own[f]:
                    e = own_rec["saved"]["tight"]["avg"][f"{L}|{f}"]
                    z = npz("sol", e["weights"], e["weights_sha256"])
                    out[f].append((L, 0.5, z["W"].astype(np.float64).reshape(-1), float(z["b"].reshape(-1)[0]), z["mean"].astype(np.float64).reshape(-1),
                                   z["std"].astype(np.float64).reshape(-1)))
            return out, {}
        if kind == "own_a_cmd":
            c = own_rec["saved"]["cmd"]["cmdA"]
            zt, zo = npz("sol", c["own_a_tight_weights"]["path"], c["own_a_tight_weights"]["sha256"]), npz("sol", c["own_a_weights"]["path"], c["own_a_weights"]["sha256"])
            out, which = {}, {}
            for f in range(5):
                z = zt if f"own_a__f{f}__W" in zt else zo
                which[f] = "tight" if z is zt else "outer"
                g = lambda p, z=z, f=f: z[f"own_a__f{f}__{p}"].astype(np.float64).reshape(-1)  # noqa: E731
                out[f] = [(63, 1.0, g("W"), float(g("b")[0]), g("mean"), g("std"))]
            return out, {"weights_used": which}
        ro = pid[-1]  # famexp
        out = {}
        for f in range(5):
            e = cfg["weights"]["own_frac100_L48"][f"{ro}|{f}"]
            t = stn.load(read("p2", e["path"], e["sha256"]))
            out[f] = [(48, 1.0, t["W"].astype(np.float64).reshape(-1), float(t["b"].reshape(-1)[0]), t["mean"].astype(np.float64).reshape(-1),
                       t["std"].astype(np.float64).reshape(-1))]
        return out, {}

    winfo = {}
    for pid, _, _, _, kind in PROBES:
        W8[pid], winfo[pid] = comps_of(pid, kind)
        for f, cs in W8[pid].items():
            for L, a, w, b, m, s in cs:
                if not (w.shape == m.shape == s.shape == (5120,) and np.isfinite(w).all() and np.isfinite(m).all() and (s > 0).all()):
                    raise SystemExit(f"G1: {pid} f{f} L{L}: bad weight arrays")
    note("weights loaded")

    # ---------------- activations, layer by layer (chunk files loaded whole, rows kept) ----------------
    need = {}
    for pid, _, _, ro, _ in PROBES:
        for cs in W8[pid].values():
            for c in cs:
                need.setdefault(c[0], set()).add(ro)
    inv = {ro: sorted(tables[ro].items(), key=lambda kv: kv[1]) for ro in tables}
    by_chunk = {}
    for ro in tables:
        for (rt, ch, q), ti in inv[ro]:
            by_chunk.setdefault((ro, rt, ch), ([], []))
            by_chunk[(ro, rt, ch)][0].append(q)
            by_chunk[(ro, rt, ch)][1].append(ti)
    MEAN, MAXV, REC, RECC = {}, {}, {pid: np.zeros(len(M[pid]["order"])) for pid in M}, {}
    timing = {}
    for L in sorted(need):
        tl, nbytes = time.time(), 0
        X = {ro: np.zeros((len(tables[ro]), 5120), np.float32) for ro in need[L]}
        jobs = [(k, v) for k, v in by_chunk.items() if k[0] in need[L]]

        def load(job, L=L, X=X):
            (ro, rt, ch), (rows, ti) = job
            p = Path("/b") / rt / "by_layer" / f"L{L}" / f"{ch}.npy"
            A = np.load(p)
            if A.ndim != 2 or A.shape[1] != 5120 or max(rows) >= A.shape[0]:
                raise SystemExit(f"G3: {p}: shape {A.shape} vs row {max(rows)}")
            X[ro][np.array(ti)] = A[np.array(rows)].astype(np.float32)
            return p.stat().st_size

        with ThreadPoolExecutor(16) as ex:
            nbytes = sum(ex.map(load, jobs))
        for ro in X:
            if not np.isfinite(X[ro]).all() or (np.abs(X[ro]).max(1) == 0).any():
                raise SystemExit(f"G3: L{L} {ro}: a non-finite or all-zero row")
        for pid, _, _, ro, _ in PROBES:
            fg = {f: [c for c in cs if c[0] == L] for f, cs in W8[pid].items()}
            if not any(fg.values()):
                continue
            mp = M[pid]
            Xp = X[ro][mp["trow"]].astype(np.float64)
            MEAN[(pid, L)] = np.add.reduceat(Xp, mp["starts"], axis=0) / np.diff(np.r_[mp["starts"], len(Xp)])[:, None]
            MAXV[(pid, L)] = Xp[mp["maxpos"]]
            uf = U[pid]["fold"][mp["order"]]
            lg = np.full(len(Xp), np.nan)
            for f, cs in fg.items():
                for _, a, w, b, m, s in cs:
                    sel = uf == f
                    lg[sel] = ((Xp[sel] - m) / s) @ w + b
                    REC[pid][sel] += a * lg[sel]
            if U[pid]["comp"]:
                RECC[(pid, L)] = lg
            del Xp
        del X
        timing[L] = {"seconds": round(time.time() - tl, 1), "chunk_bytes": int(nbytes), "chunks": len(jobs), "readouts": sorted(need[L])}
        note(f"L{L}: {len(jobs)} chunks, {nbytes / 1e9:.2f} GB, {timing[L]['seconds']} s")

    # ---------------- G2: reproduction of every cohort unit's stored score ----------------
    g2 = {}
    for pid, _, _, _, kind in PROBES:
        o = M[pid]["order"]
        d = float(np.max(np.abs(REC[pid] - U[pid]["lin"][o])))
        g2[pid] = {"units": int(len(o)), "max_abs_diff_linear": d}
        ok = d <= TOL_SCORE
        if kind == "fibs":
            lgs = np.stack([RECC[(pid, L)] for L in FIBS_LAYERS], 1)
            dl = float(np.max(np.abs(lgs - np.stack([U[pid]["comp"][L][o] for L in FIBS_LAYERS], 1))))
            ds = float(np.max(np.abs((1 / (1 + np.exp(-lgs))).mean(1) - U[pid]["score"][o])))
            g2[pid] |= {"max_abs_diff_layer_logits": dl, "max_abs_diff_sigmoid_score": ds}
            ok = ok and dl <= TOL_SCORE and ds <= TOL_SIG
        if kind == "own_a_avg":
            dl = 0.0
            for L, v in U[pid]["comp"].items():
                sel = ~np.isnan(v[o])
                dl = max(dl, float(np.max(np.abs(RECC[(pid, L)][sel] - v[o][sel]))))
            g2[pid]["max_abs_diff_layer_files"] = dl
            ok = ok and dl <= TOL_SCORE
        gate(ok, f"G2 {pid} reproduction", g2[pid])
    note("G2 passed")

    # ---------------- analysis ----------------
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    ND, NT = cfg["bootstrap"]["draws"], len(cohort)
    CNT = np.stack([np.bincount(rng.integers(0, NT, NT), minlength=NT) for _ in range(ND)]).astype(np.float64)  # [draws, tasks]

    def ci(v):
        v = np.asarray(v, np.float64)
        f = v[np.isfinite(v)]
        return {"ci95": [float(np.quantile(f, 0.025)), float(np.quantile(f, 0.975))] if len(f) else None, "finite_draws": int(len(f)), "failed_draws": int(len(v) - len(f))}

    extra = cfg.get("addendum")
    res, table = {}, []
    for pid, label, role, ro, kind in PROBES:
        mp, u = M[pid], U[pid]
        ft = mp["fold_task"]
        groups = [-1] if (u["fold"] == -1).all() else list(range(5))
        gt = {g: np.flatnonzero(ft == g) for g in groups}
        sd = np.array([honest_sd[pid][ft[t]]["sd"] for t in range(NT)])
        ssd = np.array([honest_sd[pid][ft[t]]["score_sd"] for t in range(NT)])
        lin, scr = u["lin"][mp["order"]], u["score"][mp["order"]]
        ends = np.r_[mp["starts"][1:], len(lin)]
        lin_run = {"mean": np.array([lin[s:e].mean() for s, e in zip(mp["starts"], ends)]), "max": lin[mp["maxpos"]]}
        scr_run = {"mean": np.array([scr[s:e].mean() for s, e in zip(mp["starts"], ends)]), "max": scr[mp["maxpos"]]}
        d_dim = len(W8[pid][groups[0]]) * 5120
        out = {"label": label, "role": role, "readout": {"R": "output_end_mean (reply)", "C": "content-only call end"}[ro], "kind": kind,
               "layers_by_fold": {str(g): [c[0] for c in W8[pid][g]] for g in groups}, "d": d_dim, "random_direction_abs_cos_ref": float(1 / np.sqrt(d_dim)),
               "honest_sd": {str(k): v for k, v in honest_sd[pid].items()}, "weights_info": winfo[pid], "contrasts": {}}
        g5 = 0.0
        for A, B in CONTRASTS + ((("sim", "run2"),) if (extra and role == "own-real") else ()):
            cname = f"{A}_minus_{B}"
            ia, ib = A3.index(A) * NT, A3.index(B) * NT
            cres = {}
            for summ, V in (("mean_unit", MEAN), ("max_unit", MAXV)):
                Ds, ps = {}, np.zeros(NT)
                for g in groups:
                    cs = W8[pid][g]
                    tt = gt[g]
                    D = np.concatenate([(V[(pid, c[0])][ia + tt] - V[(pid, c[0])][ib + tt]) / c[5] for c in cs], 1)  # [n_g, d]
                    uvec = np.concatenate([c[1] * c[2] for c in cs])
                    ps[tt] = D @ uvec
                    Ds[g] = (D, uvec)
                key = summ.split("_")[0]
                stored = lin_run[key][ia:ia + NT] - lin_run[key][ib:ib + NT]
                g5 = max(g5, float(np.max(np.abs(ps - stored))))
                P = ps / sd
                Sd = (scr_run[key][ia:ia + NT] - scr_run[key][ib:ib + NT]) / ssd
                cos_raw, cos_deb, n_raw, n_deb = {}, {}, {}, {}
                for g, (D, uvec) in Ds.items():
                    v = D.mean(0)
                    nu, nv = float(np.linalg.norm(uvec)), float(np.linalg.norm(v))
                    if not (nu > 0 and nv > 0):
                        raise SystemExit(f"{pid} {cname} {summ} fold {g}: zero weight or shift norm")
                    tr = float(((D - v) ** 2).sum() / (len(D) - 1))
                    n2 = nv ** 2 - tr / len(D)
                    cos_raw[g] = float(uvec @ v / (nu * nv))
                    cos_deb[g] = float(uvec @ v / (nu * np.sqrt(n2))) if n2 > 0 else None  # an unbounded estimate; None = magnitude unresolved
                    n_raw[g], n_deb[g] = nv, (float(np.sqrt(n2)) if n2 > 0 else None)
                nw = {g: len(gt[g]) for g in groups}
                wmean = lambda d, f=(lambda x: x): (float(sum(nw[g] * f(d[g]) for g in groups) / sum(nw.values())) if all(d[g] is not None for g in groups) else None)  # noqa: E731
                bc, bca, bw = np.zeros(ND), np.zeros(ND), np.zeros(ND)
                for g, (D, uvec) in Ds.items():
                    Cg = CNT[:, gt[g]]
                    tot = Cg.sum(1)
                    for s0 in range(0, ND, 250):
                        Vb = (Cg[s0:s0 + 250] @ D) / np.where(tot[s0:s0 + 250] > 0, tot[s0:s0 + 250], np.nan)[:, None]
                        cb = (Vb @ uvec) / (np.linalg.norm(uvec) * np.linalg.norm(Vb, axis=1))
                        bc[s0:s0 + 250] += tot[s0:s0 + 250] * cb
                        bca[s0:s0 + 250] += tot[s0:s0 + 250] * np.abs(cb)
                    bw += tot
                okd = np.all([CNT[:, gt[g]].sum(1) > 0 for g in groups], 0)
                bcos, bcos_abs = np.where(okd, bc / bw, np.nan), np.where(okd, bca / bw, np.nan)
                cres[summ] = {
                    "projection_honest_sd": {"point": float(P.mean())} | ci((CNT @ P) / NT),
                    "projection_raw_logit": float(ps.mean()),
                    "stored_score_diff_score_sd": {"point": float(Sd.mean())} | ci((CNT @ Sd) / NT),
                    "cos_raw": {"weighted": wmean(cos_raw), "by_fold": {str(g): cos_raw[g] for g in groups}} | ci(bcos),
                    "abs_cos_raw": {"weighted": wmean(cos_raw, abs)} | ci(bcos_abs),
                    "cos_noise_corrected": {"weighted": wmean(cos_deb), "weighted_abs": wmean(cos_deb, abs), "by_fold": {str(g): cos_deb[g] for g in groups},
                                            "folds_abs_gt_1": [str(g) for g in groups if cos_deb[g] is not None and abs(cos_deb[g]) > 1],
                                            "folds_unresolved": [str(g) for g in groups if cos_deb[g] is None]},
                    "shift_norm_raw": {"weighted": wmean(n_raw), "by_fold": {str(g): n_raw[g] for g in groups}},
                    "shift_norm_noise_corrected": {"weighted": wmean(n_deb), "by_fold": {str(g): n_deb[g] for g in groups}},
                    "n_tasks_by_fold": {str(g): int(nw[g]) for g in groups},
                }
            out["contrasts"][cname] = cres
        gate(g5 <= TOL_LIN, f"G5 {pid} linearity", {"max_abs_diff": g5})
        res[pid] = out
        for summ in ("mean_unit", "max_unit"):
            a, b = out["contrasts"]["sim_minus_run1"][summ], out["contrasts"]["run2_minus_run1"][summ]
            table.append({"probe": label, "units": summ, "P_sim_run1": a["projection_honest_sd"], "P_run2_run1": b["projection_honest_sd"],
                          "cos_raw_sim_run1": a["cos_raw"], "abs_cos_raw_sim_run1": a["abs_cos_raw"], "cos_corr_sim_run1": a["cos_noise_corrected"]["weighted"],
                          "abs_cos_corr_sim_run1": a["cos_noise_corrected"]["weighted_abs"], "cos_raw_run2_run1": b["cos_raw"]["weighted"],
                          "norm_corr_sim_run1": a["shift_norm_noise_corrected"]["weighted"], "norm_corr_run2_run1": b["shift_norm_noise_corrected"]["weighted"],
                          "rand_cos": out["random_direction_abs_cos_ref"]})
            if "sim_minus_run2" in out["contrasts"]:
                c2 = out["contrasts"]["sim_minus_run2"][summ]
                table[-1] |= {"P_sim_run2": c2["projection_honest_sd"], "cos_raw_sim_run2": c2["cos_raw"], "abs_cos_raw_sim_run2": c2["abs_cos_raw"]}
    note("analysis done")

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        return x

    rec = {"label": "descriptive add-on, 0d 02:1x panel 3 (RULE.md); PRELIMINARY until 09, Codex and Aswin review", "meta": meta, "name": name,
           "cohort": {"n_tasks": NT, "tasks": cohort, "source": "FPR sweep record probes.<p>.cohort"},
           "bootstrap": cfg["bootstrap"] | {"cnt_sha256": hashlib.sha256(CNT.tobytes()).hexdigest()},
           "a2_detector_cosine": cfg["a2_detector"], "gates": gates, "gates_pass": True, "read_sha256": read_sha, "timing": timing, "log": log,
           "content_only_skip": {"fixed_ids": sorted(fixed_ids), "dropped_rows_per_staged_file_cohort": dropped}, "probes": res, "table": table}
    rec = clean(rec)
    def leaves(x, pre=""):  # every container also leaves a structural marker (type + keys or length), compared exactly
        if isinstance(x, dict):
            out = {f"{pre}#struct": ("dict", tuple(sorted(x)))}
            for k, v in x.items():
                out |= leaves(v, f"{pre}/{k}")
            return out
        if isinstance(x, list):
            out = {f"{pre}#struct": ("list", len(x))}
            for i, v in enumerate(x):
                out |= leaves(v, f"{pre}[{i}]")
            return out
        return {pre: x}

    def compare_blocks(rp, fp, allowed_extra=()):
        """Per (probe, contrast) of either side: each float leaf's |diff| (list elements are fields), every other leaf exactly (type and
        value), every dict's key set and list's length exactly; one-sided keys are mismatches unless in allowed_extra (new contrasts)."""
        per = {}
        keys = sorted({(pid, c) for src in (rp, fp) for pid in src for c in src[pid].get("contrasts", {})} | {(pid, "<probe>") for pid in set(rp) ^ set(fp)})
        for pid, c in keys:
            if (pid, c) in allowed_extra and pid in rp and pid in fp and c in rp[pid]["contrasts"] and c not in fp[pid]["contrasts"]:
                continue
            if c == "<probe>" or pid not in rp or pid not in fp or c not in rp[pid]["contrasts"] or c not in fp[pid]["contrasts"]:
                per[f"{pid}|{c}"] = {"n_leaves_first": 0, "n_float_fields": 0, "max_abs_float_diff": None, "float_field_abs_diff": {},
                                     "nonfloat_or_missing_mismatch": ["<probe or contrast present on one side only>"]}
                continue
            if True:
                a, b = leaves(rp[pid]["contrasts"][c]), leaves(fp[pid]["contrasts"][c])
                fd = {k: abs(a[k] - b[k]) for k in b if k in a and type(a[k]) is float and type(b[k]) is float}
                nf = sorted(k for k in set(a) | set(b) if k not in fd and (k not in a or k not in b or type(a[k]) is not type(b[k]) or a[k] != b[k]))
                per[f"{pid}|{c}"] = {"n_leaves_first": len(b), "n_float_fields": len(fd), "max_abs_float_diff": max(fd.values()) if fd else None,
                                     "float_field_abs_diff": fd, "nonfloat_or_missing_mismatch": nf}
        return per

    def compare_summary(per, cnt_same):
        mx = max((v["max_abs_float_diff"] for v in per.values() if v["max_abs_float_diff"] is not None), default=None)
        return {"keys": len(per), "max_abs_float_diff": mx, "all_float_diffs_le_1e-9": (mx is None or mx <= 1e-9) and len(per) > 0,
                "all_nonfloat_exact": not any(v["nonfloat_or_missing_mismatch"] for v in per.values()), "cnt_sha256_equal": cnt_same,
                "keys_with_any_float_diff": sorted(k for k, v in per.items() if v["max_abs_float_diff"])}

    if cfg.get("diagnostic"):  # G6 diagnostic: differences only, no result numbers
        first = json.loads(blobs["first_record"])
        per = compare_blocks(rec["probes"], first["probes"])
        cnt_same = rec["bootstrap"]["cnt_sha256"] == first["bootstrap"]["cnt_sha256"]
        summ = compare_summary(per, cnt_same)
        drec = {"label": "G6 diagnostic only (0d's OK 10-15 03:1x via 2e): per-key |diff| of the first record's contrast blocks; no result numbers",
                "comparison_rule": ("every leaf, list elements included, is a field: floats by |diff| (0d: <= 1e-9), every other leaf exactly (type and value); "
                                    "every dict's key set and every list's length exactly; probes or contrasts present on one side only are mismatches"),
                "name": name, "meta": meta, "first_record": cfg["diagnostic"]["first_record"], "gates_G1_to_G5": rec["gates"], "summary": summ, "per_key": per}
        blob = json.dumps(drec, allow_nan=False).encode()
        (od / f"{name}.json").open("xb").write(blob)
        sol_vol.commit()
        return {"file": f"{name}.json", "bytes": blob, "sha256": hashlib.sha256(blob).hexdigest(), "table": [], "summary": summ}
    if extra:  # G6 AMENDED after a failure (RULE.md Addendum 1, amendment of 10-15 03:1x, 0d's OK): floats <= 1e-9, non-floats and count-matrix sha exact
        first = json.loads(blobs["first_record"])
        own = {pid for pid, _, role, _, _ in PROBES if role == "own-real"}
        per = compare_blocks(rec["probes"], first["probes"], allowed_extra={(pid, "sim_minus_run2") for pid in own})
        g6 = compare_summary(per, rec["bootstrap"]["cnt_sha256"] == first["bootstrap"]["cnt_sha256"])
        g6["expected_keys"] = g6["keys"] == sum(len(v["contrasts"]) for v in first["probes"].values())
        gate(g6["all_float_diffs_le_1e-9"] and g6["all_nonfloat_exact"] and g6["cnt_sha256_equal"] and g6["expected_keys"],
             "G6 (amended) first record reproduced: floats <= 1e-9, non-floats and count-matrix sha exact", {k: v for k, v in g6.items() if k != "keys_with_any_float_diff"})
        rec["gates"] = clean(gates)
        rec["addendum"] = extra
    blob = json.dumps(rec, default=float, allow_nan=False).encode()
    (od / f"{name}.json").open("xb").write(blob)
    sol_vol.commit()
    return {"file": f"{name}.json", "bytes": blob, "sha256": hashlib.sha256(blob).hexdigest(), "table": rec["table"]}


@app.function(image=image, cpu=(8, 8), memory=(32768, 32768), timeout=TIMEOUT, retries=0, volumes={"/sol": sol_vol, "/p2": p2_vol, "/b": b_vol})
def run_remote(name: str, cfg: dict, blobs: dict, meta: dict) -> dict:
    od = Path("/sol/xfer/projection_why") / name
    od.mkdir(parents=True, exist_ok=False)  # the cost guard: a rescheduled attempt finds the folder and stops before any work
    sol_vol.commit()
    try:
        return _compute(name, cfg, blobs, meta, od)
    except BaseException as e:  # a diagnostic (no result numbers) for any failure that is not a gate's own; then re-raised
        if not (od / f"{name}_GATE_FAILED.json").exists():
            (od / f"{name}_FAILED.json").write_text(json.dumps({"label": "FAILED (no result numbers): stop and ask 2e", "meta": meta,
                                                                "error": f"{type(e).__name__}: {str(e)[:2000]}"}, indent=1))
            sol_vol.commit()
        raise


@app.local_entrypoint()
def main(config: str = "config/inputs.json", addendum: bool = False, diagnostic: bool = False):
    from zoneinfo import ZoneInfo

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip() or git("status", "--porcelain", "--", str(HERE)).strip():
        raise SystemExit("uncommitted tracked changes, or uncommitted files in this experiment folder")
    head = git("rev-parse", "HEAD").decode().strip()  # read once at launch, before any spawn
    cfg = json.loads((HERE / config).read_text())
    blobs = {}
    for k, spec in cfg["git"].items():
        b = git("show", f"{spec['commit']}:{spec['path']}")
        if hashlib.sha256(b).hexdigest() != spec["sha256"]:
            raise SystemExit(f"G1: {k} sha256 mismatch")
        blobs[k] = b
    if addendum and diagnostic:
        raise SystemExit("--addendum and --diagnostic are exclusive")
    if "addendum" in cfg or "diagnostic" in cfg:
        raise SystemExit("the config file must not set addendum or diagnostic (flags only)")
    if addendum or diagnostic:
        first = {"commit": "dc9f2079d52bcca5f9466867fbe9b76408fda016", "path": "experiments/2026-10-15_projection_why_small_gaps/results/projection_why_2026-10-15_023002.json",
                 "sha256": "00590c2da8b596d2ddc24aa2debde064f395983c28d5cc101be3249482557f6b"}
        b = git("show", f"{first['commit']}:{first['path']}")
        if hashlib.sha256(b).hexdigest() != first["sha256"]:
            raise SystemExit("G1: first record sha256 mismatch")
        blobs["first_record"] = b
        if addendum:
            cfg = cfg | {"addendum": {"item": "RULE.md Addendum 1, item 3: sim - run2 for the own-real readouts (paired per task)", "first_record": first}}
        else:
            cfg = cfg | {"diagnostic": {"item": "G6 diagnostic (0d's OK 10-15 03:1x via 2e)", "first_record": first}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    name = f"projection_why_{'addendum_' if addendum else ('g6diag_' if diagnostic else '')}{stamp}"
    meta = {"git": head, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "config_sha256": hashlib.sha256((HERE / config).read_bytes()).hexdigest(),
            "rule": "experiments/2026-10-15_projection_why_small_gaps/RULE.md", "stamp_pt": stamp, "app": APP,
            "inputs": {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()}}
    print(json.dumps({"name": name, "git": head}), flush=True)
    out = run_remote.remote(name, cfg, blobs, meta)
    dest = HERE / "results" / out["file"]
    dest.parent.mkdir(exist_ok=True)
    dest.open("xb").write(out["bytes"])
    print(json.dumps({"record": str(dest.relative_to(HERE.parents[1])), "sha256": out["sha256"], "volume": f"{SOLVOL}:/xfer/projection_why/{name}/"}), flush=True)
    if "summary" in out:
        print(json.dumps({"g6_diagnostic_summary": out["summary"]}), flush=True)
    for row in out["table"]:
        print(json.dumps(row), flush=True)
