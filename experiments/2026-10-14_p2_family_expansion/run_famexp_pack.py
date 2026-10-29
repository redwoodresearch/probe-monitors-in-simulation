"""RULE.md Jobs 4 and 6 (probe-family expansion): X_err v7's extra-probe input, one npz per (variant, readout), in two phases.
Phase 1 (`--part open`): non-sim rows only; the pack holds no sim key anywhere (sim coverage as counts). Phase 2 (`--part
sim_sealed`): sim rows, only after a pushed commit holds the eligibility record that names this open pack (checked in-job).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach <this file> --spec <prep stamp> \\
      --run <fit stamp> --part open | --part sim_sealed --elig-commit <sha> --elig-path <path>
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
APP = "probe-shift_aswin-surya_sub-p2probes_famexppack_kill-2026-11-13"
FRACTIONS, LAYERS_OWN, KS = (0.10, 0.25, 1.0), (16, 32, 48), (5, 10, 20)  # as run_famexp.py
OWN = {f"own_frac{int(round(v * 100))}_L{L}": {"family": "own_a", "fraction": v, "layer": L} for v in FRACTIONS for L in LAYERS_OWN}
AD = {f"ad_k{k}": {"family": "adapted", "k": k} for k in KS}
# RULE.md "recipe_id" (Append 15 §7; Amendment 7b item 5): primary = one recipe per own_a LAYER (fractions are nested draws);
# the single own_a recipe is the labelled sensitivity (Codex round 5 and 0d's lean)
RECIPE = {n: f"own_a_L{v['layer']}" for n, v in OWN.items()} | {n: "adapted_sol_plus_own" for n in AD} | {"b_asis": "b_asis"}
RECIPE_SENS = {n: "own_a" for n in OWN} | {n: RECIPE[n] for n in AD} | {"b_asis": "b_asis"}
B_ASIS = {"R": ("xfer/asis_2026-10-14_135701/rows_2026-10-14_142228/b_asis_tight__avg_top2.npz", "ca90be5ee621a8bddd1f562bc4d309ac7a454fa76b45588e0bc30fcb4e4a6a0d"),
          "C": ("xfer/asis_2026-10-14_135701/rows_2026-10-14_142228/b_asis_tight__cmd_A.npz", "368b7cb20c9bc979ac4891cb56c669e0b86a18cdff96f2f656e24bffd82465de")}
LABEL_RUN1 = "run1 is the training launch (sim shares run1's launch)"  # Append 5 / Amendment 7b: own_a and adapted train on run1
POOLS = {"B": "B", "BuC": "BuC"}  # exactly v6's / v7's labels (09, about 16:5x)
POOL_RULE = "X_err v7: C-sameness per variant x readout on same-fold pairs (BuC if the C-vs-AuB 95% interval is inside [0.45, 0.55], else B); per-fold z (ddof 0, draw-0 moments)"
app = modal.App(APP)
OUT = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2)
SOL = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2).read_only()
img = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


def _rows(files, combine_layers, problems):
    """Concatenate a variant x readout's per-fold files; adapted R averages its fold's two layers row by row (avg_top2). A fold
    without both layers is left out and recorded in `problems` (never substituted by the surviving layer)."""
    import numpy as np

    by_f = {}
    for q in files:
        z = np.load(q)
        by_f.setdefault(int(q.stem.split("__")[3][1:]), []).append({k: z[k] for k in ("key", "unit", "call_index", "score", "fold")})
    parts = []
    for f, zs in sorted(by_f.items()):
        if combine_layers:
            if len(zs) != 2 or any(not np.array_equal(zs[0][k], zs[1][k]) for k in ("key", "unit", "call_index")):
                problems.append(f"fold {f}: avg_top2 needs both layers with identical rows; fold left out")
                continue
            zs = [zs[0] | {"score": (zs[0]["score"] + zs[1]["score"]) / 2}]
        parts += zs
    return {k: np.concatenate([p[k] for p in parts]) for k in ("key", "unit", "call_index", "score", "fold")} if parts else None


def _save(path, r, readout):
    """One npz per (variant, readout): per-row key/unit/call_index/score/fold and run-level run_key/run_score/run_fold
    (reply = mean of rows, command = MAX of rows; v7 recomputes them and must match to 1e-9)."""
    import numpy as np

    keys, inv = np.unique(r["key"], return_inverse=True)
    rs = (np.bincount(inv, weights=r["score"]) / np.bincount(inv)) if readout == "R" else np.full(len(keys), -np.inf)
    if readout == "C":
        np.maximum.at(rs, inv, r["score"])
    rf = np.zeros(len(keys), dtype=np.int64)
    rf[inv] = r["fold"]
    if any(len(set(r["fold"][inv == i].tolist())) != 1 for i in range(len(keys))):
        raise ValueError("a run has rows from two folds")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, key=r["key"], unit=r["unit"], call_index=r["call_index"], score=r["score"], fold=r["fold"], run_key=keys, run_score=rs, run_fold=rf)
    path.open("xb").write(buf.getvalue())
    return {"path": str(path.relative_to("/out")), "sha256": hashlib.sha256(buf.getvalue()).hexdigest(), "rows": int(len(r["key"])), "runs": int(len(keys))}


@app.function(image=img, cpu=(4, 4), memory=(32768, 32768), timeout=600, retries=0, scaledown_window=2, volumes={"/out": OUT, "/sol": SOL})
def pack(spec_stamp: str, run_stamp: str, part: str, stamp: str, split: dict, auth: dict | None = None) -> dict:
    return pack_core(spec_stamp, run_stamp, part, stamp, split, auth, OUT.commit)


def pack_core(spec_stamp: str, run_stamp: str, part: str, stamp: str, split: dict, auth: dict | None, commit) -> dict:
    """The pack itself (plain code, /out and /sol mounted by the caller): run_famexp.py's orchestrator calls it right after the fits."""
    import numpy as np

    base = Path("/out/famexp") / spec_stamp
    if part == "sim_sealed":  # the firewall: verified before any sim file is opened
        auth = auth or {}
        rec_ = auth.get("result", auth).get("phase1_eligibility_record", auth.get("result", auth))  # v7 may wrap its record
        auth = rec_ | {"rows": rec_.get("eligibility_table", rec_.get("rows", [])), "eligibility_record_sha256": auth.get("eligibility_record_sha256"),
                       "commit": auth.get("commit"), "path": auth.get("path")}
        op = base / f"pack_open_{auth.get('open_pack_stamp', 'missing')}" / "manifest.json"
        if hashlib.sha256(op.read_bytes()).hexdigest() != auth.get("open_pack_sha256") or (auth.get("spec_stamp"), auth.get("run_stamp")) != (spec_stamp, run_stamp):
            raise SystemExit("the eligibility record does not name this spec, run and open pack")
        om = json.loads(op.read_text())
        rows = {(x.get("variant"), x.get("readout"), str(x.get("alpha"))): x for x in auth.get("rows", [])}
        want = {(n, r, a) for n in list(OWN) + list(AD) + ["b_asis"] for r in ("R", "C") for a in ("0.01", "0.05")}
        ok = lambda x: (x.get("status") in ("eligible", "ineligible") and isinstance(x.get("real_tpr"), (int, float)) and x.get("pool") in POOLS) or \
            (x.get("status") == "incomplete" and bool(x.get("reason")))  # noqa: E731  (v7 phase 1 statuses; 7b item 5: incomplete is listed, not decided)
        if (om.get("part"), om.get("spec_stamp"), om.get("run_stamp")) != ("open", spec_stamp, run_stamp) or set(rows) != want or not all(map(ok, rows.values())):
            raise SystemExit("the eligibility record is not a complete 52-row table (decision or explicit incomplete) for this open pack")
    spec = json.loads((base / f"spec_{spec_stamp}.json").read_text())
    elig, cohort, main_of = set(split["eligibility_half"]), split["cohort"], split["main_of"]
    sel = sorted(t for t in cohort if t in elig)
    ex_mains = sorted({main_of[t] for t in elig})
    est = sorted(t for t in cohort if t not in elig and main_of[t] not in set(ex_mains))
    if len(elig) != 97 or len(sel) != 93:  # in-job check of the pinned v6 split (Amendment 7b)
        raise SystemExit(f"v6 split mismatch: {len(elig)} eligibility-half tasks, {len(sel)} cohort tasks in it (want 97 / 93)")
    man = json.loads((base / f"fit_{run_stamp}.json").read_text())
    fits = [x for g in man["groups"] for x in g["fits"]]
    d = base / f"run_{run_stamp}" / part
    fails = [g["group"] for g in man["failed_groups"]]  # ids only: no exception text reaches a pack (Codex round 5)
    out = {"part": part, "spec_stamp": spec_stamp, "run_stamp": run_stamp, "pack_stamp": stamp, **({"open_pack_stamp": stamp} if part == "open" else {}),
           "failed_groups": fails,
           "authorization": None if part == "open" else {  # v7 FINAL @ 4970745b: exactly these four, equal to the phase-1 record's values
               "eligibility_record_sha256": auth.get("eligibility_record_sha256"), "open_manifest_sha256": auth.get("open_pack_sha256"),
               "spec_stamp": auth.get("spec_stamp"), "run_stamp": auth.get("run_stamp")},
           "eligibility_record_ref": None if part == "open" else {"commit": auth.get("commit"), "path": auth.get("path")},
           "selection_tasks": sel, "estimation_tasks_7b": est, "excluded_mains_7b": ex_mains, "variants": {}}
    for name in list(OWN) + list(AD) + ["b_asis"]:
        for readout in ("R", "C"):
            e = {"recipe_id": RECIPE[name], "recipe_id_sensitivity": RECIPE_SENS[name], "family": (OWN | AD).get(name, {}).get("family", "b_asis"),
                 "readout": readout, "pool_rule": POOL_RULE, "sealed": part == "sim_sealed", **{k: v for k, v in (OWN | AD).get(name, {}).items() if k != "family"}}
            probs = [f"fit group failed: {g}" for g in fails if g[1] == readout and (g[0] == "ad" if name in AD else g[0] == "own" and g[2] == OWN.get(name, {}).get("layer"))]
            if name == "b_asis":  # 2e's stored tight rows, split here (its stored sim catches were seen before the roster: RULE.md)
                p_, h_ = B_ASIS[readout]
                raw = Path("/sol", p_).read_bytes()
                if hashlib.sha256(raw).hexdigest() != h_:
                    raise SystemExit(f"b_asis {readout}: stored rows sha256 differs from the pin")
                z = np.load(io.BytesIO(raw))
                r = {k: z[k] for k in ("key", "unit", "call_index", "score", "fold")}
                e |= {"source": {"volume_path": p_, "sha256": h_}, "layer": "avg_top2 (34/46/47/49, Sol (a)'s)" if readout == "R" else 63,
                      "folds": None, "certified_by": "2e's tight b_asis fits (asis_fit_2026-10-14_135701; Amendment 23 C2 reading 'tight primary'; P: Append 24)"}
            else:
                e |= {"label": LABEL_RUN1, "layer": e.get("layer") or ("Sol (a)'s per-fold avg_top2" if readout == "R" else 63),
                      "folds": {f"{x['fold']}|L{x['layer']}": {k: x.get(k) for k in ("fold", "layer", "certified", "max_grad", "n", "n_evals", "sha256")}
                                for x in fits if x["variant"] == name and x["readout"] == readout}}
                files = sorted(d.glob(f"{name}__{readout}__L*__f*.npz"))
                r = _rows(files, name in AD and readout == "R", probs)
                exp = {(f, L) for f, pair in enumerate(spec["layer_choice_a"]) for L in pair} if name in AD and readout == "R" else \
                    {(f, 63 if name in AD else OWN[name]["layer"]) for f in range(5)}
                got = {(int(q.stem.split("__")[3][1:]), int(q.stem.split("__")[2][1:])): q for q in files}
                cert = {(v["fold"], v["layer"]): v for v in e["folds"].values()}
                probs += [f"no file for fold {f} L{L}" for f, L in sorted(set(exp) - set(got))] + [f"unexpected file fold {f} L{L}" for f, L in sorted(set(got) - set(exp))]
                probs += [f"fold {f} L{L} not certified" for f, L in sorted(exp) if not cert.get((f, L), {}).get("certified")]
                for (f, L), q in got.items():  # the file must be the one the certified fit wrote, with its own fold and layer inside
                    z_ = np.load(q)
                    want_h = (cert.get((f, L), {}).get("sha256") or {}).get(part)  # a fit without a hash map is incomplete (round 7)
                    if not want_h or hashlib.sha256(q.read_bytes()).hexdigest() != want_h or set(z_["fold"].tolist()) - {f} or set(z_["layer"].tolist()) - {L}:
                        probs.append(f"fold {f} L{L}: file differs from its certified fit (hash, fold or layer)")
            if r is not None:
                world = np.array([k.split("|", 1)[0] for k in r["key"].tolist()])
                keep = (world == "sim") if part == "sim_sealed" else (world != "sim")
                r = {k: v[keep] for k, v in r.items()}
            have = set() if r is None else set(r["key"].tolist())
            ws = ("sim",) if part == "sim_sealed" else ("run1", "run2")
            miss = [f"{w}|{t}" for t in cohort for w in ws if f"{w}|{t}" not in have]
            need_h = {f"{w}|{t}" for k, (w, t) in spec["map"].items() if w in ("honest_a", "honest_b", "honest_c") and spec["n_rows"][k][readout == "C"] > 0}
            miss += sorted(need_h - have) if part == "open" else []
            probs += [f"{len(miss)} required keys without a score"] if miss else []
            e |= {"complete": not probs and r is not None, "incomplete_reasons": probs,
                  "missing": {"n": len(miss), "keys": miss if part == "sim_sealed" else [m for m in miss if not m.startswith("sim|")]}}  # 7b item 5: listed, never dropped
            if part == "open":  # sim coverage as COUNTS only: the open pack holds no sim key (v7 phase 1 refuses one)
                sk = {k for q in (d.parent / "sim_sealed").glob(f"{name}__{readout}__L*__f*.npz") for k in np.load(q)["key"].tolist()} if name != "b_asis" \
                    else {k for k in np.load(io.BytesIO(Path("/sol", B_ASIS[readout][0]).read_bytes()))["key"].tolist() if k.startswith("sim|")}
                e["sim_cohort_runs_present"] = sum(f"sim|{t}" in sk for t in cohort)
            if r is None:  # every roster entry gets a schema-valid npz, empty if need be (v7 reads e["npz"] unconditionally; Codex round 9)
                r = {"key": np.array([], dtype="<U1"), "unit": np.array([], dtype=np.int64), "call_index": np.array([], dtype=np.int64),
                     "score": np.array([], dtype=np.float64), "fold": np.array([], dtype=np.int64)}
            e["npz"] = _save(base / f"pack_{part}_{stamp}" / f"{name}__{readout}.npz", r, readout)
            out["variants"][f"{name}|{readout}"] = e
    if part == "open" and "sim|" in json.dumps(out):
        raise SystemExit("internal: a sim key reached the open manifest")
    mp = base / f"pack_{part}_{stamp}" / "manifest.json"
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.open("x").write(json.dumps(out, indent=1, default=float))
    commit()
    return {"manifest": "/p2/" + str(mp.relative_to("/out")), "sha256": hashlib.sha256(mp.read_bytes()).hexdigest(), "selection": len(sel), "estimation": len(est),
            "incomplete": [k for k, v in out["variants"].items() if not v["complete"]]}


def load_split():
    """The pinned v6 split (sha256-checked) and the T196 cell -> main map: the inputs of the in-job 7b checks."""
    pin = json.loads((HERE / "v6_pin.json").read_text())
    raw = subprocess.run(["git", "-C", str(REPO), "show", f"{pin['commit']}:{pin['path']}"], capture_output=True, check=True).stdout
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise SystemExit("the pinned X_err v6 record differs from its sha256")
    v6 = json.loads(raw)["result"]
    t196 = json.loads(subprocess.run(["git", "-C", str(REPO), "show", f"4d1eec572:{pin['t196']}"], capture_output=True, check=True).stdout)
    return pin, {"eligibility_half": v6["universe_and_split"]["eligibility_half"], "cohort": v6["cohort"]["tasks"], "main_of": {c["key"]: c["main"] for c in t196["cells"]}}


@app.local_entrypoint()
def main(spec: str, run: str, part: str, elig_commit: str = "", elig_path: str = "", elig_sha256: str = ""):
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    me = [str((HERE / f).relative_to(REPO)) for f in ("run_famexp_pack.py", "RULE.md", "v6_pin.json")]
    if git("status", "--porcelain", "--untracked-files=no") or git("status", "--porcelain", "--untracked-files=all", "--", *me):
        raise SystemExit("uncommitted changes, or this launcher / RULE.md is untracked: commit and push first (frozen analysis)")
    pin, split = load_split()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if part == "open":
        out = pack.remote(spec, run, "open", stamp, split)
    elif part == "sim_sealed" and elig_commit and elig_path and len(elig_sha256) == 64:  # phase 2: 09's record, verified on origin (fetch first)
        if git("cat-file", "-t", elig_commit) != "commit" or not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", elig_commit).splitlines()):
            raise SystemExit("--elig-commit is not a commit on origin (fetch 09's branch first)")
        rb = subprocess.run(["git", "-C", str(REPO), "show", f"{elig_commit}:{elig_path}"], capture_output=True, check=True).stdout
        if hashlib.sha256(rb).hexdigest() != elig_sha256:
            raise SystemExit("the eligibility record's sha256 differs from the one 09 sent")
        out = pack.remote(spec, run, "sim_sealed", stamp, split, json.loads(rb) | {"commit": elig_commit, "path": elig_path, "eligibility_record_sha256": elig_sha256})
    else:
        raise SystemExit("--part open | sim_sealed --elig-commit <sha> --elig-path <path> --elig-sha256 <sha256>")
    rec = {"git": git("rev-parse", "HEAD"), "stamp (PT)": stamp, "app": APP, "part": part, "v6_pin": pin, "result": out}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"pack_{part}_{stamp}.json").open("x").write(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, indent=1, default=float)[:3000])
