"""Rule out a primary/secondary label swap in the DT rows of the v5 record (0d's recheck, 2e 14:24). Read-only, from the raw files the record pins.

Per DT design and arm (honest A/B/C, run 1, run 2, sim attack, sim-honest b1/b2): file path, sha256, npz fields, the layer values and the fold -> layer
map, every manifest entry that names the file or its sha, and every manifest leaf under a 'weights' key. Then three fingerprints, same design vs cross design:
  1. per-task Spearman of the sim-attack run score (max over units, the record's run score) with run 1 and run 2;
  2. exact score coincidences: units of one world whose score equals (|diff| <= 1e-6) a unit score of the same task in another world;
  3. per-fold raw-score moments of the sim-honest pool next to the real honest B pool (each probe has its own score scale).
Also: the attack tasks with a sim-honest run that are not in M (0d's 143 vs 142). Modal CPU, seconds.

  PYTHONPATH=src <modal 1.6.1> run experiments/2026-10-15_e2e_sim_pipeline/check_dt_swap.py
"""

import hashlib
import io
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-e2esim_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scipy==1.15.3")
V = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14")


@app.function(image=image, cpu=4, memory=16384, timeout=900, retries=0, volumes={"/convaudit": V.read_only()})
def check(cfg_rows: list, coh: list, univ: list) -> dict:  # noqa: C901
    import numpy as np
    from scipy.stats import spearmanr

    def walk(o, path=()):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from walk(v, path + (k,))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                yield from walk(v, path + (i,))
        else:
            yield path, o

    def mentions(man, name, sha):
        """Every manifest leaf whose path or value names the file (basename) or holds its sha, with the parent object."""
        hits = []
        for path, v in walk(man):
            sv = str(v)
            if sha in sv or any(name == str(p_) or name in str(p_) for p_ in path) or (isinstance(v, str) and v.endswith(name)):
                par = man
                for p_ in path[:-1]:
                    par = par[p_]
                hits.append({"path": list(map(str, path)), "value": v if len(sv) < 300 else sv[:300], "parent": json.loads(json.dumps(par, default=str))
                             if len(json.dumps(par, default=str)) < 3000 else "large"})
        return hits

    task = lambda k: k.split("|", 1)[1] if "|" in k else k.split("/", 1)[1]  # noqa: E731
    out = {"designs": {}, "arms": [], "manifests": {}}
    data = {}
    for p in cfg_rows:
        d = p["label"].removesuffix(" C")
        data[d] = {}
        for w, spec in list(p["world_files"].items()) + [(f"simh_{bt['batch']}", {"volume": "conv", "path": bt["path"], "manifest": bt["manifest"]}) for bt in p["simh"]["batches"]]:
            fp = Path("/convaudit") / spec["path"].lstrip("/")
            b = fp.read_bytes()
            sha = hashlib.sha256(b).hexdigest()
            if "sha256" in spec:
                assert sha == spec["sha256"], (d, w)
            z = np.load(io.BytesIO(b), allow_pickle=False)
            if "manifest" in spec:
                mps = [Path("/convaudit") / spec["manifest"]["path"].lstrip("/")]
            else:  # the scale-up manifests: every manifest*.json in the file's folder and up to three folders above
                mps = sorted({m_ for up in (fp.parent, fp.parent.parent, fp.parent.parent.parent, fp.parent.parent.parent.parent) for m_ in up.glob("manifest*.json")})
            ment = {}
            for mp_ in mps:
                mb = mp_.read_bytes()
                man = json.loads(mb)
                out["manifests"][str(mp_)] = {"sha256": hashlib.sha256(mb).hexdigest(), "bytes": len(mb), "top_keys": list(man)[:40] if isinstance(man, dict) else None,
                                              "weights_leaves": {"/".join(map(str, pa)): v for pa, v in walk(man) if any("weight" in str(q).lower() for q in pa) and len(str(v)) < 200}}
                ment[str(mp_)] = mentions(man, Path(spec["path"]).name, sha)
            arm = {"design": d, "arm": w, "path": spec["path"], "sha256": sha, "fields": list(z.files), "rows": int(len(z["score"])),
                   "layer_values": sorted(set(np.asarray(z["layer"]).ravel().tolist())) if "layer" in z.files else None,
                   "fold_to_layer": {str(g): sorted(set(np.asarray(z["layer"])[z["fold"] == g].tolist())) for g in sorted(set(z["fold"].tolist()))} if "layer" in z.files else None,
                   "string_fields": {f: sorted(set(np.asarray(z[f]).astype(str).ravel().tolist()))[:5] for f in z.files if z[f].dtype.kind in "US" and f not in ("key", "irid")},
                   "manifests_searched": [str(m_) for m_ in mps], "manifest_mentions": ment}
            out["arms"].append(arm)
            keys = [str(k) for k in z["key"]]
            data[d][w] = {"task": np.array([task(k) for k in keys]), "score": z["score"].astype(float), "fold": z["fold"].astype(int)}
    D = list(data)
    # 1. per-task Spearman of run scores (max over units)
    rmax = {d: {w: {} for w in data[d]} for d in D}
    for d in D:
        for w, x in data[d].items():
            for t_, s_ in zip(x["task"].tolist(), x["score"].tolist()):
                rmax[d][w][t_] = max(rmax[d][w].get(t_, -np.inf), s_)
    sp = []
    for d1 in D:
        for d2 in D:
            for w2 in ("run1", "run2"):
                ts = sorted(set(coh) & set(rmax[d1]["sim"]) & set(rmax[d2][w2]))
                r_, pv = spearmanr([rmax[d1]["sim"][t_] for t_ in ts], [rmax[d2][w2][t_] for t_ in ts])
                sp.append({"sim_attack_of": d1, "real_of": d2, "real_world": w2, "same_design": d1 == d2, "n_tasks": len(ts), "spearman": float(r_), "p": float(pv)})
    out["spearman_sim_vs_real_run_scores"] = sp
    # 2. exact score coincidences within a task, across worlds and designs
    co = []
    pairs = [("sim", "run1"), ("sim", "run2"), ("simh_b1", "honest_b"), ("simh_b1", "honest_c"), ("run1", "run2"), ("honest_b", "honest_c")]
    for d1 in D:
        for d2 in D:
            for w1, w2 in pairs:
                a, b_ = data[d1][w1], data[d2][w2]
                by_t = {}
                for t_, s_ in zip(b_["task"].tolist(), b_["score"].tolist()):
                    by_t.setdefault(t_, []).append(s_)
                by_t = {t_: np.sort(np.array(v)) for t_, v in by_t.items()}
                n_eq, n_cmp, vals, tasks_eq = 0, 0, set(), set()
                for t_, s_ in zip(a["task"].tolist(), a["score"].tolist()):
                    v = by_t.get(t_)
                    if v is None:
                        continue
                    n_cmp += 1
                    i_ = np.searchsorted(v, s_)
                    hit = min(abs(v[max(i_ - 1, 0)] - s_), abs(v[min(i_, len(v) - 1)] - s_)) <= 1e-6
                    n_eq += bool(hit)
                    if hit:  # Codex: count distinct matching values and tasks too, so a repeated constant cannot carry the count
                        vals.add(round(s_, 5))
                        tasks_eq.add(t_)
                co.append({"a_design": d1, "a_world": w1, "b_design": d2, "b_world": w2, "same_design": d1 == d2, "units_compared": n_cmp, "units_with_equal_score": n_eq,
                           "distinct_equal_values": len(vals), "tasks_with_an_equal_score": len(tasks_eq)})
    out["exact_score_coincidences"] = co
    # 3. per-fold raw-score moments: sim-honest (b1+b2) vs real honest B, every design
    mom = []
    for d in D:
        sh = {k: np.concatenate([data[d][w][k] for w in data[d] if w.startswith("simh_")]) for k in ("score", "fold")}
        hb = data[d]["honest_b"]
        mom.append({"design": d, "simh": [[float(sh["score"][sh["fold"] == g].mean()), float(sh["score"][sh["fold"] == g].std())] for g in range(5)],
                    "honest_b": [[float(hb["score"][hb["fold"] == g].mean()), float(hb["score"][hb["fold"] == g].std())] for g in range(5)],
                    "sim_attack": [[float(data[d]["sim"]["score"][data[d]["sim"]["fold"] == g].mean())] for g in range(5)],
                    "run1": [[float(data[d]["run1"]["score"][data[d]["run1"]["fold"] == g].mean())] for g in range(5)]})
    out["fold_moments_raw"] = mom
    # 0d (c): attack tasks with a sim-honest run, not in M (M = sim-honest tasks with an honest B and an honest C run, in the universe)
    u = set(univ)
    for d in D:
        st = set(data[d]["simh_b1"]["task"].tolist()) | set(data[d]["simh_b2"]["task"].tolist())
        hb, hc = set(data[d]["honest_b"]["task"].tolist()) & u, set(data[d]["honest_c"]["task"].tolist()) & u
        M = st & hb & hc
        out["designs"][d] = {"simh_tasks": len(st), "M": len(M), "simh_in_cohort": len(st & set(coh)), "M_in_cohort": len(M & set(coh)),
                             "simh_in_cohort_not_in_M": [{"task": t_, "honest_b": t_ in hb, "honest_c": t_ in hc, "in_universe": t_ in u} for t_ in sorted((st & set(coh)) - M)],
                             "simh_not_in_M": [{"task": t_, "honest_b": t_ in hb, "honest_c": t_ in hc, "in_universe": t_ in u} for t_ in sorted(st - M)]}
    return out


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    cfg = json.loads((HERE / "config" / "inputs_e2e.json").read_text())
    rows = [r for r in cfg["rows"] if r.get("source") == "dt"]
    assert len(rows) == 4
    loc = cfg["local"]
    rec_b, v6 = (HERE / loc["rec_b"]["path"]).read_bytes(), git("show", loc["v6"]["git"])
    for b, k in ((rec_b, "rec_b"), (v6, "v6")):
        assert hashlib.sha256(b).hexdigest() == loc[k]["sha256"], k
    coh = json.loads(rec_b)["probes"]["sol_a|avg_top2|corrected|none|tight"]["cohort"]
    univ = json.loads(v6)["result"]["universe_and_split"]["universe"]
    out = check.remote(rows, coh, univ)
    stamp = subprocess.run(["date", "+%Y-%m-%d_%H%M%S"], capture_output=True, check=True, text=True, env={"TZ": "America/Los_Angeles"}).stdout.strip()
    out |= {"label": "DT primary/secondary swap check (read-only, raw files); POST HOC", "git": git("rev-parse", "HEAD").decode().strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "attack_cohort": len(coh)}
    dest = HERE / "results" / f"check_dt_swap_{stamp}.json"
    assert not dest.exists()
    dest.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(dest)
