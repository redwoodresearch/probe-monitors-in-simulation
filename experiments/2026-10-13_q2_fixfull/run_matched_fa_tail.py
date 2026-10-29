"""Fix-full (PLAN.md Amendment 8), POST HOC sensitivity, Modal CPU: the matched-false-alarm intervals at 1% under 2e's
smoothed bootstrap ("smooth", provisional) and, as a labelled diagnostic on the separately calibrated matched gaps, the
Gaussian tail ("gauss"), beside the committed percentile intervals, for Qwen3-32B and Seed-OSS-36B, full and N.
Inputs: the gated array dumps of analyze_matched_fa_fx.py / analyze_matched_fa_seed.py (--dump; sha256 as recorded by
the Amendment 7 dig records) and the committed matched-FA results. tail_threshold.py is read from 2e's branch at TT_SHA
(git show; blob sha256 pinned) and imported in the container from that text. Same 2,000 stratified task draws (seed
20261015) as the committed runs; smoothing noise seed 20261017 (one generator per case, worlds and models in a fixed
order). Gates: the module's "order" method reproduces each committed 1% interval exactly (main, as is - real e2, paired).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run \\
      experiments/2026-10-13_q2_fixfull/run_matched_fa_tail.py --dump-dir <dir holding qwen3_full.npz ...>
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q2-fixfull_mfatail_kill-2026-11-13"
TT_SHA, TT_PATH, TT_BLOB = "7a6f5cd8", "src/probe_shift/tail_threshold.py", "c87bd02971ef6b06678a6f45cb62be20d05db938691f1015c8ea79dc40bd53dc"
CASES = {  # dump -> (its sha256 as recorded in the dig record, the committed matched-FA result)
    "qwen3_full": ("a095442e7b758a703f254cbb45a08a7aca172f1dddd2aea7377bce3fc2ade077", "matched_fa_fx_full_POSTHOC_2026-10-13_160201.json"),
    "qwen3_N": ("8aa551549085fb994c55d213612cf333c00f2c21d23044183b5b69a5896a805f", "matched_fa_fx_N_POSTHOC_2026-10-13_160216.json"),
    "seed_full": ("ba8efd01c8256a175f189bbb6d0a59da651eb30c21c6ac7e71492527c4a7b4d1", "matched_fa_seed_full_POSTHOC_2026-10-13_163806.json"),
    "seed_N": ("c99c1cc5f3790475c29fd5e22db6060c9a7accc0fd0be52152784daba01a03fc", "matched_fa_seed_N_POSTHOC_2026-10-13_163820.json")}
B, SEED, NOISE_SEED, RATE = 2000, 20261015, 20261017, 0.01
WORLDS = ("real", "orig", "edited", "frozen")
GAPS = {"edited_minus_real_e2 (main)": (("fix", "edited"), ("e2", "real")), "orig_minus_real_e2": (("sim", "orig"), ("e2", "real")),
        "edited_minus_orig (paired)": (("fix", "edited"), ("sim", "orig"))}
LABEL_SMOOTH = "provisional: the best tested on matched gaps (about 88-90% coverage in 2e's fitted world), not nominal 95%"
LABEL_GAUSS = ("diagnostic only. Gaussian-tail coverage on matched gaps (91-92% in 2e's fitted world) rests on the simulated and real honest "
               "pools having similar tail shapes, so threshold bias partly cancels; nothing guarantees that on new data. Codex's per-world "
               "check (2e round 2): it runs hot on the simulated honest pools (held-out 1% false alarm 1.60-1.77% vs 1.09-1.11% real)")

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scipy")


@app.function(image=image, cpu=4, memory=8192, timeout=1800, retries=0)
def run(tt_src: str, dumps: dict, published: dict) -> dict:
    import io
    import types

    import numpy as np
    from scipy.stats import norm
    tt = types.ModuleType("tail_threshold"); exec(compile(tt_src, "tail_threshold.py", "exec"), tt.__dict__)  # noqa: S102

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    def catch(S, F, T):
        valid = np.arange(5)[None, :] != F[:, None]
        return np.where(valid, S > T[F], False).sum(1) / valid.sum(1)
    out = {}
    for case in sorted(dumps):
        D = np.load(io.BytesIO(dumps[case])); P = published[case]["results"]["0.01"]
        S = {n: D[f"S_{n}"] for n in ("sim", "fix", "e2")}; F = {n: D[f"F_{n}"].astype(int) for n in S}
        atask = D["atask"]
        W = {w: (D[f"{w}_ne"], D[f"{w}_fold"].astype(int), D[f"{w}_task"]) for w in WORLDS}
        A, Hs = set(atask), set(W["orig"][2])
        strata = [sorted(A & Hs), sorted(A - Hs), sorted(Hs - A)]
        g = np.random.default_rng(SEED)
        draws = [{t: 1.0 for st in strata for t in st}]
        for _ in range(B):
            cnt = {}
            for st in strata:
                if not st:
                    continue
                for k in g.integers(0, len(st), len(st)):
                    cnt[st[k]] = cnt.get(st[k], 0.0) + 1.0
            draws.append(cnt)
        aw = np.array([[d.get(t, 0.0) for t in atask] for d in draws])
        noise = np.random.default_rng(NOISE_SEED)
        T = {m: {w: np.full((B + 1, 5, 5), np.nan) for w in WORLDS} for m in ("order", "smooth", "gauss")}
        tail = {w: {"held-out 1% false alarms (share)": [], "upper-quartile QQ correlation": []} for w in ("real", "orig", "edited")}
        for w in WORLDS:
            ne, fo, ta = W[w]
            for f in range(5):
                for v in range(5):
                    if v == f:
                        continue
                    ii = np.flatnonzero((fo == v) & np.isfinite(ne[f, v]))
                    s = ne[f, v, ii]; wts = np.array([[d.get(t, 0.0) for t in ta[ii]] for d in draws])
                    for m in ("order", "smooth", "gauss"):
                        T[m][w][:, f, v] = tt.threshold(s, wts, RATE, m, None, noise if m == "smooth" else None)
                    if w != "frozen":
                        te = (fo == f) & np.isfinite(ne[f, v])
                        tail[w]["held-out 1% false alarms (share)"].append(float((ne[f, v, te] > T["gauss"][w][0, f, v]).mean()))
                        up = np.sort(s)[int(0.75 * len(s)):]
                        tail[w]["upper-quartile QQ correlation"].append(float(np.corrcoef(up, norm.ppf((np.arange(int(0.75 * len(s)), len(s)) + 0.5) / len(s)))[0, 1]))
        m_ = {}
        for meth in ("order", "smooth", "gauss"):
            for n, w in {("fix", "edited"), ("sim", "orig"), ("e2", "real"), ("fix", "frozen"), ("sim", "frozen")}:
                C = np.array([catch(S[n], F[n], T[meth][w][b]) for b in range(B + 1)])
                m_[(meth, n, w)] = (aw * C).sum(1) / aw.sum(1)
        res, gate = {}, {}
        for name, (x, y) in GAPS.items():
            d = {meth: m_[(meth, *x)] - m_[(meth, *y)] for meth in ("order", "smooth", "gauss")}
            gate[name] = bool(abs(d["order"][0] - P[name]["value"]) < 1e-12 and ci(d["order"][1:]) == P[name]["ci95"])
            res[name] = {"committed (percentile, redrawn thresholds)": {"value": P[name]["value"], "ci95": P[name]["ci95"]},
                         "smooth": {"value": float(d["smooth"][0]), "ci95": ci(d["smooth"][1:])}, "gauss (diagnostic)": {"value": float(d["gauss"][0]), "ci95": ci(d["gauss"][1:])}}
        for meth in ("order", "smooth"):
            sm = m_[(meth, "fix", "edited")] - m_[(meth, "sim", "orig")]; sf = m_[(meth, "fix", "frozen")] - m_[(meth, "sim", "frozen")]
            vals = {"frozen_shift_minus_matched_shift": sf - sm, "share_of_frozen_shift_surviving": sm / sf}
            for name, v in vals.items():
                if meth == "order":
                    gate[name] = bool(abs(v[0] - P[name]["value"]) < 1e-12 and ci(v[1:]) == P[name]["ci95"])
                    res[name] = {"committed (percentile, redrawn thresholds)": {"value": P[name]["value"], "ci95": P[name]["ci95"]}}
                else:
                    res[name]["smooth"] = {"value": float(v[0]), "ci95": ci(v[1:])}
        changes = []
        for name, r in res.items():
            excl = lambda c: c[0] > 0 or c[1] < 0  # noqa: E731
            a, b_ = excl(r["committed (percentile, redrawn thresholds)"]["ci95"]), excl(r["smooth"]["ci95"])
            r["excludes 0: committed / smooth"] = [bool(a), bool(b_)]
            if a != b_:
                changes.append(name)
        out[case] = {"gates (module 'order' reproduces the committed 1% interval exactly)": gate, "rows at 1%": res, "reading changes (smooth vs committed)": changes,
                     "gauss tail check per world (1%)": {w: {k: {"mean": float(np.mean(v)), "range": [float(min(v)), float(max(v))], "models": len(v)} for k, v in t.items()} for w, t in tail.items()},
                     "smooth bandwidth": "Silverman of each model's calibration scores (tail_threshold default)"}
    return out


@app.local_entrypoint()
def main(dump_dir: str):
    for f in (Path(__file__), HERE / "PLAN.md"):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
                or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    tt_src = subprocess.run(["git", "show", f"{TT_SHA}:{TT_PATH}"], cwd=HERE, capture_output=True, check=True).stdout
    if hashlib.sha256(tt_src).hexdigest() != TT_BLOB:
        raise SystemExit("tail_threshold.py at the pinned commit differs from its recorded blob sha256")
    dumps, published, ins = {}, {}, {}
    for case, (dsha, pub) in CASES.items():
        b = (Path(dump_dir) / f"{case}.npz").read_bytes()
        if hashlib.sha256(b).hexdigest() != dsha:
            raise SystemExit(f"{case}.npz differs from the dump the dig recorded")
        blob = subprocess.run(["git", "show", f"HEAD:experiments/2026-10-13_q2_fixfull/results/{pub}"], cwd=HERE, capture_output=True, check=True).stdout
        dumps[case], published[case] = b, json.loads(blob)
        ins[case] = {f"{case}.npz": dsha, pub: hashlib.sha256(blob).hexdigest()}
    out = run.remote(tt_src.decode(), dumps, published)
    rec = {"POST HOC": "PLAN.md Amendment 8 (sensitivity; the committed percentile intervals stay the headline)",
           "status": "preliminary until Codex", "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip(),
           "tail_threshold.py": {"branch": "aswin/robust-1pct-posthoc", "commit": TT_SHA, "blob sha256": TT_BLOB}, "rate": RATE, "draws": B,
           "seeds": {"task draws": SEED, "smoothing noise": NOISE_SEED}, "labels": {"smooth": LABEL_SMOOTH, "gauss": LABEL_GAUSS},
           "wording": "worlds are separately calibrated (each on its own honest runs), not independent", "inputs sha256": ins, "cases": out}
    rec["gates all"] = all(all(v["gates (module 'order' reproduces the committed 1% interval exactly)"].values()) for v in out.values())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"matched_fa_tail_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(dest, "gates all:", rec["gates all"])
    if not rec["gates all"]:
        raise SystemExit(1)
