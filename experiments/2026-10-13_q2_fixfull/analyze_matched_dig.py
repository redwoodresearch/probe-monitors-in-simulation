"""Fix-full (PLAN.md Amendment 7), POST HOC, preliminary until Codex: why the matched-false-alarm intervals are wide and why
little of the variant's gain survives at 1%. Reads one `--dump` of analyze_matched_fa_fx.py (Qwen3-32B) or
analyze_matched_fa_seed.py (Seed-OSS-36B): attack rows S_<n> [cells, 5] at fold F_<n>; honest worlds real / orig (as
simulated) / edited (the variant) / frozen as nested [f, v, run] with fold, task and key. Local, $0.
  1. counts per model (f, v), rate and world: calibration runs n, the rule's index k = floor(alpha n), runs above it;
  2. movers at 1%: the variant's honest runs above the as-is threshold whose as-is score was not; each one's effect when its
     scores are reverted to as is; the matched shift with the top 1, top 3 and all movers reverted;
  3a. pooled calibration (per-model standardization by the frozen world's calibration mean and SD, one threshold per world);
  3b. partial AUROC over false alarms 0-5%, per test fold, averaged;
  4. the width split: tasks only, thresholds only, both (both must equal the published intervals: a gate).
Draws: the published 2,000 stratified task draws (seed 20261015).

  uv run --no-project --python 3.12 --with numpy python experiments/2026-10-13_q2_fixfull/analyze_matched_dig.py \\
      --dump <npz> --published <matched_fa_*_POSTHOC_*.json>
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
RATES, B, SEED = ("0.01", "0.05"), 2000, 20261015
WORLDS = ("real", "orig", "edited", "frozen")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def rule(s, w, alpha):
    """the published weighted rule: the (k+1)-th largest positive-weight score, k = floor(alpha x weight)."""
    m = w > 0
    s, w = s[m], w[m]
    o = np.argsort(-s, kind="mergesort")
    c = np.cumsum(w[o])
    return float(s[o][min(int((c <= np.floor(alpha * c[-1] + 1e-9)).sum()), len(s) - 1)])


def catch(S, F, T):
    valid = np.arange(5)[None, :] != F[:, None]
    return np.where(valid, S > T[F], False).sum(1) / valid.sum(1)


def cal(ne, fo, ta, allow_nan):
    out = {}
    for f in range(5):
        for v in range(5):
            if v != f:
                ii = np.flatnonzero(fo == v)
                ok = np.isfinite(ne[f, v, ii])
                if not allow_nan and not ok.all():
                    raise SystemExit("a non-finite calibration score")
                out[f, v] = (ne[f, v, ii[ok]], ta[ii[ok]], ii[ok])
    return out


def thr(H, alpha, tw):
    T = np.full((5, 5), np.nan)
    for fv, (s, t, _) in H.items():
        T[fv] = rule(s, np.array([tw.get(x, 0.0) for x in t], float), alpha)
    return T


def pauc(pos, wp, neg, wn, fmax=0.05):
    """mean true-positive rate over false alarms [0, fmax] (step ROC, weighted)."""
    o = np.argsort(-neg, kind="mergesort"); n_, w_ = neg[o], wn[o]
    Fc = np.concatenate([[0.0], np.cumsum(w_) / w_.sum()])
    tot, Wp = 0.0, wp.sum()
    for i in range(len(n_)):
        lo, hi = Fc[i], min(Fc[i + 1], fmax)
        if lo >= fmax:
            break
        if hi > lo:
            tot += (wp[pos > n_[i]].sum() / Wp) * (hi - lo)
    return tot / fmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, type=Path); ap.add_argument("--published", required=True, type=Path)
    a = ap.parse_args()
    if subprocess.run(["git", "ls-files", "--error-unmatch", __file__], cwd=HERE, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", __file__], cwd=HERE).returncode:
        raise SystemExit("commit this script before running")
    D = np.load(a.dump)
    pub = json.loads(a.published.read_text())
    if str(D["variant"]) != pub["variant"] or not json.loads(str(D["gates"]))["all"]:
        raise SystemExit("the dump and the published result differ in variant, or the dump's gates failed")
    S = {n: D[f"S_{n}"] for n in ("sim", "fix", "e1", "e2")}; F = {n: D[f"F_{n}"].astype(int) for n in S}
    atask = D["atask"]
    W = {w: (D[f"{w}_ne"], D[f"{w}_fold"].astype(int), D[f"{w}_task"]) for w in WORLDS}
    H = {w: cal(*W[w], allow_nan=(w == "frozen")) for w in WORLDS}
    hrow = {w: (np.stack([W[w][0][fo, :, j] for j, fo in enumerate(W[w][1])]), W[w][1], W[w][2]) for w in ("real", "orig", "edited")}
    # the published draws
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
    ones = draws[0]
    out = {"POST HOC": True, "status": "preliminary until Codex", "plan": "PLAN.md Amendment 7", "reader": str(D["reader"]), "variant": pub["variant"],
           "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip(),
           "inputs sha256": {a.dump.name: sha(a.dump), a.published.name: sha(a.published)}, "gates": {}, "rates": {}}
    hk = {w: [str(k) for k in D[f"{w}_key"]] for w in ("orig", "edited")}
    if sorted(hk["orig"]) != sorted(hk["edited"]):
        raise SystemExit("orig and edited honest keys differ")
    oi = {k: j for j, k in enumerate(hk["orig"])}
    for al in RATES:
        alf = float(al); r = {}
        T0 = {w: thr(H[w], alf, ones) for w in WORLDS}
        Tb = [{w: (T0[w] if b == 0 else thr(H[w], alf, draws[b])) for w in ("real", "orig", "edited", "frozen")} for b in range(B + 1)]
        C = {(n, w): np.array([catch(S[n], F[n], Tb[b][w]) for b in range(B + 1)]) for n, w in
             {("fix", "edited"), ("sim", "orig"), ("e2", "real"), ("fix", "frozen"), ("sim", "frozen")}}
        m = {k: (aw * v).sum(1) / aw.sum(1) for k, v in C.items()}
        main_ = m[("fix", "edited")] - m[("e2", "real")]; paired = m[("fix", "edited")] - m[("sim", "orig")]
        fshift = m[("fix", "frozen")] - m[("sim", "frozen")]
        P = pub["results"][al]
        gate = (abs(main_[0] - P["edited_minus_real_e2 (main)"]["value"]) < 1e-12 and ci(main_[1:]) == P["edited_minus_real_e2 (main)"]["ci95"]
                and ci(paired[1:]) == P["edited_minus_orig (paired)"]["ci95"])
        out["gates"][f"published main and paired intervals reproduced at {al}"] = bool(gate)
        if not gate:
            raise SystemExit(f"the published intervals are not reproduced at {al}")
        # 1. counts
        cnt = {}
        for w in WORLDS:
            rows = []
            for (f, v), (s, t, _) in sorted(H[w].items()):
                n = len(s); k = int(np.floor(alf * n + 1e-9)); T = T0[w][f, v]
                rows.append({"model": f"{f}_{v}", "n": n, "k": k, "threshold rank (1 = top)": k + 1, "above": int((s > T).sum()), "threshold is the top score": bool(T == s.max())})
            cnt[w] = {"models": rows, "n range": [min(x["n"] for x in rows), max(x["n"] for x in rows)], "k values": sorted({x["k"] for x in rows}),
                      "above values": sorted({x["above"] for x in rows}), "models whose threshold is the top score": sum(x["threshold is the top score"] for x in rows)}
        r["1. counts"] = cnt
        # 2. movers (computed at every rate; asked at 1%)
        ne_o, ne_e = W["orig"][0], W["edited"][0]
        ej = {k: j for j, k in enumerate(hk["edited"])}
        movers = {}
        for (f, v), (s, t, ii) in H["edited"].items():
            for j in ii:
                k = hk["edited"][j]
                if ne_e[f, v, j] > T0["orig"][f, v] and not ne_o[f, v, oi[k]] > T0["orig"][f, v]:
                    movers.setdefault(k, []).append(f"{f}_{v}")
        top_changed = sum(hk["edited"][H["edited"][fv][2][np.argmax(H["edited"][fv][0])]] != hk["orig"][H["orig"][fv][2][np.argmax(H["orig"][fv][0])]] for fv in H["edited"])
        base_c = float(catch(S["fix"], F["fix"], T0["edited"]).mean())

        def reverted(keys_):
            ne_r = ne_e.copy()
            for k in keys_:
                ne_r[:, :, ej[k]] = ne_o[:, :, oi[k]]
            return thr(cal(ne_r, W["edited"][1], W["edited"][2], False), alf, ones)
        eff = []
        for k, models in movers.items():
            Tr = reverted([k])
            eff.append({"key": k, "models where it is above the as-is threshold": models, "max threshold drop when reverted": float(np.nanmax(T0["edited"] - Tr)),
                        "variant matched catch change when reverted (points)": 100 * (float(catch(S["fix"], F["fix"], Tr).mean()) - base_c)})
        eff.sort(key=lambda e: -e["variant matched catch change when reverted (points)"])
        sim_o = float(catch(S["sim"], F["sim"], T0["orig"]).mean()); fz = float(fshift[0])
        rev = {}
        for lab, ks in (("none", []), ("top 1", [e["key"] for e in eff[:1]]), ("top 3", [e["key"] for e in eff[:3]]), ("all", [e["key"] for e in eff])):
            sh = float(catch(S["fix"], F["fix"], reverted(ks) if ks else T0["edited"]).mean()) - sim_o
            rev[lab] = {"runs reverted": len(ks), "matched shift, variant - as is (points)": 100 * sh, "share of the frozen shift": sh / fz}
        r["2. movers"] = {"models whose top honest run differs (variant vs as is)": int(top_changed), "of models": len(H["edited"]),
                          "movers (runs)": len(eff), "movers by effect": eff, "frozen shift (points)": 100 * fz, "matched shift with movers reverted": rev}
        # 3a. pooled calibration: per-model standardization by the frozen world's calibration mean and SD
        mu = np.full((5, 5), np.nan); sd = np.full((5, 5), np.nan)
        for (f, v), (s, _, _) in H["frozen"].items():
            mu[f, v], sd[f, v] = s.mean(), s.std(ddof=1)

        def pooled_T(w, tw):
            z = np.concatenate([(s - mu[fv]) / sd[fv] for fv, (s, _, _) in H[w].items()])
            wt = np.concatenate([np.array([tw.get(x, 0.0) for x in t]) / 4.0 for _, (_, t, _) in H[w].items()])
            tz = rule(z, wt, alf)
            return mu + tz * sd, tz
        Pb = [{w: pooled_T(w, draws[b])[0] for w in ("real", "orig", "edited")} for b in range(B + 1)]
        Cp = {(n, w): np.array([catch(S[n], F[n], Pb[b][w]) for b in range(B + 1)]) for n, w in {("fix", "edited"), ("sim", "orig"), ("e2", "real")}}
        mp = {k: (aw * v).sum(1) / aw.sum(1) for k, v in Cp.items()}
        pm, pp = mp[("fix", "edited")] - mp[("e2", "real")], mp[("fix", "edited")] - mp[("sim", "orig")]
        per_model = {}
        for w in ("real", "orig", "edited"):
            Tw = Pb[0][w]
            per_model[w] = {"pooled threshold (standardized)": pooled_T(w, ones)[1],
                            "per-model share of the world's calibration runs above (min, max)": [float(min((s > Tw[fv]).mean() for fv, (s, _, _) in H[w].items())),
                                                                                                  float(max((s > Tw[fv]).mean() for fv, (s, _, _) in H[w].items()))],
                            "achieved false alarms, out of fold (points)": 100 * float(catch(hrow[w][0], hrow[w][1], Tw).mean())}
        r["3a. pooled calibration"] = {"variant - real e2 (points)": [100 * float(pm[0]), [100 * x for x in ci(pm[1:])]],
                                        "variant - as is (points)": [100 * float(pp[0]), [100 * x for x in ci(pp[1:])]],
                                        "share of the frozen shift surviving": [float(pp[0] / fshift[0]), ci(pp[1:] / fshift[1:])],
                                        "per world": per_model,
                                        "pooled-scale caveat": "one standardized threshold for all 20 models: tails differ by model, so per-model false alarms differ (see per world)"}
        # 3b. partial AUROC 0-5%, per test fold, averaged over folds
        def run_score(Srow, Fv):
            return np.array([np.mean([Srow[i, v] for v in range(5) if v != Fv[i]]) for i in range(len(Fv))])
        pairs = {"real (e2 vs real honest)": ("e2", "real"), "as is (simulated attack vs simulated honest)": ("sim", "orig"), "variant": ("fix", "edited")}
        pa = {}
        for lab, (n, w) in pairs.items():
            ps, ns_ = run_score(S[n], F[n]), run_score(hrow[w][0], hrow[w][1])
            vals = []
            for b in range(B + 1):
                d = draws[b]; wpa = np.array([d.get(t, 0.0) for t in atask]); wna = np.array([d.get(t, 0.0) for t in hrow[w][2]])
                fv = []
                for f in range(5):
                    mp_, mn_ = (F[n] == f) & (wpa > 0), (hrow[w][1] == f) & (wna > 0)
                    fv.append(pauc(ps[mp_], wpa[mp_], ns_[mn_], wna[mn_]))
                vals.append(np.mean(fv))
            pa[lab] = np.array(vals)
        r["3b. partial AUROC 0-5% (mean true-positive rate)"] = {k: [float(v[0]), ci(v[1:])] for k, v in pa.items()} | {
            "variant - as is (paired)": [float(pa["variant"][0] - pa["as is (simulated attack vs simulated honest)"][0]),
                                         ci(pa["variant"][1:] - pa["as is (simulated attack vs simulated honest)"][1:])]}
        # 4. width split
        Tfix = {w: T0[w] for w in WORLDS}
        ct = {k: catch(S[k[0]], F[k[0]], Tfix[k[1]]) for k in C}
        mt = {k: (aw * ct[k]).sum(1) / aw.sum(1) for k in C}  # tasks only
        mc = {k: v.mean(1) for k, v in C.items()}  # thresholds only
        ws = {}
        for lab, (x, y) in (("variant - real e2 (main)", (("fix", "edited"), ("e2", "real"))), ("variant - as is (paired)", (("fix", "edited"), ("sim", "orig")))):
            sp = {"both (published)": m[x][1:] - m[y][1:], "tasks only (thresholds fixed)": mt[x][1:] - mt[y][1:], "thresholds only (tasks fixed)": mc[x][1:] - mc[y][1:]}
            ws[lab] = {k: {"95% (points)": [100 * q for q in ci(v)], "width (points)": 100 * (ci(v)[1] - ci(v)[0]), "sd (points)": 100 * float(v.std(ddof=1))} for k, v in sp.items()}
            vb = float(sp["both (published)"].var(ddof=1))
            ws[lab]["variance share, tasks only / thresholds only (of both)"] = [float(sp["tasks only (thresholds fixed)"].var(ddof=1) / vb), float(sp["thresholds only (tasks fixed)"].var(ddof=1) / vb)]
        r["4. width split"] = ws
        out["rates"][al] = r
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    tag = "seed" if "Seed" in out["reader"] else "qwen3"
    with open(dest := HERE / "results" / f"matched_dig_{tag}_{out['variant']}_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
