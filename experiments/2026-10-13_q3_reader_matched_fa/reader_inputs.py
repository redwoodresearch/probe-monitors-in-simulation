"""POST HOC (RULE.md, this folder): one reader's composite detector, as arrays d9's Qwen3-32B analysis code takes. $0, local,
stored scores only; keys and numbers only.

composite[f, v, i] = run i's score under nested model (f, v) at layer[f] (both nested roles); outer[i] = run i's outer
score at layer[fold[i]]. layer[f] is the reader's committed rule, recomputed here over its seven-layer grid
(analyze_reply_break_olmo.py: the layer with the best mean validation AUROC, honest against e1, over the nested models
(f, v), each scored on its validation fold v).

load(reader, paths) returns:
  base   : the fit's 3,807 runs (runs.json order) with composite nested/outer;
  honest : a runs list in sim_honest_scores' shape (arm, key, main_task, fold, in_base) for the fit's 3,807, the 10 real
           honest runs outside the fit and the 437 simulated honest runs, with composite arrays, plus the edited rows'
           index (key -> row), for analyze_edit_honest.assemble();
  fix    : the edited simulated attack arm, keyed by base run index (fix_scores' run_index and has_fix, never positions).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

READERS = {
    "olmo": {"layers": (16, 24, 32, 40, 48, 56, 63), "committed": {0: 32, 1: 56, 2: 56, 3: 56, 4: 40}, "saved": (32, 40, 56),
             "runs_sha": "e2c86cf48983e6c93bb4ade799e47e08b2aaca0201f4a7244a5642b5de3c8784",
             "fix_sha": "7e17ba2c91be777ddfae308a4ce0a8c82480ccb2d47609d99b0578b5887a5787",
             "reading_sha": "7357056456ff503e94318d245c692c9c5365b2ae573d24a70d0f34192e6f74ca",
             "edit_effect_sha": "e620a7c45c247e1848780e14484cde72f23c8c75375fe3f29b93609eed873a12",
             "scores_sha": {16: "b73157b4fbb46432e8aa6823a4e14270d2656456321d8b89d946e597a3044e9a",
                            24: "de23c1c3f12ffa79d67d287c713345703ea45f53b2f6ca4e48e4d02f74d863ea",
                            32: "231ab6745f682def4ca8c61af2fbec80f76ab362f9df218705a7bb19e8a82e21",
                            40: "c21042eb51468460d43fef97e085c73fb6090eff35e71137672ce03f5fbb5182",
                            48: "0fab6ab5a92e184e19836908de5e9d3a6863b195282cb8c43434a1f2034c6055",
                            56: "698c9ad803212403c80b880cc5bb7ffa9ebbc5211e330d679cc85c15c5ffb20a",
                            63: "b8bac05e45d108c9da92297bcf3765e306b195fd0344c2ff793a8ef51cd9b9e0"}},
    "llama": {"layers": (20, 30, 40, 50, 60, 70, 79), "committed": {0: 60, 1: 70, 2: 60, 3: 60, 4: 70}, "saved": (60, 70),
              "runs_sha": "ad10475afccc285bde39a389a17cb8b278a15d670c5d673933a094c6ffbc5254",
              "fix_sha": "3f0018223114a1f17d9a0e8293f5e1d7a45a727f49fe6da9c147727abd048fd9",
              "reading_sha": "de6524715c9ce505b7d3e541634a908d431b49f22ae0c0ce9a8ea67d4d5caaf8",
              "edit_effect_sha": "ae24961a72693b47d03f7e50505f6b306c3018459e630a64222500358968203d",
              "scores_sha": {20: "a2855046fb03d643e66b339d0d3b4655af819bf29c5e68b7217b38f3acab4fb0",
                             30: "70e68ec851edad4184da44974d217804a0a7a5e043e6898b4cd03c0021851353",
                             40: "6fa1b3bf568b92bc7bf755580fd4640f1c05558188fe63ef552f1f03b1b162b4",
                             50: "1accda0f05c3b6495422d5d351a42db3899d83d7603d329cf3f60bb75b2b44c4",
                             60: "6d21272535104de7cf6aa179cde98070dea0cbc823fc0aee9639763c8938906b",
                             70: "f58f71edc48333cc362d3814fb83880b1abab74d3270b5a358ce3385f8b16c0c",
                             79: "ed03295de16f97290de6fc4825fd4cacbdff9dc37463c09a86664109dde28676"}},
}
SIM_RUNS_SHA = "f420ed2401dcc776cbbcfd414bbe9d6d1132a3bbd012b529a33aba7fa5ecd068"
NEST, OUT = "output_end__mean__task_nested", "output_end__mean__task_outer"
SIM_LABEL, ED_LABEL, EXTRA_LABEL = "sim-honest", "sim-honest-edited", "real-honest-extra"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frozen(*paths):
    """RULE.md: every analysis file, its dependencies and the power result are tracked and identical to HEAD."""
    import subprocess
    for p in paths:
        p = Path(p).resolve()
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=p.parent, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(p)], cwd=p.parent).returncode:
            raise SystemExit(f"{p.name} is not committed as is: commit it before running")


def auc(pos, neg):  # analyze_reply_break_olmo.auc, unchanged
    s = np.r_[pos, neg]; r = s.argsort(kind="mergesort").argsort(kind="mergesort").astype(float)
    u, inv = np.unique(s, return_inverse=True); rr = np.zeros(len(u)); np.add.at(rr, inv, r); cnt = np.bincount(inv); r = (rr / cnt)[inv]
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def layer_rule(N, arm, fold, layers):  # analyze_reply_break_olmo.analyze() step 1, unchanged
    hon, e1 = arm == "real-honest", arm == "real-attack-e1"
    layer = {}
    for f in range(5):
        fin = lambda x: x[np.isfinite(x)]  # noqa: E731
        layer[f] = max(layers, key=lambda l: np.mean([auc(fin(N[l][f, v][e1 & (fold == v)]), fin(N[l][f, v][hon & (fold == v)])) for v in range(5) if v != f]))
    return layer


def compose(nested_by_layer, outer_by_layer, fold, layer):
    """nested_by_layer[l]: [5, 5, n]; outer_by_layer[l]: [n]. composite[f, v, i] = nested[layer[f]][f, v, i];
    outer[i] = outer[layer[fold[i]]][i]."""
    n = len(fold)
    C = np.full((5, 5, n), np.nan)
    for f in range(5):
        C[f] = nested_by_layer[layer[f]][f]
    O = np.array([outer_by_layer[layer[int(fold[i])]][i] for i in range(n)], float)
    return C, O


def load(reader, fit_dir, sim_runs, simhonest_scores, fix_scores, stage_changed, edited=True):
    """edited=False (power_reader.py): the edited honest arrays are never read; the edited rows are empty."""
    R = READERS[reader]
    fit = Path(fit_dir)
    if sha(fit / "data" / "runs.json") != R["runs_sha"] or sha(sim_runs) != SIM_RUNS_SHA or sha(fix_scores) != R["fix_sha"] \
            or any(sha(fit / f"scores_L{l}.npz") != h for l, h in R["scores_sha"].items()):
        raise SystemExit("a reader input's sha256 differs from its pin")
    runs = json.loads((fit / "data" / "runs.json").read_text())
    arm, fold = np.array([r["arm"] for r in runs]), np.array([int(r["fold"]) for r in runs])
    N = {l: np.load(fit / f"scores_L{l}.npz")[NEST] for l in R["layers"]}
    Ob = {l: np.load(fit / f"scores_L{l}.npz")[OUT] for l in R["layers"]}
    layer = layer_rule(N, arm, fold, R["layers"])
    base_C, base_O = compose(N, Ob, fold, layer)
    # the honest runs read by this experiment
    z = np.load(simhonest_scores)
    srs = json.loads(Path(sim_runs).read_text())
    sim = [r for r in srs if r["arm"] == "sim-honest"]
    extra = [r for r in srs if r["arm"] == "real-honest" and not r["in_base"]]
    if [str(k) for k in z[f"{SIM_LABEL}__keys"]] != [r["key"] for r in sim] or [str(k) for k in z[f"{EXTRA_LABEL}__keys"]] != [r["key"] for r in extra]:
        raise SystemExit("the scores file's keys are not the runs files' keys in order")
    sv = R["saved"]
    parts = {}
    for lab, rs in ((SIM_LABEL, sim), (ED_LABEL, sim), (EXTRA_LABEL, extra)) if edited else ((SIM_LABEL, sim), (EXTRA_LABEL, extra)):
        fo = np.array([int(r["fold"]) for r in rs])
        parts[lab] = compose({l: z[f"L{l}__{lab}__nested"] for l in sv}, {l: z[f"L{l}__{lab}__outer"] for l in sv}, fo, layer)
    if not edited:
        parts[ED_LABEL] = (np.full((5, 5, len(sim)), np.nan), np.full(len(sim), np.nan))
    edited_keys = [r["key"] for j, r in enumerate(sim) if np.isfinite(parts[ED_LABEL][1][j])]
    if edited and sorted(edited_keys) != sorted(stage_changed):
        raise SystemExit("the edited honest scores are not exactly the stage's changed runs")
    # a runs list in sim_honest_scores' shape: the fit's 3,807 (in_base), the 10 extra real honest runs, the 437 simulated honest
    hruns = ([{"arm": r["arm"], "key": r["key"], "main_task": r["main_task"], "fold": int(r["fold"]), "in_base": True} for r in runs]
             + [{"arm": "real-honest", "key": r["key"], "main_task": r["main_task"], "fold": int(r["fold"]), "in_base": False} for r in extra]
             + [{"arm": "sim-honest", "key": r["key"], "main_task": r["main_task"], "fold": int(r["fold"]), "in_base": False} for r in sim])
    jed = [j for j, r in enumerate(sim) if r["key"] in set(edited_keys)]
    outer = np.concatenate([base_O, parts[EXTRA_LABEL][1], parts[SIM_LABEL][1], parts[ED_LABEL][1][jed]])
    nested = np.concatenate([base_C, parts[EXTRA_LABEL][0], parts[SIM_LABEL][0], parts[ED_LABEL][0][:, :, jed]], axis=2)
    n0 = len(hruns)
    edited_rows = {sim[j]["key"]: n0 + k for k, j in enumerate(jed)}
    # the edited simulated attack arm, keyed by base run index
    fz = np.load(fix_scores)
    ri, has = fz["run_index"].astype(int), fz["has_fix"].astype(bool)
    if not (arm[ri] == "sim-attack").all() or len(set(ri.tolist())) != len(ri):
        raise SystemExit("fix_scores' run_index does not map to distinct simulated attack runs")
    fixC = {}
    for j in np.flatnonzero(has):
        i = int(ri[j]); f = int(fold[i])
        fixC[i] = np.array([fz[f"L{layer[f]}"][f, v, j] for v in range(5)])
    return {"layer": layer, "runs": runs, "arm": arm, "fold": fold, "base_C": base_C, "base_O": base_O,
            "hruns": hruns, "h_outer": outer, "h_nested": nested, "edited_rows": edited_rows, "fix": fixC,
            "n_edited": len(edited_rows), "n_fix": len(fixC)}
