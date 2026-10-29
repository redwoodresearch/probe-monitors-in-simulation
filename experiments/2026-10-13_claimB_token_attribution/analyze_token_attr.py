"""Token attribution, the analysis (RULE.md in this folder), written and committed before any score is read.

Reads scores.npz from the Volume (per-token scores and token ids only) and data/staged.jsonl (local, honest). In order:
1. Checks: the job's done.json and repro.json pass; every item's token ids equal the local rendering; its window equals
   marks.layout's.
2. The gate (gutter plants, plant-2 direction of the other half): per reply the best planted token's rank fraction
   (rank - 1) / n by push within the window; G = np.median over the 30 replies; PASS iff G <= 0.01 and each of the 5
   random plant-2 directions' G > 0.01. Beside it, descriptive: gutter-line tokens alone, the final-newline token alone,
   every planted token's rank fraction, the main direction's G; the review note (planted minus twin at the training
   position, the window's last token and each closing template token, aligned from the end).
3. Only if the gate passes: the main direction on the 91 site windows (per token string and arm: n, sites, tasks, T, m;
   random T), the top 20 by T in fake replies (all tokens, and known-slip tokens excluded), candidate tells, the
   second-real control, the new/natural split, the reply-end AUROC at output_end, n-grams (S3 only), the whole runs.
Writes results/token_attr_<stamp>.json (single token strings, counts and scores only) and, if the gate passes,
data/token_attr_context_<stamp>.json (context windows, 5 seeded examples, n-gram tables) for S3 behind a manifest.

  uv run --no-project --python 3.12 --with transformers==4.56.2 --with jinja2 --with boto3 --with numpy --with scipy \\
      --with 'modal==1.6.1' python experiments/2026-10-13_claimB_token_attribution/analyze_token_attr.py
"""

import hashlib
import io
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "experiments/2026-10-09_fixed_sim")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import marks  # noqa: E402

from probe_shift.models import QWEN3_32B  # noqa: E402

RUN_SHA12 = "145c32c05f05"
OUT = f"claimB/token_attr_v1/{RUN_SHA12}"
N_RANDOM, TOP, BAR, CTX = 5, 20, 0.01, 10
EXAMPLES_SEED = "token-attr-2026-10-13:examples"
_tok = None


def _init():
    global _tok
    from transformers import AutoTokenizer

    _tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])


def prep(it):
    """Per item: ids, every tool output's window, planted tokens, known-slip tokens (simulated items only)."""
    ids, off, spans = marks.layout(_tok, it["messages"])
    last = max(spans)  # the last tool output (a whole run can end with the agent's text)
    r = {"item": it["item"], "ids": ids, "window": spans[last][1:], "spans": [s[1:] for _, s in sorted(spans.items())]}
    if it["kind"] == "plant":
        r["planted"] = marks.planted_tokens(_tok, it)
        if it["slip"] == "gutter":  # line tokens and the newline token, each from its own characters
            r["line_tokens"] = marks.tokens_at(off, [spans[last][0] + c for c in it["planted_chars"][:-1]])
            r["newline_token"] = marks.tokens_at(off, [spans[last][0] + it["planted_chars"][-1]])
    if it["arm"] == "fake":
        which = {last} if it["kind"] == "site" else {i for i, m in enumerate(it["messages"]) if m["role"] == "tool"}
        r["slips"] = sorted(set().union(*map(set, marks.known_slips(_tok, it["messages"], which)[0].values())))
    return r


def rank_frac(p):
    """(rank - 1) / n by push, rank 1 = the largest push."""
    return (rankdata(-p, method="min") - 1) / len(p)


def auroc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    import modal

    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    read = lambda p: b"".join(vol.read_file(f"{OUT}/{p}"))  # noqa: E731
    done, repro = json.loads(read("done.json")), json.loads(read("repro.json"))
    raw_scores = read("scores.npz")
    z = np.load(io.BytesIO(raw_scores))
    staged = HERE / "data/staged.jsonl"
    items = [json.loads(x) for x in staged.read_text().splitlines()]
    sha = hashlib.sha256(staged.read_bytes()).hexdigest()
    if not repro["pass"] or done["staged_sha256"] != sha or done["script_sha256"][:12] != RUN_SHA12 or done["items"] != len(items):
        raise SystemExit("the job's outputs do not match the staged inputs or the refit did not reproduce")
    meta = json.loads(str(z["meta"]))
    cols = meta[0]["columns"]
    main_c, plant_c = cols.index("fixed_vs_e1|v"), cols.index("plant2|v")
    rnd = {"main": [cols.index(f"fixed_vs_e1|r{i + 1}") for i in range(N_RANDOM)], "plant2": [cols.index(f"plant2|r{i + 1}") for i in range(N_RANDOM)]}
    with ProcessPoolExecutor(os.cpu_count(), initializer=_init) as ex:  # n_jobs = every core
        P = {r["item"]: r for r in ex.map(prep, items, chunksize=4)}
    _init()
    tok = _tok
    S, IDS = {}, {}
    by_kind = {it["item"]: it["kind"] for it in items}  # whole runs are cut at 32k; their windows are clipped below
    for m_, st, ln in zip(meta, z["starts"], z["lens"]):
        i, w0 = m_["item"], m_["w0"]
        S[i], IDS[i] = z["scores"][st: st + ln].astype(np.float64), z["ids"][st: st + ln]
        if list(IDS[i]) != P[i]["ids"][w0: w0 + ln] or (by_kind[i] != "whole" and tuple(m_["window"]) != tuple(P[i]["window"])):
            raise SystemExit(f"item {i}: token ids or window differ from the local rendering")
    W0 = {m_["item"]: m_["w0"] for m_ in meta}

    def push(i, a, b, c):
        """Push p_t = s_t - s_{t-1} for t in [a, b) along column c (absolute positions)."""
        s = S[i][:, c]
        return s[a - W0[i]: b - W0[i]] - s[a - 1 - W0[i]: b - 1 - W0[i]]

    def score_at(i, t, c):
        return float(S[i][t - W0[i], c])

    # ---- 2. the gate ----
    gut = [it for it in items if it["kind"] == "plant" and it["slip"] == "gutter"]
    gate, per = {}, defaultdict(list)
    for name, cs in (("plant2", [plant_c] + rnd["plant2"]), ("main", [main_c] + rnd["main"])):
        for j, c in enumerate(cs):
            rs, rs_lines, rs_nl, allr = [], [], [], []
            for it in gut:
                i = it["item"]
                a, b = P[i]["window"]
                rf = rank_frac(push(i, a, b, c))
                pl = [t - a for t in P[i]["planted"]]
                nl = [t - a for t in P[i]["newline_token"]]
                lines = [t - a for t in P[i]["line_tokens"]]
                rs.append(float(rf[pl].min()))
                rs_lines.append(float(rf[lines].min()) if lines else float("nan"))
                rs_nl.append(float(rf[nl].min()))
                allr.append([float(x) for x in rf[pl]])
            key = name if j == 0 else f"{name}_random{j}"
            gate[key] = {"G": float(np.median(rs)), "per_reply_r": rs, "per_reply_r_gutter_lines": rs_lines, "per_reply_r_final_newline": rs_nl,
                         "G_gutter_lines_only": float(np.nanmedian(rs_lines)),
                         "G_final_newline_only": float(np.median(rs_nl)),
                         "newline_is_best_planted": int(sum(n_ <= l_ for n_, l_ in zip(rs_nl, rs_lines)))}
            if j == 0:
                per[name] = allr
    gate["pass"] = bool(gate["plant2"]["G"] <= BAR and all(gate[f"plant2_random{j}"]["G"] > BAR for j in range(1, N_RANDOM + 1)))
    gate["all_planted_rank_fractions"] = {"plant2": per["plant2"], "main": per["main"]}
    note = {}
    pairs = defaultdict(dict)
    for it in items:
        if it["kind"] in ("plant", "twin") and it["slip"] == "note":
            pairs[(it["run"], it["msg"])][it["kind"]] = it["item"]
    for name, c in (("plant2", plant_c), ("main", main_c)):
        d_end, d_win, tails = [], [], defaultdict(list)
        for pr in pairs.values():
            p_, t_ = pr["plant"], pr["twin"]
            e_p, e_t = [x for x in meta if x["item"] == p_][0], [x for x in meta if x["item"] == t_][0]
            d_end.append(score_at(p_, e_p["reply"][1] - 1, c) - score_at(t_, e_t["reply"][1] - 1, c))
            d_win.append(score_at(p_, P[p_]["window"][1] - 1, c) - score_at(t_, P[t_]["window"][1] - 1, c))
            kp, kt = e_p["n_tokens"] - P[p_]["window"][1], e_t["n_tokens"] - P[t_]["window"][1]
            for k in range(1, min(kp, kt) + 1):  # closing template tokens, aligned from the end
                tails[k].append((score_at(p_, e_p["n_tokens"] - k, c) - score_at(t_, e_t["n_tokens"] - k, c),
                                 tok.decode([int(IDS[p_][-k])]), tok.decode([int(IDS[t_][-k])])))
        note[name] = {"pairs": len(d_end), "output_end_planted_minus_twin": {"mean": float(np.mean(d_end)), "median": float(np.median(d_end))},
                      "window_last_planted_minus_twin": {"mean": float(np.mean(d_win)), "median": float(np.median(d_win))},
                      "template_tail": {f"-{k}": {"mean": float(np.mean([x[0] for x in v])), "median": float(np.median([x[0] for x in v])),
                                                 "tokens_planted": sorted({x[1] for x in v}), "tokens_twin": sorted({x[2] for x in v})}
                                        for k, v in sorted(tails.items())}}
    out = {"label": "token attribution (preliminary until reviewed)", "gate": gate, "note_descriptive": note,
           "refit": repro, "job": done}
    ctx = None
    # ---- 3. main analysis, only if the gate passes ----
    if gate["pass"]:
        sites = [it for it in items if it["kind"] == "site"]
        rng_ex = random.Random(EXAMPLES_SEED)

        def table(group, cs, slip_excl):
            """Per token string: per arm n, sites, tasks, T, m; per random direction T."""
            t = defaultdict(lambda: defaultdict(lambda: {"n": 0, "sites": set(), "tasks": set(), "T": 0.0, "Tr": np.zeros(N_RANDOM)}))
            occ = defaultdict(list)
            for it in group:
                i = it["item"]
                end = W0[i] + len(IDS[i])  # whole runs: outputs inside the 32k cut, a crossing one clipped at the cut
                spans = [P[i]["window"]] if it["kind"] == "site" else [(s[0], min(s[1], end)) for s in P[i]["spans"] if s[0] < end and s[1] > s[0]]
                excl = set(P[i].get("slips", [])) if slip_excl else set()
                for a, b in spans:
                    ps = [push(i, a, b, c) for c in cs]
                    for k, tpos in enumerate(range(a, b)):
                        if tpos in excl:
                            continue
                        w = tok.decode([int(IDS[i][tpos - W0[i]])])
                        e = t[w][it["arm"]]
                        e["n"] += 1
                        e["sites"].add(it.get("site"))
                        e["tasks"].add(it["task"])
                        e["T"] += ps[0][k]
                        e["Tr"] += [p[k] for p in ps[1:]]
                        if it["arm"] == "fake" and it["kind"] == "site":
                            occ[w].append((ps[0][k], i, tpos))
            return t, occ

        def top_list(t, role, base="e1"):
            rows = []
            for w, arms in t.items():
                f = arms.get(role)
                if not f or f["n"] == 0:
                    continue
                b = arms.get(base, {"n": 0, "T": 0.0})
                row = {"token": w, "T_" + role: f["T"], "n": {a: arms[a]["n"] for a in ("e1", "e2", "fake") if a in arms},
                       "m": {a: arms[a]["T"] / arms[a]["n"] for a in ("e1", "e2", "fake") if a in arms and arms[a]["n"]},
                       "sites": len(f["sites"]), "tasks": len(f["tasks"]), "T_random_max": float(f["Tr"].max()), "T_random_mean": float(f["Tr"].mean())}
                ok_n = f["n"] >= 5 and row["sites"] >= 3 and row["tasks"] >= 3
                beats = f["T"] > f["Tr"].max()
                if b["n"] < 3:
                    row["candidate"] = "rare in real replies" if ok_n and beats else False
                else:
                    row["candidate"] = bool(ok_n and beats and f["T"] / f["n"] > b["T"] / b["n"])
                rows.append(row)
            rows.sort(key=lambda r: -r["T_" + role])
            return rows[:TOP]

        def summarize(group, label):
            t, occ = table(group, [main_c] + rnd["main"], slip_excl=False)
            tx, _ = table(group, [main_c] + rnd["main"], slip_excl=True)
            fake, fake_x, ctrl = top_list(t, "fake"), top_list(tx, "fake"), top_list(t, "e2")
            n_c = lambda rows: sum(r["candidate"] is True for r in rows)  # noqa: E731
            return {"sites": len({it["site"] for it in group}), "top_fake_all_tokens": fake, "top_fake_known_slips_excluded": fake_x,
                    "control_top_second_real": ctrl, "candidates_fake": n_c(fake), "candidates_fake_excluded": n_c(fake_x),
                    "candidates_control": n_c(ctrl)}, occ, t

        main_res, occ, t_all = summarize(sites, "all")
        out["main"] = {"all_sites": main_res,
                       "new_sites": summarize([it for it in sites if it["sim_source"] == "new"], "new")[0],
                       "natural_sites": summarize([it for it in sites if it["sim_source"] == "natural"], "natural")[0]}
        ends = defaultdict(dict)
        runon = defaultdict(dict)
        for it in sites:
            m_ = [x for x in meta if x["item"] == it["item"]][0]
            ends[it["arm"]][it["site"]] = score_at(it["item"], m_["reply"][1] - 1, main_c)
            if P[it["item"]]["window"][1] > m_["reply"][1]:
                runon[it["arm"]][it["site"]] = score_at(it["item"], P[it["item"]]["window"][1] - 1, main_c)
        out["main"]["reply_end"] = {"mean_score": {a: float(np.mean(list(v.values()))) for a, v in ends.items()},
                                    "auroc_fake_vs_first_real": auroc(np.array(list(ends["fake"].values())), np.array(list(ends["e1"].values()))),
                                    "auroc_second_vs_first_real": auroc(np.array(list(ends["e2"].values())), np.array(list(ends["e1"].values()))),
                                    "run_on_token_count": {a: len(v) for a, v in runon.items()},
                                    "run_on_token_mean_score": {a: float(np.mean(list(v.values()))) for a, v in runon.items() if v}}
        whole = [it for it in items if it["kind"] == "whole"]
        tw, _ = table(whole, [main_c] + rnd["main"], slip_excl=False)
        twx, _ = table(whole, [main_c] + rnd["main"], slip_excl=True)
        out["main"]["whole_runs"] = {"runs_per_arm": len(whole) // 3, "top_simulated_all_tokens": top_list(tw, "fake"),
                                     "top_simulated_known_slips_excluded": top_list(twx, "fake"), "control_top_second_real": top_list(tw, "e2")}
        grams = {}
        for n in (2, 3):
            g = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
            for it in sites:
                i = it["item"]
                a, b = P[i]["window"]
                s = S[i][:, main_c]
                for tpos in range(a + n - 1, b):
                    key = tuple(tok.decode([int(IDS[i][x - W0[i]])]) for x in range(tpos - n + 1, tpos + 1))
                    e = g[key][it["arm"]]
                    e[0] += 1
                    e[1] += s[tpos - W0[i]] - s[tpos - n - W0[i]]
            rows = sorted(({"gram": list(k), "n": {a: v[0] for a, v in arms.items()}, "T": {a: v[1] for a, v in arms.items()}}
                           for k, arms in g.items() if "fake" in arms), key=lambda r: -r["T"]["fake"])[:TOP]
            grams[n] = rows
        out["main"]["ngrams_counted"] = {n: len(v) for n, v in grams.items()}

        def window(i, tpos):
            lo, hi = max(tpos - CTX, W0[i]), min(tpos + CTX + 1, W0[i] + len(IDS[i]))
            return {"before": tok.decode([int(x) for x in IDS[i][lo - W0[i]: tpos - W0[i]]]), "token": tok.decode([int(IDS[i][tpos - W0[i]])]),
                    "after": tok.decode([int(x) for x in IDS[i][tpos + 1 - W0[i]: hi - W0[i]]])}

        fake_sites = sorted((it for it in sites if it["arm"] == "fake"), key=lambda x: x["item"])
        examples = []
        for it in rng_ex.sample(fake_sites, 5):
            i = it["item"]
            a, b = P[i]["window"]
            p = push(i, a, b, main_c)
            examples.append({"site": it["site"], "top3": [window(i, a + int(k)) | {"push": float(p[k])} for k in np.argsort(-p)[:3]]})
        ctx = {"contexts_top_fake": {r["token"]: [window(i, tpos) | {"push": float(v)} for v, i, tpos in sorted(occ[r["token"]], key=lambda x: -x[0])[:3]]
                                     for r in main_res["top_fake_all_tokens"]},
               "examples": examples, "examples_seed": EXAMPLES_SEED, "ngrams": grams}
    out |= {"inputs": {"scores_npz_sha256": hashlib.sha256(raw_scores).hexdigest(), "staged_sha256": sha},
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if ctx is not None:
        cpath = HERE / "data" / f"token_attr_context_{stamp}.json"
        with open(cpath, "x") as fh:
            fh.write(json.dumps(ctx) + "\n")
        out["context_file"] = {"path": str(cpath.relative_to(REPO)), "sha256": hashlib.sha256(cpath.read_bytes()).hexdigest(), "where": "S3 only"}
    with open(HERE / "results" / f"token_attr_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1, default=float) + "\n")
    print(f"results/token_attr_{stamp}.json written; gate pass: {gate['pass']}")


if __name__ == "__main__":
    main()
