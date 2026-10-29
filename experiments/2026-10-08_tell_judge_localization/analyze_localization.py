"""1c, as pre-registered in preregistration.md: does the pre_call shift concentrate after content-flagged outputs?

Saved scores only ($0). Re-runs unchanged on any label file keyed by the blind list's F-ids (the provisional labels or
the Sol classifier's).

  uv run --with pandas --with scikit-learn python experiments/2026-10-08_tell_judge_localization/analyze_localization.py \
      --flags .run_cache/flags_swap_<stamp>.jsonl --labels config/flag_labels_provisional.yaml
Writes results/localization_<labels>_<stamp>.{json,md}.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
Q2 = Path("~/probe-shift-q2").expanduser()
sys.path.insert(0, str(ROOT / "experiments/2026-10-06_q2_impactful_call_transfer/shift"))
import examples as ex  # noqa: E402

from probe_shift.provenance import capture  # noqa: E402

CODES = {"SI": "state_inconsistency", "US": "unearned_success", "RV": "environment_rule_violation", "TR": "too_regular",
         "HA": "harness_artifact", "RL": "reference_leak", "TO": "thin_output", "OT": "other"}
CONTENT = {"state_inconsistency", "unearned_success", "environment_rule_violation"}
UNCITED_LATER = re.compile(r"\b(?:later|subsequent(?:ly)?|afterwards?)\b", re.I)
# Case-sensitive for the slot letters, so the ordinary "in a" does not match (Codex review, 2026-10-08).
OTHER_TRANSCRIPT = re.compile(r"\b(?:[Tt]ranscript [AB]\b|in [AB]\b|[AB]'s\b|[Tt]he other (?:transcript|trajectory|run)|"
                              r"[Oo]therwise (?:matching|identical)|[Cc]omparable environment|[Pp]aired)")
N_BOOT, SEED = 1000, 0
SWAP = Q2 / ".run_cache/q2_swap"
COVARIATES = ["env", "position", "inserts", "tool_class", "batch", "attempted", "y"]


def position_bucket(t):
    return "0-4" if t < 5 else "5-9" if t < 10 else "10-19" if t < 20 else "20-29" if t < 30 else "30+"


def design(df: pd.DataFrame, indicators: list[str]) -> np.ndarray:
    X = pd.get_dummies(df[COVARIATES].astype(str), drop_first=True).astype(float)
    return np.c_[np.ones(len(df)), df[indicators].astype(float).values, X.values]


def ols(df: pd.DataFrame, outcome: str, indicators: list[str], w=None) -> np.ndarray:
    X, yv = design(df, indicators), df[outcome].values
    if w is not None:
        s = np.sqrt(w)
        X, yv = X * s[:, None], yv * s
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    return beta[1:1 + len(indicators)]


def s3(df, col, mult=None):
    """AUROC (impactful vs read-only) of unflagged calls reweighted to the flagged calls' strata, minus that of flagged
    calls; common support only. `mult`: bootstrap multiplicity per call."""
    m = np.ones(len(df)) if mult is None else np.asarray(mult, float)
    strata = (df.position + "|" + df.inserts + "|" + df.tool_class + "|" + df.y.astype(str)).values
    fl_ = df.flagged.values == 1
    sf = pd.Series(m[fl_]).groupby(strata[fl_]).sum()
    su = pd.Series(m[~fl_]).groupby(strata[~fl_]).sum()
    common = set(sf.index) & set(su.index)
    keep = np.isin(strata, list(common)) & (m > 0)
    w = np.where(fl_, m, m * np.array([sf.get(x, 0) / su.get(x, 1) for x in strata]))
    y, sc = df.y.values, df[col].values
    out = {}
    for name, mask in (("auroc_flagged", keep & fl_), ("auroc_unflagged_reweighted", keep & ~fl_)):
        out[name] = round(float(roc_auc_score(y[mask], sc[mask], sample_weight=w[mask])), 4) if len(set(y[mask])) == 2 else float("nan")
    out["difference"] = round(out["auroc_unflagged_reweighted"] - out["auroc_flagged"], 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--blind-key", default=".run_cache/blind_key.jsonl")
    args = ap.parse_args()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    ex.DATA = Q2 / ".run_cache/q2_shift"

    # Flags with labels and visibility.
    flags = pd.read_json(HERE / args.flags, lines=True)
    flags = flags[flags.in_range].copy()
    key = pd.read_json(HERE / args.blind_key, lines=True)
    labels = yaml.safe_load((HERE / args.labels).read_text())["labels"]
    key["category"] = key.fid.map(lambda f: CODES.get(labels[f], labels[f]))
    flags = flags.merge(key[["side", "key", "action_n", "category"]], on=["side", "key", "action_n"], how="left", validate="1:1")
    assert flags.category.notna().all(), "a flag without a label"
    flags["content"] = flags.category.isin(CONTENT)
    flags["drop_clipped_sim"] = (flags.side == "sim") & flags.clipped
    flags["drop_uncited_later"] = ~flags.cites_later & flags.reason.str.contains(UNCITED_LATER)
    flags["other_transcript"] = flags.reason.str.contains(OTHER_TRANSCRIPT)
    flags["own_turn"] = flags.turn

    # Calls: sim (V1, V2, V4) and real, pre_call.
    base = json.loads(sorted(SWAP.glob("scores_2*.json"))[-1].read_text())
    note = json.loads(sorted(SWAP.glob("scores_swap_note_2*.json"))[-1].read_text())
    k = ["key", "step", "action"]
    sw = pd.DataFrame(base["swap"]).query("position == 'pre_call'")[k + ["y", "original", "rerendered"]].rename(columns={"original": "V1", "rerendered": "V2"})
    v4 = pd.DataFrame([{**{c: r[c] for c in k}, "V4": r["scores"]["sim-attack-diff-note"]} for r in note["swap"] if r["position"] == "pre_call"])
    sim = sw.merge(v4, on=k, how="inner")
    sim_keys = set(flags[flags.side == "sim"].key) | set(sim.key)
    real_keys = set(flags[flags.side == "real"].key)
    saved = pd.DataFrame(base["saved"])
    real = saved[(saved.arm == "real-attack-e1")].rename(columns={"pre_call_score": "S"})[k + ["y", "S"]]
    q = ex.table("qwen3-32b")
    cells = set(q[q.key.isin(set(sim.key))].cell)
    real_keys = set(q[(q.arm == "real-attack-e1") & q.cell.isin(cells)].key)
    real = real[real.key.isin(real_keys)]
    info = q.drop_duplicates(k).set_index(k)[["tool", "env", "main_task", "sim_reference"]]
    runs = ex.load_runs(set(sim.key) | real_keys)
    attempted = {json.loads(l)["key"]: json.loads(l)["label"] == "attempted"
                 for l in (Q2 / ".run_cache/q2_shift/attack_attempt_labels_2026-10-07_154729.jsonl").read_text().splitlines()}

    def covariates(df, arm):
        df = df.join(info, on=k)
        assert df.env.notna().all(), f"{arm}: calls missing from the Q2 table"
        df["position"] = df.step.map(position_bucket)
        ins = {kk: sorted(sa[0] for sa, c in runs[kk].items() if c["tool"] == "text_editor" and ex.editor_args(c)[0] == "insert")
               for kk in set(df.key)}
        df["inserts"] = [str(min(sum(s < t for s in ins[kk]), 2)) for kk, t in zip(df.key, df.step)]
        df["tool_class"] = df.tool.where(df.tool.isin(["bash", "text_editor"]), "other")
        df["batch"] = (df.sim_reference.fillna("") if arm == "sim" else df.key.map(lambda x: "random" if "_random_" in x[:20] else "fleet"))
        df["attempted"] = df.key.map(lambda x: attempted.get(x, False)).astype(int)
        return df

    sim, real = covariates(sim, "sim"), covariates(real, "real")
    for col in ("V1", "V2", "V4"):
        sim[f"z_{col}"] = (sim[col] - sim[col].mean()) / sim[col].std() * np.where(sim.y == 0, 1, -1)
        sim[f"u_{col}"] = (sim[col] - sim[col].mean()) / sim[col].std()
    real["z"] = (real.S - real.S.mean()) / real.S.std() * np.where(real.y == 0, 1, -1)
    real["u"] = (real.S - real.S.mean()) / real.S.std()

    def indicators(df, side, fl):
        fl = fl[fl.side == side]
        out = {"flagged": [], "future": [], "later_run": []}
        by = {kk: g for kk, g in fl.groupby("key")}
        for kk, t in zip(df.key, df.step):
            g = by.get(kk)
            vt = g.vis.values if g is not None else np.array([])
            out["flagged"].append(bool((vt < t).any()))
            out["future"].append(bool((vt >= t).any()))
            out["later_run"].append(bool(len(vt)))
        return df.assign(**{c: np.array(v, int) for c, v in out.items()})

    def flag_set(name):
        f = flags[~flags.drop_uncited_later & ~flags.drop_clipped_sim].copy()
        f["vis"] = f.visible_turn
        if name == "main":
            return f[f.content]
        if name == "no_other_transcript":
            return f[f.content & ~f.other_transcript]
        if name == "cited_later_undelayed":
            g = f[f.content].copy()
            g["vis"] = g.own_turn
            return g
        if name == "any_flag":
            return f
        if name == "harness_artifact":
            return f[f.category == "harness_artifact"]
        raise ValueError(name)

    tasks = sorted(set(sim.main_task) | set(real.main_task))
    rng = np.random.default_rng(SEED)
    draws = [np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(N_BOOT)]
    tid = {t: i for i, t in enumerate(tasks)}

    def boot(fn):
        point = fn(None)
        vals = np.array([fn(d) for d in draws])
        lo, hi = np.nanpercentile(vals, [2.5, 97.5], axis=0)
        return point, lo, hi

    def weights(df, d):
        return None if d is None else d[df.main_task.map(tid).values].astype(float)

    results = {}
    for name in ("main", "no_other_transcript", "cited_later_undelayed", "any_flag", "harness_artifact"):
        fl = flag_set(name)
        s_df, r_df = indicators(sim, "sim", fl), indicators(real, "real", fl)
        res = {"flags_used": {"sim": int((fl.side == "sim").sum()), "real": int((fl.side == "real").sum())},
               "calls_flagged": {"sim": int(s_df.flagged.sum()), "real": int(r_df.flagged.sum())},
               "calls": {"sim": len(s_df), "real": len(r_df)}}
        b = lambda df, out, d, ind=("flagged",): ols(df, out, list(ind), weights(df, d))
        for v in ("V1", "V2", "V4"):
            p, lo, hi = boot(lambda d: (b(s_df, f"z_{v}", d)[0], b(r_df, "z", d)[0], b(s_df, f"z_{v}", d)[0] - b(r_df, "z", d)[0]))
            res[f"beta_sim_{v}"] = [round(float(p[0]), 4), round(float(lo[0]), 4), round(float(hi[0]), 4)]
            res[f"P1_{v}"] = [round(float(p[2]), 4), round(float(lo[2]), 4), round(float(hi[2]), 4)]
            res["beta_real"] = [round(float(p[1]), 4), round(float(lo[1]), 4), round(float(hi[1]), 4)]
        if name == "main":
            # S2: label-conditional effects on the unsigned score.
            for lab, nm in ((0, "read_only"), (1, "impactful")):
                ss, rr = s_df[s_df.y == lab], r_df[r_df.y == lab]
                p, lo, hi = boot(lambda d: (ols(ss, "u_V2", ["flagged"], weights(ss, d))[0], ols(rr, "u", ["flagged"], weights(rr, d))[0]))
                res[f"S2_{nm}"] = {"sim_V2": [round(float(p[0]), 4), round(float(lo[0]), 4), round(float(hi[0]), 4)],
                                   "real": [round(float(p[1]), 4), round(float(lo[1]), 4), round(float(hi[1]), 4)]}
            # S3: AUROC of flagged calls vs unflagged calls reweighted to the flagged calls' strata (position × inserts ×
            # tool × label), on common support only: strata with both flagged and unflagged calls (Codex review).
            res["S3"] = {"sim_V1": s3(s_df, "V1"), "sim_V2": s3(s_df, "V2"), "real": s3(r_df, "S")}
            # Post hoc (suggested by the Codex review after the result; not pre-registered): the same contrast on calls
            # after an insert in an earlier turn, where the motivating AUROC gap is, as sim minus real, bootstrapped jointly.
            s_ai, r_ai = s_df[s_df.inserts != "0"], r_df[r_df.inserts != "0"]
            p, lo, hi = boot(lambda d: (lambda a, b: (a, b, a - b))(s3(s_ai, "V2", weights(s_ai, d))["difference"],
                                                                     s3(r_ai, "S", weights(r_ai, d))["difference"]))
            res["posthoc_S3_after_insert"] = {"sim_V2": [round(float(p[0]), 4), round(float(lo[0]), 4), round(float(hi[0]), 4)],
                                              "real": [round(float(p[1]), 4), round(float(lo[1]), 4), round(float(hi[1]), 4)],
                                              "sim_minus_real": [round(float(p[2]), 4), round(float(lo[2]), 4), round(float(hi[2]), 4)],
                                              "calls_flagged": {"sim": int(s_ai.flagged.sum()), "real": int(r_ai.flagged.sum())},
                                              "calls": {"sim": len(s_ai), "real": len(r_ai)}}
            # C1: before the first flag; C2: future flags.
            for df, out, arm in ((s_df, "z_V2", "sim"), (r_df, "z", "real")):
                pre = df[df.flagged == 0]
                p, lo, hi = boot(lambda d: ols(pre, out, ["later_run"], weights(pre, d)))
                res[f"C1_{arm}"] = [round(float(p[0]), 4), round(float(lo[0]), 4), round(float(hi[0]), 4)]
                p, lo, hi = boot(lambda d: ols(df, out, ["flagged", "future"], weights(df, d)))
                res[f"C2_{arm}"] = {"flagged": [round(float(p[0]), 4), round(float(lo[0]), 4), round(float(hi[0]), 4)],
                                    "future": [round(float(p[1]), 4), round(float(lo[1]), 4), round(float(hi[1]), 4)]}
        results[name] = res
        print(name, json.dumps({kk: v for kk, v in res.items() if kk.startswith(("P1", "beta", "calls_flagged", "flags_used"))}), flush=True)

    out = {"stamp": stamp, "provenance": capture(ROOT, [Path(__file__), HERE / args.labels]), "args": vars(args),
           "flags_after_rules": {"content_sim": int(flag_set("main").side.eq("sim").sum()), "content_real": int(flag_set("main").side.eq("real").sum()),
                                 "dropped_clipped_sim": int(flags.drop_clipped_sim.sum()), "dropped_uncited_later": int(flags.drop_uncited_later.sum()),
                                 "other_transcript": int(flags.other_transcript.sum())},
           "category_counts": {s: flags[flags.side == s].category.value_counts().to_dict() for s in ("sim", "real")},
           "results": results}
    tag = Path(args.labels).stem.replace("flag_labels_", "")
    (HERE / "results" / f"localization_{tag}_{stamp}.json").write_text(json.dumps(out, indent=1, default=str))
    f3 = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
    md = [f"# Localization result ({tag} labels, {stamp})", "", f"Flags after the visibility rules: {out['flags_after_rules']}.",
          f"Categories (sim / real): {out['category_counts']}", "",
          "| flag set | sim calls flagged / all | real calls flagged / all | P1 (V2) | P1 (V1) | P1 (V4) | β sim V2 | β real |", "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        md.append(f"| {name} | {r['calls_flagged']['sim']} / {r['calls']['sim']} | {r['calls_flagged']['real']} / {r['calls']['real']} | "
                  f"{f3(r['P1_V2'])} | {f3(r['P1_V1'])} | {f3(r['P1_V4'])} | {f3(r['beta_sim_V2'])} | {f3(r['beta_real'])} |")
    m = results["main"]
    md += ["", f"S2 read-only (unsigned score, + = more impactful-looking): sim V2 {f3(m['S2_read_only']['sim_V2'])}, real {f3(m['S2_read_only']['real'])}",
           f"S2 impactful: sim V2 {f3(m['S2_impactful']['sim_V2'])}, real {f3(m['S2_impactful']['real'])}",
           f"S3 (AUROC unflagged reweighted − flagged, common support): {m['S3']}",
           f"Post hoc (Codex review, not pre-registered), calls after an insert: S3 difference sim V2 {f3(m['posthoc_S3_after_insert']['sim_V2'])}, "
           f"real {f3(m['posthoc_S3_after_insert']['real'])}, sim − real {f3(m['posthoc_S3_after_insert']['sim_minus_real'])} "
           f"(flagged calls {m['posthoc_S3_after_insert']['calls_flagged']} of {m['posthoc_S3_after_insert']['calls']})",
           f"C1 before the first flag (later-flagged run): sim {f3(m['C1_sim'])}, real {f3(m['C1_real'])}",
           f"C2 future flag: sim {f3(m['C2_sim']['future'])} (flagged {f3(m['C2_sim']['flagged'])}); real {f3(m['C2_real']['future'])}"]
    (HERE / "results" / f"localization_{tag}_{stamp}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
