"""Examples: where TF-IDF and the probe disagree at level 0, and what each finds most real or simulated at level 7.

Scores are held-out-environment log-odds (real e1 vs simulated) on outputs both methods scored, standardized per
method over real-e1 and simulated outputs (z). Positive z means "simulated".

(A) Level 0. Signed confidence c = z if the output is simulated, -z if real (large c: confidently right). "TF-IDF
    confident, probe not": largest c_tfidf - c_probe; the other way round, largest c_probe - c_tfidf. At most one
    output per run and two per main task, so one long run cannot fill a list. Also per-tool and per-editor-command
    AUROC for each method, the methods' rank correlation, and what the 500 strongest disagreements each way are.
(B) Level 7. For each method, the 20 outputs with the highest z (most simulated) and the 20 lowest (most real), same
    caps, shown as level-7 text. Written blind (no true arm) to results/level7_blind_<stamp>.md for hand labels;
    `--labels config/level7_labels.yaml` then unblinds and counts classes. Also simple content markers per arm at
    levels 0 and 7 (error and success words, empty outputs, length) and how each tracks the two scores.

  uv run python experiments/2026-10-07_what_each_method_sees/analyze_examples.py [--labels config/level7_labels.yaml]
"""

import argparse
import gzip
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from probe_shift.provenance import capture
from probe_shift.scrub import EDITOR_DIFF, GUTTER, mask_times, scrub
from probe_shift.scoring import auroc_ci

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
E1, SIM = "real-attack-e1", "sim-attack"
METHODS = ("tfidf", "probe")
NAMES = {"tfidf": "TF-IDF", "probe": "the probe"}
CLASSES = ("residual formatting", "invented or implausible content", "missing error", "code-content difference", "unclear")
MARKERS = {  # simple content markers, matched on the text at each level
    "error word": re.compile(r"\b(?:error|failed|failure|traceback|exception|fatal|denied|not found|cannot|can't|unable)\b", re.I),
    "warning word": re.compile(r"\bwarn(?:ing)?s?\b", re.I),
    "success word": re.compile(r"\b(?:success(?:ful(?:ly)?)?|passed|ok|done|complete(?:d)?|resolved|ready|running)\b|✓", re.I),
    "empty": re.compile(r"\A\s*\Z"),
}


# Rare tells found in the level-7 extremes, each a regex on the output as stored (level 0); counted per arm over all outputs.
TELLS = {
    "view_range error (editor)": re.compile(r"Invalid `view_range`"),
    "argument-validation error (harness)": re.compile(r"validation errors? parsing tool input arguments"),
    "dpkg percentage progress": re.compile(r"\(Reading database \.\.\. \d+%"),
    "harness wrapper visible": re.compile(r"timeout -k \S+ \S+ bash --login -c"),
    "ANSI colour codes": re.compile(r"\x1b\[|\[\d+(?:;\d+)*m"),
}
POSITION_BUCKETS = ((1, 1), (2, 3), (4, 10), (11, 25), (26, 10_000))


def pick(order, key, main_task, n, per_task=2):
    """The first n rows of `order`, at most one per run and `per_task` per main task."""
    runs, tasks, out = set(), Counter(), []
    for i in order:
        if key[i] in runs or tasks[main_task[i]] >= per_task:
            continue
        runs.add(key[i])
        tasks[main_task[i]] += 1
        out.append(int(i))
        if len(out) == n:
            break
    return out


def clip(text, head=600, tail=200):
    text = text if len(text) <= head + tail + 20 else f"{text[:head]}\n[... {len(text) - head - tail} chars ...]\n{text[-tail:]}"
    return text.replace("```", "'''")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config/what_sees.yaml"))
    ap.add_argument("--split", default="env")
    ap.add_argument("--n-disagree", type=int, default=15)
    ap.add_argument("--n-extreme", type=int, default=20)
    ap.add_argument("--labels", default="")
    ap.add_argument("--blind-stamp", default="", help="with --labels: the stamp of the blind list the labels were made on")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    files = [Path(__file__), Path(args.config)] + ([Path(args.labels)] if args.labels else [])
    out = {"stamp": stamp, "provenance": capture(ROOT, files), "args": vars(args)}
    T = dict(np.load(HERE / ".run_cache/joined.npz"))
    staging = Path(cfg["staging_dir"]).expanduser()
    text = {(r["key"], r["step"], r["action"]): r["readout_text"]
            for r in map(json.loads, gzip.open(staging / "outputs.jsonl.gz", "rt")) if r["kept"]}
    key, arm, task = T["key"], T["arm"], T["main_task"]
    tool = np.where(T["tool"] == "text_editor", np.char.add("text_editor ", T["editor_command"]), T["tool"])
    txt = lambda i: text[(key[i], int(T["step"][i]), int(T["action"][i]))]

    def z_scores(level):
        s = {m: T[f"{m}_L{level}_{args.split}_real_vs_sim"] for m in METHODS}
        rows = np.isin(arm, [E1, SIM]) & ~np.isnan(s["tfidf"]) & ~np.isnan(s["probe"])
        z = {m: np.where(rows, (s[m] - s[m][rows].mean()) / s[m][rows].std(), np.nan) for m in METHODS}
        return rows, s, z

    def card(i, level, s, z, show_arm=True, label=""):
        head = f"**{label}** `{key[i][:70]}` step {T['step'][i]}.{T['action'][i]}, {tool[i]}, {T['environment'][i]}"
        if show_arm:
            head += f", **{arm[i]}**"
        scores = ", ".join(f"{m} z {z[m][i]:+.2f} (log-odds {s[m][i]:+.2f})" for m in METHODS)
        return [head + f"  \n{scores}", "", "```", clip(scrub(txt(i), level)), "```", ""]

    # ---- (A) level 0 -----------------------------------------------------------------------------------------------
    rows, s, z = z_scores(0)
    y = np.where(arm == SIM, 1.0, -1.0)
    c = {m: z[m] * y for m in METHODS}
    idx = np.flatnonzero(rows)
    yy = arm[idx] == SIM
    A = {"rows": int(rows.sum()), "per_output_auroc": {m: auroc_ci(yy, s[m][idx], task[idx]) for m in METHODS},
         "spearman_all": round(float(spearmanr(s["tfidf"][idx], s["probe"][idx])[0]), 3),
         "spearman_within_real": round(float(spearmanr(s["tfidf"][idx][~yy], s["probe"][idx][~yy])[0]), 3),
         "spearman_within_sim": round(float(spearmanr(s["tfidf"][idx][yy], s["probe"][idx][yy])[0]), 3), "by_tool": {}}
    for t_ in sorted(set(tool[idx])):
        mm = idx[tool[idx] == t_]
        if len(set(arm[mm])) == 2 and len(mm) >= 30:
            A["by_tool"][t_] = {"n_real": int((arm[mm] == E1).sum()), "n_sim": int((arm[mm] == SIM).sum()),
                                **{m: round(float(roc_auc_score(arm[mm] == SIM, s[m][mm])), 3) for m in METHODS}}
    md = [f"# Examples ({stamp}); scores are held-out-environment log-odds, z standardized per method, + = simulated", ""]
    md += ["## (A) Level 0: what each method keys on", "",
           f"Per-output AUROC on {A['rows']:,} outputs both scored: TF-IDF {A['per_output_auroc']['tfidf']['auroc']}, probe "
           f"{A['per_output_auroc']['probe']['auroc']}. Rank correlation of the two scores {A['spearman_all']} (within real "
           f"{A['spearman_within_real']}, within simulated {A['spearman_within_sim']}).", "",
           "| tool / editor command | real | simulated | TF-IDF AUROC | probe AUROC |", "|---|---|---|---|---|"]
    md += [f"| {t_} | {v['n_real']} | {v['n_sim']} | {v['tfidf']:.3f} | {v['probe']:.3f} |" for t_, v in A["by_tool"].items()]
    md.append("")
    base_clock = round(float(np.mean([mask_times(txt(i)) != txt(i) for i in idx])), 3)
    base_gutter = round(float(np.mean([EDITOR_DIFF in txt(i) and bool(GUTTER.search(txt(i))) for i in idx])), 3)
    # Position in the run: the probe reads everything before an output, TF-IDF only the output. On `create` outputs the
    # text is one template in both arms, so any separation there has to come from context.
    order_all = np.lexsort((T["action"], T["step"], key))
    ordinal = np.empty(len(key), int)
    first = np.r_[True, key[order_all][1:] != key[order_all][:-1]]
    run_start = np.maximum.accumulate(np.where(first, np.arange(len(key)), 0))
    ordinal[order_all] = np.arange(len(key)) - run_start + 1
    A["by_position"] = {}
    for level in (0, 7):
        r_, s_, _ = z_scores(level)
        for subset, m_ in (("all outputs", r_), ("text_editor create", r_ & (T["editor_command"] == "create"))):
            for lo, hi in POSITION_BUCKETS:
                mm = np.flatnonzero(m_ & (ordinal >= lo) & (ordinal <= hi))
                if len(set(arm[mm])) == 2:
                    A["by_position"][f"L{level} | {subset} | {lo}-{hi if hi < 10_000 else 'end'}"] = {
                        "n": int(len(mm)), **{mt: round(float(roc_auc_score(arm[mm] == SIM, s_[mt][mm])), 3) for mt in METHODS}}
    md += ["Per-output AUROC by position in the run (held-out environment):", "",
           "| level, outputs, position | n | TF-IDF | probe |", "|---|---|---|---|"]
    md += [f"| {k_} | {v['n']} | {v['tfidf']:.3f} | {v['probe']:.3f} |" for k_, v in A["by_position"].items()]
    md.append("")
    for a, b in (("tfidf", "probe"), ("probe", "tfidf")):
        gap = np.where(rows, c[a] - c[b], -np.inf)
        order = np.argsort(-gap)
        top500 = order[:500]
        A[f"{a}_confident_{b}_not"] = {
            "top500_tools": {str(k): v for k, v in Counter(tool[top500]).most_common(8)},
            "top500_arms": {str(k): v for k, v in Counter(arm[top500]).items()},
            "top500_median_chars": float(np.median(T["readout_chars"][top500])),
            "top500_share_with_clock": round(float(np.mean([mask_times(txt(i)) != txt(i) for i in top500])), 3),
            "top500_share_editor_diff_with_gutter": round(float(np.mean([EDITOR_DIFF in txt(i) and bool(GUTTER.search(txt(i))) for i in top500])), 3),
            "examples": [{"key": key[i], "step": int(T["step"][i]), "action": int(T["action"][i]), "arm": arm[i], "tool": tool[i],
                          **{f"z_{m}": round(float(z[m][i]), 3) for m in METHODS}}
                         for i in pick(order, key, task, args.n_disagree)]}
        d = A[f"{a}_confident_{b}_not"]
        d["base_share_with_clock"] = base_clock
        d["base_share_editor_diff_with_gutter"] = base_gutter
        md += [f"### {NAMES[a]} more right than {NAMES[b]}: the {args.n_disagree} largest gaps c_{a} - c_{b} (one per run, two per main task)", "",
               f"Among the 500 largest gaps: tools {d['top500_tools']}; arms {d['top500_arms']}; median length "
               f"{d['top500_median_chars']:.0f} characters; with a clock time or date {d['top500_share_with_clock']} (all outputs "
               f"{base_clock}); editor diffs with a line-number gutter {d['top500_share_editor_diff_with_gutter']} (all outputs "
               f"{base_gutter}).", ""]
        for j, e in enumerate(d["examples"]):
            i = int(np.flatnonzero((key == e["key"]) & (T["step"] == e["step"]) & (T["action"] == e["action"]))[0])
            md += card(i, 0, s, z, label=f"A{'T' if a == 'tfidf' else 'P'}{j + 1:02d}")
    out["A_level0"] = A

    # ---- (B) level 7 -----------------------------------------------------------------------------------------------
    rows7, s7, z7 = z_scores(7)
    B = {"rows": int(rows7.sum()), "extremes": []}
    blind = [f"# Level-7 extremes, blind ({stamp}): label each with one of {list(CLASSES)}", ""]
    for m in METHODS:
        for side, sign in (("most_sim", -1), ("most_real", 1)):
            order = np.argsort(sign * np.where(rows7, z7[m], np.nan))  # NaN rows sort last
            order = order[rows7[order]]
            for j, i in enumerate(pick(order, key, task, args.n_extreme)):
                ex_id = f"{m[0].upper()}{'S' if side == 'most_sim' else 'R'}{j + 1:02d}"
                B["extremes"].append({"id": ex_id, "method": m, "side": side, "key": key[i], "step": int(T["step"][i]),
                                      "action": int(T["action"][i]), "arm": arm[i], "tool": tool[i], "environment": T["environment"][i],
                                      **{f"z_{mm}": round(float(z7[mm][i]), 3) for mm in METHODS},
                                      "level7_chars": len(scrub(txt(i), 7))})
                blind += [f"### {ex_id} ({m}, {side.replace('_', ' ')}), {tool[i]}, {T['environment'][i]}, step {T['step'][i]}", "",
                          "```", clip(scrub(txt(i), 7)), "```", ""]
    # Content markers per arm, levels 0 and 7, and how they track each score.
    every = np.flatnonzero(np.isin(arm, [E1, SIM]))
    markers = {}
    for level in (0, 7):
        texts = [scrub(txt(i), level) for i in every]
        sim = arm[every] == SIM
        for name, pat in MARKERS.items():
            has = np.array([bool(pat.search(t_)) for t_ in texts])
            r = {"share_real": round(float(has[~sim].mean()), 4), "share_sim": round(float(has[sim].mean()), 4)}
            for mth in METHODS:  # mean z of outputs with the marker vs without, within each arm (does the score follow it?)
                zz = (T[f"{mth}_L{level}_{args.split}_real_vs_sim"][every])
                ok = ~np.isnan(zz)
                zz = (zz - np.nanmean(zz)) / np.nanstd(zz)
                for a_name, a_mask in (("real", ~sim), ("sim", sim)):
                    w, wo = ok & has & a_mask, ok & ~has & a_mask
                    r[f"{mth}_z_with_minus_without_within_{a_name}"] = (
                        round(float(zz[w].mean() - zz[wo].mean()), 3) if w.any() and wo.any() else None)
            markers[f"L{level} {name}"] = r
        length = np.array([len(t_) for t_ in texts])
        markers[f"L{level} length (chars)"] = {"median_real": float(np.median(length[~sim])), "median_sim": float(np.median(length[sim])),
                                               "auroc_longer_is_sim": round(float(roc_auc_score(sim, length)), 4)}
    B["markers"] = markers
    # The rare tells: how often each arm shows them, and the level-7 run AUROC with those outputs dropped (saved scores,
    # no refit: the models are unchanged, the runs just lose those outputs).
    every_text = [txt(i) for i in every]
    sim = arm[every] == SIM
    hit_any = np.zeros(len(every), bool)
    ablation = {}

    def run_auc(mask_rows, method, split):
        sc = T[f"{method}_L7_{split}_real_vs_sim"][mask_rows]
        ok = ~np.isnan(sc) & ~np.isnan(T[f"{METHODS[1] if method == METHODS[0] else METHODS[0]}_L7_{split}_real_vs_sim"][mask_rows])
        k_, inv = np.unique(key[mask_rows][ok], return_inverse=True)
        mean = np.bincount(inv, sc[ok]) / np.bincount(inv)
        firsts = np.unique(inv, return_index=True)[1]
        return auroc_ci(arm[mask_rows][ok][firsts] == SIM, mean, task[mask_rows][ok][firsts])

    hits = {name: np.array([bool(pat.search(t_)) for t_ in every_text]) for name, pat in TELLS.items()}
    hit_any = np.logical_or.reduce(list(hits.values()))
    ablations = {}
    for split in ("env", "task"):
        ablation = {"none dropped": {m_: run_auc(every, m_, split) for m_ in METHODS}}
        for name, hit in list(hits.items()) + [("all five dropped", hit_any)]:
            ablation[name] = {"outputs_real": int((hit & ~sim).sum()), "outputs_sim": int((hit & sim).sum()),
                              **{m_: run_auc(every[~hit], m_, split) for m_ in METHODS}}
        ablations[split] = ablation
    B["tell_ablation_run_auroc"] = ablations
    out["B_level7"] = B
    md += ["## (B) Level 7: the rare tells, and run AUROC without them (saved scores, no refit)", "",
           "| outputs dropped | real e1 outputs | simulated outputs | TF-IDF, held-out env | probe, held-out env | TF-IDF, main folds | probe, main folds |",
           "|---|---|---|---|---|---|---|"]
    fmt = lambda v: f"{v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
    md += [f"| {k_} | {v.get('outputs_real', '')} | {v.get('outputs_sim', '')} | {fmt(v['tfidf'])} | {fmt(v['probe'])} | "
           f"{fmt(ablations['task'][k_]['tfidf'])} | {fmt(ablations['task'][k_]['probe'])} |" for k_, v in ablations["env"].items()]
    md += ["", "## (B) Level 7: content markers per arm (all real-e1 and simulated outputs)", "",
           "Within each arm: mean z of outputs with the marker minus without, so a positive number means the score calls "
           "outputs with the marker more simulated.", "",
           "| level, marker | share of real | share of simulated | TF-IDF z shift (real / sim) | probe z shift (real / sim) |",
           "|---|---|---|---|---|"]
    for name, r in markers.items():
        if "share_real" in r:
            f = lambda v: "n/a" if v is None else f"{v:+.2f}"
            md.append(f"| {name} | {r['share_real']:.3f} | {r['share_sim']:.3f} | {f(r['tfidf_z_with_minus_without_within_real'])} / "
                      f"{f(r['tfidf_z_with_minus_without_within_sim'])} | {f(r['probe_z_with_minus_without_within_real'])} / "
                      f"{f(r['probe_z_with_minus_without_within_sim'])} |")
        else:
            md.append(f"| {name} | median {r['median_real']:.0f} | median {r['median_sim']:.0f} | AUROC (longer = simulated) "
                      f"{r['auroc_longer_is_sim']} | |")
    md.append("")

    if args.labels:  # unblind: join the hand labels made on the blind list of `--blind-stamp`
        labels = yaml.safe_load(Path(args.labels).read_text())
        made_on = json.loads((HERE / "results" / f"examples_{args.blind_stamp}.json").read_text())["B_level7"]["extremes"]
        mine = [{k: e[k] for k in ("id", "key", "step", "action")} for e in B["extremes"]]
        assert mine == [{k: e[k] for k in ("id", "key", "step", "action")} for e in made_on], "extremes differ from the blind list"
        assert set(labels) == {e["id"] for e in B["extremes"]}, "every extreme needs exactly one label"
        counts = {}
        for e in B["extremes"]:
            lab = labels[e["id"]]
            assert lab["class"] in CLASSES, f"{e['id']}: unknown class {lab['class']}"
            e |= {"class": lab["class"], "note": lab["note"], "right_side": (e["arm"] == SIM) == (e["side"] == "most_sim")}
            counts.setdefault(f"{e['method']} {e['side']}", Counter())[lab["class"]] += 1
            counts.setdefault(f"{e['method']} {e['side']} right side", Counter())[str(e["right_side"])] += 1
        B["counts"] = {k: dict(v) for k, v in counts.items()}
        md += ["## (B) Level 7: the extremes, labelled blind and unblinded", "",
               "| method, side | " + " | ".join(CLASSES) + " | on the right side |", "|---|" + "---|" * (len(CLASSES) + 1)]
        for mth in METHODS:
            for side in ("most_sim", "most_real"):
                cc = counts[f"{mth} {side}"]
                right = counts[f"{mth} {side} right side"].get("True", 0)
                md.append(f"| {mth}, {side.replace('_', ' ')} | " + " | ".join(str(cc.get(k, 0)) for k in CLASSES) +
                          f" | {right} of {sum(cc.values())} |")
        md.append("")
        for e in B["extremes"]:
            i = int(np.flatnonzero((key == e["key"]) & (T["step"] == e["step"]) & (T["action"] == e["action"]))[0])
            md += card(i, 7, s7, z7, label=f"{e['id']} [{e['class']}; {'right' if e['right_side'] else 'wrong'} side]")
            md.insert(len(md) - 4, f"_{e['note']}_")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"examples_{stamp}.json").write_text(json.dumps(out, indent=1, default=str))
    (HERE / "results" / f"examples_{stamp}.md").write_text("\n".join(md))
    if not args.labels:
        (HERE / "results" / f"level7_blind_{stamp}.md").write_text("\n".join(blind))
    print(f"wrote results/examples_{stamp}.json, .md" + ("" if args.labels else f", level7_blind_{stamp}.md"))


if __name__ == "__main__":
    main()
