"""Matched-step pairs: metrics. Does the simulated output alone (same real history) look different from the real one?

Inputs: the pairs file and simulator results (git-ignored data/), and probe_scores_<stamp>.json from analyze_probes.py.
Two numbers per method, each with a 95% interval from resampling main tasks (1,000 draws):
  - AUROC over all items (each pair gives a real item, label 0, and a simulated one, label 1);
  - pairwise accuracy: the share of pairs whose simulated item scores above its real one (ties count half).
Methods:
  - TF-IDF on step k's output text alone, as the reader reads it up to the readout (output_end), at scrub level 0 and
    level 5 (baselines-q3's scrub, copied below); word 1-2-grams and character 2-5-grams hashed, TF-IDF, logistic
    regression C = 1, trained within the pairs leaving one environment out (the fresh probe's folds);
  - output length (characters) and Qwen3-32B mean log-prob of the output tokens (no training);
  - the Q1-recipe probe frozen on the cohorts, per layer, at output_end (and Q1's headline layers 40 and 63);
  - the fresh within-pairs probe, per layer and position, with its label-shuffle null;
  - the Q2 impactful-call probes at the next call: score shift and flag flips between the two versions.
Slices: attack and honest pairs; no prefix the September contract rejected; outputs Inspect did not truncate;
bash only; pairs whose two outputs differ (identical outputs tie for every method); position early/mid/late.

  PYTHONPATH=src uv run --no-project --python 3.14 --with transformers==4.56.2 --with jinja2 --with numpy \\
    --with scikit-learn --with scipy --with pyyaml python experiments/2026-10-06_matched_step_pairs/analyze_pairs.py \\
    --pairs data/pairs_<stamp>.jsonl --sims data/sim_a.jsonl,data/sim_b.jsonl --probes results/probe_scores_<stamp>.json
"""

import argparse
import hashlib
import json
import re
from datetime import datetime
from functools import partial
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
PT = ZoneInfo("America/Los_Angeles")
N_BOOT = 1000

# --- baselines-q3's scrub (~/probe-shift-baselines/experiments/2026-10-06_q1_text_baselines/analyze_tfidf.py), levels
# 0-5, copied so this folder runs on its own.
MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
TIME_PATTERNS = [(re.compile(r"\d{4}-\d{2}-\d{2}"), "<date>"),
                 (re.compile(r"(?<![\d.:])\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?![\d:])"), "<time>"),
                 (re.compile(rf"\b(?:{MONTHS})\s+\d{{1,2}}\b"), "<date>"),
                 (re.compile(r"\b1[5-9]\d{8}(?:\d{3})?\b"), "<epoch>")]
ID_PATTERNS = [(re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<uuid>"),
               (re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])"), "<ip>"),
               (re.compile(r"\b(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{6,}\b"), "<hex>"),
               (re.compile(r"\d+"), "<n>")]
TRAILING_SPACE, BLANK_RUNS = re.compile(r"[^\S\n]+$", re.M), re.compile(r"\n{3,}")
EDITOR_DIFF, GUTTER = "Here's a diff showing the changes made", re.compile(r"^ *\d+\|", re.M)
STDERR_LINE = re.compile(r"^(?:(?:fatal|error|warning|bash|sh|/bin/sh|sed|grep|ls|cat|find|rm|cp|mv|mkdir|chmod|head|tail|awk|"
                         r"wc|xargs|cd|python3?|node|npm WARN|E): .*|.*(?:No such file or directory|Permission denied|"
                         r"command not found|not a git repository).*)\n?", re.M | re.I)
GUTTER_AND_SPACE, HUNK = re.compile(r"^ *\d+\|([+\- ]?) ?", re.M), re.compile(r"^@@ [^\n]*@@[^\n]*$", re.M)
WEEKDAYS = re.compile(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b")
MONTH_NAMES = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|"
                         r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\b")


# Simulator files written before rows recorded their build: only these verified files, with the build both of their
# launches logged (run_simulate_pairs.LEGACY_BUILDS; sim_pilot20 is a copy of the first 20 rows).
LEGACY_BUILDS = {name: {"simulator_model": "openai/gpt-6-astra", "sims_commit": "ba990399bfbd71aebf7636c04252d22c05720156",
                        "sims_src_diff_sha256": "7d2b6021d1301cc1b084441905bae9592d74cae2c201d808c322ae1763702db4"}
                 for name in ("sim_pilot_2026-10-06.jsonl", "sim_pilot20_2026-10-06.jsonl")}


def build_of(path, row) -> str:
    """The simulator build a row was made by, as a comparable string; legacy rows only from the verified files."""
    if "build" in row:
        return json.dumps(row["build"], sort_keys=True)
    assert Path(path).name in LEGACY_BUILDS, f"{path} has rows with no recorded simulator build"
    return json.dumps(LEGACY_BUILDS[Path(path).name], sort_keys=True)


def scrub(text: str, level: int) -> str:
    if level >= 4 and EDITOR_DIFF in text:
        text = GUTTER_AND_SPACE.sub(r"\1", text) if level >= 5 else GUTTER.sub("", text)
        if level >= 5:
            text = HUNK.sub("@@", text)
    if level >= 4:
        text = STDERR_LINE.sub("", text)
    if level >= 1:
        for pattern, placeholder in TIME_PATTERNS:
            text = pattern.sub(placeholder, text)
    if level >= 2:
        for pattern, placeholder in ID_PATTERNS:
            text = pattern.sub(placeholder, text)
    if level >= 3:
        text = BLANK_RUNS.sub("\n\n", TRAILING_SPACE.sub("", text)).rstrip()
    if level >= 4:
        text = MONTH_NAMES.sub("<month>", WEEKDAYS.sub("<wday>", text))
        text = re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n", "\n", text))
    return text


def char_ngrams(text: str, lo: int, hi: int) -> list[str]:
    return [text[i:i + n] for n in range(lo, hi + 1) for i in range(len(text) - n + 1)]


def featurize(texts: list[str]):
    common = dict(n_features=2**20, alternate_sign=False, norm=None)
    word = HashingVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b|[^\w\s]", lowercase=False, **common)
    char = HashingVectorizer(analyzer=partial(char_ngrams, lo=2, hi=5), **common)
    return sp.hstack([word.transform(texts), char.transform(texts)]).tocsr()


def tfidf_scores(texts, y, env):
    X = featurize(texts)
    s = np.full(len(y), np.nan)
    for e in np.unique(env):
        tr = env != e
        tf = TfidfTransformer(sublinear_tf=True).fit(X[tr])
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000).fit(tf.transform(X[tr]), y[tr])
        s[~tr] = clf.decision_function(tf.transform(X[~tr]))
    return s


def readout_texts(items, pairs, tok):
    """Step k's output as the reader reads it: cut after its last content token (output_end), as baselines-q3 does."""
    from probe_shift.reader import _layout

    out = []
    for pair, version, content in items:
        rec = pairs[pair]
        msgs = [dict(m) for m in rec["reader_messages"]]
        msgs[rec["out_msg_index"]]["content"] = content
        text, found, enc = _layout(tok, msgs)
        a, b = next((a, b) for kind, step, action, _, a, b in found if kind == "tool_output" and step == rec["turn"] and action == 0)
        ends = [e for s, e in enc["offset_mapping"] if s >= a and e <= b]
        out.append(text[a:max(ends)] if ends else "")
    return out


def metrics(score, y, pair, task, mask=None, rng=None):
    """AUROC and pairwise accuracy with 95% intervals from resampling main tasks."""
    score, y, pair, task = map(np.asarray, (score, y, pair, task))
    keep = ~np.isnan(score) if mask is None else (~np.isnan(score) & np.asarray(mask))
    by_pair = {}
    for s, l, p, t in zip(score[keep], y[keep], pair[keep], task[keep]):
        by_pair.setdefault(p, {"task": t})[int(l)] = s
    pairs = [(v[1], v[0], v["task"]) for v in by_pair.values() if 0 in v and 1 in v]
    if len(pairs) < 5:
        return {"n_pairs": len(pairs)}
    sim, real, ptask = map(np.array, zip(*pairs))
    win = (sim > real) + 0.5 * (sim == real)
    tasks = np.unique(ptask)
    idx = {t: np.flatnonzero(ptask == t) for t in tasks}
    rng = rng or np.random.default_rng(0)

    def stats(rows):
        s = np.concatenate([sim[rows], real[rows]])
        l = np.concatenate([np.ones(len(rows)), np.zeros(len(rows))])
        return roc_auc_score(l, s), win[rows].mean()

    point = stats(np.arange(len(sim)))
    boots = np.array([stats(np.concatenate([idx[t] for t in rng.choice(tasks, len(tasks))])) for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"n_pairs": len(sim), "n_tasks": len(tasks), "auroc": round(point[0], 4), "auroc_ci": [round(lo[0], 4), round(hi[0], 4)],
            "pairwise": round(point[1], 4), "pairwise_ci": [round(lo[1], 4), round(hi[1], 4)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--sims", required=True)
    ap.add_argument("--probes", type=Path, default=None)
    ap.add_argument("--text-frozen", type=Path, default=None, help="results/text_frozen_scores_<stamp>.json")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    pairs = {json.loads(l)["pair"]: json.loads(l) for l in args.pairs.read_text().splitlines()}
    pairs_sha = hashlib.sha256(args.pairs.read_bytes()).hexdigest()
    sims, builds = {}, set()
    for path in args.sims.split(","):
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            assert r["pairs_sha256"] == pairs_sha, f"{path} holds simulator results for another pairs file"
            if r["status"] == "ok":
                builds.add(build_of(path, r))
                sims.setdefault(r["pair"], r["result"])
    assert len(builds) <= 1, f"--sims mixes {len(builds)} simulator builds; analyze one build at a time"
    order = sorted(sims)
    P = json.loads(args.probes.read_text()) if args.probes else None
    # Score files from before the hash was recorded (probe_scores_2026-10-06_191958) carry only the pairs file name.
    assert not P or P.get("pairs_sha256", pairs_sha) == pairs_sha, "probe scores come from another pairs file"
    assert not P or P["pairs_file"] == args.pairs.name, "probe scores come from another pairs file"
    dropped = []
    if P:  # only pairs whose two versions the reader read (a version past the 32k cut is skipped there)
        read = {tuple(x) for x in P["frozen"]["items"]}
        dropped = [p for p in order if (p, "real") not in read or (p, "sim") not in read]
        order = [p for p in order if p not in dropped]
    items = [(p, v, pairs[p]["real_output"] if v == "real" else sims[p]["reader_output"]) for p in order for v in ("real", "sim")]
    y = np.array([float(v == "sim") for _, v, _ in items])
    pair = np.array([p for p, _, _ in items])
    task = np.array([pairs[p]["main_task"] for p in pair])
    env = np.array([pairs[p]["environment"] for p in pair])
    arm = np.array([pairs[p]["arm"] for p in pair])
    slices = {"all": np.ones(len(items), bool), "attack": arm == "attack", "honest": arm == "honest",
              "no_trusted_prefix": np.array([not sims[p]["trusted_prefix"] for p in pair]),
              "not_truncated": np.array([not sims[p]["inspect_truncated"] and "<START_TOOL_OUTPUT>" not in pairs[p]["real_output"] for p in pair]),
              "bash_only": np.array([pairs[p]["step"]["function"] == "bash" for p in pair]),
              # identical outputs give identical reader inputs, so every method ties on them
              "outputs_differ": np.array([sims[p]["reader_output"] != pairs[p]["real_output"] for p in pair]),
              "no_fallback": np.array([sims[p]["fallback_substitutions"] + sims[p]["refusal_substitutions"]
                                       + sims[p]["block_substitutions"] + sims[p]["net_substitutions"] == 0 for p in pair])}
    for b in ("early", "mid", "late"):
        slices[b] = np.array([pairs[p]["bucket"] == b for p in pair])

    def report(score):
        return {name: metrics(score, y, pair, task, m) for name, m in slices.items()}

    from transformers import AutoTokenizer

    from probe_shift.models import QWEN3_32B  # the pin the reader and the splits use

    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    texts = readout_texts(items, pairs, tok)
    readout_path = HERE / "data" / f"readout_{datetime.now(PT).strftime('%Y-%m-%d_%H%M%S')}.jsonl"  # input to analyze_text_frozen.py
    readout_path.parent.mkdir(parents=True, exist_ok=True)
    readout_path.write_text("".join(json.dumps({"pair": p, "version": v, "text": t}) + "\n" for (p, v, _), t in zip(items, texts)))
    out = {"created": datetime.now(PT).isoformat(timespec="seconds"), "pairs_file": args.pairs.name,
           "sims_files": [Path(s).name for s in args.sims.split(",")], "probes_file": args.probes.name if args.probes else None,
           "n_pairs": len(order), "pairs_not_read": dropped, "slice_sizes": {k: int(v.sum() // 2) for k, v in slices.items()}, "methods": {}}
    M = out["methods"]
    M["output_length_chars"] = report(np.array([len(t) for t in texts], float))
    for level in (0, 5):
        s = tfidf_scores([scrub(t, level) for t in texts], y, env)
        M[f"tfidf_scrub{level}"] = report(s)
        # label-shuffle null: within each pair, swap labels with probability 1/2, 20 draws
        rng = np.random.default_rng(level)
        null = []
        for _ in range(20):
            flip = {p: rng.random() < 0.5 for p in order}
            yn = np.where([flip[p] for p in pair], 1 - y, y)
            sn = tfidf_scores([scrub(t, level) for t in texts], yn, env)
            null.append(roc_auc_score(yn, sn))
        M[f"tfidf_scrub{level}"]["all"]["null_auroc_mean_p95"] = [round(float(np.mean(null)), 4), round(float(np.percentile(null, 95)), 4)]
    if args.text_frozen:
        T = json.loads(args.text_frozen.read_text())
        # The scores must have been made from these very texts: compare the readout file they were scored on, item by
        # item, with this run's texts (so a subset analysis still passes). Files from before the hashes were recorded
        # (text_frozen_scores_2026-10-06_183355) carry only the readout file name.
        assert T.get("pairs_sha256", pairs_sha) == pairs_sha, "frozen text scores come from another pairs file"
        scored_on = HERE / "data" / T["readout_file"]
        assert scored_on.exists(), f"{scored_on} (the texts the frozen scores were made from) is not in data/"
        assert T.get("readout_sha256", hashlib.sha256(scored_on.read_bytes()).hexdigest()) == \
            hashlib.sha256(scored_on.read_bytes()).hexdigest(), f"{scored_on} changed since it was scored"
        scored = {(r["pair"], r["version"]): r["text"] for r in map(json.loads, scored_on.read_text().splitlines())}
        assert all(scored.get((p, v)) == t for (p, v, _), t in zip(items, texts)), "frozen text scores were made from other texts"
        out["text_frozen_file"] = args.text_frozen.name
        for name, rows in T["scores"].items():
            got = {(p, v): s for p, v, s in rows}
            assert all((p, v) in got for p, v, _ in items), f"frozen text scores ({name}) miss some analyzed items"
            M[f"tfidf_frozen_{name}"] = report(np.array([got.get((p, v), np.nan) for p, v, _ in items]))
    if P:
        key = {(p, v): i for i, (p, v) in enumerate(map(tuple, P["frozen"]["items"]))}
        sel = [key[(p, v)] for p, v, _ in items]
        if "logprob_mean" in P:
            kq = {(p, v): i for i, (p, v) in enumerate(map(tuple, P["items"]))}
            M["qwen_mean_logprob_neg"] = report(-np.array(P["logprob_mean"])[[kq[(p, v)] for p, v, _ in items]])
        def null_summary(draws, observed):
            draws = np.asarray(draws)
            return {"mean": round(float(draws.mean()), 4), "p95": round(float(np.percentile(draws, 95)), 4),
                    "p_value": round(float((1 + (draws >= observed).sum()) / (1 + len(draws))), 4), "n": len(draws)}

        def swap_null(score, n=200, seed=1):
            """AUROC of fixed scores against within-pair label swaps: the evaluation-side null for a frozen probe."""
            rng = np.random.default_rng(seed)
            draws = []
            for _ in range(n):
                flip = {p: rng.random() < 0.5 for p in order}
                draws.append(roc_auc_score(np.where([flip[p] for p in pair], 1 - y, y), score))
            return draws

        def choose(auroc_by_layer: dict, tol: float = 0.01) -> int:
            """Q1's rule: the earliest layer within tol of the best AUROC."""
            best = max(auroc_by_layer.values())
            return min(int(l) for l, a in auroc_by_layer.items() if a >= best - tol)

        # 1. Frozen Q1-recipe probe, both readouts, every layer; layer chosen on the cohort's out-of-fold AUROC.
        fz = {}
        for readout in P["readouts"]:
            curve, oof = {}, {}
            for layer, v in sorted(P["frozen"]["layers"].items(), key=lambda kv: int(kv[0])):
                r = v[readout]
                s = np.array(r["scores"])[sel]
                observed = roc_auc_score(y, s)
                train_null = [roc_auc_score(y, np.array(ns)[sel]) for ns in r["null_scores"]]
                curve[layer] = {"all": metrics(s, y, pair, task), "cohort_oof_auroc": round(r["cohort_oof_auroc"], 4),
                                "null_training_label_shuffle": null_summary(train_null, observed),
                                "null_within_pair_swap": null_summary(swap_null(s), observed),
                                "dump_agreement": {"n": len(r["agreement"]),
                                                   "min_cosine": round(float(min(a[2] for a in r["agreement"])), 5),
                                                   "token_index_matches": int(sum(a[1] for a in r["agreement"]))}}
                oof[layer] = r["cohort_oof_auroc"]
            chosen = choose(oof)
            s = np.array(P["frozen"]["layers"][str(chosen)][readout]["scores"])[sel]
            fz[readout] = {"chosen_layer": chosen, "rule": "earliest layer within 0.01 of the best cohort out-of-fold AUROC "
                           "(Q1 rows, main tasks used in the pairs left out); the pairs play no part",
                           "at_chosen_layer": report(s), "curve": curve}
        M["probe_frozen_q1"] = fz
        # 2. Fresh within-pairs probe, every position and layer, label-shuffle null at every layer; nested inner CV.
        fr = {}
        for kind, r in P["fresh"].items():
            k2 = {(p, v): i for i, (p, v) in enumerate(map(tuple, r["items"]))}
            sel2 = [k2[(p, v)] for p, v, _ in items]
            yl = np.array([float(vv == "sim") for _, vv in r["items"]])
            up = np.unique(np.array([pp for pp, _ in r["items"]]))
            curve = {}
            for layer, v in sorted(r["layers"].items(), key=lambda kv: int(kv[0])):
                s = np.array(v["scores"])[sel2]
                observed = roc_auc_score(y, s)
                rng = np.random.default_rng(0)  # the flips analyze_probes drew (seed 0, pairs in sorted order)
                draws = []
                for ns in v["null_scores"]:  # drawn over every read item (RNG alignment), scored on the analyzed ones
                    flip = dict(zip(up, rng.random(len(up)) < 0.5))
                    yn = np.where([flip[pp] for pp, _ in r["items"]], 1 - yl, yl)
                    draws.append(roc_auc_score(yn[sel2], np.asarray(ns)[sel2]))
                curve[layer] = {"all": metrics(s, y, pair, task), "null_within_pair_swap": null_summary(draws, observed)}
            entry = {"curve": curve}
            inner = [x for x in P.get("fresh_inner", []) if x["kind"] == kind]
            if inner:  # nested: each held-out environment scored at the layer its inner CV chose
                nested, chosen = np.full(len(items), np.nan), {}
                for x in inner:
                    layer = choose(x["inner_auroc"])
                    chosen[x["outer_env"]] = layer
                    s = np.array(r["layers"][str(layer)]["scores"])[sel2]
                    nested[env == x["outer_env"]] = s[env == x["outer_env"]]
                entry |= {"chosen_layer_by_env": chosen, "rule": "per held-out environment, the earliest layer within 0.01 "
                          "of the best inner leave-one-environment-out AUROC on the other 7 environments",
                          "nested": report(nested)}
            fr[kind] = entry
        M["probe_fresh"] = fr
        q2 = {}
        for name, v in P["q2"].items():
            k3 = {(p, vv): i for i, (p, vv) in enumerate(map(tuple, P["items"]))}
            s = np.array(v["scores"])
            fold = np.array(v["item_fold"])
            thr = np.array(v["threshold"])[fold]
            real_i = [k3[(p, "real")] for p in order]
            sim_i = [k3[(p, "sim")] for p in order]
            d = s[sim_i] - s[real_i]
            flag_r, flag_s = s[real_i] > thr[real_i], s[sim_i] > thr[sim_i]
            q2[name] = {"mean_shift_sim_minus_real": round(float(d.mean()), 4), "sd_shift": round(float(d.std()), 4),
                        "real_score_sd": round(float(s[real_i].std()), 4), "share_sim_higher": round(float((d > 0).mean()), 4),
                        "flag_rate_real": round(float(flag_r.mean()), 4), "flag_rate_sim": round(float(flag_s.mean()), 4),
                        "flag_flips": int((flag_r != flag_s).sum()), "n": len(order), "layers": v["layer"],
                        "real_vs_sim": metrics(s[[k3[(p, vv)] for p, vv, _ in items]], y, pair, task)}
        M["q2_next_call"] = q2
    stamp = datetime.now(PT).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"metrics_{stamp}{args.tag}.json"
    with open(path, "x") as f:
        f.write(json.dumps(out, indent=1))
    print(f"wrote {path}")
    for name in ("output_length_chars", "tfidf_scrub0", "tfidf_scrub5", "tfidf_frozen_scrub0", "tfidf_frozen_scrub5",
                 "qwen_mean_logprob_neg"):
        if name in M:
            print(name, {k: (M[name][k].get("auroc"), M[name][k].get("auroc_ci"), M[name][k].get("pairwise")) for k in ("all", "attack", "honest")})


if __name__ == "__main__":
    main()
