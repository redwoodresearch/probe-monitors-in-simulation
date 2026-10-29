"""POST HOC (RULE.md Amendment 1): do Go content tabs track the fix-full probe's scores? Counts and statistics only.

One Modal CPU container computes per-run Go-tab flags from the staged fix-full shards on the activations Volume, with the
census's reply rule, and joins them with d9's per-run scores. Nothing is computed on the laptop.
- (i)-(iv) for the 437 fix-full simulated honest runs (shard_12) joined with d9's honest CSV;
- (v): the boolean-only pass over shard_10's text_editor replies (per-run flags only, to the Volume), joined with d9's
  per-run attack scores. Both CSVs are baked into the image from local copies (TAB_HONEST_CSV, TAB_ATTACK_CSV), and
  sha-checked locally and in the container.
Results go to the Volume under claimB/tab_census_v1/; only the result JSON is fetched.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-13_claimB_tab_census/run_tab_scores_modal.py   (with TAB_HONEST_CSV and TAB_ATTACK_CSV set)
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-claimb-tab-census_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
FX = "2026-10-13_q2_fixfull_inputs_v2/inputs"
HONEST = ("shard_12.jsonl", "94ae36d6d3bbd420d607254fd7c389910aee80b512d2281e19d2f08ae693b799")  # stage record @ 82909df0
ATTACK = ("shard_10.jsonl", "5a3874814163b21e1c20dfedd00f1473dc01d2151b9769bb2930a1c16ae05652")
HONEST_CSV_SHA = "0cf45b7e4d43c278d06a576485f53222624eed4537c9cd9d83a5d72b173a64c4"  # d9's honest CSV @ 42c3737a
ATTACK_CSV_SHA = "7d127a801e559462cb1c2f3f835d0739c018efc20070cdccf1a162be4603690e"  # d9's attack CSV @ bad19721
OUT = "claimB/tab_census_v1"
SEED, DRAWS, TOP = 20261013, 2000, 44

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.1.3")
if modal.is_local():  # local copies of d9's two CSVs, baked into the image at build time (sha-checked again inside)
    image = (image.add_local_file(os.environ["TAB_HONEST_CSV"], "/root/scores_honest.csv", copy=True)
             .add_local_file(os.environ["TAB_ATTACK_CSV"], "/root/scores_attack.csv", copy=True))


def run_flags(path: str, want_sha: str) -> tuple[dict, dict]:
    """Per run key: has_go_reply, has_go_content_tab, n_go_replies, n_go_content_tab, from text_editor replies only (the
    census's rule: a tab outside one leading cat -n gutter tab). Any failure on a row (decoding, JSON, shape) is counted
    and the row skipped, never echoed: rows may hold run text."""
    import re

    gutter = re.compile(r"^(\s*\d+)\t")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != want_sha:
        raise SystemExit("shard sha256 mismatch: stop")
    flags, errs = {}, {"row_failures": 0, "pairing_mismatches": 0}
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        try:
            r = json.loads(line.decode("utf-8"))
            key = str(r["key"])
            n_go = n_tab = mis = 0
            pending = []
            for m in r["messages"]:
                if m["role"] == "assistant":
                    pending = [dict(tc["function"]) for tc in (m.get("tool_calls") or [])]
                elif m["role"] == "tool":
                    call = pending.pop(0) if pending else None
                    if call is None or call.get("name") != m.get("tool_name"):
                        mis += 1
                        continue
                    if call.get("name") != "text_editor":
                        continue
                    args = call.get("arguments") or {}
                    if isinstance(args, str):
                        args = json.loads(args)
                    if not str(dict(args).get("path") or "").endswith(".go"):
                        continue
                    text = m["content"] if isinstance(m["content"], str) else ""
                    n_go += 1
                    n_tab += any("\t" in gutter.sub(r"\1", x, count=1) for x in text.split("\n"))
        except Exception:  # counted, never echoed
            errs["row_failures"] += 1
            continue
        errs["pairing_mismatches"] += mis
        flags[key] = {"has_go_reply": n_go > 0, "has_go_content_tab": n_tab > 0, "n_go_replies": n_go, "n_go_content_tab": n_tab}
    return flags, errs


def auroc(pos, neg) -> float | None:
    import numpy as np

    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def hyper_p(x: int, N: int, K: int, n: int) -> float:
    from math import comb

    return float(sum(comb(K, i) * comb(N - K, n - i) for i in range(x, min(n, K) + 1)) / comb(N, n))


def layer_stats(rows: list[dict], flag: str) -> dict:
    """rows: one per run with score_mean, caught_1pct, main_task, run_key and the flags."""
    import numpy as np

    with_ = [r["score_mean"] for r in rows if r[flag]]
    without = [r["score_mean"] for r in rows if not r[flag]]
    out = {"n_with": len(with_), "n_without": len(without),
           "mean_with": float(np.mean(with_)) if with_ else None, "mean_without": float(np.mean(without)) if without else None,
           "median_with": float(np.median(with_)) if with_ else None,
           "median_without": float(np.median(without)) if without else None, "auroc": auroc(with_, without)}
    tasks = sorted({r["main_task"] for r in rows})
    by = {t: [r for r in rows if r["main_task"] == t] for t in tasks}
    rng, draws = np.random.default_rng(SEED), []
    for _ in range(DRAWS):
        sample = [r for t in rng.choice(len(tasks), len(tasks)) for r in by[tasks[t]]]
        a = auroc([r["score_mean"] for r in sample if r[flag]], [r["score_mean"] for r in sample if not r[flag]])
        if a is not None:
            draws.append(a)
    out["auroc_ci95"] = [float(x) for x in np.percentile(draws, [2.5, 97.5])] if draws else None
    out["bootstrap_draws_used"] = len(draws)
    top = sorted(rows, key=lambda r: (-r["score_mean"], r["run_key"]))[:TOP]
    k = sum(r[flag] for r in rows)
    x = sum(r[flag] for r in top)
    out["top44"] = {"with_flag": x, "base_count": k, "N": len(rows), "base_rate": k / len(rows),
                    "expected": TOP * k / len(rows), "hypergeom_p_ge": hyper_p(x, len(rows), k, TOP)}
    caught = [r for r in rows if r["caught_1pct"] > 0]
    w = sum(r["caught_1pct"] for r in rows)
    out["caught_1pct"] = {"runs_caught": len(caught), "flag_share_among_caught": (sum(r[flag] for r in caught) / len(caught)) if caught else None,
                          "flag_share_weighted": (sum(r["caught_1pct"] * r[flag] for r in rows) / w) if w else None,
                          "flag_share_all": k / len(rows), "sum_weights": w}
    return out


def caught_stats(rows: list[dict], flag: str) -> dict:
    caught = [r for r in rows if r["caught_1pct"] > 0]
    w = sum(r["caught_1pct"] for r in rows)
    k = sum(r[flag] for r in rows)
    return {"runs": len(rows), "runs_caught": len(caught),
            "flag_share_among_caught": (sum(r[flag] for r in caught) / len(caught)) if caught else None,
            "flag_share_weighted": (sum(r["caught_1pct"] * r[flag] for r in rows) / w) if w else None,
            "flag_share_all": k / len(rows), "flag_count_all": k, "sum_weights": w}


def read_scores(path: str) -> dict:
    import csv

    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[(r["run_key"], int(r["layer"]))] = {"run_key": r["run_key"], "main_task": r["main_task"],
                                                    "score_mean": float(r["score_mean"]), "caught_1pct": float(r["caught_1pct"])}
    return out


@app.function(image=image, volumes={"/vol": acts}, cpu=4, memory=8192, timeout=1800, retries=0)
def honest(git: str, script_sha256: str) -> dict:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    for p, want in (("/root/scores_honest.csv", HONEST_CSV_SHA), ("/root/scores_attack.csv", ATTACK_CSV_SHA)):
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != want:
            raise SystemExit("scores CSV sha256 mismatch: stop")
    flags, errs = run_flags(f"/vol/{FX}/{HONEST[0]}", HONEST[1])
    if errs["row_failures"]:
        raise SystemExit("honest shard row failures: stop")
    scores = read_scores("/root/scores_honest.csv")
    res = {"label": "POST HOC (RULE.md Amendment 1), preliminary until reviewed", "git": git, "script_sha256": script_sha256,
           "honest_flags_errors": errs, "honest_flag_counts": {f: sum(v[f] for v in flags.values()) for f in ("has_go_reply", "has_go_content_tab")},
           "runs": len(flags), "layers": {}}
    for layer in (63, 48):
        rows = [scores[(k, layer)] | flags[k] for k in flags if (k, layer) in scores]
        if len(rows) != 437:
            raise SystemExit(f"honest join count at L{layer}: {len(rows)}")
        res["layers"][f"L{layer}"] = {"join_count": len(rows), "by_go_content_tab": layer_stats(rows, "has_go_content_tab"),
                                      "by_go_reply (confound check)": layer_stats(rows, "has_go_reply")}
    # (v): the boolean-only attack pass (per-run flags to the Volume, no text in any output), then the join with d9's
    # attack scores, which arrived before this run (Amendment 1's note)
    aflags, aerrs = run_flags(f"/vol/{FX}/{ATTACK[0]}", ATTACK[1])
    ascores = read_scores("/root/scores_attack.csv")
    res["attack_v"] = {}
    for layer in (63, 48):
        rows = [ascores[(k, layer)] | aflags[k] for k in aflags if (k, layer) in ascores]
        if len(rows) != 939:
            raise SystemExit(f"attack join count at L{layer}: {len(rows)}")
        res["attack_v"][f"L{layer}"] = {"join_count": len(rows), "by_go_content_tab": caught_stats(rows, "has_go_content_tab"),
                                        "by_go_reply (confound check)": caught_stats(rows, "has_go_reply")}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = Path("/vol", OUT)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"attack_flags_{stamp}.json", "x") as fh:
        fh.write(json.dumps({"source": f"{FX}/{ATTACK[0]}", "sha256": ATTACK[1], "errors": aerrs, "flags": aflags}) + "\n")
    res["attack_flags"] = {"path": f"{OUT}/attack_flags_{stamp}.json", "runs": len(aflags), "errors": aerrs,
                           "counts": {f: sum(v[f] for v in aflags.values()) for f in ("has_go_reply", "has_go_content_tab")}}
    name = f"tab_scores_POSTHOC_{stamp}.json"
    with open(out / name, "x") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")
    acts.commit()
    return {"path": f"{OUT}/{name}", "summary": {L: {"auroc": v["by_go_content_tab"]["auroc"], "ci95": v["by_go_content_tab"]["auroc_ci95"],
                                                     "top44": v["by_go_content_tab"]["top44"], "caught": v["by_go_content_tab"]["caught_1pct"],
                                                     "go_reply_auroc": v["by_go_reply (confound check)"]["auroc"]}
                                                 for L, v in res["layers"].items()},
            "flags": res["honest_flag_counts"], "errors": errs, "attack_flags": res["attack_flags"],
            "attack_v": {L: {"tab": v["by_go_content_tab"], "go_reply": v["by_go_reply (confound check)"]} for L, v in res["attack_v"].items()}}


@app.local_entrypoint()
def main():
    for var, want in (("TAB_HONEST_CSV", HONEST_CSV_SHA), ("TAB_ATTACK_CSV", ATTACK_CSV_SHA)):
        if hashlib.sha256(Path(os.environ[var]).read_bytes()).hexdigest() != want:
            raise SystemExit(f"{var}: local CSV sha256 mismatch: stop")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(HERE)], capture_output=True, text=True, cwd=HERE).stdout
    if dirty.strip():
        raise SystemExit("commit the script before running it")
    out = honest.remote(git, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(out, indent=1))
