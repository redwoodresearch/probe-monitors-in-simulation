"""POST HOC, descriptive (RULE.md Amendment 1 @ 266ecf07, items 1 and 3, committed before this script). Item 1: the character
1-5-gram classifier rerun on replies with times and dates masked (probe_shift.scrub level 1) after the repairs: fake vs
first real, second real vs first real as the control. Item 3: a planted positive control on the masked replies: the
real tool's review note appended to a seeded 28 of the 91 first-real replies, classified against the second-real
replies. Same sites, repairs, classifier and null as analyze_POSTHOC_fake_reply_diffs.py (imported unchanged). $0, local.
Writes results/masked_planted_POSTHOC_<PT stamp>.json: numbers and n-grams of at most 5 characters, no reply text.

  PYTHONPATH=src uv run --no-project --python 3.12 --with numpy --with scipy --with scikit-learn --with joblib \\
      --with pyyaml --with 'modal>=1.4.3' python \\
      experiments/2026-10-13_claimB_fake_reply_diffs/analyze_POSTHOC_masked_planted.py --data <meaning judge data dir>
"""

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_POSTHOC_fake_reply_diffs as F  # noqa: E402  (load, repair, classifier: unchanged)
import rerender  # noqa: E402  (REVIEW: the real tool's review note; on sys.path via fixes.py)
from probe_shift.scrub import scrub  # noqa: E402

PLANTED_N, SEED = 28, 20261013
INSERT = "\n" + rerender.REVIEW
# clock cues the level-1 scrub leaves (Amendment 1, item 1): counted per world after masking
MON = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
LEFTOVER = {"weekday": re.compile(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b"), "year": re.compile(r"\b(?:19|20)\d\d\b"),
            "zone": re.compile(r"\b(?:UTC|GMT|PST|PDT|EST|EDT|CET|CEST|JST)\b"), "day_month": re.compile(rf"\b\d{{1,2}} (?:{MON})\b")}


def plant(t: str) -> str:
    """The review note on its own line after the reply's last non-newline character; the trailing newlines stay."""
    body = t.rstrip("\n")
    return body + INSERT + t[len(body):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="the meaning judge's data dir (sites.jsonl, sim_new.jsonl)")
    args = ap.parse_args()
    for f in (Path(__file__).resolve(), Path(F.__file__).resolve()):
        if subprocess.run(["git", "status", "--porcelain", str(f)], capture_output=True, text=True, check=True, cwd=F.REPO).stdout:
            raise SystemExit(f"commit {f.name} first: the result records the commit that ran")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=F.REPO).stdout.strip()
    sites = F.load(args.data)
    tasks = sorted({s["task"] for s in sites})
    groups = np.array([tasks.index(s["task"]) for s in sites])
    after = {w: [] for w in ("sim", "e1", "e2")}
    for s in sites:
        for w in after:
            t, _, has_range = F.repair(s["text"][w], s["command"])
            assert not has_range  # the first run dropped no site (RULE.md); stop if that changed
            after[w].append(t)
    masked = {w: [scrub(t, 1) for t in ts] for w, ts in after.items()}
    out = {"label": "POST HOC, descriptive, preliminary until reviewed", "rule": "RULE.md Amendment 1 (items 1 and 3)",
           "git": git, "input_digest": {f: hashlib.sha256((args.data / f).read_bytes()).hexdigest() for f in ("sites.jsonl", "sim_new.jsonl")},
           "sites": len(sites), "tasks": len(tasks), "shuffles": F.SHUFFLES,
           "replies_changed_by_mask": {w: int(sum(a != m for a, m in zip(after[w], masked[w]))) for w in after},
           "identical_after_mask": {"sim_eq_e1": int(sum(a == b for a, b in zip(masked["sim"], masked["e1"]))),
                                    "e2_eq_e1": int(sum(a == b for a, b in zip(masked["e2"], masked["e1"])))},
           "leftover_clock_cues_after_mask": {w: {k: int(sum(len(rx.findall(t)) for t in ts)) for k, rx in LEFTOVER.items()}
                                              for w, ts in masked.items()},
           "classifier": {}}
    for comp, (a, names) in {"fake_vs_e1|masked": ("sim", ("fake", "e1")), "e2_vs_e1|masked": ("e2", ("e2", "e1"))}.items():
        out["classifier"][comp] = F.classifier(masked[a], masked["e1"], groups, names)

    chosen = sorted(np.random.default_rng(SEED).choice(len(sites), PLANTED_N, replace=False).tolist())
    planted = [plant(t) if i in chosen else t for i, t in enumerate(masked["e1"])]
    res = F.classifier(planted, masked["e2"], groups, ("e1_planted", "e2"))
    top = res["top_toward_e1_planted"][0]
    top_gram = ast.literal_eval(top["ngram"])  # the repr() of a str of at most 5 characters
    res["planted_sites"] = chosen
    res["insert_chars"] = len(INSERT)
    res["top_feature_in_insert"] = top_gram in INSERT
    res["top_feature_in_first_real_before_planting"] = int(sum(t.count(top_gram) for t in masked["e1"]))
    res["top_feature_in_second_real"] = int(sum(t.count(top_gram) for t in masked["e2"]))
    res["pass_top_feature"] = bool(res["top_feature_in_insert"] and res["top_feature_in_first_real_before_planting"] == 0
                                   and res["top_feature_in_second_real"] == 0)
    res["pass_auroc"] = res["above_null_p95"]
    res["pass"] = res["pass_top_feature"] and res["pass_auroc"]
    out["classifier"]["planted_e1_vs_e2|masked"] = res
    for k, c in out["classifier"].items():
        print(f"{k}: AUROC {c['auroc']} null mean {c['null_mean']} p95 {c['null_p95']} p {c['p']}", flush=True)
    print("planted pass:", res["pass"], "(top feature", res["pass_top_feature"], "auroc", res["pass_auroc"], ")")
    print("changed by mask:", out["replies_changed_by_mask"], "identical after mask:", out["identical_after_mask"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"masked_planted_POSTHOC_{stamp}.json"
    with open(path, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
