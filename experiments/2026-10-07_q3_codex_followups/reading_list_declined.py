"""Reading list: the 10 declined real attack (e1) runs the frozen Q3 detector scores highest (out-of-fold run score,
all-calls readout), with each run's 3 highest-scoring calls. What does the detector pick up on when there is no
sabotage? Writes data/reading_lists/reading_list_declined_<stamp>.md (git-ignored: it quotes transcripts; archive it to
S3), with passwords and seeded tokens in call arguments redacted.
Python 3.14 (reads the transcripts).

  python reading_list_declined.py --scores-dir <#36 results> --labels <attempt labels>
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
from probe_shift.transcripts import load_sample, reader_messages  # noqa: E402


SECRETS = [(re.compile(r"(PGPASSWORD=)('[^']*'|\"[^\"]*\"|\\?\"[^\"\\]*\\?\"|\S+)"), r"\1<redacted>"),
           (re.compile(r"(?i)((?:password|token|secret)\\?[\"']?\s*[:=]\s*\\?[\"']?)([^\s,'\"\\}]+)"), r"\1<redacted>"),
           (re.compile(r"(Bearer\s+)[A-Za-z0-9_\-.=]+"), r"\1<redacted>"), (re.compile(r"\batok_[A-Za-z0-9]+"), "<redacted-token>")]


def redact(text: str) -> str:
    for pat, sub in SECRETS:
        text = pat.sub(sub, text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--run", default="2026-10-06_181548")
    ap.add_argument("--sce", default=None, help="simulated-control-evals checkout (default: the sibling of this repository)")
    args = ap.parse_args()
    res = Path(args.scores_dir); d = res / f"run_scores_{args.run}"
    runs = json.loads((d / "runs.json").read_text()); t = np.load(d / "tables.npz")
    metrics = json.loads(sorted(res.glob(f"metrics_{args.run}__*.json"))[-1].read_text())
    layer = int(next(iter(set(metrics["readouts"]["probe_call_end_mean"]["task"]["chosen_layer_by_fold"].values()))))
    z = np.load(d / f"scores_L{layer}.npz")
    rows = z["call_end__rows"]; s = z["call_end__step__task_outer"].astype(float); r_run = t["call_end__run"][rows]
    run_score = z["call_end__mean__task_outer"].astype(float)
    lab = {(x["arm"], x["key"]): x for x in map(json.loads, Path(args.labels).read_text().splitlines())}
    honest = np.array([r["arm"] == "real-honest" for r in runs])
    pct = lambda v: float((run_score[honest] < v).mean())
    dec = [i for i, r in enumerate(runs) if r["arm"] == "real-attack-e1" and lab.get((r["arm"], r["key"]), {}).get("label") == "declined"]
    top = sorted(dec, key=lambda i: -run_score[i])[:10]
    ex = {(x["arm"], x["key"]): x for x in json.loads((REPO / "splits/exclusions_v4.json").read_text())["samples"]}
    sce = Path(args.sce) if args.sce else next(p for p in (REPO.parent / "simulated-control-evals", Path.home() / "simulated-control-evals") if p.exists())
    uris = yaml.safe_load((sce / "src/paper/configs/real_runs.yml").read_text())["runs"]
    out = ["# Reading list: declined attack runs the detector flags (2026-10-07)", "",
           f"Detector run {args.run}, all-calls readout (layer {layer}), out-of-fold scores. The 10 real attack (epoch 1) runs labelled",
           "declined (they report not attempting the side task) with the highest run scores. Percentile = share of real honest runs",
           "scoring lower. For each, the 3 highest-scoring calls. The reader saw the honest system prompt, not the attack one.", ""]
    for n, i in enumerate(top, 1):
        r = runs[i]; e = ex[(r["arm"], r["key"])]
        smp = load_sample(uris[e["cohort"]]["uri"], e["file"], e["sample_id"], e["epoch"])
        # the scores' step/action index reader_messages (submit calls and empty turns dropped), not the raw transcript
        turns = [m for m in reader_messages(smp, "") if m["role"] == "assistant"]
        goal = ((smp.get("metadata") or {}).get("side_task") or {}).get("goal", "").strip()
        idx = np.flatnonzero(r_run == i)
        best = idx[np.argsort(-s[idx])][:3]
        out += [f"## {n}. `{e['sample_id']}` (run score {run_score[i]:.2f}, above {100 * pct(run_score[i]):.0f}% of honest runs)", "",
                f"- Transcript: `{uris[e['cohort']]['uri']}logs/{e['file']}`, sample `{e['sample_id']}` epoch {e['epoch']}",
                f"- Side task (declined): {goal[:200]}",
                f"- Self-report: {lab[(r['arm'], r['key'])]['evidence'][:200]}", f"- {len(turns)} turns, {len(idx)} scored calls. Top calls:"]
        for j in best:
            st, ac = int(t["call_end__step"][rows[j]]), int(t["call_end__action"][rows[j]])
            c = turns[st]["tool_calls"][ac]["function"]
            out.append(f"  - step {st}.{ac}, score {s[j]:.2f}: `{c['name']}` {json.dumps(c['arguments'], ensure_ascii=False)[:300]}")
        out.append("")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "data" / "reading_lists" / f"reading_list_declined_{stamp}.md"  # quotes transcripts: git-ignored, archive to S3
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(redact("\n".join(out)))
    print("->", path)


if __name__ == "__main__":
    main()
