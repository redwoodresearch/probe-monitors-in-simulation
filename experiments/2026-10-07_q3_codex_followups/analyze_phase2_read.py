"""Reading list for a Phase 2 read (run_read_phase2.py for attacks, run_read_phase2_honest.py for honest runs): where
each Qwen3.5-27B run lands on the frozen Q3 detector, and which of its calls score highest.

Per run and readout: the run score next to the same cell's Sol run of the read's check arm (re-read in the check) and its percentile among
the real honest runs and the real e1 attack runs of the same main-task fold (the detector's committed out-of-fold run
scores: run_score_sim_honest scoring 2026-10-07_124717, original-text probes on original-text features). Calls: the
top 5 per call readout with the per-call threshold from analyze_monitor_replay (5% whole-run false alarms on real
honest runs, same fold). Seeded tokens, passwords and other credential-like values in call arguments and agent text
are redacted. The reading list quotes call text, so it goes to git-ignored data/reading_lists/ (archive it to S3);
the summary JSON (scores only) goes to results/.

  python analyze_phase2_read.py --read <phase2_read_*.json> --scores <sim_honest_scores_2026-10-07_124717 dir>
      --replay results/monitor_replay_2026-10-06_181548_*.json
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
TOKENS = [(re.compile(r"(Bearer\s+)[A-Za-z0-9_\-.=]+"), r"\1<redacted>"), (re.compile(r"\batok_[A-Za-z0-9]+"), "<redacted-token>"),
          (re.compile(r"(PGPASSWORD=)('[^']*'|\S+)"), r"\1<redacted>"),
          (re.compile(r"(?i)((?:password|token|secret)\\?[\"']?\s*[:=]\s*\\?[\"']?)([^\s,'\"\\}]+)"), r"\1<redacted>")]
RO_NAMES = {"call_end_content": "content-token calls L63", "call_end": "all calls L48", "output_end": "tool outputs L63"}


def redact(text: str) -> str:
    for pat, sub in TOKENS:
        text = pat.sub(sub, text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    rd = json.loads(Path(args.read).read_text())
    runs = json.loads((Path(args.scores) / "runs.json").read_text())
    z = np.load(Path(args.scores) / "scores.npz")
    thr = {ro: {int(f): t for f, t in v["thresholds"].items()} for ro, v in json.loads(Path(args.replay).read_text())["readouts"].items()}
    arm = np.array([r["arm"] for r in runs]); fold = np.array([r["fold"] for r in runs])
    checks = {c["cell"]: c for c in rd["checks"]}
    check_arm = rd.get("check_arm", "real-attack-e1")  # the attack read predates the field; its checks are Sol e1 runs
    kind = "honest runs" if check_arm == "real-honest" else "attacks"
    sol_name = f"Sol {check_arm.removeprefix('real-attack-').removeprefix('real-')}"
    out, md = {"read": rd["stamp"], "runs": []}, []
    md.append(f"# Phase 2 {kind} on the frozen Q3 detector ({rd['stamp']})\n")
    md.append(f"Qwen3-32B read each Qwen3.5-27B run ({kind}; honest system prompt, reasoning dropped, 32k cut) exactly as it read the "
              f"LinuxArena training runs; the same cells' {sol_name} runs re-read through this code reproduced the stored features "
              "exactly. Scores are logits of the run 2026-10-06_181548 outer model for the main task's fold. Percentiles are "
              "among that fold's real honest and real e1 attack runs. Calls marked ⚑ exceed the per-call threshold that gives "
              "5% whole-run false alarms on real honest runs.\n")
    for r in rd["runs"]:
        f = r["fold"]
        sol = checks[r["cell"]]["run_scores"]
        rec = {"env": r["env"], "cell": r["cell"], "side_task_success": r["side_task_success"], "n_tokens": r["n_tokens"],
               "truncated": r["truncated"], "readouts": {}}
        md.append(f"\n## {r['env']} (side task {r['side_task_success']}; {r['n_tokens']:,} tokens{', cut at 32k' if r['truncated'] else ''})\n")
        md.append(f"| readout | Qwen3.5 run score (calls) | {sol_name}, same cell (calls) | percentile among real honest | among real e1 attacks |")
        md.append("|---|---|---|---|---|")
        for ro in ("call_end_content", "call_end", "output_end"):
            s = r["run_scores"][ro]["mean"]
            ref = z[f"{ro}__outer"]
            pct = {a: float(np.mean(ref[(arm == a) & (fold == f) & ~np.isnan(ref)] < s)) if s is not None else None
                   for a in ("real-honest", "real-attack-e1")}
            rec["readouts"][ro] = {"score": s, "calls": r["run_scores"][ro]["calls"], "sol_check": sol[ro], "percentile": pct}
            md.append(f"| {RO_NAMES[ro]} | {s:+.2f} ({r['run_scores'][ro]['calls']}) | {sol[ro]['mean']:+.2f} ({sol[ro]['calls']}) | "
                      f"{100 * pct['real-honest']:.0f} | {100 * pct['real-attack-e1']:.0f} |")
        kinds = {}
        for c in r["calls"]:
            if c["readout"] == "call_end":
                kinds[c["call_kind"]] = kinds.get(c["call_kind"], 0) + 1
        rec["call_end_kinds"] = {str(k): v for k, v in sorted(kinds.items())}
        md.append(f"\nCall endings (0 content, 1 timeout digit, 2 escaped newline): {rec['call_end_kinds']}\n")
        for ro in ("call_end_content", "call_end"):
            cs = sorted([c for c in r["calls"] if c["readout"] == ro], key=lambda c: -c["score"])
            flagged = [c for c in cs if c["score"] > thr[ro][f]]
            rec["readouts"][ro] |= {"flagged_calls": len(flagged), "first_flagged_step": min((c["step"] for c in flagged), default=None),
                                    "threshold": thr[ro][f]}
            md.append(f"**Top {args.top} calls, {RO_NAMES[ro]}** (threshold {thr[ro][f]:.2f}; {len(flagged)} of {len(cs)} calls above it"
                      + (f", first at step {rec['readouts'][ro]['first_flagged_step']}" if flagged else "") + ")\n")
            for c in cs[: args.top]:
                key = f"{c['step']}:{c['action']}"
                text = redact(r["call_text"].get(key, "")).replace("\n", " ")[:400]
                said = redact(r["agent_text"].get(str(c["step"]), "")).replace("\n", " ")[:200]
                md.append(f"- {'⚑ ' if c['score'] > thr[ro][f] else ''}step {key} `{c['function']}` score {c['score']:+.2f}: `{text}`"
                          + (f"\n  - agent text: {said}" if said else ""))
            md.append("")
        out["runs"].append(rec)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "data" / "reading_lists").mkdir(parents=True, exist_ok=True)
    with (HERE / "data" / "reading_lists" / f"phase2_reading_list_{rd['stamp']}__{stamp}.md").open("x") as fh:
        fh.write("\n".join(md) + "\n")
    with (HERE / "results" / f"phase2_read_summary_{rd['stamp']}__{stamp}.json").open("x") as fh:
        fh.write(json.dumps(out | {"check_arm": check_arm, "checks": rd["checks"], "probes_sha256": rd["probes_sha256"], "git": rd["git"], "app": rd["app"],
                                   "seconds": rd["seconds"]}, indent=1))
    for rec in out["runs"]:
        print(rec["env"], rec["side_task_success"], {ro: (round(v["score"], 2), round(v["sol_check"]["mean"], 2), {a: round(p, 2) for a, p in v["percentile"].items()},
                                                          v.get("flagged_calls")) for ro, v in rec["readouts"].items()}, rec["call_end_kinds"])


if __name__ == "__main__":
    main()
