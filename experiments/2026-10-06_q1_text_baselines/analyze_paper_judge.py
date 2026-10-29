"""The LinuxArena paper's real-vs-simulated judge, re-read from its stored verdicts and joined to our paired cells.

The judge (simulated-control-evals `metrics.discriminator`, prompt `prompts/pair_2afc.md`) sees one real and one
simulated transcript of the same cell, in both orders, and names the real one. It sees only tool calls and outputs:
whole transcripts, no 32k cut, no system prompt. So it is not the probe's context; the cells with no run over 32k are
the closest comparison. Pair accuracy is the mean over the two orders; the paper reports 1 - accuracy as the
simulator's win rate.

Step 1 (this script, no model calls): recompute the paper's number from the stored per-pair verdicts, then report
accuracy on our cells, matched by sample ID, with 95% intervals from bootstrapping main tasks.

  uv run python experiments/2026-10-06_q1_text_baselines/analyze_paper_judge.py
"""

import gzip
import json
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from probe_shift.scoring import pairwise_accuracy_ci

HERE = Path(__file__).parent
E1, SIM = "real-attack-e1", "sim-attack"


def fetch(cfg) -> Path:
    """The judge runs' folders from the paper's evidence archive, extracted once into .run_cache/."""
    root = HERE / ".run_cache" / "paper_judge"
    base = root / cfg["archive_dir"]
    if not all((base / run / "pairs.json").exists() for runs in cfg["judges"].values() for run in runs):
        root.mkdir(parents=True, exist_ok=True)
        wanted = [f"{cfg['archive_dir']}/{run}/" for runs in cfg["judges"].values() for run in runs]
        proc = subprocess.Popen(["aws", "s3", "cp", "--profile", cfg["aws_profile"], "--quiet", cfg["archive"], "-"],
                                stdout=subprocess.PIPE)
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:  # streamed: one pass over the archive
            for member in tar:
                if member.isfile() and any(member.name.removeprefix("./").startswith(w) for w in wanted):
                    tar.extract(member, root, filter="data")
        assert proc.wait() == 0, f"aws s3 cp failed for {cfg['archive']}"
    return base


def main():
    cfg = yaml.safe_load((HERE / "config" / "paper_judge.yaml").read_text())
    base = fetch(cfg)
    ex = HERE / ".run_cache" / cfg["examples"]
    runs = [json.loads(l) for l in (ex / "runs.jsonl").read_text().splitlines()]
    sample_of = {}
    for line in gzip.open(ex / "outputs.jsonl.gz", "rt"):
        r = json.loads(line)
        sample_of.setdefault(r["key"], r["sample_id"])
    by = {(r["arm"], r["cell"]): r for r in runs}
    cells = sorted(c for (a, c) in by if a == SIM)
    over_32k = {r["cell"] for r in runs if r["truncated"]}
    slices = {"all_cells": cells, "base200": [c for c in cells if by[(SIM, c)]["sim_reference"] == "base200"],
              "overlap800": [c for c in cells if by[(SIM, c)]["sim_reference"] == "overlap800"],
              "no_run_over_32k": [c for c in cells if c not in over_32k]}

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    summary = {"config": cfg, "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
               "judges": {}}
    (HERE / "results").mkdir(exist_ok=True)
    out_pairs = HERE / "results" / f"paper_judge_pairs_{stamp}.jsonl"
    with out_pairs.open("w") as fh:
        for judge, run_dirs in cfg["judges"].items():
            pairs, stored, provenance = [], [], []
            for run in run_dirs:
                payload = json.loads((base / run / "pairs.json").read_text())
                pairs += payload["two_afc"]["pairs"]
                stored.append(json.loads((base / run / "summary.json").read_text()) | {"run": run})
                provenance.append(payload["_provenance"]["run"] | {"drops": payload["drops"]})
            acc = np.array([p["accuracy"] for p in pairs], float)
            # Reproduction: the stored summaries' simulator win rate, recomputed from the stored pairs.
            recomputed = {s["run"]: round(1 - float(np.mean([p["accuracy"] for p in json.loads((base / s["run"] / "pairs.json").read_text())["two_afc"]["pairs"]])), 6)
                          for s in stored}
            res = {"provenance": provenance, "stored_summaries": stored, "recomputed_sim_win_per_run": recomputed,
                   "all_paper_pairs": {"n_pairs": len(pairs), "sim_win": round(1 - float(acc.mean()), 4),
                                       "judge_accuracy": round(float(acc.mean()), 4),
                                       "orders_per_pair": dict(zip(*map(lambda a: a.tolist(), np.unique([p["n_orders"] for p in pairs], return_counts=True))))}}
            index = {(p["id"], p["sim_id"]): p for p in pairs}
            real_ids = {p["id"] for p in pairs}
            matched, status = {}, {"matched": 0, "real_run_scored_with_another_sim_run": 0, "not_scored": 0}
            for c in cells:
                rid, sid = sample_of[by[(E1, c)]["key"]], sample_of[by[(SIM, c)]["key"]]
                if (rid, sid) in index:
                    matched[c] = index[(rid, sid)]
                    status["matched"] += 1
                    fh.write(json.dumps({"judge": judge, "cell": c, "main_task": c.split(":")[0], "real_key": by[(E1, c)]["key"],
                                         "sim_key": by[(SIM, c)]["key"], "accuracy": matched[c]["accuracy"],
                                         "n_orders": matched[c]["n_orders"], "orders_correct": matched[c]["trials"]}) + "\n")
                else:
                    status["real_run_scored_with_another_sim_run" if rid in real_ids else "not_scored"] += 1
            res["our_cells"] = {"coverage": status, "unmatched_cells": sorted(set(cells) - set(matched))}
            for name, cs in slices.items():
                cs = [c for c in cs if c in matched]
                res["our_cells"][name] = pairwise_accuracy_ci([matched[c]["accuracy"] for c in cs], [c.split(":")[0] for c in cs],
                                                              cfg["bootstrap"], cfg["seed"])
            summary["judges"][judge] = res
            a = res["our_cells"]["all_cells"]
            print(f"{judge}: paper pairs {len(pairs)}, sim win {res['all_paper_pairs']['sim_win']} (stored {[s['sim_win'] for s in stored]}, "
                  f"recomputed {list(recomputed.values())}); our cells {status} accuracy {a['accuracy']} {a['ci95']}")
    out = HERE / "results" / f"paper_judge_summary_{stamp}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"-> {out}\n-> {out_pairs}")


if __name__ == "__main__":
    main()
