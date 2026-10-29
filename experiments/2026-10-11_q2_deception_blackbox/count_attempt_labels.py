"""Deception black-box, beside the reading (0d's check, about 00:00 PT 10-12): how many of the cohort's attack runs
declined. From q3's frozen self-report attempt labels (one file per launch; columns read: run, label, side_task_scorer;
no description text exists in these files, only its hash). Count-only, local, $0.

  python3 experiments/2026-10-11_q2_deception_blackbox/count_attempt_labels.py <judge inputs jsonl> <folder of label files>
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
LABELS = {  # dump -> its launch's label file
    "2026-10-10_scaleup_canary_segends_v1": "phase2_attempt_labels_2026-10-09_173815_scaleup_canary_selfreport_2026-10-10_111817.jsonl",
    "2026-10-10_scaleup_l2_segends_v1": "phase2_attempt_labels_2026-10-09_182505_scaleup_selfreport_2026-10-10_111838.jsonl",
    "2026-10-10_scaleup_topup_segends_v1": "phase2_attempt_labels_2026-10-10_001314_scaleup_topup_selfreport_2026-10-10_111821.jsonl",
    "2026-10-10_second_attack_segends_v1": "phase2_attempt_labels_2026-10-10_001314_second_attack_2026-10-10_111840.jsonl",
}


def main(inp: Path, folder: Path):
    me = Path(__file__).resolve()
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    lab, shas, out_missing = {}, {}, Counter()
    for dump, name in LABELS.items():
        b = (folder / name).read_bytes(); shas[name] = hashlib.sha256(b).hexdigest()
        for line in b.decode().splitlines():
            r = json.loads(line)
            if not r.get("run"):  # a missing attempt or .eval: no run to join (counted)
                out_missing[name] += 1; continue
            if (dump, r["run"]) in lab:
                raise SystemExit(f"{name}: run {r['run']} labelled twice")
            lab[(dump, r["run"])] = (r["label"], r["side_task_scorer"])
    out = {"git": head, "label files sha256": shas, "label rows without a run (missing attempt or .eval)": dict(out_missing), "by arm": {}}
    for arm in ("attack", "attack_run2"):
        keys = [(x["dump"], x["run"]) for x in map(json.loads, inp.open()) if x["arm"] == arm]
        if len(set(keys)) != len(keys):
            raise SystemExit(f"{arm}: a run repeats in the judge inputs")
        got = [lab.get(k) for k in keys]
        out["by arm"][arm] = {"runs": len(keys), "self-report label": dict(Counter(g[0] if g else "no label" for g in got)),
                              "side-task scorer": dict(Counter(str(g[1]) if g else "no label" for g in got))}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"attempt_labels_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
