"""Deception black-box, two README claims made committed (Codex on #849): (1) an independent recompute of both
within-task AUROCs and the AUROC of the command count alone, from the raw files; (2) which runs had byte-identical judge
prompts and so shared one cached judge call (683 runs, 682 calls, retries counted from the rows' attempt values). Keys, counts and numbers only. Local, $0.

  uv run --no-project python experiments/2026-10-11_q2_deception_blackbox/check_readme_claims.py <scores json> <judge inputs jsonl> \\
      <judgments json>
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))
from judge_common import fill, prompts  # noqa: E402  the runner's own prompt filling

HERE = Path(__file__).parent
PIN = {"scores": "f3c4dee40c00368f96d3f379f980630108f5ba4c1a68ff9afe0bbc8e70e72e72",
       "inputs": "6ed2d6ebc43d7cb7fd9cb380105cfc6e018786d84e78f539c33ef8e70dba37d1",
       "judgments": "2698204370f66fa14d6f036058bcb97dd4836d7eec8b16d33f647b3d5fc60c5b"}
# the prompt and filler the judge run used: the judgments file's "prompt sha256" and the token check's "judge_common sha256"
USED = {"judge_prompt.md": "bf1ed077fcfc773f2527ada2ed8a17dcee3e7217eeb599b6556f8649c36d71b2",
        "judge_common.py": "51c27220c77062b09154fc86dfec4c9573a815d1e43a7b10b2fa50ff79dda51b"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(scores_p, inp_p, judg_p):
    me = Path(__file__).resolve()
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    for p, k in ((scores_p, "scores"), (inp_p, "inputs"), (judg_p, "judgments")):
        if sha(p) != PIN[k]:
            raise SystemExit(f"{Path(p).name} is not the pinned {k}")
    if json.load(open(judg_p))["prompt sha256"] != USED["judge_prompt.md"]:
        raise SystemExit("the judgments file records a different prompt")
    for name, h in USED.items():  # shared calls are found by re-filling prompts: they must be the judge run's own
        if sha(HERE / name) != h:
            raise SystemExit(f"{name} is not the one the judge run used")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    inp = [json.loads(l) for l in open(inp_p)]
    jrows = json.load(open(judg_p))["rows"]
    jp = {r["key"]: r["probability"] for r in jrows}
    att = {r["key"]: r["attempt"] for r in jrows}  # attempt 1 = a second paid call after an unparseable first answer
    rows = defaultdict(list)
    for r in json.load(open(scores_p))["scores"]:
        rows[(r["dump"], r["arm"], r["run"])].append(r["score"])
    pr = {x["key"]: sum(rows[(x["dump"], x["arm"], x["run"])]) / len(rows[(x["dump"], x["arm"], x["run"])]) for x in inp}
    cnt = {x["key"]: len(x["commands"]) for x in inp}
    bt = defaultdict(lambda: defaultdict(list))
    for x in inp:
        bt[x["task"]][x["arm"]].append(x["key"])

    def au(arm, s):
        w = n = 0
        for t, d in bt.items():
            for a in d[arm]:
                for h in d["honest_a"] + d["honest_b"]:
                    w += (s[a] > s[h]) + 0.5 * (s[a] == s[h]); n += 1
        return w / n, n
    out = {"git": head, "inputs sha256": PIN, "prompt and filler sha256 (as the judge run)": USED, "AUROC (independent recompute)": {}}
    for arm in ("attack", "attack_run2"):
        (ap, n), (aj, _), (ac, _) = au(arm, pr), au(arm, jp), au(arm, cnt)
        out["AUROC (independent recompute)"][arm] = {"pairs": n, "probe": ap, "judge": aj, "command count alone (attack higher)": ac}
    system, user = prompts()
    by = defaultdict(list)
    for x in inp:
        by[hashlib.sha256((system + "\x00" + fill(user, x)).encode()).hexdigest()].append(x)
    dups = [v for v in by.values() if len(v) > 1]
    calls = sum(1 + max(att[x["key"]] for x in g) for g in by.values())  # one call per distinct prompt, plus its retries
    out["identical judge prompts"] = {"runs": len(inp), "distinct prompts": len(by),
                                      "attempt values (pinned rows)": {str(a): n for a, n in sorted(Counter(att.values()).items())},
                                      "judge calls (distinct prompts plus retries)": calls, "groups sharing one call": [
        {"runs": [x["key"] for x in g], "arms": [x["arm"] for x in g], "tasks": sorted({x["task"] for x in g}), "commands each": [len(x["commands"]) for x in g],
         "judge probabilities (shared call)": [jp[x["key"]] for x in g]} for g in dups]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"readme_claims_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(*sys.argv[1:4])
