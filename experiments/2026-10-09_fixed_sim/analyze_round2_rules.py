"""Round two's derivation (PLAN.md amendment 6, bars committed in 79afb6c2 before any count): the 5 DEV environments
only, filtered when the runs are read, so no test-environment output is ever counted. Free, local, counts only (no text).
Per rule (fixes_round2.py):
  real form share  among the real dev outputs the rule acts on (text, not harness-truncated), pooled over real e1 and e2
                   (the decision) and per epoch: R8 create outputs ending without a newline; R9 full-file views ending
                   with an empty numbered line in the strict real form (also reported over `cat -n` file views only); R10 non-editor outputs with >= 2 non-empty lines and >= 1 error-style
                   line in which every error-style line comes before every other non-empty line
  dry run          the rule applied to every real dev run: outputs it would change (must be 0)
  kept             real form share >= 0.95 AND dry run 0
  simulated        outputs the rule changes per dev environment (the v2 text, round one already applied)
Then the kept rules are applied in order to every simulated dev run: a rule's unchanged outputs must stay byte-identical.
Writes results/round2_rules_<stamp>.json.

  PYTHONPATH=src:experiments/2026-10-09_fixed_sim uv run --no-project --with numpy python \\
      experiments/2026-10-09_fixed_sim/analyze_round2_rules.py <local 2026-10-09_la_fixed-sim_v2>
"""

import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import fixes  # noqa: E402
import fixes_round2 as F  # noqa: E402

DEV = set(json.loads((HERE / "split.json").read_text())["dev"])
REAL = ("real-attack-e1", "real-attack-e2")
BAR = 0.95


def acted_on(rule, c, text):
    """Is this output one the rule acts on (text, not truncated), and is it in the real form."""
    if not isinstance(text, str) or fixes.TRUNCATED in text:
        return None
    if rule == "R8":
        return (not text.endswith("\n")) if c["tool"] == "text_editor" and c["args"].get("command") == "create" else None
    if rule in ("R9", "R9_cat_n"):  # R9: every full-file view, as PLAN words it (the decision); R9_cat_n: `cat -n` file views only
        full_view = c["tool"] == "text_editor" and c["args"].get("command") == "view" and "view_range" not in c["args"]
        return F.last_is_empty_numbered(text) if full_view and (rule == "R9" or text.startswith(F.CAT_N)) else None
    if c["tool"] == "text_editor":
        return None
    flags = [(l.strip() != "", e) for l, e in zip(text.rstrip("\n").split("\n"), F.error_flags(text))]
    nonempty = [e for ne, e in flags if ne]
    if len(nonempty) < 2 or not any(nonempty):
        return None
    first_other = next((i for i, e in enumerate(nonempty) if not e), len(nonempty))
    return not any(nonempty[first_other:])  # every error-style line before every other non-empty line


def main(run_dir):
    runs = []
    for f in sorted((Path(run_dir) / "inputs").glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["environment"] in DEV:  # dev only, before anything else is read
                runs.append(r)
    envs = sorted({r["environment"] for r in runs})
    assert set(envs) <= DEV
    out = {"dev_environments": envs, "runs": dict(Counter(r["arm"] for r in runs)), "bar": BAR, "rules": {}}
    for rule, fn in F.RULES.items():
        form, sim_acted, form_cat_n = Counter(), Counter(), Counter()
        dry, sim_changed = Counter(), defaultdict(int)
        for r in runs:
            calls = fixes.calls_of(r["messages"])
            for c in calls:
                v = acted_on(rule, c, r["messages"][c["msg"]]["content"])
                if v is not None:
                    form[(r["arm"], v)] += 1
                if rule == "R9" and (w := acted_on("R9_cat_n", c, r["messages"][c["msg"]]["content"])) is not None:
                    form_cat_n[(r["arm"], w)] += 1
            _, recs = fn(r["messages"])
            n = sum(x["changed"] for x in recs)
            if r["arm"] in REAL:
                dry[r["arm"]] += n
            else:
                sim_changed[r["environment"]] += n
                sim_acted[r["environment"]] += sum(x["category"] not in ("skipped_non_str", "skipped_truncated") for x in recs)
        share = {a: (form[(a, True)] / max(1, form[(a, True)] + form[(a, False)]), form[(a, True)] + form[(a, False)]) for a in (*REAL, "sim-attack")}
        pooled_n = sum(form[(a, b)] for a in REAL for b in (True, False))
        pooled = sum(form[(a, True)] for a in REAL) / max(1, pooled_n)
        kept = pooled >= BAR and sum(dry.values()) == 0
        out["rules"][rule] = {"real_form_share_pooled": round(pooled, 4), "real_outputs_acted_on": pooled_n,
                              "real_form_share": {a: [round(s, 4), n] for a, (s, n) in share.items()},
                              "dry_run_real_changed": dict(dry), "simulated_changed_per_env": dict(sim_changed),
                              "simulated_looked_at_per_env": dict(sim_acted), "kept": kept}
        if rule == "R9":  # the same share over `cat -n` file views only (directory listings and other text left out)
            n9 = sum(form_cat_n[(a, b)] for a in REAL for b in (True, False))
            out["rules"][rule]["real_form_share_pooled_cat_n_only"] = [round(sum(form_cat_n[(a, True)] for a in REAL) / max(1, n9), 4), n9]
    kept = [r for r, v in out["rules"].items() if v["kept"]]
    identical, changed_total = 0, 0
    for r in runs:
        if r["arm"] in REAL:
            continue
        new, recs = F.apply(r["messages"], kept)
        changed_msgs = {x["msg"] for x in recs if x["changed"]}
        if len(new) != len(r["messages"]) or any(r["messages"][i] == new[i] for i in changed_msgs):
            raise ValueError(f"{r['key']}: messages added or removed, or a recorded change left its message as it was")
        for i, (a, b) in enumerate(zip(r["messages"], new)):
            if i in changed_msgs:
                changed_total += 1
            elif a != b:
                raise ValueError(f"{r['key']}: message {i} changed but no record says so")
            else:
                identical += 1
    out["kept"] = kept
    out["apply_check"] = {"simulated_messages_changed": changed_total, "unchanged_messages_byte_identical": identical}
    out["fixes_round2_sha256"] = hashlib.sha256((HERE / "fixes_round2.py").read_bytes()).hexdigest()
    out["git"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for rule, v in out["rules"].items():
        print(f"{rule}: real form {v['real_form_share_pooled']:.4f} of {v['real_outputs_acted_on']} real dev outputs "
              f"(per arm {v['real_form_share']}); dry run on real {v['dry_run_real_changed']}; simulated changed "
              f"{sum(v['simulated_changed_per_env'].values())} {v['simulated_changed_per_env']} -> {'KEPT' if v['kept'] else 'DROPPED'}")
    print("kept:", kept, "| apply check:", out["apply_check"], "| fixes_round2.py sha256", out["fixes_round2_sha256"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"round2_rules_{stamp}.json"
    with open(path, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main(sys.argv[1])
