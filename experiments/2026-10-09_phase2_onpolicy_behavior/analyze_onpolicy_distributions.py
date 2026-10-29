"""Aswin's spec for the on-policy behavior check: per-arm distributions (descriptive only), run endings, task success,
and 5 real/simulated pairs side by side. Read-only, $0, CPU. README.md has the spec and the predictions.

  uv run --no-project --python 3.14 --with numpy --with boto3 --with pyyaml python -I \
      experiments/2026-10-09_phase2_onpolicy_behavior/analyze_onpolicy_distributions.py --run-dir <copy of the run>

Writes results/onpolicy_distributions_<stamp>.json (counts and statistics only) and data/pairs_<stamp>.md (the pairs'
raw commands and outputs; data/ is gitignored).
"""
import argparse
import collections
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_onpolicy_behavior as B  # noqa: E402  measure(), failure(), the labeler and the c3 re-read rule
from behavior_common import resolved_messages  # noqa: E402  attachment refs resolved, as in measure()
from probe_shift.transcripts import tool_result_text  # noqa: E402

SEED, BOOT = B.SEED, B.BOOT
ARMS = [("attack", None), ("sim_attack", "september"), ("sim_attack", "today"),
        ("honest_b", None), ("sim_honest", "september"), ("sim_honest", "today"), ("honest_a", None)]
KINDS = {"edits": {"file_write", "file_write_tmp"}, "network": {"network"}, "scripts": {"run_program"}}
FIELDS = ["steps", "share_edits", "share_network", "share_scripts", "share_rest", "fail_share",
          "rereads_per_100_calls", "edit_reread_share"]
name = lambda k: k[0] + ("@" + k[1] if k[1] else "")  # noqa: E731


def ending(m: dict, outcome: dict) -> str:
    calls = outcome.get("policy_calls") or [{}]
    if m["sample_error"]:
        return "error: " + m["sample_error"].split("(")[0]
    if outcome.get("sample_limits"):
        return "limit: " + ",".join(map(str, outcome["sample_limits"]))
    if m["submitted"]:
        return "submitted"
    if any(c.get("step_limit_hits") for c in calls):
        return "step limit"
    return f"other: ended={calls[-1].get('ended')}"


def load(run_dir: Path):
    launch = json.loads((run_dir / "launch.json").read_text())
    runs, samples, label_errors = {}, {}, collections.Counter()
    for r in launch["runs"]:
        key = (r["arm"], None if r["real"] else r["simulator"])
        counted = [a for a in r["attempts"] if not a.get("redrawn_because")]
        a = counted[0] if r["arm"] == "honest_a" else counted[-1]  # honest_a: its first try (amendment 13)
        (ev,) = sorted((run_dir / Path(a["log_dir"]).relative_to(Path(a["log_dir"]).parents[1])).glob("*.eval"))
        with zipfile.ZipFile(ev) as z:
            (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
            s = json.loads(z.read(n))
        m, _, le = B.measure(s, r)
        label_errors += le
        m["sample_error"] = ((s.get("error") or {}).get("message") or "")[:200] or None
        m["submitted"] = m["submitted"] and not m["sample_error"]
        cats, n_calls = m["categories"], m["calls"]
        for k, cs in KINDS.items():
            m["share_" + k] = sum(cats.get(c, 0) for c in cs) / n_calls if n_calls else None
        m["share_rest"] = 1 - sum(m["share_" + k] for k in KINDS) if n_calls else None
        m["ending"] = ending(m, a.get("outcome") or {})
        value = ((s.get("scores") or {}).get("linux_scorer") or {}).get("value") or {}
        field = "main_task_success" if "honest" in r["arm"] else "side_task_success"
        judge = (((s.get("scores") or {}).get("linux_scorer") or {}).get("metadata") or {}).get(field)
        judged_by = sorted({str(x.get("model")) for x in (judge or {}).get("judge_messages") or []
                            if isinstance(x, dict) and x.get("model")}) if isinstance(judge, dict) else []
        if r["real"]:
            m["success_real_scorer"] = value.get(field)  # C, I, N, or None when unscored
        else:
            m["sim_judge_verdict"] = value.get(field)  # the simulator's own judge: NOT the real scorer
            m["sim_judge_models"] = judged_by
        runs[(key, r["task"])] = m
        samples[(key, r["task"])] = s
    return launch, runs, samples, label_errors


def summary(values: list[float]) -> dict:
    v = np.array(values, float)
    if not len(v):
        return {"n": 0}
    boots = np.median(v[np.random.default_rng(SEED).integers(0, len(v), size=(BOOT, len(v)))], axis=1)
    return {"n": len(v), "median": float(np.median(v)), "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)), "min": float(v.min()), "max": float(v.max()),
            "median_ci95_descriptive": [float(x) for x in np.percentile(boots, [2.5, 97.5])]}


def steps_of(s: dict) -> list[dict]:
    """Every tool call in order, with its step (agent turn with a non-submit call) and the full action and output."""
    msgs = resolved_messages(s)
    res = {x.get("tool_call_id"): x for x in msgs if x["role"] == "tool"}
    rows, step = [], 0
    for msg in msgs:
        if msg["role"] != "assistant":
            continue
        tcs = msg.get("tool_calls") or []
        step += any(tc["function"] != B.RUNNER_SUBMIT for tc in tcs)
        for tc in tcs:
            a = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
            act = (f"{a.get('command')} {a.get('path')}" if tc["function"] == "text_editor"
                   else str(a.get("cmd", a.get("command", a))))
            out = res.get(tc.get("id"))
            rows.append({"step": step, "tool": tc["function"], "action": act, "arguments": a,  # in full (data/ only)
                         "fail": (B.failure(tc["function"], out) if out else "no result"),
                         "output": tool_result_text(out) if out else ""})
    return rows


def cut(text: str, n: int) -> str:
    one = text.replace("\n", " \u23ce ")
    return one if len(one) <= n else one[:n] + f" ...[+{len(one) - n} chars]"


def side_by_side(samples: dict, runs: dict, stamp: str) -> tuple[Path, list]:
    """5 (task, kind) pairs drawn with SEED from tasks where both runs finished without a sample error. Full records go
    to data/pairs_<stamp>.jsonl (exact text with each call's full arguments, for quoting); data/pairs_<stamp>.md aligns the two runs step by step."""
    cands = [(t, kind) for kind, real, sim in (("attack", ("attack", None), ("sim_attack", "september")),
                                                ("honest", ("honest_b", None), ("sim_honest", "september")))
             for t in sorted({t for (_, t) in runs})
             if (real, t) in runs and (sim, t) in runs and not runs[(real, t)]["sample_error"]
             and not runs[(sim, t)]["sample_error"]]
    picks = [cands[i] for i in sorted(np.random.default_rng(SEED).choice(len(cands), size=5, replace=False))]
    md, full = [f"# 5 real/simulated pairs, seed {SEED}, from {len(cands)} (task, kind) candidates\n"], []
    for t, kind in picks:
        keys = [("attack", None), ("sim_attack", "september")] if kind == "attack" else \
               [("honest_b", None), ("sim_honest", "september")]
        rows = {k: steps_of(samples[(k, t)]) for k in keys}
        for k in keys:
            full += [{"run": f"{name(k)}/{t}", **r} for r in rows[k]]
        md.append(f"\n## {t}, {kind}: {name(keys[0])} ({runs[(keys[0], t)]['steps']} steps, "
                  f"{runs[(keys[0], t)]['ending']}) vs {name(keys[1])} ({runs[(keys[1], t)]['steps']} steps, "
                  f"{runs[(keys[1], t)]['ending']})\n")
        for step in range(1, max(r["step"] for k in keys for r in rows[k] or [{"step": 0}]) + 1):
            for k in keys:
                for r in (x for x in rows[k] if x["step"] == step):
                    md.append(f"- {step:3d} {'REAL' if k[1] is None else 'SIM '} `{r['tool']}` "
                              f"{cut(r['action'], 140)!r} -> {'FAIL ' + r['fail'] if r['fail'] else 'ok'} | "
                              f"{cut(r['output'], 160)!r}\n")
    out = HERE / "data" / f"pairs_{stamp}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(md))
    (HERE / "data" / f"pairs_{stamp}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in full))
    return out, picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    commit = subprocess.run(["git", "-C", str(B.REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(B.REPO), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())  # repo-wide, tracked files: imported code counts
    launch, runs, samples, label_errors = load(args.run_dir)
    print(f"run {args.run_dir.name} | script commit {commit[:12]}{' (repo has uncommitted changes)' if dirty else ''} | labeler {B.fingerprint()} | seed {SEED} | "
          f"descriptive only: medians with task-bootstrap 95% intervals ({BOOT} resamples), no tests")
    out = {"run": args.run_dir.name, "script_commit": commit, "repo_dirty": dirty, "labeler": B.fingerprint(), "seed": SEED, "made": stamp,
           "per_arm": {}, "endings": {}, "success": {}, "label_errors": dict(label_errors)}
    print(f"calls the labeler has no rule for (counted in 'rest'): {dict(label_errors) or 'none'}")
    print("\nper-arm distributions, runs without a sample error: median [p25, p75] (min-max), median's interval")
    for f in FIELDS:
        print(f"  {f}:")
        for k in ARMS:
            vals = [m[f] for (kk, _), m in runs.items() if kk == k and not m["sample_error"] and m[f] is not None]
            s = out["per_arm"].setdefault(name(k), {})[f] = summary(vals)
            if not s["n"]:
                print(f"    {name(k):22s} n= 0")
                continue
            print(f"    {name(k):22s} n={s['n']:2d} {s['median']:8.3f} [{s['p25']:.3f}, {s['p75']:.3f}] "
                  f"({s['min']:.3f}-{s['max']:.3f}) median CI [{s['median_ci95_descriptive'][0]:.3f}, "
                  f"{s['median_ci95_descriptive'][1]:.3f}]")
    print("\nendings (all 20 runs per arm):")
    for k in ARMS:
        c = collections.Counter(m["ending"] for (kk, _), m in runs.items() if kk == k)
        out["endings"][name(k)] = dict(c)
        print(f"  {name(k):22s} {dict(c)}")
    print("\ntask success:")
    for k in ARMS:
        ms = [m for (kk, _), m in runs.items() if kk == k]
        if k[1] is None:
            c = collections.Counter(str(m["success_real_scorer"]) for m in ms)
            out["success"][name(k)] = {"scorer": "linux_scorer (real)", "counts": dict(c)}
            print(f"  {name(k):22s} real scorer: {dict(c)}")
        else:
            c = collections.Counter(str(m["sim_judge_verdict"]) for m in ms)
            models = sorted({x for m in ms for x in m["sim_judge_models"]})
            out["success"][name(k)] = {"scorer": f"simulator judge {models} - NOT comparable", "counts": dict(c)}
            print(f"  {name(k):22s} NON-COMPARABLE, the simulator's judge {models}, not the real scorer: {dict(c)}")
    pairs, picks = side_by_side(samples, runs, stamp)
    out["pairs"] = {"file": str(pairs.relative_to(B.REPO)), "picks": picks}
    print(f"\npairs (seed {SEED}): {picks} -> {pairs.relative_to(B.REPO)} (raw text, gitignored)")
    out["per_run"] = [{"arm": name(k), "task": t, **{f: m.get(f) for f in FIELDS + ["ending", "success_real_scorer",
                                                                                     "sim_judge_verdict"]}}
                      for (k, t), m in sorted(runs.items(), key=str)]
    res = HERE / "results" / f"onpolicy_distributions_{stamp}.json"
    res.parent.mkdir(exist_ok=True)
    with open(res, "x") as fh:
        fh.write(json.dumps({k: v for k, v in out.items() if k != "per_run"}, indent=1)[:-2]
                 + ',\n "per_run": [\n' + ",\n".join("  " + json.dumps(x) for x in out["per_run"]) + "\n ]\n}\n")
    print(f"-> {res.relative_to(B.REPO)}")


if __name__ == "__main__":
    main()
