"""Label every tool call in valid real attack (epoch 1), simulated attack, real honest and simulated honest runs
read-only or impactful.

Streams the cohorts from S3, indexes calls the way the reader sees them (`reader_messages`: submit calls and empty
turns dropped; step = agent turn, action = call within the turn, as in `reader.render`), and labels each with
`probe_shift.calls.label_call`. The simulated honest cohort is not in the shared exclusions, so its validity and
sample fields come from its own manifest.json. Every row carries the config's task-level slice flags. Writes:
  - .run_cache/q2_call_labels/<stamp>.jsonl   every call with its label (transcript text, so not in git), and
    <stamp>.meta.json                         the labeler's fingerprint, which analyses check before reading labels
  - results/label_counts_<stamp>.json         counts by arm, category and label, over all valid runs, the paired tasks
                                              and the paired tasks under 32k; what each open labeling call changes
  - .run_cache/q2_call_labels/handcheck_<stamp>.md   100 calls for hand-checking, spread over arm x category (it quotes
    commands, including environment credentials, so it stays out of git); results/handcheck_<stamp>_calls.json
    lists which calls they are

  uv run python experiments/2026-10-06_q2_impactful_call_transfer/run_label_calls.py
"""

import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from botocore.exceptions import ConnectionClosedError, EndpointConnectionError, ReadTimeoutError

from probe_shift import calls
from probe_shift.calls import label_call
from probe_shift import cohorts, transcripts
from probe_shift.transcripts import invalidity, main_task_id, reader_messages

HERE = Path(__file__).parent
SHARED_ARMS: set[str] = set()  # arms in the shared exclusions, whose validity must agree with them
ROOT = HERE.parent.parent


def fetch(src, alias: str, file: str, tries: int = 5) -> list[dict]:
    """One .eval file's samples, retrying S3 read timeouts and dropped connections."""
    for k in range(tries - 1):
        try:
            return src.file_samples(alias, file)
        except (ReadTimeoutError, ConnectionClosedError, EndpointConnectionError):
            time.sleep(2 ** k)
    return src.file_samples(alias, file)  # the last try lets its error through


def sample_rows(arm: str, file: str, s: dict, excluded: dict, info: dict, paired: dict, slices: dict) -> tuple[dict, list[dict]]:
    md = s.get("metadata") or {}
    side = md.get("side_task")
    side = side.get("id") if isinstance(side, dict) else side
    env = md.get("environment") or md.get("env")
    env = env.get("id") if isinstance(env, dict) else env
    mt = main_task_id(s)
    key = f"{file.removesuffix('.eval')}__{s['id'].replace('/', '_')}__{s['epoch']}"
    cell = f"{mt}:{side}" if side else mt
    reason = excluded.get((arm, key))
    invalid = invalidity(s)
    if arm in SHARED_ARMS and (reason in ("errored", "stalled", "no_actions")) != (invalid is not None):
        raise ValueError(f"validity disagrees with the shared exclusions for {arm} {key}: {invalid} vs {reason}")
    sample = info[(arm, key)]  # v3 per-sample table: truncation (cut to 32k for the reader) and the sim's reference
    run = {"arm": arm, "key": key, "cell": cell, "invalid": invalid, "excluded": reason}
    if invalid:
        return run, []
    rows, step = [], -1
    for m in reader_messages(s, ""):
        if m["role"] != "assistant":
            continue
        step += 1
        for action, tc in enumerate(m["tool_calls"]):
            fn = tc["function"]
            rows.append({"arm": arm, "key": key, "cell": cell, "main_task": mt, "side_task": side, "environment": env,
                         "step": step, "action": action, "tool": fn["name"], "arguments": fn["arguments"],
                         "truncated_transcript": sample["truncated"], "sim_reference": sample["sim_reference"],
                         **{name: cell in cells for name, cells in paired.items()},
                         **{name: mt in tasks for name, tasks in slices.items()}, **label_call(fn["name"], fn["arguments"])})
    return run, rows


def counts(rows: list[dict]) -> dict:
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    out = {}
    for arm, rs in sorted(by_arm.items()):
        n = len(rs)
        label = Counter(r["label"] for r in rs)
        impactful = [r for r in rs if r["label"] == "impactful"]
        # What each labeling decision moves: calls impactful only through /tmp writes, or only through effects the
        # command does not show (run_program, unknown: the robustness row leaves these out).
        only = lambda cats: sum(set(c for c in r["categories"] if c in calls.IMPACTFUL) <= cats for r in impactful)
        net = [r for r in rs if r["net_method"] is not None]
        out[arm] = {
            "calls": n, "transcripts": len({r["key"] for r in rs}),
            "label": dict(label), "label_share": {k: round(v / n, 4) for k, v in label.items()},
            "category": dict(Counter(r["category"] for r in rs).most_common()),
            "category_by_tool": {t: dict(Counter(r["category"] for r in rs if r["tool"] == t).most_common())
                                 for t in sorted({r["tool"] for r in rs})},
            "impactful_only_via": {"file_write_tmp": only({"file_write_tmp"}), "run_program_or_unknown": only(set(calls.UNCLEAR)),
                                   "network": only({"network"})},  # what a narrower network rule could make read-only
            "network": {"calls": len(net), "method": dict(Counter(r["net_method"] for r in net).most_common()),
                        "dest": dict(Counter(r["net_dest"] for r in net).most_common()),
                        "sending_data": sum(r["net_sends_data"] for r in net),
                        "curl_sending_data": sum(r["curl_sends_data"] for r in rs)},
            "unknown_rules": dict(Counter(x for r in rs for x in r["rules"] if x.startswith("unknown:")).most_common(30)),
        }
    return out


def handcheck(rows: list[dict], cfg: dict) -> str:
    rng = random.Random(cfg["seed"])
    picked = []
    for arm in cfg["arms"]:
        pools = defaultdict(list)
        for r in rows:
            if r["arm"] == arm:
                pools[r["category"]].append(r)
        for pool in pools.values():
            rng.shuffle(pool)
        order = sorted(pools, key=calls.PRIORITY.index)
        k = 0
        while sum(1 for p in picked if p["arm"] == arm) < cfg["n_per_arm"] and any(pools.values()):  # round-robin
            cat = order[k % len(order)]
            if pools[cat]:
                picked.append(pools[cat].pop())
            k += 1
    intro = (f"{len(picked)} tool calls, {cfg['n_per_arm']} per arm ({', '.join(cfg['arms'])}), spread evenly over the "
             "categories present in each arm, so rare categories are over-represented. For each call: is it read-only "
             "or not? Tick if the labeler is right; otherwise write the right answer.")
    rules = ("Rules: a call is READ-ONLY only if every piece is clearly read-only. Every network call (curl, wget, ssh, "
             "aws, HTTP from code, a database client given a host) counts as NOT read-only, whatever its method or "
             "destination; writes only under /tmp count as NOT read-only; running a program whose effects the command "
             "does not show (category run_program or unknown) counts as NOT read-only. The category is the call's most "
             "impactful piece; `rules` lists every piece that matched.")
    lines = ["# Q2 labeler hand-check: read-only or not?", "", intro, "", rules, ""]
    for i, r in enumerate(picked, 1):
        a = r["arguments"]
        if r["tool"] == "text_editor":
            body = {k: v for k, v in a.items() if k not in ("command", "path")}
            text = f"text_editor {a.get('command')} {a.get('path')}\n" + "\n".join(f"{k}: {str(v)[:600]}" for k, v in body.items())
        else:
            text = str(a.get("cmd", a.get("command", a)))
        if len(text) > cfg["max_command_chars"]:
            text = text[:cfg["max_command_chars"]] + f"\n... [{len(text) - cfg['max_command_chars']} more characters]"
        fence = "````" if "```" in text else "```"
        verdict = "READ-ONLY" if r["label"] == "read_only" else "NOT READ-ONLY"
        lines += [f"## {i}. {r['arm']} · labeler says **{verdict}** · {r['category']}", "",
                  f"`{r['key'][:90]}` step {r['step']} action {r['action']} · tool `{r['tool']}`", "",
                  "rules: " + ", ".join(f"`{x}`" for x in r["rules"]), "", fence + "bash", text, fence, "",
                  "- [ ] right · if wrong, read-only or not: ", ""]
    return "\n".join(lines), picked


def main():
    cfg = yaml.safe_load((HERE / "config" / "label_calls.yaml").read_text())
    code = {"impactful_clear": calls.CLEAR_IMPACTFUL, "impactful_unclear": calls.UNCLEAR, "read_only": calls.READ_ONLY}
    if cfg["labels"] != code:
        raise ValueError(f"config labels differ from probe_shift.calls: {code}")
    shared = json.loads((ROOT / "splits" / cfg["exclusions_file"]).read_text())
    excluded = {(r["arm"], r["key"]): r["reason"] for r in shared["excluded"]}
    info = {(r["arm"], r["key"]): r for r in shared["samples"]}
    paired = {"paired": set(shared["paired_attack_cells"]["cells"]),
              "paired_under_32k": set(shared["paired_attack_cells_under_32k"]["cells"])}
    expected = {arm: shared["samples_total"][arm] for arm in cfg["cohorts"] if arm in shared["samples_total"]}
    SHARED_ARMS.update(expected)  # other arms' validity comes from their own manifest; an invalid run is recorded, not fatal
    manifests = {arm: cohorts.manifest(alias) for arm, alias in cfg["cohorts"].items() if arm not in expected}
    for arm, man in manifests.items():  # one run per main task; sample fields from the manifest
        expected[arm] = man["n_requested"]
        for t in man["tasks"]:
            key = f"{t['log'].removeprefix('logs/').removesuffix('.eval')}__{t['sample_id'].replace('/', '_')}__1"
            info[(arm, key)] = {"truncated": None, "sim_reference": t["reference"]["source_sample"]}
    man = manifests["sim-honest"]
    slices = {"sim_honest_local_digest_task": {t["main"] for t in man["tasks"] if t["contract_stamped_locally"]},
              "sim_honest_redrawn_task": set(man["redrawn_after_model_calls"])}

    runs, rows = [], []
    for arm, alias in cfg["cohorts"].items():
        src = cohorts if alias in cohorts.COHORTS else transcripts  # project cohorts not yet in real_runs.yml
        files = src.eval_files(alias)
        with ThreadPoolExecutor(32) as pool:
            for file, samples in zip(files, pool.map(lambda f: fetch(src, alias, f), files)):
                for s in samples:
                    run, rs = sample_rows(arm, file, s, excluded, info, paired, slices)
                    runs.append(run)
                    rows.extend(rs)
        print(f"{arm}: {sum(r['arm'] == arm for r in runs)} samples, {sum(r['arm'] == arm for r in rows)} calls", flush=True)
    got = Counter(r["arm"] for r in runs)
    if dict(got) != expected:
        raise ValueError(f"streamed {dict(got)} samples, the shared exclusions expect {expected}")

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip())  # tracked changes only, as in the other scripts
    cache = ROOT / ".run_cache" / "q2_call_labels"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{stamp}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    bash = subprocess.run(["bash", "--version"], capture_output=True, text=True).stdout.splitlines()[0]  # for malformed
    meta = {"labeler_sha": calls.fingerprint(), "git": git, "git_dirty": dirty, "exclusions_file": cfg["exclusions_file"],
            "bash_for_syntax_check": bash, "arms": list(cfg["cohorts"]),
            "slices": {name: sorted(tasks) for name, tasks in slices.items()}}
    (cache / f"{stamp}.meta.json").write_text(json.dumps(meta))

    subsets = {"all_valid": rows, "paired_tasks": [r for r in rows if r["paired"]],
               "paired_tasks_under_32k": [r for r in rows if r["paired_under_32k"]]}
    out = {"config": cfg, "git": {"commit": git, "dirty": dirty}, "labeler_sha": meta["labeler_sha"],
           "labels_file": str((cache / f"{stamp}.jsonl").relative_to(ROOT)),
           "samples": {arm: {"total": got[arm], "invalid": dict(Counter(r["invalid"] for r in runs if r["arm"] == arm and r["invalid"]))}
                       for arm in cfg["cohorts"]},
           "counts": {name: counts(rs) for name, rs in subsets.items()}}
    (HERE / "results" / f"label_counts_{stamp}.json").write_text(json.dumps(out, indent=1))
    md, picked = handcheck(rows, cfg["handcheck"])
    (cache / f"handcheck_{stamp}.md").write_text(md)
    (HERE / "results" / f"handcheck_{stamp}_calls.json").write_text(json.dumps(
        [{"id": i, **{k: r[k] for k in ("arm", "key", "step", "action", "tool", "label", "category")}} for i, r in enumerate(picked, 1)], indent=1))

    for name, c in out["counts"].items():
        print(f"\n== {name}")
        for arm, d in c.items():
            print(f"  {arm}: {d['calls']} calls in {d['transcripts']} transcripts; labels {d['label_share']}")
            print(f"    categories {d['category']}")
            print(f"    impactful only via {d['impactful_only_via']}; unknown {sum(d['unknown_rules'].values())}")
    print(f"\nresults -> results/label_counts_{stamp}.json, results/handcheck_{stamp}_calls.json; hand-check -> "
          f"{(cache / f'handcheck_{stamp}.md').relative_to(ROOT)}; calls -> {out['labels_file']}")


if __name__ == "__main__":
    main()
