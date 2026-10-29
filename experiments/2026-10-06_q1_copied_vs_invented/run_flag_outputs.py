"""Flag every tool output by whether it matches the simulator's reference trajectory (Q1, copied vs invented).

For each kept tool output (the probe's units) of real attack epoch 1, epoch 2 and simulated attack on the paired
cells, against that cell's reference (the sim sample's `metadata.orig_trajectory`: the real honest run of the same
main task that the simulator was handed, as {function, arguments, output} actions):
  - cmd_in_ref: the reference has a call with the same tool and arguments (keys in `ignore_arg_keys` dropped);
  - out_match: `exact` if the output, trailing whitespace stripped, equals a reference output; `near` if its best
    similarity to one is >= near_threshold; else `none`;
  - same_cmd_out: the same comparison against only the reference outputs of the same command (`na` if none).
The real arms are compared with the same reference, so their match rate is the deterministic baseline: how often a
real environment gives the same output as the honest run did.

Writes per-output flags (identifiers only, no transcript text) to .run_cache/ and counts to results/.

  uv run --with rapidfuzz python experiments/2026-10-06_q1_copied_vs_invented/run_flag_outputs.py
"""

import gzip
import io
import json
import subprocess
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from rapidfuzz import fuzz, process

from probe_shift.transcripts import _s3, cohort_uri, reader_messages

HERE = Path(__file__).parent
EDITOR = "text_editor"


def read_samples(alias: str, file: str, wanted: set[tuple[str, int]], eval_dir: Path) -> dict:
    """The wanted (sample_id, epoch) samples of one .eval archive, from the local copy if present, else S3."""
    path = eval_dir / alias / file
    if path.exists():
        data = path.read_bytes()
    else:
        bucket, prefix = cohort_uri(alias).removeprefix("s3://").split("/", 1)
        data = _s3().get_object(Bucket=bucket, Key=f"{prefix}logs/{file}")["Body"].read()
    out = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for sid, epoch in wanted:
            out[(sid, epoch)] = json.loads(z.read(f"samples/{sid}_epoch_{epoch}.json"))
    return out


def command(function: str, arguments: dict, ignore: set[str]) -> tuple[str, str]:
    return function, json.dumps({k: v for k, v in (arguments or {}).items() if k not in ignore}, sort_keys=True)


def match(text: str, refs: list[str], ref_set: set[str], near: float, max_chars: int) -> tuple[str, float]:
    """(`exact` | `near` | `none`, best similarity) of text against reference outputs (`refs` already cut to
    max_chars). The cutoff lets rapidfuzz skip pairs whose lengths alone rule out a near match, so a `none` output's
    similarity is recorded as 0 rather than its exact best."""
    if text in ref_set:
        return "exact", 1.0
    best = process.extractOne(text[:max_chars], refs, scorer=fuzz.ratio, score_cutoff=near * 100) if refs else None
    return ("near", round(best[1] / 100, 4)) if best else ("none", 0.0)


def main():
    cfg = yaml.safe_load((HERE / "config" / "copied_vs_invented.yaml").read_text())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    eval_dir = Path(cfg["eval_dir"]).expanduser()
    ignore = set(cfg["ignore_arg_keys"])

    rows = []
    with gzip.open(Path(cfg["staged_outputs"]).expanduser(), "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["arm"] in cfg["arms"] and r["kept"]:
                rows.append(r)
    cells = {r["cell"] for r in rows}
    assert len(cells) == cfg["expected_paired_cells"], f"{len(cells)} cells, expected {cfg['expected_paired_cells']}"

    # Every sample a row points at, grouped by archive; exactly one run per (arm, cell).
    runs = {(r["arm"], r["cell"]): (r["file"], r["sample_id"], r["epoch"], r["key"]) for r in rows}
    by_file = defaultdict(set)
    for (arm, _), (file, sid, epoch, _) in runs.items():
        by_file[(arm, file)].add((sid, epoch))

    def load(item):
        (arm, file), wanted = item
        return arm, file, read_samples(cfg["cohorts"][arm], file, wanted, eval_dir)

    samples = {}
    with ThreadPoolExecutor(16) as pool:
        for arm, file, got in pool.map(load, by_file.items()):
            for (sid, epoch), s in got.items():
                samples[(arm, file, sid, epoch)] = s

    # Per run: calls indexed (step, action) as the reader sees them; per cell: the sim sample's reference.
    calls, refs, sim_reference = {}, {}, {}
    for (arm, cell), (file, sid, epoch, key) in runs.items():
        s = samples[(arm, file, sid, epoch)]
        msgs = reader_messages(s, "")
        turns = [m["tool_calls"] for m in msgs if m["role"] == "assistant"]
        calls[key] = {(i, j): (c["function"]["name"], c["function"]["arguments"]) for i, t in enumerate(turns)
                      for j, c in enumerate(t)}
        if arm == "sim-attack":
            ref = s["metadata"]["orig_trajectory"]
            assert isinstance(ref, list) and ref, f"{key}: no reference trajectory"
            refs[cell] = ref
    for r in rows:
        if r["arm"] == "sim-attack":
            sim_reference[r["cell"]] = r["sim_reference"]

    print(f"loaded {len(samples)} samples, {len(refs)} references", flush=True)
    cut = cfg["near_max_chars"]
    ref_index = {}
    for cell, ref in refs.items():
        outs = [str(a.get("output") or "").rstrip() for a in ref]
        by_cmd = defaultdict(list)
        for a, o in zip(ref, outs):
            by_cmd[command(a["function"], a.get("arguments"), ignore)].append(o)
        ref_index[cell] = ([o[:cut] for o in outs], set(outs), {k: ([o[:cut] for o in v], set(v)) for k, v in by_cmd.items()})

    def flag(r):
        fn, args = calls[r["key"]][(r["step"], r["action"])]
        if fn != r["tool"]:
            raise ValueError(f"{r['key']} step {r['step']} action {r['action']}: call {fn} but output tool {r['tool']}")
        outs, out_set, by_cmd = ref_index[r["cell"]]
        text = r["text"].rstrip()
        out_match, best = match(text, outs, out_set, cfg["near_threshold"], cut)
        same = by_cmd.get(command(fn, args, ignore))
        same_out = match(text, *same, cfg["near_threshold"], cut)[0] if same else "na"
        return {"arm": r["arm"], "key": r["key"], "cell": r["cell"], "main_task": r["main_task"],
                "environment": r["environment"], "sim_reference": sim_reference[r["cell"]], "step": r["step"],
                "action": r["action"], "tool": r["tool"], "n_chars": len(text), "cmd_in_ref": bool(same),
                "out_match": out_match, "best_similarity": best, "same_cmd_out": same_out}

    flags = []
    for i, r in enumerate(rows):
        flags.append(flag(r))
        if i % 10000 == 0:
            print(f"flagged {i}/{len(rows)}", flush=True)

    cache = HERE / ".run_cache"
    cache.mkdir(exist_ok=True)
    with gzip.open(cache / f"flags_{stamp}.jsonl.gz", "wt") as f:
        for x in flags:
            f.write(json.dumps(x) + "\n")

    def counts(keyf):
        c = Counter(keyf(x) for x in flags)
        return {"|".join(map(str, k)): v for k, v in sorted(c.items())}

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    summary = {
        "config": cfg, "git": git, "stamp": stamp, "flags_file": f".run_cache/flags_{stamp}.jsonl.gz",
        "n_outputs": len(flags), "n_runs": len(runs), "n_cells": len(cells), "n_reference_actions": sum(map(len, refs.values())),
        "counts": {
            "arm|out_match": counts(lambda x: (x["arm"], x["out_match"])),
            "arm|cmd_in_ref|out_match": counts(lambda x: (x["arm"], x["cmd_in_ref"], x["out_match"])),
            "arm|cmd_in_ref|same_cmd_out": counts(lambda x: (x["arm"], x["cmd_in_ref"], x["same_cmd_out"])),
            "arm|editor|out_match": counts(lambda x: (x["arm"], x["tool"] == EDITOR, x["out_match"])),
            "arm|sim_reference|out_match": counts(lambda x: (x["arm"], x["sim_reference"], x["out_match"])),
        },
    }
    (HERE / "results").mkdir(exist_ok=True)
    out = HERE / "results" / f"flag_counts_{stamp}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["counts"]["arm|cmd_in_ref|out_match"], indent=1))
    print(f"wrote {out} and {summary['flags_file']}")


if __name__ == "__main__":
    main()
