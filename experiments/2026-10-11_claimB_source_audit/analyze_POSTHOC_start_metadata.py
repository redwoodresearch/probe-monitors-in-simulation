"""POST HOC (source audit follow-up 2; 0d's go, $0, laptop, read-only on S3): how does each September claim B run start?
Fresh from the task start, or forked from a prefix of another run (the confound David Zhang found: real runs started
fresh while simulated runs forked)?

For every sample of the simulated attack cohort and both real attack epochs, it counts:
- whether `metadata.fork` exists, its `fraction` (of the source's actions), `after` (whole steps adopted) and the
  number of `step_texts`, which must equal `after` (sims' `sim.steps.Fork` at ba990399);
- the prefix actually seeded into the run: `len(metadata.seeded_prefix)` (sims `runner.build.arena_dataset.seed_metadata`);
- tool-call ids the run shares with its simulator reference (`metadata.orig_trajectory`), both raw and after the
  replay normalization (`runner.simulate.seeding._call_id`: characters outside `[a-zA-Z0-9_-]` become `_`).
A missing field is counted as missing, never as zero. Completeness is asserted: unique (file, id, epoch); at least
1,000 samples per real cohort (e2 holds 1,001); exactly 1,000 simulated samples, 200 base200 plus 800 overlap800.
The samples are loaded whole through `probe_shift.transcripts` (which caches them locally as usual); only counts are
written.

  cd <worktree root> && PYTHONPATH="src:$HOME/simulated-control-evals/src" uv run --no-project --python 3.14 \\
      --with boto3 --with pyyaml python experiments/2026-10-11_claimB_source_audit/analyze_POSTHOC_start_metadata.py
"""

import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from probe_shift.transcripts import eval_files, file_samples

HERE = Path(__file__).parent
COHORTS = {"sim-attack": "la_public10_sim_astra_attack_n1000", "real-attack-e1": "la_public10_real_attack_n1000",
           "real-attack-e2": "la_public10_real_attack_reroll_n1000"}


def norm(x) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(x))


def call_ids(sample: dict) -> set:
    return {tc["id"] for m in sample["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls") or []}


def main():
    out = {}
    for arm, alias in COHORTS.items():
        c, seen = Counter(), set()
        files = eval_files(alias)
        assert files, alias
        for file in files:
            family = file.split("_")[0] if arm == "sim-attack" else "all"
            for s in file_samples(alias, file):
                ident = (file, s["id"], s["epoch"])
                assert ident not in seen, ident
                seen.add(ident)
                c[(family, "samples")] += 1
                md = s.get("metadata")
                if not isinstance(md, dict):
                    c[(family, "metadata_missing")] += 1
                    continue
                fork = md.get("fork")
                if fork is None:
                    c[(family, "fork_record_absent")] += 1
                else:
                    assert isinstance(fork, dict) and "fraction" in fork and "after" in fork, (ident, sorted(fork))
                    assert fork["after"] == len(fork.get("step_texts") or []), (ident, fork["after"])
                    c[(family, f"fork_fraction={fork['fraction']}")] += 1
                    c[(family, f"fork_after_steps={fork['after']}")] += 1
                if "seeded_prefix" not in md:
                    c[(family, "seeded_prefix_absent")] += 1
                else:
                    c[(family, f"seeded_prefix_actions={len(md['seeded_prefix'] or [])}")] += 1
                ref = md.get("orig_trajectory")
                if not ref:
                    c[(family, "reference_absent")] += 1
                    continue
                ref_ids = [a.get("id") for a in ref]
                c[(family, "reference_actions_without_id")] += sum(i is None for i in ref_ids)
                ids = call_ids(s)
                raw = ids & {i for i in ref_ids if i is not None}
                normed = {norm(i) for i in ids} & {norm(i) for i in ref_ids if i is not None}
                c[(family, "samples_sharing_call_ids_with_reference_raw")] += bool(raw)
                c[(family, "samples_sharing_call_ids_with_reference_normalized")] += bool(normed)
                c[(family, "shared_call_ids_normalized_total")] += len(normed)
        n = sum(v for (fam, k), v in c.items() if k == "samples")
        out[arm] = {"|".join(k): v for k, v in sorted(c.items())}
        print(arm, json.dumps(out[arm]))
        # The first run (21:00 PT) stopped here on e2's 1,001 samples (unique file, id, epoch); the count is reported.
        assert n >= 1000, (arm, n)
        if arm == "sim-attack":
            assert n == 1000, (arm, n)
            assert c[("base200", "samples")] == 200 and c[("overlap800", "samples")] == 800, dict(c)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    res = {"label": "POST HOC (source audit follow-up 2): fork records, seeded prefixes and reference-shared call ids per sample (counts only)",
           "cohorts": COHORTS, "counts": out, "git": git,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")  # laptop local time (PT)
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"start_metadata_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
