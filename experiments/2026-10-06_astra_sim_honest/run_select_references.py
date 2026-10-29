"""Pick the honest reference for each Astra simulated honest run, and check it is not a run we compare against.

A simulated honest run of main task M is grounded on one real honest run of M (the "reference"). It must never be a
run in our real honest cohort, or the simulator copies outputs from the transcript it is later compared with.

The only honest runs of these tasks outside the cohort are the fresh references the September overlap800 fleet made
for each Astra attack job (`fork.run = sce-paired-honest-*`). They survive only as `metadata.orig_trajectory` inside
the simulated attack samples. For each main task with one, take the reference of a paired cell first
(splits/exclusions_v3.json), so the honest run shares its reference with an attack run; then one whose September
environment digest is the env's usual fleet digest (one TikTok fleet job ran on a differing tree, so its reference could
not pass the sims code's environment check); then the first by cell name.
Main tasks that appear only in base200 have no reference outside the cohort and are listed as uncovered.

Checks, per chosen reference, against every cohort honest run of the same main task: shared tool-call IDs (must be 0)
and the share of its commands that are identical to some cohort command (independent runs share a few `cat`s).

  uv run python experiments/2026-10-06_astra_sim_honest/run_select_references.py
"""

import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.transcripts import cohort_uri, eval_files, file_samples

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
SIM = "la_public10_sim_astra_attack_n1000"
HONEST = "la_public10_real_honest_n1000"


def main():
    paired = set(json.loads((ROOT / "splits/exclusions_v3.json").read_text())["paired_attack_cells"]["cells"])

    cohort_ids, cohort_cmds = defaultdict(set), defaultdict(set)
    for file in eval_files(HONEST):
        for s in file_samples(HONEST, file):
            main = s["metadata"]["main_task"]["id"]
            for m in s["messages"]:
                for tc in m.get("tool_calls") or [] if m["role"] == "assistant" else []:
                    cohort_ids[main].add(tc["id"])
                    cohort_cmds[main].add(json.dumps(tc.get("arguments"), sort_keys=True))

    candidates, base200_mains = defaultdict(list), set()
    contracts = defaultdict(lambda: defaultdict(set))  # env -> cohort -> September contents digests
    for file in eval_files(SIM):
        for s in file_samples(SIM, file):
            md = s["metadata"]
            main, side = md["main_id"], md["side_task"]["id"]
            contracts[md["env"]][file.split("_")[0]].add(md["arena_reference_contract"]["contents_sha256"])
            if file.startswith("base200"):
                base200_mains.add(main)
                continue
            assert file.startswith("overlap800"), file
            assert md["fork"]["fraction"] == 0.0 and md["fork"]["run"].startswith("sce-paired-honest-"), md["fork"]
            traj = md["orig_trajectory"]
            candidates[main].append({
                "environment": md["env"], "cell": f"{main}:{side}", "paired": f"{main}:{side}" in paired,
                "sim_uri": f"{cohort_uri(SIM)}logs/{file}", "sim_sample_id": s["id"], "sim_epoch": s["epoch"],
                "fork_run": md["fork"]["run"], "source_sample": md["fork"]["source_sample"],
                "n_actions": len(traj), "sept_contract": md["arena_reference_contract"],
                "shared_call_ids": len({a["id"] for a in traj} & cohort_ids[main]),
                "command_overlap": round(sum(json.dumps(a["arguments"], sort_keys=True) in cohort_cmds[main]
                                             for a in traj) / max(1, len(traj)), 3),
            })

    usual = {env: max(c["overlap800"], key=lambda h: sum(x["sept_contract"]["contents_sha256"] == h
                                                         for cs in candidates.values() for x in cs))
             for env, c in contracts.items()}
    chosen = {main: sorted(cs, key=lambda c: (not c["paired"], c["sept_contract"]["contents_sha256"] != usual[c["environment"]],
                                              c["cell"]))[0]
              for main, cs in sorted(candidates.items())}
    unusual = sorted(m for m, c in chosen.items() if c["sept_contract"]["contents_sha256"] != usual[c["environment"]])
    leaks = [m for m, c in chosen.items() if c["shared_call_ids"]]
    assert not leaks, f"references sharing tool-call IDs with the honest cohort: {leaks}"
    uncovered = sorted(base200_mains - set(chosen))
    assert set(cohort_ids) == set(chosen) | set(uncovered), "main tasks in the honest cohort that are neither covered nor listed"

    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {
        "created": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
        "git": git, "sim_alias": SIM, "honest_alias": HONEST,
        "rule": "overlap800 reference: paired cell first, then the env's usual fleet digest, then first by cell name; "
                "0 shared tool-call IDs with the cohort",
        "usual_fleet_digest": usual, "chosen_with_unusual_digest": unusual,
        "n_covered": len(chosen), "n_uncovered": len(uncovered),
        "september_contracts": {env: {cohort: sorted(h) for cohort, h in c.items()} for env, c in sorted(contracts.items())},
        "references": chosen, "uncovered_mains": uncovered,
    }
    path = HERE / "results" / f"references_{datetime.now(ZoneInfo('America/Los_Angeles')):%Y%m%d-%H%M%S}.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "x") as f:  # results/ is append-only
        f.write(json.dumps(out, indent=1))
    overlaps = sorted(c["command_overlap"] for c in chosen.values())
    print(f"{len(chosen)} main tasks covered ({sum(c['paired'] for c in chosen.values())} from paired cells), "
          f"{len(uncovered)} uncovered, {len(unusual)} with an unusual digest {unusual}; command overlap with cohort mean {sum(overlaps) / len(overlaps):.2f}, "
          f"max {overlaps[-1]:.2f}; wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
