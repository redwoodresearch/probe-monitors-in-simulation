"""Reproduce the LinuxArena paper's judge on a random sample of our pairs, before trusting its stored verdicts.

Re-runs the paper's exact setup, copied from its archived script (experiments/2026-09-21_linuxarena_sol_discriminator/
run.py in the paper's evidence archive): `Pair` + `full_sim_actions` + `discriminate_pair` from simulated-control-evals
`metrics.discriminator`, prompt `pair_2afc.md` (digest checked against the stored runs), gpt-5.6-sol at medium effort,
both orders per pair. Transcripts are built in memory from the cohort samples in S3 (`sample_transcript`), the same
samples the paper scored (pair IDs are checked). Then compares the new verdicts with the stored ones on the same pairs.

  uv run python experiments/2026-10-06_q1_text_baselines/run_judge_repro.py          # preflight: builds pairs, no calls
  op run --env-file=.env.tpl -- uv run python experiments/2026-10-06_q1_text_baselines/run_judge_repro.py --run
With --config config/judge_missing.yaml: the same judge on every staged cell the paper never judged (no comparison).
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from inspect_ai.log import EvalSample
from metrics.contract import prompt_digest
from metrics.discriminator.drops import DropTally
from metrics.discriminator.pairing import PAIR_PROMPT, Pair, build_pair_prompt, full_sim_actions
from metrics.discriminator.two_afc import DiscriminatorConfig, discriminate_pair
from runner.metrics.clients import JudgeClient, ModelSettings
from sim.adapters.inspect_eval import sample_transcript
from sim.identity import Mode
from sim.transcript import Corpus

from probe_shift.scoring import pairwise_accuracy_ci
from probe_shift.transcripts import load_sample

HERE = Path(__file__).parent


def transcript(alias: str, row: dict, simulated: bool):
    sample = EvalSample.model_validate(load_sample(alias, row["file"], row["sample_id"], row["epoch"]))
    corpus = Corpus(name="simulated" if simulated else "real", simulated=simulated, setting="linuxarena:public10")
    t = sample_transcript(sample, corpus=corpus, simulated=simulated, mode=Mode.ATTACK, epoch=row["epoch"])
    assert t.session_id == row["sample_id"], (t.session_id, row["sample_id"])
    return t


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="make the paid calls after the preflight")
    parser.add_argument("--config", default="config/judge_repro.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((HERE / args.config).read_text())
    stored_summary = json.loads((HERE / "results" / cfg["stored_summary"]).read_text())
    stored_digest = {p["prompt_sha256"] for p in stored_summary["judges"][cfg["stored_judge"]]["provenance"]}
    assert stored_digest == {prompt_digest(PAIR_PROMPT)}, f"prompt changed since the paper's run: {stored_digest}"
    stored = [json.loads(l) for l in (HERE / "results" / cfg["stored_pairs"]).read_text().splitlines()]
    stored = {r["cell"]: r for r in stored if r["judge"] == cfg["stored_judge"]}
    if cfg.get("select") == "missing":  # every staged cell the paper never judged, with the staging's e1 and sim runs
        runs = [json.loads(l) for l in (HERE / ".run_cache" / cfg["examples"] / "runs.jsonl").read_text().splitlines()]
        run_key = {(r["arm"], r["cell"]): r["key"] for r in runs}
        chosen = sorted({r["cell"] for r in runs} - set(stored))
        pair_keys = {c: (run_key[("real-attack-e1", c)], run_key[("sim-attack", c)]) for c in chosen}
    else:  # a random sample of the cells the paper judged, with the paper's runs
        cells = sorted(stored)
        chosen = [cells[i] for i in sorted(np.random.default_rng(cfg["seed"]).choice(len(cells), cfg["n_pairs"], replace=False))]
        pair_keys = {c: (stored[c]["real_key"], stored[c]["sim_key"]) for c in chosen}

    meta = {}
    for line in gzip.open(HERE / ".run_cache" / cfg["examples"] / "outputs.jsonl.gz", "rt"):
        r = json.loads(line)
        meta.setdefault(r["key"], {k: r[k] for k in ("file", "sample_id", "epoch")})
    with ThreadPoolExecutor(16) as ex:
        reals = list(ex.map(lambda c: transcript(cfg["cohorts"]["real"], meta[pair_keys[c][0]], False), chosen))
        sims = list(ex.map(lambda c: transcript(cfg["cohorts"]["simulated"], meta[pair_keys[c][1]], True), chosen))
    pairs = [Pair(id=r.session_id, sim_id=s.session_id, real_actions=tuple(r.actions), sim_actions=full_sim_actions(s), source="attack")
             for r, s in zip(reals, sims)]
    assert all(p.real_actions and p.sim_actions for p in pairs)
    chars = sum(len(build_pair_prompt(p, slot)) for p in pairs for slot in ("A", "B"))
    est = chars / cfg["chars_per_token"] * cfg["price_per_mtok_input"] / 1e6 + 2 * len(pairs) * cfg["output_tokens_per_call"] * cfg["price_per_mtok_output"] / 1e6
    print(f"{len(pairs)} pairs, {2 * len(pairs)} calls, {chars:,} prompt characters; estimated ${est:.0f} "
          f"(paper's measured rate: ${cfg['paper_usd_per_pair'] * len(pairs):.0f})", flush=True)
    if not args.run:
        return

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    run_name = f"{cfg['run_name']}_{stamp}"
    # Save every rendered prompt (both orders) before any call, with both repos' revisions.
    prompts = HERE / ".run_cache" / f"{cfg.get('name', 'judge_repro')}_{stamp}" / "prompts.jsonl.gz"
    prompts.parent.mkdir(parents=True)
    with gzip.open(prompts, "wt") as fh:
        for c, p in zip(chosen, pairs):
            for slot in ("A", "B"):
                fh.write(json.dumps({"cell": c, "real_id": p.id, "sim_id": p.sim_id, "real_slot": slot,
                                     "prompt": build_pair_prompt(p, slot)}) + "\n")
    revisions = {name: subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=path).stdout.strip()
                 + ("-dirty" if subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=path).stdout.strip() else "")
                 for name, path in (("probe-shift", HERE), ("simulated-control-evals", HERE.parents[2] / "simulated-control-evals"))}
    patches = {}  # a "-dirty" revision's uncommitted changes, saved next to the prompts before any call
    for name, path in (("probe-shift", HERE), ("simulated-control-evals", HERE.parents[2] / "simulated-control-evals")):
        if revisions[name].endswith("-dirty"):  # from the repo root: the whole diff, and every untracked source file in full
            root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, cwd=path, check=True).stdout.strip())
            diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True, cwd=root, check=True).stdout
            new = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True, cwd=root,
                                 check=True).stdout.splitlines()
            sources = "".join(f"\n# untracked {f}:\n{(root / f).read_text()}" for f in new if f.endswith((".py", ".yaml")))
            patch = prompts.parent / f"{name}.patch"
            patch.write_text(diff + sources + "".join(f"\n# untracked (not saved): {f}" for f in new if not f.endswith((".py", ".yaml"))))
            patches[name] = {"path": str(patch.relative_to(HERE)), "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}
    judge = JudgeClient(model=cfg["model"], settings=ModelSettings(reasoning_effort=cfg["reasoning_effort"], run_name=run_name))
    config, tally = DiscriminatorConfig(reasoning_effort=cfg["reasoning_effort"], max_concurrent=cfg["max_concurrent"]), DropTally()
    semaphore = asyncio.Semaphore(config.max_concurrent)
    outcomes = await asyncio.gather(*(discriminate_pair(p, judge, config, semaphore, tally) for p in pairs))

    rows = []
    for c, outcome in zip(chosen, outcomes):
        new = outcome.as_dict() if outcome is not None else None
        rows.append({"cell": c, "main_task": c.split(":")[0], "real_key": pair_keys[c][0], "sim_key": pair_keys[c][1],
                     "stored_accuracy": stored[c]["accuracy"] if c in stored else None,
                     "stored_orders_correct": stored[c]["orders_correct"] if c in stored else None, "new": new,
                     "new_accuracy": None if new is None else new["accuracy"]})
    scored = [r for r in rows if r["new_accuracy"] is not None]
    same_orders = [a == b for r in scored if r["stored_orders_correct"] and len(r["new"]["trials"]) == 2 == len(r["stored_orders_correct"])
                   for a, b in zip(r["new"]["trials"], r["stored_orders_correct"])]
    with_stored = [r for r in scored if r["stored_accuracy"] is not None]
    groups = [r["main_task"] for r in scored]
    summary = {"config": cfg, "run_name": run_name, "revisions": revisions, "patches": patches,
               "prompts": {"path": str(prompts.relative_to(HERE)), "sha256": hashlib.sha256(prompts.read_bytes()).hexdigest()},
               "n_pairs": len(rows), "n_scored": len(scored), "drops": tally.as_dict(),
               "new": pairwise_accuracy_ci([r["new_accuracy"] for r in scored], groups, cfg["bootstrap"], cfg["seed"]),
               "stored_same_pairs": pairwise_accuracy_ci([r["stored_accuracy"] for r in with_stored], [r["main_task"] for r in with_stored],
                                                         cfg["bootstrap"], cfg["seed"]) if with_stored else None,
               "per_order_agreement_with_stored": round(float(np.mean(same_orders)), 4) if same_orders else None,
               "pair_accuracy_equal_to_stored": round(float(np.mean([r["new_accuracy"] == r["stored_accuracy"] for r in with_stored])), 4)
               if with_stored else None}
    out = HERE / "results" / f"{cfg.get('name', 'judge_repro')}_{stamp}.json"
    out.write_text(json.dumps(summary | {"pairs": rows}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    asyncio.run(main())
