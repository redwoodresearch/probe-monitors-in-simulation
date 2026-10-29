"""POST HOC (RULE.md Amendment 1 @ 266ecf07, item 2(ii)): which scrub level each reader's leftover-gap pipelines read, and
so whether times and dates were masked. Reads configs, stagers and extractors at cited commits with `git show`; opens no data.
The pipeline list came from a repo search (labelled "from agent summary"); this script checks each entry against the code.
A staged input whose level can't be read from code is "unverified". Writes results/pipeline_levels_POSTHOC_<PT stamp>.json.

  uv run --no-project --python 3.12 --with pyyaml python experiments/2026-10-13_claimB_fake_reply_diffs/analyze_POSTHOC_pipeline_levels.py
"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

HERE = Path(__file__).resolve().parent
MAIN = "565997f1"  # this branch's base on main
G3 = "a3efa7ea"  # origin/aswin/q2-g3-reader head (run_extract_g3.py is not on main)
N36, V4, S1 = "2026-10-06_la_segment-end_qwen3-32b_v1", "2026-10-06_la_segment-end-v4_qwen3-32b_v1", "2026-10-06_la_segment-end-scrub1_qwen3-32b_v1"
FIX, PLANT, EDIT = "2026-10-11_q3_fix939_inputs_v1", "2026-10-12_q2_plantslip_reply_{which}_inputs_v1", "2026-10-12_q3_edithonest_inputs_v1"
BL1, AL1 = "2026-10-10_la_fixed-sim-v2_probe-inputs-b-l1_v1", "2026-10-10_la_fixed-sim-v2_probe-inputs-l1_v1"
# claim B step A's level-1 fixed sim: these exact lines of its stager make RUN_L1 the level-1 scrub of the fixed sim
AL1_SNIPPETS = ['RUN_L0, RUN_L1 = "2026-10-10_la_fixed-sim-v2_probe-inputs-l0_v1", "2026-10-10_la_fixed-sim-v2_probe-inputs-l1_v1"',
                'level1 = ([row(r, "sim-attack-fixed", scrubbed(r["messages"], 1)) for r in sim]',
                'for run, items in ((RUN_L0, level0), (RUN_L1, level1)):']
SCRUB_CALL = re.compile(r"\b(?:scrub|scrubbed|mask_times)\(")
# staged run -> how its level is set: ("yaml", config) | ("derived", stager, source run) | ("claimb", stager)
STAGED = {N36: ("yaml", "experiments/2026-10-06_segment_end_extraction/config/segment_end.yaml"),
          V4: ("yaml", "experiments/2026-10-06_segment_end_extraction/config/segment_end_v4.yaml"),
          S1: ("yaml", "experiments/2026-10-06_segment_end_extraction/config/segment_end_scrub1.yaml"),
          FIX: ("derived", "experiments/2026-10-08_q3_swap_cue/run_fix939_stage.py", N36),
          PLANT: ("derived", "experiments/2026-10-12_plantslip_transfer/q2_reply_stage.py", N36),
          EDIT: ("derived", "experiments/2026-10-12_q3_edit_honest_fa/stage_edit_honest.py", V4),
          BL1: ("claimb", "experiments/2026-10-09_fixed_sim/run_stage_probe_inputs_b_modal.py"),
          AL1: ("snippets", "experiments/2026-10-09_fixed_sim/run_stage_probe_inputs_modal.py", AL1_SNIPPETS, 1)}
# reader -> [(pipeline, file, commit)]: YAML configs name their staged input in `inputs_from`; Python extractors by string
PIPELINES = {
    "Qwen3-32B": [("#36 segment-end (q2 reply / q3 attack detectors)", "experiments/2026-10-06_all_layers_extraction/config/alllayers.yaml", MAIN),
                  ("fix939 both-fixes copies", "experiments/2026-10-06_all_layers_extraction/config/alllayers_q3_fix939.yaml", MAIN),
                  ("v4 five arms (n437 simulated honest)", "experiments/2026-10-06_all_layers_extraction/config/alllayers_v4.yaml", MAIN),
                  ("claim B (b), level 1", "experiments/2026-10-06_all_layers_extraction/config/alllayers_fixedsim_v2_b_scrub1.yaml", MAIN)],
    "Qwen3.5-27B": [("q2 reply readout (#36 inputs)", "experiments/2026-10-06_qwen35_replication/config/qwen35_alllayers.yaml", MAIN),
                    ("q2 repaired (fix939 inputs)", "experiments/2026-10-12_q2_qwen35_repaired/config/q35fix_repaired.yaml", MAIN),
                    ("v4 five arms (n437 simulated honest)", "experiments/2026-10-06_qwen35_replication/config/qwen35_alllayers_v4.yaml", MAIN),
                    ("claim B (b), level 1", "experiments/2026-10-06_qwen35_replication/config/reader_fixedsim_v2_l1.yaml", MAIN)],
    "Olmo-3.1-32B": [("q2 reader", "experiments/2026-10-12_q2_olmo_reader/run_extract_olmo.py", MAIN),
                     ("q3rmfa simulated honest", "experiments/2026-10-13_q3_reader_matched_fa/run_extract_simhonest_olmo.py", "3d0a61cd")],
    "Llama-3.3-70B": [("q2 reader", "experiments/2026-10-12_q2_llama_reader/run_extract_llama.py", MAIN),
                      ("q3rmfa simulated honest", "experiments/2026-10-13_q3_reader_matched_fa/run_extract_simhonest_llama.py", "9a38165e")],
    "Seed-OSS-36B": [("G3 reader", "experiments/2026-10-13_g3_new_reader/run_extract_g3.py", G3)],
}


def show(path: str, ref: str) -> str | None:
    """The file at ref, or None when git can't read it (the caller reports that pipeline as unverified)."""
    p = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, cwd=HERE)
    return p.stdout if p.returncode == 0 else None


def level_of(run: str, seen=()) -> dict:
    """The scrub level of a staged run's tool outputs, with how it was read."""
    if run not in STAGED:
        return {"run": run, "level": None, "how": "unverified: no known stager"}
    kind, path, *rest = STAGED[run]
    src = show(path, MAIN)
    if src is None:
        return {"run": run, "level": None, "how": f"unverified: {path}@{MAIN} unreadable"}
    if kind == "snippets":
        ok = all(x in src for x in rest[0])
        return {"run": run, "level": rest[1] if ok else None, "how": f"{path}@{MAIN} staging lines {'present' if ok else 'NOT found (unverified)'}"}
    if kind == "yaml":
        cfg = yaml.safe_load(src)
        assert cfg["run"] == run, (path, cfg["run"])
        return {"run": run, "level": int(cfg.get("scrub_level", 0)), "how": f"{path}@{MAIN} scrub_level ({'set' if 'scrub_level' in cfg else 'absent = 0'})"}
    if kind == "derived":
        calls = len(SCRUB_CALL.findall(src))
        if run.split("{")[0] not in src or rest[0] not in src or calls:
            return {"run": run, "level": None, "how": f"unverified: {path}@{MAIN} names run {run.split('{')[0] in src}, source {rest[0] in src}, scrub calls {calls}"}
        base = level_of(rest[0])
        return {"run": run, "level": base["level"], "how": f"{path}@{MAIN} copies {rest[0]} with no scrub call; source: {base['how']}"}
    m = re.search(r'1: \("([^"]+)", "([^"]+)", "([^"]+)"\)', src)  # claim B's SOURCES[1]: (real arms, fixed sim A, B's run)
    if not m or m.group(3) != run:
        return {"run": run, "level": None, "how": f"unverified: {path}@{MAIN} SOURCES[1] not found"}
    real, sim = level_of(m.group(1)), level_of(m.group(2))
    return {"run": run, "level": 1 if real["level"] == 1 and sim["level"] == 1 else None,
            "how": f"{path}@{MAIN} SOURCES[1]: real arms from {m.group(1)} ({real['how']}); fixed sim from {m.group(2)} ({sim['how']})"}


def main():
    me = Path(__file__).resolve()
    if subprocess.run(["git", "status", "--porcelain", str(me)], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit(f"commit {me.name} first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    scrub_src = show("src/probe_shift/scrub.py", MAIN)
    out = {"label": "POST HOC; code and configs only", "rule": "RULE.md Amendment 1 @ 266ecf07, item 2(ii)", "git": git,
           "scrub_masks_times_at_level_1": "    if level >= 1:\n        text = mask_times(text)\n" in scrub_src, "readers": {}}
    for reader, pipes in PIPELINES.items():
        rows = []
        for label, path, ref in pipes:
            src = show(path, ref)
            if src is None:
                rows.append({"pipeline": label, "file": f"{path}@{ref}", "times_masked": "unverified", "note": "unreadable at this commit"})
                print(f"{reader:14s} | {label:48s} | unreadable: unverified")
                continue
            if path.endswith(".yaml"):
                runs = [yaml.safe_load(src)["inputs_from"]]
                own_calls = 0
            else:
                runs = [r for r in STAGED if r.split("{")[0] in src]
                own_calls = len(SCRUB_CALL.findall(src))
            levels = [level_of(r) for r in runs]
            lv = {x["level"] for x in levels}
            masked = ("unverified" if None in lv or own_calls or not runs else
                      "yes" if lv == {1} or min(lv) >= 1 else "no" if lv == {0} else "mixed")
            rows.append({"pipeline": label, "file": f"{path}@{ref}", "staged_inputs": levels, "extractor_scrub_calls": own_calls,
                         "times_masked": masked})
            print(f"{reader:14s} | {label:48s} | levels {sorted(lv, key=str)} | own scrub calls {own_calls} | times masked: {masked}")
        out["readers"][reader] = rows
    print("scrub masks times at level >= 1:", out["scrub_masks_times_at_level_1"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"pipeline_levels_POSTHOC_{stamp}.json"
    with open(path, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
