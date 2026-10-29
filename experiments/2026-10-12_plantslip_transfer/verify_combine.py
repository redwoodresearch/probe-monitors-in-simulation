"""Verify the combined planted stagings that the four extraction configs pin (read-only, $0; counts and hashes only).

The combine step is `combine` in run_plantslip_onpolicy.py (the runner; the function is byte-identical at f1d4f2de,
the commit deployed when it ran at 13:48-13:49 PT 10-12, and at the as-run runner 9a6bc4e8). It concatenated each
output's source planted stagings (run_plant_splice.py plant mode) in the order recorded in COMBINE_RECORD.md. This
script re-reads the four combined files and their sources from the acts Volume and checks, per output:
- sha256 of the combined file equals the config's pin;
- the combined file's lines equal the sources' lines concatenated in the recorded order (byte for byte);
- epochs and token counts equal the ones combine returned.

  uv run --no-project --with modal==1.6.1 --with pyyaml python verify_combine.py
"""

import hashlib
import json
import sys
from pathlib import Path

import modal
import yaml

HERE = Path(__file__).parent
CFG = HERE.parents[1] / "experiments/2026-10-08_q3_onpolicy_figures/config"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
RUNS = {  # output -> (config, sources in order, combine's returned epochs and tokens)
    "plantslip_p1_run1": ("extract_plantslip_p1_run1.yaml", ["rerun_strict_segends_v2_plantslip", "scaleup_canary_segends_v1_plantslip",
                                                              "scaleup_l2_segends_v1_plantslip", "scaleup_topup_segends_v1_plantslip"], 169, 9350398),
    "plantslip_p1_run2": ("extract_plantslip_p1_run2.yaml", ["second_attack_segends_v1_plantslip"], 169, 9414940),
    "plantslip_p78_run1": ("extract_plantslip_p78_run1.yaml", ["rerun_strict_segends_v2_scrub1_swapsys_plantslip", "scaleup_canary_segends_v1_scrub1_swapsys_plantslip",
                                                                "scaleup_l2_segends_v1_scrub1_swapsys_plantslip", "scaleup_topup_segends_v1_scrub1_swapsys_plantslip"], 169, 8223259),
    "plantslip_p78_run2": ("extract_plantslip_p78_run2.yaml", ["second_attack_segends_v1_scrub1_swapsys_plantslip"], 169, 8280050),
}

v = modal.Volume.from_name(ACTS, version=2)
read = lambda p: b"".join(v.read_file(p))  # noqa: E731
ok = True
for out, (cfg, sources, epochs, tokens) in RUNS.items():
    pin = yaml.safe_load((CFG / cfg).read_text())["inputs_sha256"]["attack.jsonl"]
    combined = read(f"{out}/inputs/attack.jsonl")
    src_lines = [line for s in sources for line in read(f"{s}/inputs/attack.jsonl").decode().splitlines()]
    rebuilt = ("\n".join(src_lines) + "\n").encode()
    lines = combined.decode().splitlines()
    n_tok = sum(len(json.loads(line)["ids"]) for line in lines)
    res = {"output": out, "sha256": hashlib.sha256(combined).hexdigest(), "pin_matches": hashlib.sha256(combined).hexdigest() == pin,
           "equals_ordered_concatenation_of_sources": combined == rebuilt, "epochs": len(lines), "epochs_match": len(lines) == epochs,
           "tokens": n_tok, "tokens_match": n_tok == tokens}
    ok &= res["pin_matches"] and res["equals_ordered_concatenation_of_sources"] and res["epochs_match"] and res["tokens_match"]
    print(json.dumps(res))
print("ALL VERIFIED" if ok else "MISMATCH")
if not ok:
    sys.exit(1)  # a mismatch fails the check (#1041 review)
