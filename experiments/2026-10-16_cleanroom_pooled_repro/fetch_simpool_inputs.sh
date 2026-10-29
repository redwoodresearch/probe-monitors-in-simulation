#!/usr/bin/env bash
# Addendum 19: git show the 7 pinned inputs of cleanroom_simpool.py into $1 (a new, empty folder) and check each sha256. Keys-only tool.
set -euo pipefail
out=${1:?give an empty output folder}
mkdir -p "$out"; [ -z "$(ls -A "$out")" ] || { echo "STOP: $out is not empty"; exit 1; }
g() { git show "$1:$2" > "$out/$3"; }
g 16682ef28 experiments/2026-10-15_fpr_sweep_507_prep/results/routec_pwm_2026-10-18/alltasks_CRD_FULL_r2_allorigins_pooledworld_catalogue_e12_2026-10-18_112818_catalogue.json.gz cat.json.gz
gunzip "$out/cat.json.gz"
g 596bc767e experiments/2026-10-16_honest_sim_detector_all/results/honest_all_routec_2026-10-18_121211.json honest_all_routec.json
g ff8ea35a7 experiments/2026-10-16_attack_prompt_provenance/results/openai_usage_2026-10-16_185845_scan_rows.json scan_rows.json
g de0bf0301 experiments/2026-10-16_attack_prompt_provenance/results/openai_usage_reanalysis_2026-10-16_191817.json usage.json
g 4326d76e9 experiments/2026-10-15_e2e_sim_pipeline/results/simh_cont_headers_2026-10-17_081840.json cont_headers.json
g 4af62db74 experiments/2026-10-14_q3_q35_simhonest_purchase/results/manifest_simh_eval_upload_pilot.json pilot_manifest.json
g 104087089 experiments/2026-10-20_routec_simpool_numbers/results/routec_simpool_numbers_2026-10-20_020501.json simpool_record.json
cd "$out" && shasum -a 256 -c - <<SUMS
72432318a3703d4675a5c4530d77f598a003af351340d895d305b9bd1050adc8  cat.json
10ed5645e0a8357150e07ac6416a06c6196e995d01eb5088b94de1d554d73c46  honest_all_routec.json
0d0392fb773b1804a3ded3b4f3fd2f4dc8e7ae33b1ee660dd7200d0550410197  scan_rows.json
a0bcb2cef1ec21d598dda1795cc8b027eaf49235ba191f5d4c9643e108701102  usage.json
a7638c2d19f6a56f2953babca0ea98a7fd63854cbec21be6f1861ea452941e5d  cont_headers.json
875167b9916560444f80be78bf516beba6711837a811bd677b82f66e704d182b  pilot_manifest.json
2e17de0c852612d8a2b1d83468997c0cb6f787dcfee9a3c93eef6edee711fd11  simpool_record.json
SUMS
