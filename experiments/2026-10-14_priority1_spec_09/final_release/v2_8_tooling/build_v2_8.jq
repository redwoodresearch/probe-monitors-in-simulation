# v2.8 = v2.7 + .realline_v2_8 (and .counts_v2_8 when given), by key. Usage:
#   jq --slurpfile rl realline_v2_8_block.json --slurpfile ct counts_v2_8_block.json --arg date "<PT>" --arg v27sha <sha> \
#      --arg bsha <this file's sha256> -f build_v2_8.jq ../routec_FINAL_release_v2.7_2026-10-21.json
. as $v27
| ($v27 | keys) as $k27
| $v27
| .version = "2.8"
| .date_pt = $date
| .supersedes = {"path": "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.7_2026-10-21.json",
                 "commit": "13ca31b07742dd5b115c9ed4c389337bc9efbd08", "sha256": $v27sha,
                 "origin_branches": ["origin/aswin/routec-final-release"], "kind": "release_record", "version": "2.7"}
| .builder = {"note": "v2.8 = v2.7 + realline_v2_8 (+ counts_v2_8), built by key (build_v2_8.jq) from the committed v2.7 record; every other key is v2.7's, unchanged",
              "script_sha256": $bsha}
| .repo_path = "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.8_2026-10-26.json"
| .realline_v2_8 = $rl[0]
| (if ($ct | length) > 0 then .counts_v2_8 = $ct[0] else . end)
