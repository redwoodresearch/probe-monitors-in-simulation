# 2026-10-11_q2_new_real_runs

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

What can new real GPT-5.6-Sol honest runs resolve for the data-transfer, destructive-command and credential monitors, what do they cost, and (once bought) do they pass the stage gates?

## What we did

Pre-registered sizing and cost projections (\$0), then a staged purchase of new real honest runs for the credential-monitor comparison: 548 real honest runs completed on 274 tasks, with zero errored runs reported across stages. About \$718 was approved; recorded spending was about \$398.7, plus extraction of stages 2–6. The earlier 233-task (\$611) sizing was superseded. Stage-1 gates on credential and data-transfer rates, a label-calibration amendment, and a seal keeping the runs unread.

## What we found

README: the first part (transfer and destructive at plus or minus 5 points) too few even with spending; the credential-monitor comparison was first sized at 233 tasks (\$611 with buffer), then run on 274 tasks (548 real honest runs, zero errored); stage 1 failed the calibrated credential floor by 0.004 per run and the transfer gate (0.36-0.46 of baseline rates), recorded as deviations; the project lead chose to continue.

## In the post?

Not in the post.

## Status

Exploratory (runs sealed)

## Builds on

[2026-10-11_q2_zoo_plans](../2026-10-11_q2_zoo_plans/); [2026-10-11_credential_stage1_labels](../2026-10-11_credential_stage1_labels/); [2026-10-11_credential_calibration_panel](../2026-10-11_credential_calibration_panel/)

## Main scripts

- [`analyze_width_check.py`](analyze_width_check.py)
- [`analyze_c_sizing.py`](analyze_c_sizing.py)
- [`make_batch_spec.py`](make_batch_spec.py)
- [`check_stage1_labels_real.py`](check_stage1_labels_real.py)
- [`make_stage1_label_sheet.py`](make_stage1_label_sheet.py)
- [`check_c_stage1_credentials.py`](check_c_stage1_credentials.py)
- [`analyze_c_calibrated_gate.py`](analyze_c_calibrated_gate.py)
- [`analyze_c_guard.py`](analyze_c_guard.py)

Result files the index names:

- `width_check_2026-10-11_150754.json` (not copied: result left out to keep the release under 120 MB)
- [`c_sizing_2026-10-11_150659.json`](results/c_sizing_2026-10-11_150659.json)
- [`batch_spec_2026-10-11_151207.json`](results/batch_spec_2026-10-11_151207.json)
- `stage1_labels_real_2026-10-11_190808.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- [`c_calibrated_gate_2026-10-11_231648.json`](results/c_calibrated_gate_2026-10-11_231648.json)
- `c_stage1_credential_gate_2026-10-11_204528.json` (not copied: result left out to keep the release under 120 MB)

## Files here

11 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-11_q2_new_real_runs/` at commit `b21c3677bbfd3d49014ccfb40d5f23ed326675aa` of our internal repository. 17 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
