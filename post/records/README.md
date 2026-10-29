# The result files

The 50 files in this folder are the saved results the post was written from. `../rebuild_post.py` rebuilds the post's figures
and numbers from them alone.

Each file except `realline_draws.npz` is a byte-for-byte copy of a file in our research repository (`pertask` is gzipped here).
That repository is not public. `../MANIFEST.json` gives each file's sha256, its original path and the commit it was copied from.
The files keep their original names and metadata, because the checks pin them by sha256; `../README.md` has a glossary of the
working labels in the names.

How to read the tables:

- **Short name**: the name `../MANIFEST.json`, `../number_rules.json` and `../config/slot_runs.json` use for the file.
- **Made by**: the experiment folder (under `experiments/` in our research repository), the script that wrote the file and the
  commit of that script. Where the script ran at a different commit from the file's own, `../MANIFEST.json` has both.
- **Used for**: the figures whose plotting scripts read the file; the statistics tables (`../config/slot_runs.json`) built from
  it; and the post lines whose rebuilt numbers come from it, with the number of numbers. Line numbers are those of the post's
  source text as checked, and the number ids in `../number_rules.json` start with them (`L100.2` is the second number on line
  100). A number read from a statistics table is counted under the result file that the table's code reads for it.

## The main result and what is computed from it

| Short name and file | What it holds | Made by (folder in our research repository, script, commit) | Used for (figures; statistics tables; post lines) |
|---|---|---|---|
| `A`<br>`routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz` | The main result: for every probe and both kinds of input (commands, tool outputs), catch rates on real run 1, real run 2 and the simulated attack runs at the 1% and 5% alarm levels, their differences, the real-threshold check, and 95% intervals from 10,000 resamples of main tasks. | `2026-10-15_fpr_sweep_507_prep/run_peraction311.py` at `f2d47f19f` | Figures 1, 4, 5; input of tables rc1a, rc1b, rc1b_detl48; 246 numbers: TL;DR (6, 7); setup section (35, 40); section on catch rates (46, 48, 50); section on detecting the simulation (62); footnote on task counts (86); footnote on the four probes (100); footnote on tool outputs (102); footnote on the real threshold (104); Appendix A (118, 149, 155, 157); Appendix B (Tables 3, 4) (182); Appendix C (199, 201, 203, 205, 207, 209); Table 5 (217, 218, 219, 220, 221, 222, 223, 224); Appendix C (226, 230); Appendix D (238, 257, 263, 271); Appendix F (301, 305, 307, 311); Table 8 (317, 318, 319, 320, 321, 322, 323, 324); Appendix F (326) |
| `B`<br>`routec_B_alltasks_CRD_FULL_r2_allorigins_sweep15_pooledworld_scored_e12_2026-10-19_180652.json.gz` | Record A's analysis at 15 alarm levels from 0.1% to 5% (Appendix C; gzipped). | `2026-10-15_fpr_sweep_507_prep/run_peraction311.py` at `f2d47f19f` | input of table rc2; 4 numbers: Appendix C (228) |
| `R37`<br>`routec_R37_alltasks_CRD_FULL_r2_allorigins_rule37_pooledworld_scored_e12_2026-10-19_181100.json.gz` | Record A's analysis at the strictest alarm levels: 0.01%, 0.02% and 0.05% (Appendix C; gzipped). | `2026-10-15_fpr_sweep_507_prep/run_peraction311.py` at `f2d47f19f` | input of table rc2; 5 numbers: Appendix C (228) |
| `group_catch`<br>`group_catch_2026-10-20_003430.json` | Catch rates averaged over the six probes and over the four probes, computed from record A, with 95% intervals. | `2026-10-15_fpr_sweep_507_prep/run_group_catch.py` at `74edde46f` | Figures 1, 5; input of tables rc1a, rc1b, rc1b_detl48; 4 numbers: Table 5 (223, 224); Table 8 (323, 324) |
| `group_excess`<br>`group_excess_2026-10-20_003418_a6cada8b0fa546d890b6976f675f7246.json` | Figure 4's comparison for the four-probe average, computed from record A: how far simulated catch sits from the two real runs, minus how far the two real runs sit from each other. | `2026-10-15_fpr_sweep_507_prep/run_group_excess.py` at `74edde46f` | Figure 4; input of tables rc1a, rc1b, rc1b_detl48; 3 numbers: Appendix D (257) |
| `pw_shares`<br>`pw_shares_pooled_all_origins_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327_routeC_2026-10-20_003446.json` | For each probe, the share of honest runs that raise at least one alarm (the tick marks in Figures 1 and 5), computed from record A. | `2026-10-15_fpr_sweep_507_prep/run_pw_shares_routec.py` at `74edde46f` | Figures 1, 5 |
| `pertask`<br>`pertask_full_2026-10-20_004115.json.gz` | For every probe, task and attack run, the score of the run's most suspicious command or tool output, which Figures 6 and 7 compare with the 1% alarm threshold (gzipped here). | `2026-10-15_fpr_sweep_507_prep/run_peraction311.py` at `7797f2cc2` | Figures 6, 7 |
| `anchor`<br>`pertask_full_2026-10-20_004115_anchor_2026-10-20_011851_ap-Ul4ZPPM7A0z6LFMNMfGTqh.json` | Per-task R² between real run 1 and the simulated run, and between real run 1 and real run 2, with 95% intervals (Figure 7 and Appendix H). | `2026-10-15_fpr_sweep_507_prep/run_pertask_anchor.py` at `d8b087792` | Figures 6, 7; input of tables rc1b, rc1b_detl48; 14 numbers: Appendix H (356, 358, 362, 366) |
| `kappa`<br>`kappa_CRD_2026-10-20_004454.json.gz` | Per-task agreement (Cohen's kappa) between real and simulated caught-or-missed outcomes, for each probe (gzipped). | `2026-10-15_fpr_sweep_507_prep/run_peraction311.py` at `7797f2cc2` | input of tables rc1b, rc1b_detl48; 5 numbers: Appendix H (352) |
| `kappa_extra`<br>`kappa_extra_kappa_CRD_2026-10-20_004454_2026-10-20_011442_ap-lSODwCn5gn3jLeCaCx38fX.json` | The kappa averaged over five probes, as a sensitivity check on the six-probe average (Appendix H). | `2026-10-15_fpr_sweep_507_prep/run_kappa_extra.py` at `10d3b2e28` | input of tables rc1b, rc1b_detl48; 5 numbers: Appendix H (354) |
| `detdrop`<br>`detector_vs_drop_2026-10-20_011851_ap-aYMAJlzcASqKL6EFhdlTkf.json` | A simulation detector's score (layer 63, reading commands) for each task against that task's drop in catch rate from real to simulated runs. No printed number uses it; it is an input of statistics table rc1b, which must be reproduced exactly. | `2026-10-15_fpr_sweep_507_prep/run_detector_vs_drop.py` at `d8b087792` | input of table rc1b |
| `detdrop_l48`<br>`detector_vs_drop_2026-10-20_125606_ap-NuUKhGaNZaLhqdKopouezY.json` | The same as detdrop with the detector at layer 48; Appendix C uses its rank correlation. | `2026-10-15_fpr_sweep_507_prep/run_detector_vs_drop.py` at `76032545d` | input of table rc1b_detl48; 4 numbers: Appendix C (209) |
| `plant_rows`<br>`panel_plant_rows_plant_rows_POOLED_FULL_routeC_2026-10-20_004247_2026-10-20_004833.json` | The planted-error test: every simulated attack run's scores shifted so that simulated catch falls or rises by about 3, 5, 8 or 12 points, and which measures notice it (Table 6). | `2026-10-15_panel_noise_matched/run_plant_rows.py` at `fd12eca35` | input of tables rc1b, rc1b_detl48; 23 numbers: section on catch rates (50); Appendix D (236); Table 6 (246, 247, 248, 249); Appendix D (259) |
| `realline`<br>`realline_boot_read_2026-10-25_180356.json` | The six-probe average of simulated minus real catch when simulated runs are judged against the real threshold, with 95% intervals; the rebuild recomputes these from realline_draws and requires an exact match. | `2026-10-25_realline_final_read/run_realline_boot_read.py` at `9eb78b6cd` | 4 numbers: footnote on the real threshold (104) |
| `realline_draws`<br>`realline_draws.npz` | The 12 resample arrays (10,001 values each) behind realline's intervals, copied unchanged from record A's full resample file, which is too large to include. The one file here that is not a copy of a committed file. | `2026-10-25_realline_final_read/run_extract_realline_draws.py` at `5bcec4bab` | 4 numbers: footnote on the real threshold (104) |

## The simulation detector

| Short name and file | What it holds | Made by (folder in our research repository, script, commit) | Used for (figures; statistics tables; post lines) |
|---|---|---|---|
| `refit3`<br>`attack_all_refit3_2026-10-16_141813.json` | The simulation detector on attack runs: how well a probe reading Qwen3.5-27B's activations tells simulated runs from real ones (Figure 2, Table 7 and Appendix E), after 3 of its 40 fits were refitted to meet the convergence bar. | `2026-10-16_attack_refit3/run_attack_all.py (the detector, first written at 8ea4a6a5d) and run_refit3.py` at `4a9322473` | Figure 2; 25 numbers: section on detecting the simulation (56, 60); Appendix E (277, 279, 281); Table 7 (289, 290, 291); Appendix E (295); Appendix F (303) |
| `honest_all`<br>`honest_all_routec_2026-10-18_121211.json` | The simulation detector applied to honest runs: simulated versus real honest runs, and two sets of real honest runs as a control (Table 7 and Appendix E). | `2026-10-16_honest_sim_detector_all/run_honest_all.py` at `4c29bb573` | 4 numbers: Appendix E (283); Table 7 (292) |

## Runs, tasks, behaviour and worked examples

| Short name and file | What it holds | Made by (folder in our research repository, script, commit) | Used for (figures; statistics tables; post lines) |
|---|---|---|---|
| `pooledmetrics`<br>`pooledmetrics_FULL_E12_r2_union_2026-10-16_145637_ap-CtBFalAop8o369oES3vfrt.json` | Each probe's AUROC (Table 1). Made by an earlier version of the main pipeline; an internal review of the code changes (2026-10-18) found that the final analysis leaves these values unchanged. | `2026-10-15_fpr_sweep_507_prep/run_peraction311.py` at `7be398471` | 6 numbers: Table 1 (28, 29, 30, 31, 32, 33) |
| `behaviour`<br>`main_bootstrap_2026-10-16_100211.json` | The agent's behaviour in simulated and real runs (steps, tool calls, output tokens, how often it attempted the side task), with intervals from resampling main tasks (Appendix G). | `2026-10-16_behaviour_full/run_main_bootstrap.py` at `4a0acbc17` | 20 numbers: footnote on attempts (84); Appendix G (334, 336, 338) |
| `behaviour_medians`<br>`main_bootstrap_2026-10-16_144720.json` | Median steps of runs that did not attempt the side task, with intervals (Appendix G). | `2026-10-16_behaviour_full/run_main_bootstrap.py` at `c7dd37380` | 5 numbers: Appendix G (338) |
| `tables`<br>`examples_v81_tables_pooled_2026-10-20_003359.json` | Scores, thresholds and outcome counts behind the worked examples of Tables 3 and 4 (no command text). | `2026-10-16_examples_v81/run_tables_v81.py` at `f4f575ce6` | 20 numbers: Appendix B (Tables 3, 4) (165, 169, 175, 176, 177, 178, 179, 180, 182, 184, 190, 191, 192) |
| `exclusions`<br>`exclusions_2026-10-16_102159.json` | The tasks left out of the comparison, with their class and the reason (Appendix I and the task counts). | `2026-10-15_fpr_sweep_507_prep/make_exclusions.py` at `6414c492b` | 24 numbers: footnote on task counts (86); Appendix A (118); Appendix I (372, 374, 376, 377, 378, 379, 380, 381, 382, 383) |
| `appI_counts`<br>`appI_real_counts_2026-10-20_021136.json` | Counts of real honest and real attack runs that errored, by environment (Appendix I). | `2026-10-20_appI_counts/count_appI_real.zsh` at `7e0abf8c3` | 6 numbers: Appendix I (387) |
| `appI_counts_e2`<br>`appI_real_counts_2026-10-20_022807.json` | The same counts, adding honest attempts in which no model call gave a usable reply (Appendix I). | `2026-10-20_appI_counts/count_appI_real.zsh` at `b86d7b12f` | 1 number: Appendix I (387) |
| `appendix_a_rates`<br>`appendix_a_rates_2026-10-16_121443.json` | Failure rates of real honest runs (the footnote on failed runs). | `2026-10-15_fpr_sweep_507_prep/make_appendix_a_rates.py` at `35dff7132` | 2 numbers: footnote on failed runs (92) |
| `simpool`<br>`routec_simpool_numbers_2026-10-20_020501.json` | Counts from the final pool of simulated honest runs: the share in which GPT-5.6-Sol wrote part of the simulated output, the failed-run rate, and the run counts of Table 7's honest rows. | `2026-10-20_routec_simpool_numbers/run_routec_simpool_numbers.py` at `8030f8c29` | 6 numbers: footnote on the simulator (82); footnote on failed runs (92); Appendix E (283); Table 7 (293) |
| `attack_sol_share`<br>`attack_sol_share_2026-10-20_023418.json` | The share of simulated attack runs in which GPT-5.6-Sol wrote part of the simulated output. | `2026-10-20_routec_simpool_numbers/run_attack_sol_share.py` at `1c263d842` | 1 number: footnote on the simulator (82) |
| `cunits_by_tool`<br>`cunits_by_tool_2026-10-20_044328.json` | The share of scored commands that are file edits rather than shell commands, in honest and attack runs (Appendix A). | `2026-10-20_routec_fileedit_share/run_routec_fileedit_share.py` at `a6d37c024` | 2 numbers: Appendix A (149) |

## Probe fits and training runs

| Short name and file | What it holds | Made by (folder in our research repository, script, commit) | Used for (figures; statistics tables; post lines) |
|---|---|---|---|
| `read_fit_highstakes`<br>`read_fit_2026-10-14_131233.json` | The high-stakes probe's fit on Qwen3.5-27B: the layer chosen and its held-out scores. | `2026-10-14_p2_highstakes/run_read_fit.py` at `fd47141c3` | 1 number: Appendix A (136) |
| `read_fit_refusal_harm`<br>`read_fit_2026-10-14_125514.json` | The harmfulness and refusal directions' fit on Qwen3.5-27B: the layers chosen and their scores. | `2026-10-14_p2_refusal_harm/run_read_fit.py` at `ecc07824a` | 2 numbers: Appendix A (135) |
| `sola_fit`<br>`sola_uncut_reselect_fit_2026-10-16_030230_cutfull_491db0e7e2c9.json` | The off-policy probe's fit: the layers chosen for each of its five fold copies. | `2026-10-14_q35_sol_avg_POSTHOC/run_avg_transfer.py` at `199f50846` | 3 numbers: Appendix A (133) |
| `sola_keys`<br>`sola_uncut_reselect_training_keys_2026-10-16_030230_cutfull_491db0e7e2c9.json` | The runs the off-policy probe was trained on (run identifiers only). | `2026-10-14_q35_sol_avg_POSTHOC/run_avg_transfer.py` at `199f50846` | 2 numbers: Appendix A (133) |
| `oaru_keys`<br>`oaru_training_keys_2026-10-16_132653.json` | The runs the on-policy probe was trained on (run identifiers only). | `2026-10-16_examples_v81/run_examples_v81.py` at `4cc8da1ba` | 2 numbers: Appendix A (132) |

## The four saved statistics tables

These hold the formatted values the post's statistics were copied from. The rebuild recomputes each table with the original `slot_table.py` (`../slots/`) and requires every entry to match the saved file.

| Short name and file | What it holds | Made by (folder in our research repository, script, commit) | Used for (figures; statistics tables; post lines) |
|---|---|---|---|
| `slots_rc1a`<br>`slots_routeC_writing_rc1a.json` | Saved statistics table rc1a: the post's formatted numbers made by slot_table.py from records A, group_catch and group_excess. The rebuild recomputes it and requires an exact match. | `2026-10-17_routec_slot_readout/run_slot_readout.py and slot_table.py` at `283a711ab` | compared with table rc1a as recomputed here; 232 numbers are read from the recomputed table |
| `slots_rc1b`<br>`slots_routeC_writing_rc1b.json` | Saved statistics table rc1b: as rc1a, plus kappa, kappa_extra, detdrop, anchor and plant_rows. | `2026-10-17_routec_slot_readout/run_slot_readout.py and slot_table.py` at `283a711ab` | compared with table rc1b as recomputed here; 47 numbers are read from the recomputed table |
| `slots_rc1b_detl48`<br>`slots_routeC_writing_rc1b_detl48.json` | Saved statistics table rc1b_detl48: rc1b with the layer-48 detector (detdrop_l48) in place of detdrop. | `2026-10-17_routec_slot_readout/run_slot_readout.py and slot_table.py` at `a0a8cc70c` | compared with table rc1b_detl48 as recomputed here; 4 numbers are read from the recomputed table |
| `slots_rc2`<br>`slots_routeC_writing_rc2.json` | Saved statistics table rc2: the post's formatted numbers from records B and R37. | `2026-10-17_routec_slot_readout/run_slot_readout.py and slot_table.py` at `6f86f2b0b` | compared with table rc2 as recomputed here; 9 numbers are read from the recomputed table |

## Companion label files

Small files that name a result file's run and its sha256. The plotting scripts and `slot_table.py` check them before reading a result file, so that a figure or table cannot be built from the wrong file.

| Short name and file | What it holds | Made by (folder in our research repository, script, commit) | Used for (figures; statistics tables; post lines) |
|---|---|---|---|
| `labels_171`<br>`LABELS_171pool_2026-10-17.json` | The registry of labels that the table script checks to tell the final records apart from records built on an earlier pool of 171 simulated honest runs. | no script identified; added in its own commit (`429232cee`) | input of tables rc1a, rc1b, rc1b_detl48, rc2 |
| `A_label`<br>`routec_A_label_sidecar_2026-10-19_181355.json` | Companion file naming record A's run and its sha256; the plotting scripts and the table script check it. | no script identified; added in its own commit (`095a9db4c`) | Figures 1, 4, 5, 6, 7 (checked); input of tables rc1a, rc1b, rc1b_detl48 |
| `B_label`<br>`routec_B_label_sidecar_2026-10-19_181355.json` | Companion file naming record B's run and its sha256. | no script identified; added in its own commit (`e4cb2a190`) | input of table rc2 |
| `R37_label`<br>`routec_R37_label_sidecar_2026-10-19_181355.json` | Companion file naming record R37's run and its sha256. | no script identified; added in its own commit (`03ec92d1a`) | input of table rc2 |
| `group_catch_label`<br>`group_catch_2026-10-20_003430_LABEL.json` | Companion file naming group_catch's run and its sha256. | `2026-10-15_fpr_sweep_507_prep/run_group_catch.py` at `74edde46f`; committed with its record; the commit message says the same run wrote both | Figures 1, 5 (checked); input of tables rc1a, rc1b, rc1b_detl48 |
| `group_excess_label`<br>`group_excess_2026-10-20_003418_a6cada8b0fa546d890b6976f675f7246_LABEL.json` | Companion file naming group_excess's run and its sha256. | `2026-10-15_fpr_sweep_507_prep/run_group_excess.py` at `74edde46f`; committed with its record; the commit message says the same run wrote both | Figure 4 (checked); input of tables rc1a, rc1b, rc1b_detl48 |
| `pw_shares_label`<br>`pw_shares_pooled_all_origins_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327_routeC_2026-10-20_003446_LABEL.json` | Companion file naming pw_shares's run and its sha256. | `2026-10-15_fpr_sweep_507_prep/run_pw_shares_routec.py` at `74edde46f`; committed with its record; the commit message says the same run wrote both | Figures 1, 5 (checked) |
| `anchor_label`<br>`pertask_full_2026-10-20_004115_anchor_2026-10-20_011851_ap-Ul4ZPPM7A0z6LFMNMfGTqh_LABEL.json` | Companion file naming anchor's run and its sha256. | `2026-10-15_fpr_sweep_507_prep/run_pertask_anchor.py` at `d8b087792`; committed with its record; the commit message says the same run wrote both | Figures 6, 7 (checked); input of tables rc1b, rc1b_detl48 |
| `kappa_label`<br>`routec_kap_label_sidecar_2026-10-20_004454.json` | Companion file naming kappa's run and its sha256. | no script identified; added in its own commit (`10d3b2e28`) | input of tables rc1b, rc1b_detl48 |
| `kappa_extra_label`<br>`routec_kx_label_sidecar_2026-10-20_004454.json` | Companion file naming kappa_extra's run and its sha256. | no script identified; added in its own commit (`82194c0c5`) | input of tables rc1b, rc1b_detl48 |
| `detdrop_label`<br>`detector_vs_drop_2026-10-20_011851_ap-aYMAJlzcASqKL6EFhdlTkf_LABEL.json` | Companion file naming detdrop's run and its sha256. | `2026-10-15_fpr_sweep_507_prep/run_detector_vs_drop.py` at `d8b087792`; committed with its record; the commit message says the same run wrote both | input of table rc1b |
| `detdrop_l48_label`<br>`detector_vs_drop_2026-10-20_125606_ap-NuUKhGaNZaLhqdKopouezY_LABEL.json` | Companion file naming detdrop_l48's run and its sha256. | `2026-10-15_fpr_sweep_507_prep/run_detector_vs_drop.py` at `76032545d`; committed with its record (commit a66fa8d80) | input of table rc1b_detl48 |
| `plant_rows_label`<br>`routec_plant_rows_label_sidecar_2026-10-20_004247.json` | Companion file naming plant_rows's run and its sha256. | no script identified; added in its own commit (`2d69ba990`) | input of tables rc1b, rc1b_detl48 |
