# 2026-10-13_whitening_POSTHOC

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Can real honest runs help a probe trained only on simulated attack vs simulated honest runs, without being used as labels, by whitening the features with real honest statistics?

## What we did

Refit the simulation-only tool-output probe (Qwen3-32B reading GPT-5.6-Sol runs) after diagonal, Mahalanobis or ZCA whitening estimated on real honest runs, with retuned regularization, shuffle nulls, inner-fold refits, a training objective rescaled by the regularization strength and tight-tolerance spot checks (13 amendments). Modal.

## What we found

README Amendment 12 (converged refits): at 1% ZCA real catch 70.3 vs plain 67.0; ZCA-minus-plain change in second-real-run catch +3.3 [-7.1, +7.2] and gap +1.1 [-1.3, +2.8], so 'helps' fails under all three gates at 1% and 5%; earlier retuned numbers came from early-stopped fits and are superseded.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/) (simulated/simulated cell); [2026-10-13_why_steps](../2026-10-13_why_steps/) (packs)

## Main scripts

- [`run_whitening.py`](run_whitening.py)
- [`run_zca.py`](run_zca.py)
- [`run_zca_c.py`](run_zca_c.py)
- [`run_mahal.py`](run_mahal.py)
- [`run_null.py`](run_null.py)
- [`run_converged.py`](run_converged.py)
- [`run_converged_nulls.py`](run_converged_nulls.py)
- [`run_cscaled.py`](run_cscaled.py)
- [`run_tight.py`](run_tight.py)
- [`run_innerfold.py`](run_innerfold.py)
- [`run_cequiv_all.py`](run_cequiv_all.py)

Result files the index names:

- `cscaled_POSTHOC_2026-10-14_055510.printout.txt` (not copied: lines over 300 characters)
- [`converged_POSTHOC_2026-10-14_061158.printout.txt`](results/converged_POSTHOC_2026-10-14_061158.printout.txt)
- [`convnulls_POSTHOC_2026-10-14_062021.printout.txt`](results/convnulls_POSTHOC_2026-10-14_062021.printout.txt)
- [`tight_POSTHOC_2026-10-14_063303.printout.txt`](results/tight_POSTHOC_2026-10-14_063303.printout.txt)

## Files here

21 code and configuration files and 4 result files, copied byte for byte from `experiments/2026-10-13_whitening_POSTHOC/` at commit `4af3665c6b7ed1c398b2bf61fdc65bb1851ad675` of our internal repository. 56 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
