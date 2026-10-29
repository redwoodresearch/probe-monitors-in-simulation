# 2026-10-12_claimB_provenance

2026-10-12 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Bookkeeping: keep every commit cited by the October 12 fixed-simulator check branches reachable from main.

## What we did

A merge that changes no files (git merge -s ours) recording the as-run commits of six stacked branches of fixed-simulator checks (TF-IDF environment-halves test (training on one half of the environments and testing on the other, in both directions), the operating-point planted check, the edit-count artifact check, the planted-check-1 layer sweep, Llama-3.2-1B environment-halves test (training on one half of the environments and testing on the other, in both directions), the convergence check) and four other sources.

## What we found

README table of 10 cited commits; no files or numbers.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-12_claimB_tfidf_halves](../2026-10-12_claimB_tfidf_halves/); [2026-10-12_claimB_planted_oppoint](../2026-10-12_claimB_planted_oppoint/); [2026-10-12_claimB_edit_artifact](../2026-10-12_claimB_edit_artifact/); [2026-10-12_claimB_plant1_layers](../2026-10-12_claimB_plant1_layers/); [2026-10-12_claimB_1b_halves](../2026-10-12_claimB_1b_halves/); [2026-10-12_claimB_convergence_check](../2026-10-12_claimB_convergence_check/)

## Main scripts

- none

## Files here

0 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_claimB_provenance/` at commit `a5821536ddd10e80e7e7c0766028cfd6a0f60976` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
