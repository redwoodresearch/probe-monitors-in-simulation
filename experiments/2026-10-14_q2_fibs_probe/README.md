# 2026-10-14_q2_fibs_probe

2026-10-14 · part of [Training and scoring the probes](../../docs/experiments.md#training-and-scoring-the-probes)

## Question

Make FAR.AI's published FIBS deception probes for Qwen3.5-27B usable without running untrusted pickle code.

## What we did

One isolated Modal CPU container downloaded the pinned model.pt files for the six linear probes (layers 19, 27, 35, 43, 51, 58), checked hashes, walked the pickle opcodes without unpickling, read the raw float32 storages and wrote safetensors plus a record.

## What we found

Six safetensors files written (layers 19–58); the experiment's written rules parked scoring until the main figure was out; later scoring ([2026-10-14_p2_fibs_pair](../2026-10-14_p2_fibs_pair/)) checks each file's sha256 against this conversion.

## In the post?

In the post: Table 1 and Appendix A (deception probe weights)

## Status

Current

## Builds on

Nothing earlier in this index.

## Main scripts

- [`run_convert_fibs_linear.py`](run_convert_fibs_linear.py)

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran.

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_q2_fibs_probe/` at commit `f4d30ebef93121dda450d5ab0f6dbd8de8082e8e` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
