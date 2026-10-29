# 2026-10-12_writeup_numbers

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Does every number in the 10-12 write-up outline match its result file?

## What we did

Listed each outline number with its result file, JSON key and commit (specs assembled by agents), checked each by script, listed figures with data owners, and ran independent Codex reviews of the old-Qwen, tool-error and realism-probe claims.

## What we found

Of 255 numbers, 197 match, 4 do not, 7 have no source, 61 come from unmerged work; 1 of 12 figure images matches a file; the realism-probe paragraph mixed six experiments; one old-Qwen count was wrong (774 of 1,146 should be 722 of 1,195).

## In the post?

Not in the post (checks the 10-12 outline, not the final post).

## Status

Current (for the 10-12 outline)

## Builds on

Many (the outline's sources)

## Main scripts

- [`analyze_outline_numbers.py`](analyze_outline_numbers.py)

Result files the index names:

- `NUMBERS.md` (not in this folder at the source commit)
- `numbers.csv` (not in this folder at the source commit)
- `FIGURES.md` (not copied: notes or README (internal working labels))

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_writeup_numbers/` at commit `934beb1723e350e11b2317f2164fb55de5febc0f` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
