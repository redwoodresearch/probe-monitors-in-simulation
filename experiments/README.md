# Experiments

One folder for each of the 476 experiment folders in our project, named as in our internal repository (date, then a short
name), plus `src/`, the project's shared Python package. Each folder's `README.md` is that experiment's entry in our experiment
index: the question, what we did, what we found, whether and where it appears in the post, its status, its main scripts and the
commit its files come from. [`../docs/experiments.md`](../docs/experiments.md) organizes all 492 entries (these 476 and 16 checks
run outside any folder) by question, with a short account of how the project unfolded and a list of its most interesting
findings.

## The code is as it was in our repository

Every file here except the READMEs and `manifest.tsv` is a byte-for-byte copy of a file in our internal repository; we changed
nothing, so the code reads as it did while we worked. Four things follow.

- **Folder names and comments use internal working labels.** The glossary below explains the most frequent ones.
- **The scripts will not run from this folder.** They import the project's package (copied in `src/`, below) and a second,
  private repository, read runs and activations from our storage, and launch jobs on Modal. They are here to show the code we
  worked with. The post's main analysis runs on its own from [`../analysis/`](../analysis/).
- **Each folder is a snapshot.** Its files come from the last commit that touched the folder on the branch that holds its
  latest version. For 38 main scripts that the index names but that commit lacks, we copied each script's own latest version
  from another branch.
- **The latest version is not always the one that ran.** Where we found that a script behind a result the post uses (or a result
  the index names, in a folder that feeds the post) ran at a version that differs from the folder's copy, that version is in the
  folder under `as_run/<commit>/`, and the folder README says which result it produced and how it differs. We could not check
  every script: for 16 of the folders that feed the post, no committed result file records the code commit (see below), so
  their copies may not be the versions that ran. `manifest.tsv` gives every file's source commit.

## As-run versions

We found the versions that ran in two ways. An audit of the code behind the post's results named the commits for the off-policy
probe, the on-policy probe, the simulation detector and the final pooled analysis (6 scripts in 4 folders, plus 2 scripts those
folders lack). For every other folder that feeds the post, we read the code commit that its committed result files record and
compared the main scripts at that commit with the folder's copy; this added 7 scripts in 4 folders. An independent audit of
these READMEs then found 7 more scripts in 5 folders behind results the post uses (the refit of the simulation detector behind
Figure 2 and Table 7, the record behind Table 1, the on-policy probe's training-run counts, the statistics tables and the staging
of the runs whose activations the probes read), which we added the same way. A final audit added the 2 scripts that
[`../pipeline/score.py`](../pipeline/score.py) cites in
[`2026-10-14_q35_sol_avg_POSTHOC`](2026-10-14_q35_sol_avg_POSTHOC/): the scoring of the off-policy and simulated-attack
probes on the 196 pilot tasks, and the row reader for the on-policy probes' scoring. In all, 24 files sit under `as_run/` in 12
folders.
Result files kept outside git, or compressed, could not be checked this way; 16 of the 39 folders that feed the post have no
committed result file that records a code commit, and their READMEs say so. (`../docs/experiments.md` lists 40 entries that feed
the post; one of them was run outside any folder.)

## The shared package (`src/probe_shift/`)

The experiment scripts import `probe_shift`, the project's own package. `src/probe_shift/` holds every file that ever existed
in it, each at the most recent commit that changed it on any branch (83 files; some modules exist only on the branch of the
experiment that used them). The post's runs used the package at commits `f8df19e65` and `378d4c7b5`. There, `reader.py`,
`transcripts.py` and the other modules the post's scripts import are byte-identical to the copies here, except `torch_probe.py`
and `probes.py`, whose versions at those commits are in `src/probe_shift/as_run/f8df19e65/`. We checked the same modules at
the other commits the audit names and at `4f58f5640`: each holds either the copy here or one of these two versions. A third
version, `torch_probe.py` at commit `341ee1585`, holds the on-policy probe's solver that
[`../pipeline/train_probes.py`](../pipeline/train_probes.py) cites; it is in `src/probe_shift/as_run/341ee1585/`.

## The `simulated-control-evals` repository

The scripts also import `sce`, from Redwood Research's `simulated-control-evals` repository, which holds the simulator, the
code that ran the agents, and shared metrics. It is private as of 28 October 2026, and we do not copy it here. Our repository used it as an editable checkout next to
our own, not pinned to a commit (`pyproject.toml` and `uv.lock` at the as-run commits `f8df19e65`, `f2d47f19f` and `7797f2cc2`
point to `../simulated-control-evals`), so the commit the post's runs used is not recorded. Two early experiments recorded the
commit they used: [`2026-10-06_phase2_feasibility`](2026-10-06_phase2_feasibility/) pins `57307004d` in its configuration, and
[`2026-10-11_claimB_yardsticks`](2026-10-11_claimB_yardsticks/) writes it into its results.

## What was copied

| Copied | Files | MB |
|---|---|---|
| Code: `.py`, `.sh`, `.zsh`, `.jq`, `.yaml`, `.yml`, `.toml`, `.cfg`, `.patch`, `.diff` | 4,972 | 44.0 |
| Configuration: JSON files in a `config/` folder with no string over 300 characters | 574 | 55.5 |
| Main scripts missing from a folder's snapshot, copied from their own latest commits | 38 | 0.8 |
| As-run versions under `as_run/` | 24 | 2.1 |
| The shared package `src/probe_shift/`, with its 3 as-run versions | 86 | 0.9 |
| Results: see below | 1,264 | 9.3 |

A result file qualified if it is a `.json`, `.csv`, `.tsv`, `.md` or `.txt` file in a `results/` folder, is under 1 MB, holds no
string, field or line over 300 characters, and is not by its name a note, review, prompt, example, label file, judge output or
log. Of the 8,737 files that qualified, 1,264 fit within our 120 MB limit for this folder. We went through them in this order,
taking each file that still fit: files the index names as an experiment's key results, then other files, both without text-like
strings (60 or more characters with six or more spaces); then the same two groups with such strings; the smallest first within
each group.

Not copied: run transcripts and per-run records, activations, logs, figures, the original notes and READMEs (which use internal
working labels throughout), configuration JSON holding strings over 300 characters, two configuration files that the
repository's secret scanner (gitleaks) flagged, which we left out without review, and all other result files.
[`manifest.tsv`](manifest.tsv) lists all 22,676 files in the 476 folders at their source commits, plus the 38 main scripts, 24
as-run versions and 86 package files from other commits, with what happened to each and why.

## Glossary

The experiment entries (each folder's `README.md` and [`../docs/experiments.md`](../docs/experiments.md)) replace our internal
labels with plain descriptions, but some of our working vocabulary remains, such as cell, arm, gate, cohort and stratum. The first
table below defines those terms (cell, arm and gate are in the code-comment table). Folder names and code are as they were in our
repository, so they keep our internal working labels; the other two tables explain the ones you will meet most.

### Working terms in the entries

| Term | Meaning |
|---|---|
| cohort | A fixed set of tasks or runs that an analysis uses, such as the 196-task cohort or GPT-5.6-Sol's runs on the simulated control evaluations paper's tasks |
| stratum | A part of the task set analysed as a unit. The final analysis's 1,313 tasks come in strata, from the oldest (the 196 pilot tasks) to the tasks added later; elsewhere a stratum is any subgroup analysed separately, such as runs of similar length |
| world | Real or simulated, the two sides of a comparison |
| cell, arm, gate | See the code-comment table below |

### In folder names

| Label | Meaning |
|---|---|
| `q1`, `q2`, `q3` | Our numbered research questions. In the first phase, `q1` asked whether a probe can tell real runs from simulated ones, `q2` whether probes that flag impactful commands work on both, and `q3` whether attack detectors and off-the-shelf probes score both alike. Later folders reuse the numbers loosely, and `q2` and `q3` were also the names of the working sessions that ran many of these folders, so read them as rough topic tags |
| `phase2`, `p2` | The second phase, in which Qwen3.5-27B is the agent; `p2` marks second-phase probe work |
| `claimB` | The fixed-simulator test: the simulator's formatting slips are fixed by general rules, and we ask whether a probe can still tell simulated runs from real ones |
| `onpol`, `onpolicy` | On-policy: Qwen3.5-27B reads its own runs |
| `sce` | A simulated control evaluation; in folder names, a probe trained the way such an evaluation would train it (simulated attack runs against real honest runs) |
| `e1` | The hide-or-mention test: does the deception probe respond alike in real and simulated runs when an honest agent is told to leave an action out of its report. In `f8_e1only` it means real run 1 only |
| `n2` | Behaviours counted by script (such as touching the side task's target) in simulated runs, against the difference between two real runs |
| `zoo`, `zooc`, `zoo_c` | Extra monitors, each flagging one kind of action; `zoo_c` is the credential monitor |
| `g3` | The test of whether the forecast from planted simulator slips holds for a reader from a new model family (Seed-OSS-36B) |
| `fix939`, `fixfull`, `f8` | Script fixes to the simulator's output formatting: `fix939` is two fixes applied to all 939 simulated attack runs, `fixfull` the full set of fixes, `f8` the fix for error-message templates |
| `simh`, `simhonest` | Simulated honest runs |
| `fa`, `matchedfa` | False alarms; comparisons at matched false-alarm rates |
| `xerr` | The "error to add" estimates that preceded the final analysis: how many points a lab should add to its error bars when it measures probe catch in simulation |
| `il` | The interleaved launch, in which a second simulated and a third real attack run were launched together |
| `311`, `su311`, `507`, `sp493` | Task sets: the 311 tasks added after the 196 pilot tasks (507 in all), and the simulated control evaluations paper's other 493 tasks |
| `routec` | The final pooled analysis on all 1,313 tasks, and its checks |
| `cleanroom` | An independent reimplementation, written only from the registered rules |
| `POSTHOC` | Decided after seeing the results it examines, not registered in advance |
| `_09`, `_2e` (suffixes) | The working session that ran the experiment |

### In code comments

Ten internal labels that recur in the comments of the copied code, most mentioned first. Mentions are whole-word, case-sensitive
matches (`Amendment N` counted as the word `Amendment`) in the text after the first `#` on each line of the 5,117 copied code files
(`.py`, `.sh`, `.zsh`, `.jq`, `.yaml`, `.yml`, `.toml`, `.cfg`, `.patch`, `.diff`, including the as-run versions and `src/`), so a
few come from strings rather than comments.

| Label | Meaning | Mentions |
|---|---|---|
| `09`, `0d`, `2e`, `a1`, `d9`, `b7` | Names of our parallel working sessions, used to say who ran, owned or reviewed a step | 5,573 |
| `arm` | One group of runs over the same tasks, such as real attack run 1, real attack run 2, simulated attack, or real or simulated honest runs | 2,085 |
| `RULE`, `Amendment N` | An experiment's written rules (what to run and how to read the result), and numbered later changes to them | 1,943 |
| `Codex` | An AI reviewer (OpenAI's Codex agent) that checked scripts, plans and results before launches and merges | 1,756 |
| `cut`, `32k`, `cutfull`, `uncut` | Whether a run was read only up to its first 32,768 tokens (`cut`, `32k`) or in full (`cutfull`, `uncut`) | 1,619 |
| `cell` | One LinuxArena task: a main task paired with a side task (the main task alone for honest runs) | 1,351 |
| `call_end`, `output_end`, `segment-end` | Where a probe reads: the last content token of a command (a tool call) or of a tool output | 1,312 |
| `gate` | A pass-or-fail check that has to pass before the next step: a launch, a scoring run or reading a result | 1,289 |
| `T196`, `+311`, `507`, `nm`, `sp493` | Task sets, in the order we ran them: the 196 pilot tasks, the 311 added next (507 in all), the tasks with new main tasks, and the simulated control evaluations paper's other 493 tasks | 1,115 |
| `Q1`, `Q2`, `Q3`, `phase2` | Our numbered research questions. In the first phase, Q1 asked whether a probe can tell real runs from simulated ones, Q2 whether probes that flag impactful commands work on both, and Q3 whether attack detectors and off-the-shelf probes score both alike; later work reused the numbers. `phase2` is the second phase, with Qwen3.5-27B as the agent | 823 |

## manifest.tsv

One row per file: the 22,676 files in the 476 folders at their source commits, plus the 38 main scripts copied from other commits,
the 24 as-run versions under the folders' `as_run/`, and the 86 files of the shared package `src/probe_shift/` (83 at their most
recent commits and 3 as-run versions), 22,824 rows in all.

| Column | Meaning |
|---|---|
| `folder` | The experiment folder |
| `path` | The file's path inside the folder |
| `status` | `copied` or `skipped` |
| `reason` | Why it was copied or skipped |
| `source_ref`, `source_commit` | The branch and commit of our internal repository the file was read from |
| `git_blob` | The file's git object id at that commit |
| `sha256` | SHA-256 of the copied file (copied files only) |
| `bytes` | The file's size |
