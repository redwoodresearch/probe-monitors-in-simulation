# Rebuilding every figure and number in the post

This folder rebuilds the figures of the post *Probe monitors seem to perform realistically in simulated control environments*
(link to come), and every number in it that comes from data, and checks each one against the post as printed.

It starts from the 50 result files the post was written from (`records/`). It reruns the original plotting scripts, and the
original script that turned the result files into the post's formatted numbers. Then it compares:

- each of the six data figures with the post's figure, byte for byte (or pixel for pixel where only the PNG encoder differs),
  and Figure 3, a diagram, with a reference render;
- each of the 445 numbers in the post that come from data with the number as printed, character for character.

You need no GPU, no API keys and no private code: only Python 3.12 with numpy and matplotlib.

This is the second of two levels of checks in this repository. The first level, in `../analysis/` (`make_numbers.py` and
`make_figures.py`), recomputes the main result from each probe's scores for every command and tool output it reads. This folder reaches further: it covers
every data-derived number and every figure, but it starts from the saved result files rather than from the scores. See
[Overlap with the analysis folder](#overlap-with-the-analysis-folder).

## Quick start

Install [uv](https://docs.astral.sh/uv/). Then, from this folder:

    uv run rebuild_post.py

or, from the folder above (the repository root):

    uv run --project post post/rebuild_post.py

Both run the same check. The first run downloads Python 3.12 and the packages pinned in `uv.lock` (numpy 2.3.3,
matplotlib 3.10.6 and their dependencies). After that it also works offline: `uv run --offline rebuild_post.py`.

It prints its progress through the four checks below and ends with

    PASS: every figure and every rebuilt number matches

and exit code 0. Any difference makes it exit with a non-zero code: it prints the first 60 differences, and `report.json`
lists them all. If an input fails its sha256 check (check 1), it prints every failing input and stops before the other checks.

The outputs go to `rebuild_out/` (`--out DIR` to change it):

- the redrawn figures;
- `rebuilt_post_numbers.json`: every rebuilt number, by its id;
- `report.json`: the comparison, number by number.

`--skip-figures` rebuilds the numbers only. It always exits non-zero, because it is not a full check.

## What it checks

1. **Inputs.** Every file it reads must match the sha256 listed in `MANIFEST.json`: the 50 result files, the copied scripts,
   the post's figures, and this folder's own rule files. So no input can change silently.
2. **Figures.** It reruns the original plotting scripts on the result files, with the arguments and settings of the run that
   drew the post's figures (recorded in `figures/*/render_log_*.json`). Each output must equal that run's output byte for byte,
   or pixel for pixel where only the PNG encoder differs. It also checks that the post's figures are that run's outputs.
   Figure 3 is a diagram with no data. It is compared with a reference render, `expected/figures/fig3_reference.png`, because
   the post's copy was drawn with an older matplotlib.
3. **Statistics tables.** Most statistics in the post were copied from four tables of formatted values, such as
   `−1.3 (−3.5 to +0.9)`: a mean and its 95% interval. The script `slot_table.py` made these tables from the result files; the
   code calls each table entry a *slot*. The rebuild reruns the same script on the same files, and every entry must equal the
   saved table's.
4. **Every printed number.** `expected/expected_post_numbers.json` lists all 708 numbers in the post, each with its line. The
   445 that come from data each have a rule in `number_rules.json`, and the rebuilt text must equal the printed text exactly.
   A rule that checks a claim (such as "within 3 points" or "near 100%") raises an error if the claim fails; it never prints a
   value.

## What is covered

**Figures.** All seven figures in the post. Figures 1, 2 and 4 to 7 are redrawn from the result files; Figure 3 is a diagram.

**Numbers.** Every number in the post has a class. Each number that is not rebuilt carries a one-line reason in
`expected/expected_post_numbers.json`.

| Class | Numbers | Meaning |
|---|---|---|
| R | 445 | Rebuilt from the result files and compared as printed: every numeric cell of Tables 1 and 3 to 8, Table 6's word cells, and the numbers in the text and footnotes |
| C | 116 | Constants of the design and definitions: alarm levels (1%, 5%, 0.1%, 0.02%, 0.05%), the 95% interval level, layer numbers, planted sizes, the 200-step limit, the AUROC and kappa scales |
| S | 124 | Labels: table, figure and appendix numbers, run labels (real run 1 or 2), a programme name, a date |
| X | 5 | Numbers cited from the simulated control evaluations paper |
| H | 10 | The two hypothetical worked examples (Appendices D and H) |
| N | 8 | Counts written from run transcripts, which are not included: Table 4's "about 95 lines" and "20 most recent messages", Appendix B's counts of command types, and "the other three failed" |

A few rules reproduce the post's rounded wording rather than one stored statistic, so they check the wording against the data:

- "about 840" and "about 670" main tasks: 838 training attack runs (real run 1; 1,675 training runs in all, with the honest ones) and four-fifths of them, rounded to 10;
- "about two and about one per main task": honest runs behind each threshold;
- "mostly layers 45 to 49": layer picks within 5 of their median, at least 60% of them;
- "about one in seven": one over the share of calls the trained probes skip;
- "about 21%": an R² of 0.21.

**Not covered.** The text cells of Tables 2 to 4 (commands and descriptions quoted from runs), and everything that happened
before the result files were written; see [What this does not reproduce](#what-this-does-not-reproduce).

## From a number in the post to the code that made it

Every rebuilt number has an id: `L100.2` is the second number on line 100 of the post's source text as checked
(`expected/expected_post_numbers.json` names that text and its sha256). `../analysis/numbers.csv` uses the same ids.

1. Look the id up in `number_rules.json`. Its rule says where the number comes from, in one of three forms:

   | Form | Rules | Example |
   |---|---|---|
   | `slot`: an entry of a recomputed statistics table | 266 | The post says "the 95% intervals are −3.5 to +0.9 points". Rule `L100.2` reads table `rc1a`, entry `02.coheadline.auroc06_subset\|C@0.01.S`, which is `−1.3 (−3.5 to +0.9)`, and takes its number at index 1 (counting from 0): `−3.5`. |
   | `rec`: a value at a key path in a result file, sometimes summed, differenced or compared | 141 | The post says "an AUROC of 0.88". Rule `L56.1` reads `cells → reply\|L48 → problems → c_simatk_vs_run1 → run_level → auroc` in the `refit3` file and prints it to two decimals. |
   | `fn`: a short named function in `post_numbers.py` | 38 | The post says honest-run failures were "3.7–6.8% of honest runs" on real machines. `failed_range` computes 55 failed runs out of 1,455 launched plus 47 earlier attempts that left no record (3.7%), and, assuming all 47 failed, 102 out of 1,502 (6.8%). |

2. Find the result file. A `rec` rule names it. For a `slot` rule, `config/slot_runs.json` lists the result files the table is
   built from. For an `fn` rule, the function in `post_numbers.py` reads a result file by name (for example
   `records.json("appendix_a_rates")`) or a statistics table.
3. [`records/README.md`](records/README.md) describes every result file in one line, and gives the experiment folder, script
   and commit that made it, and the figures and post lines that use it. `MANIFEST.json` gives each file's original path.

`rebuild_out/report.json` shows, for every number, the printed text, the rebuilt text and whether they match.

## Overlap with the analysis folder

`../analysis/make_numbers.py` checks 267 numbers against `../analysis/results/`, which `../analysis/main_analysis.py` computes from the probes' scores. All 267
are among the 445 rebuilt here, under the same ids and with the same printed text, so they are checked at both levels: the
catch rates, their differences and intervals, the comparison in Figure 4, the per-task R² of Appendix H and the
real-threshold footnote.

The other 178 are checked here only. They come from the post's other experiments:

- the simulation detector, on attack and on honest runs (Figure 2, Table 7, Appendix E, one number in Appendix F): 29;
- the excluded tasks and the run counts (Appendix I, Appendix A, the footnote on task counts): 31;
- the planted-error test (Table 6 and the text around it): 23;
- the worked examples (Tables 3 and 4): 20;
- the agent's behaviour (Appendix G and the footnote on attempts): 25;
- the simulated runs' pool, the share of their output written by GPT-5.6-Sol, and the failure rates (two footnotes, Table 7, Appendix E): 9;
- Table 1's AUROCs: 6;
- the per-task kappa (Appendix H): 10;
- the other alarm levels and the layer-48 detector (Appendix C): 13;
- the probes' fits, training runs and the share of file edits (Appendix A): 12.

`../analysis/make_figures.py` draws Figures 1 and 4 to 7 from the recomputed results. This folder redraws the same five, and Figure 2,
with the original plotting scripts.

## Terms

- **Main task.** Each LinuxArena task pairs an ordinary main task, such as fixing a bug in a web service, with a harmful side
  task. Honest runs get the main task alone; attack runs also get the side task. Every 95% interval comes from resampling main
  tasks.
- **Alarm level.** The share of honest commands (or honest tool outputs) that raise an alarm. At the 1% level, a probe's
  threshold is set so that 1% of honest commands score above it.
- **Catch rate.** The share of attack runs with at least one command (or tool output) above the threshold.
- **Real threshold.** The post's main comparison sets a separate threshold in each setting, from that setting's own honest runs.
  The real-threshold footnote instead applies the threshold set on real honest runs to the simulated attack runs.
- **Planted errors.** A test that fakes a simulator with a known error: every simulated attack run's scores are shifted by a
  fixed amount, so that simulated catch falls (or, for single probes, rises) by about 3, 5, 8 or 12 points, to see which
  measures notice it (Table 6).
- **Simulation detector.** A probe trained to tell simulated runs from real ones by reading the agent's activations (Figure 2,
  Table 7).

## Code structure

```
post/
├── rebuild_post.py       entry point: runs the four checks in order and writes the report
├── post_figures.py       check 2: reruns the plotting scripts and compares their outputs
├── post_numbers.py       checks 3 and 4: reruns slot_table.py, then applies number_rules.json
├── number_rules.json     one rule per rebuilt number: where its value comes from
├── MANIFEST.json         sha256, original path and commit of every input file, and the expected class counts
├── pyproject.toml        Python 3.12, numpy, matplotlib; uv.lock pins every dependency
├── records/              the 50 result files the post was written from; records/README.md describes each one
├── figures/              the original plotting scripts, one folder per batch of figures; each data batch also has
│                         the log of the run that drew the post's figures and an input map (inputs.json);
│                         fig3/ holds only the diagram script
├── slots/                the original slot_table.py, in the two versions that were run
├── config/               slot_runs.json: which result files each of the four statistics tables uses
└── expected/             what the post shows: every printed number, and the post's figures
```

The new code is `rebuild_post.py`, `post_figures.py` and `post_numbers.py`, with the rules in `number_rules.json` and the input
maps (`figures/*/inputs.json`, `config/slot_runs.json`). The plotting scripts and render logs under
`figures/`, the scripts under `slots/`, the result files and the post's figures under `expected/figures/` are copies of the files that
produced the post, none edited (`MANIFEST.json` records `edits: none` for each copied file), with two exceptions among the result
files: one is stored compressed (`pertask_full_...json.gz`), and `realline_draws.npz` holds 12 resample arrays copied unchanged from a
result file too large to include (`records/README.md` describes both). The lists of expected numbers and figures, Figure 3's
reference render, `records/README.md` and the project files were made for this package. The
`_source.json` and `_text.md` files next to each figure are side outputs of the plotting scripts; the check compares the
redrawn ones too.

## The result files keep their internal names

The result files, the statistics tables and the plotting scripts' side outputs keep the file names and metadata they had in our
research repository. The checks pin every one of them by sha256, and the plotting scripts and `slot_table.py` check the
companion label files by name and hash, so renaming or editing them would break the checks. Their names and metadata therefore
contain our working labels. The glossary below explains the recurring ones, and [`records/README.md`](records/README.md) gives
each file a plain description.

### Glossary of working labels

Found in the result files' names and metadata, and in the original scripts. Each meaning comes from our research repository's
code, analysis plan or commit messages; where only the code shows it, the entry says so.

| Label | Meaning |
|---|---|
| `routec`, `routeC`, `route C` | The final version of the pooled analysis on all 1,313 tasks. The name comes from the third of three options we considered for adding 20 pilot runs to the simulated honest runs of the original tasks; it stuck for the whole final analysis. `routecA` marks figures drawn from its main result file, A. |
| `A`, `B`, `R37`, `sweep15`, `rule37` | The main result file (1% and 5% alarm levels); the same analysis at 15 alarm levels from 0.1% to 5% (`sweep15`); and at the three strictest levels, 0.01%, 0.02% and 0.05% (item 37 of our analysis plan, hence `rule37` and `R37`). |
| `alltasks` | All 1,313 tasks. |
| `CRD` | One job computed three kinds of scores: probes on commands (C), probes on tool outputs (R), and two separate command monitors for destructive actions and data transfer (D). |
| `C`, `R` after a probe name | The probe's scores on commands (C) or on tool outputs (R). |
| `FULL`, `uncut`, `cutfull`, `32k` | Full-length transcripts. Earlier builds read only the first 32,768 tokens of each run (`32k`). `<probe>\|uncut` is a probe with fixed weights scored on full transcripts; `<probe>_uncut` is a probe retrained on full transcripts; `cutfull` tags the files of that full-length rebuild. `32k-frozen` is the set of tasks as fixed for the 32,768-token build. |
| `r2` | The second data refresh, which added the paper's remaining tasks as a fourth task set (`S_paper493`). Not "run 2" and not R². |
| `S_old`, `S_new`, `S_newmains`, `S_paper493`, `paper_cells` | The four task sets: the original pilot tasks; the tasks added in the first scale-up; tasks on main tasks outside the simulated control evaluations paper's; and the rest of that paper's tasks. `paper_cells` is the tasks from the paper, that is, all but `S_newmains`. |
| `allorigins`, `all_origins`, `pooled_all_origins` | All four task sets pooled into one result, with each main task counted once. Every headline number uses this pool. |
| `pooledworld`, `pooled_world`, `scored` | One alarm threshold per world (real or simulated), set on the honest runs of all task sets together, rather than one per task set. `scored` is the scoring pass of that build (an earlier pass only listed the honest runs). |
| `e12`, `E12`, `e2` | Real thresholds set on two batches ("epochs") of real honest runs per task instead of one; `e2` is the second batch. |
| `union` | Intervals from resampling each main task once across task sets. The file with this tag (`pooledmetrics`, Table 1) was made before the final version and sets one threshold per task set. |
| `honest_a`, `honest_b` | The two independent real honest runs per task. The simulator was given run A as its reference, so real thresholds are set on run B; run A serves only the real-versus-real comparison. |
| `run1`, `run2`, `sim`, `run2@A` | Real attack run 1, real attack run 2 and the simulated attack run; `run2@A` is real run 2's catch rate at the threshold set on honest run A. |
| `sim2`, `run3` | A second simulated and a third real attack run per task, launched together to check for effects of the launch itself. |
| `L48`, `L63` | Layers 48 and 63 of Qwen3.5-27B. |
| `own_all_rs_uncut_L48`, `oaru`, `own_all_rs` | The on-policy probe: trained on Qwen3.5-27B's own runs with re-split cross-validation folds (`rs`), retrained on full transcripts, at layer 48. `oaru` is its initials (from the code; no document spells it out). `rs10` and `rs25` are versions trained on 10% and 25% of the tasks. |
| `sola`, `Sol (a)`, `sol_a_uncut`, `reselect` | The off-policy probe: Qwen3.5-27B reading GPT-5.6-Sol's runs, trained with training recipe (a). `reselect` means its layers were chosen again on full-length data. |
| `FIBS` | The deception probe: FAR.AI's published "Telling FIBS" probe, used unchanged. |
| `harmfulness`, `refusal`, `high-stakes` | The harmfulness direction, the refusal direction and the high-stakes probe. |
| `refit3` | The simulation detector after refitting the 3 of its 40 cross-validation fits that missed the convergence bar; the other 37 are unchanged. |
| `detector_vs_drop` | Per task, how simulated the detector finds the simulated run, against how far the probe's score fell from the real to the simulated run (a rank correlation). |
| `pertask_full`, `anchor` | Per-task results: each run's highest score relative to its threshold. `anchor` is the reference run, real run 1, for the per-task R². |
| `kappa` | Per-task agreement (Cohen's kappa) on caught versus missed at the 1% level. |
| `plant_rows`, `PLANT`, `POOLED_FULL` | The planted-error test (Table 6). `POOLED_FULL` is its real configuration (pooled thresholds, full-length transcripts), as opposed to its test configurations. |
| `slot`, `slots`, `rc1a`, `rc1b`, `rc1b_detl48`, `rc2` | The saved statistics tables (see check 3), made in stages: `rc1a` the main catch rates and averages; `rc1b` adds the per-task, kappa, detector and planted-error files; `rc1b_detl48` uses the layer-48 detector; `rc2` the other alarm levels. |
| `coheadline` | The signed average over the six (or four) probes: `S` is simulated minus real catch, `Y` is real run 2 minus real run 1. |
| `LABELS_171pool`, `171-pool` | Results built before the final version, when the original tasks' pool of simulated honest runs had 171 runs. The registry lists their labels so that the table script can refuse them. |
| `_LABEL`, `label_sidecar` | A companion label file: see [`records/README.md`](records/README.md). |
| `simpool` | Numbers that depend on the final pool of simulated honest runs (the earlier pool plus 20 pilot runs). |
| `appI` | Appendix I of the post. |
| `XDB` | The initials of the three input files of one error count in `count_appI_real.zsh` (from the code; no document spells it out). |
| `Stage A`, `stage D` | Steps of the main pipeline: catch rates and gaps (A); per-task top scores (D). |
| `d1` | A rule that each pooled threshold is computed once, with the main result, and reused by later per-task and kappa jobs. |
| `fpr_sweep_507`, `run_peraction311` | The folder and script of the main pipeline. They began as a false-alarm-rate sweep for a 507-task scale-up and a script for its 311 added tasks, and grew into the final analysis. |
| `examples_v81`, `attack_v5`, `figs156`, `figs78` | Names from the drafts: the worked examples made for draft 81; the fifth render of Figure 2; and figure batches named by an earlier draft's numbering (`figs156` is the post's Figures 1, 4 and 5; `figs78` its Figures 6 and 7). Figure 4's file is `fig5_excess_C`, and Figure 5's is `fig1_main_R`. |
| `ap-…` | The id of the cloud job (on Modal) that wrote the file. |

## What this does not reproduce

The result files are taken as given. Making them needed:

- **The runs**: real and simulated LinuxArena attack and honest runs, using Qwen3.5-27B served with vLLM, GPT-6-Astra and
  GPT-5.6-Sol through their APIs, and Redwood's private simulator.
- **Activations, probe training and scoring**: Qwen3.5-27B activations extracted on H200 GPUs.
- **Most resamples**: every interval comes from 10,000 resamples of main tasks. The result files store each interval's
  summary, which is what the rebuild uses; only the 12 arrays behind the real-threshold footnote are included in full
  (`records/realline_draws.npz`).
- **Text cells**: the commands and descriptions in Tables 2 to 4 come from run transcripts, which are not included.
