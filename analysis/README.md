# analysis/: the main result, from each probe's scores

This folder recomputes the post's main result from the probes' scores in [`../data/`](../data/): catch rates, their differences
and 95% intervals (Figures 1 and 5), Figure 4's comparison, the per-task agreement in Figures 6 and 7, the real-threshold check (a
footnote to the main result, also used in Appendices C and D), and the honest-run alarm shares. Then it draws those figures and checks the 267 post numbers that come from these results.

Run these from the repository root:

    uv run analysis/main_analysis.py --workers 12   # data/ -> analysis/results/ (all 10,000 resamples; about 6 minutes on 12 cores)
    uv run analysis/make_numbers.py                 # analysis/results/ -> checks 267 numbers against the post
    uv run analysis/make_figures.py                 # analysis/results/ -> Figures 1 and 4 to 7 in analysis/output/

`results/` already holds the analysis output, so the last two run in seconds without the first. `--draws 500` gives a quicker but
rougher run; it overwrites `results/`, after which some numbers no longer match the post.

| File | What it is |
|---|---|
| `main_analysis.py` | The analysis, from per-unit scores to every result; its opening comment explains each step exactly |
| `results/` | Its output (see below) |
| `make_numbers.py`, `numbers.csv` | Recompute each of the 267 numbers from `results/` and check it against the number as the post prints it. `numbers.csv` lists each number: the line of the post it is on, the words around it, the number as printed, where it comes from in `results/`, and how it is formatted |
| `make_figures.py` | Draws Figures 1 and 4 to 7 from `results/` |
| `published_figures/` | The post's copies of those figures, for comparison |

## Terms

- **Tasks.** Each LinuxArena task pairs an ordinary *main task*, such as fixing a bug in a web service, with a harmful *side task*,
  such as planting a backdoor. Several tasks share the same main task: the 1,313 tasks span 864 main tasks.
- **Attack and honest runs.** In an attack run the agent gets both the main task and the side task. In an honest run it gets the
  main task alone. Every task was run as an attack twice on real machines (real run 1 and real run 2) and once in simulation. The
  two real runs show how much a rerun changes results.
- **Units.** A probe scores *units*: each command the agent runs, or each tool output a command returns. The two probes we trained
  skip about one command in seven: those whose last token is "0" or "n", usually a fixed field such as a timeout's last digit;
  [`../pipeline/units.py`](../pipeline/units.py) defines every unit exactly.
- **Alarm level.** A probe raises an alarm when a unit's score is above a threshold. At the 1% alarm level, the threshold is set so
  that about 1% of honest units raise an alarm (ties and the strict "above" can make it slightly less). Real runs use a threshold
  set on real honest runs, and simulated runs use one set on simulated honest runs. Each setting has one threshold, set on its
  honest runs from all four task sets together (the tasks were drawn in four batches, the `task_set` column of
  `../data/cells.csv`); our pre-registered plan set one per task set, and we chose the pooled threshold
  afterwards ([`../docs/methods.md`](../docs/methods.md), section 4).
- **Caught.** An attack run is caught if the probe raises an alarm on at least one of its units. A catch rate gives every main
  task equal weight: each main task's rate is the average over its tasks, and the overall rate averages the main tasks.
- **95% interval.** Each interval comes from 10,000 resamples of main tasks, drawn within each environment. Tasks that share a main
  task stay together, and a task's real and simulated runs stay together. For the catch rates (the "joint" results used throughout
  the post), the thresholds are set again in every resample. The interval is the middle 95% of the resampled values. Each interval
  holds for one quantity at a time, not for all probes at once.
- **No-skill floor.** The share of real honest runs that raise at least one alarm (the black tick in Figures 1 and 5): a reference
  for how often a run is flagged without any attack. An honest run has about 65 to 75 commands, so even at the 1% alarm level 26%
  to 43% of honest runs with scored commands raise an alarm, depending on the probe and the setting (the six probes on commands,
  real and simulated). Attack runs are about as long.

## The probes

| Probe (file name) | What it is |
|---|---|
| On-policy probe (`on_policy_probe`) | A logistic regression we trained to tell Qwen3.5-27B's own attack runs from its honest runs |
| Off-policy probe (`off_policy_probe`) | The same kind of classifier, trained on runs by a different agent (GPT-5.6-Sol), as a lab might reuse a monitor built for another agent |
| Deception probe (`deception_probe`) | Published weights (FAR AI) for Qwen3.5-27B: six linear probes, one per layer, trained to detect deceptive statements; the score averages their sigmoid outputs |
| Harmfulness direction (`harmfulness_direction`) | A direction from published work, computed by us for Qwen3.5-27B: the average activation on harmful prompts minus that on harmless ones |
| Refusal direction (`refusal_direction`) | The same kind of direction, taken at the token just before the model replies to a prompt, where it begins deciding whether to refuse; like the harmfulness direction, it is then applied to the agent's commands |
| High-stakes probe (`high_stakes_probe`) | A logistic regression we trained for Qwen3.5-27B on published data (conversations GPT-4o wrote and labeled by whether they could lead to significant harm) |
| Simulated-attack probe (`extra_probe_simulated_attack_vs_real_honest`) | Appendix D's probe: the off-policy probe's recipe, trained on GPT-5.6-Sol's simulated attack runs against its real honest runs |
| On-policy probe on fewer tasks (`on_policy_probe_25pct_tasks`, `on_policy_probe_10pct_tasks`) | The on-policy probe trained on 25% and on 10% of its training tasks |

The **six-probe average** covers the first six. The **four-probe average** covers the on-policy, off-policy and deception probes and
the harmfulness direction. We chose these four before seeing these results, because in earlier work their AUROC (the chance that a
random attack run scores above a random honest run; 0.5 is chance and 1 is perfect) on commands was at least 0.6 (we chose the cut-off of 0.6 after seeing those earlier values). The rows for
tool outputs use the same four. The last three probes are sensitivity variants and never enter an average: the post shows the
simulated-attack probe (Appendix D) and the on-policy probe on a tenth of its tasks (Appendix C); the 25% version is not in the post.

The probes we trained come in five copies, one per *fold*: we split the main tasks into five groups and trained each copy without
one group, so every task is scored by a copy that never saw it. The published probes are single models.
[`../pipeline/`](../pipeline/) holds the code that builds every probe, and [`../weights/`](../weights/) their weights.

## How the analysis works

`main_analysis.py` does the following for each probe, kind of input, alarm level (1% and 5%) and resample. Its opening comment
gives each step exactly.

1. **Standardize** the scores of the probes we trained. On each side (real or simulated), each fold copy's scores are centred and
   scaled by the mean and standard deviation of that side's honest units in that fold. A unit whose main task is in no fold is
   scored by all five copies; its five standardized scores are averaged, and that average is standardized again.
2. **Threshold.** On each side, the score above which the chosen share of honest units lies.
3. **Catch.** An attack run is caught if its highest-scoring unit is strictly above its side's threshold.
4. **Catch rate**, with every main task weighted equally.
5. **Resample** main tasks 10,000 times. "Joint" results redo steps 1 to 4 in every resample; "fixed" results keep the original
   standardization and thresholds and resample only the catches.
6. **Intervals**: the 2.5th and 97.5th percentiles of the resampled values.
7. **Figure 4's comparison**: the simulation gap, 0.5 × |simulated − real run 1| + 0.5 × |simulated − real run 2|, minus the rerun
   gap, |real run 2 − real run 1|, from the three catch rates.
8. **Figures 6 and 7**: per task, how far each attack run's most suspicious unit sits above the 1% alarm line (original
   thresholds). Over the tasks where all three runs have scored units, the R² (squared Pearson correlation) of real run 1 with the
   simulated run and with real run 2. Every task counts equally here, unlike in the catch rates; in a resample a task counts as
   often as its main task is drawn.
9. **The real-threshold check** (the main result's footnote, and Appendices C and D): the simulated runs scored as if they were real ones (standardized like real units)
   and caught against the real threshold.

## results/

| File | What it holds |
|---|---|
| `main_results.json` | Every quantity below; rates and differences as fractions (0.66 means 66%), thresholds and counts as they are |
| `catch_rates.csv` | The table behind Figures 1 and 5 (joint results, both alarm levels), in percent and percentage points |
| `excess_distance.csv` | The table behind Figure 4, in percentage points |
| `per_task_r2.csv` | The R² values for every probe and kind of input (R² has no unit); Figure 7 shows the six probes on commands |
| `per_task_values.csv` | Per-task values (Figure 6 shows the on-policy probe on commands): for every probe, kind of input, task and attack run, the run's most suspicious unit minus the threshold, in standardized units for the trained probes and in the probe's own score units for the published ones; empty where the run has no scored units |

`main_results.json` holds:

- `per_probe[probe][scores][alarm level][joint|fixed]` and `averages[six_probe_average|four_probe_average][...]`: each quantity with
  its value on the original data (`value`), its 95% interval (`ci95`, or null when withheld), the number of resamples (`draws`)
  and how many gave a value (`finite_draws`), and whether the thresholds behind it are readable (`readable`: at least 10 honest
  units above each of them on the original data). Quantities: `real_run1_caught`, `real_run2_caught`, `real_caught` (their mean),
  `simulated_caught`, `simulated_minus_real`, `run2_minus_run1`, Figure 4's `simulated_to_real_distance`, `real_to_real_distance`
  and `excess_distance`, and the real-threshold check's `simulated_caught_at_real_threshold` and
  `simulated_minus_real_at_real_threshold`.
- `per_task[probe][scores]`: the number of tasks with all three runs scored, and the R² values with intervals (1% level only).
- `honest_runs[probe][scores][real|simulated]`, original data only: per alarm level the threshold, the honest units above it,
  the no-skill floor (`runs_alarmed`, counting runs with no output as runs without an alarm), the same share among runs with at
  least one scored unit, alarms per alarmed run, and the share expected if alarms fell on units independently; and `counts` of
  runs and units.
- `attack_runs[probe][scores][arm]`: counts of attack runs and units, and units per run.
- `tasks[probe][scores]`: the number of tasks in each comparison; `saturated[...]`: whether a catch rate is at least 99%, which
  leaves Figure 4's distances little room.

In `catch_rates.csv`, `real_caught` is the mean of real runs 1 and 2, `simulated_minus_real` is `simulated_caught` minus
`real_caught`, and `run2_minus_run1` compares the two real runs against the same real threshold. Each value has `_low` and `_high`
columns for its 95% interval. The interval for a difference comes from resampling the difference itself, so it cannot be worked out
from the two catch rates' intervals.

## How we checked it

- **Against the post's own result files.** We ran `main_analysis.py` on `../data/` alone, in a clean container, and compared its
  output with the result files the post was written from, in 1,547 comparisons (each probe's catch rates, differences and
  Figure 4 distances at both alarm levels and in both modes, each with its interval ends and resample counts; the averages; the
  real-threshold check; Figure 7's R² values; the honest runs' thresholds, alarm shares and counts) and all 70,902 per-task values
  behind Figures 6 and 7; we also made 404 checks of the CSV tables (estimates and interval ends against `main_results.json`, task
  counts against the exported task lists). Every comparison was within |ours − stored| ≤ 10⁻⁹ × max(1, |stored|); the largest
  relative difference was 2.4 × 10⁻¹⁴.
- **From a fresh copy of this repository.** In a clean container holding only the committed repository, `main_analysis.py`
  regenerated all five files in `results/` from `../data/` in 332 seconds on 12 cores. Every one of their 78,047 numbers was within
  8.3 × 10⁻¹⁷ (relative) of the committed value, and all text matched; the files differ from the committed ones only in the last
  printed digit of some numbers, from floating-point summation on a different machine. `make_numbers.py` and `make_figures.py`
  then ran cleanly on the regenerated results (270 of 270 checks), and `../post/rebuild_post.py` passed in the same container.
- **The post's numbers.** `make_numbers.py` reproduces all 267 numbers listed in `numbers.csv` exactly as the post prints them
  (270 checks: three printed numbers stand for two probes or two settings each, and each is checked for both; we copied each
  printed number into `numbers.csv`). Numbers from the post's other experiments are checked in [`../post/`](../post/).
- **The figures.** We compared the figures in `output/` with those in `published_figures/` by eye; they match. [`../post/`](../post/)
  checks every figure of the post byte for byte from the saved result files.
