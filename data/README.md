# data/: each probe's score on every command and tool output it reads

These files are the input to [`../analysis/main_analysis.py`](../analysis/main_analysis.py): every probe's score on every unit it
scores (a command the agent ran, or a tool output it read; the two probes we trained skip about one command in seven) in the runs behind the post's main result, plus the task lists and the resampling
draws. They were exported from the scores the post's analysis used; `data_manifest.json` gives the sha256 of every file.

Load the `.npz` files with numpy (`numpy.load(path)`) and the CSV files with any CSV reader; indices into the CSV files count
data rows from 0.

| File | What it holds |
|---|---|
| `scores/<probe>__<commands or tool_outputs>.npz` | One probe and kind of input; arrays below |
| `cells.csv` | The 1,313 tasks: `cell` (the task's name), `task_set` (the batch it was drawn in: `pilot_196`, `scale_up_311`, `new_main_tasks_360` or `paper_remaining_493`, named for the number of tasks drawn, before some were left out), `main_task`, `environment` |
| `main_tasks.csv` | The 902 main tasks the resamples are drawn over, with their environment: the 864 main tasks of the 1,313 tasks, and 38 whose launched tasks were all left out of the comparison (they stay in the resampling draws the post used) |
| `multiplicities.npz` | `multiplicities`, 10,001 × 902: row 0 is the original data (all ones); each later row gives how many times each main task appears in one resample |
| `honest_runs_without_output.csv` | The honest runs in which no probe scores anything at all (one real, one simulated, on the same task); they count as runs without an alarm in the no-skill floor |
| `data_manifest.json` | The sha256 of every file, the array counts, and this schema |

Arrays in each scores file:

| Array | Meaning |
|---|---|
| `honest_real_score`, `honest_simulated_score` | Every honest unit's score on that side; these set the side's threshold |
| `honest_<side>_fold` | The unit's fold (0 to 4); −1 if its main task is in no fold; −9 for the published probes, which have no folds |
| `honest_<side>_fold_scores` | For units with fold −1: the five fold copies' scores, one row per such unit, in unit order |
| `honest_<side>_main`, `honest_<side>_run` | The unit's main task (row of `main_tasks.csv`) and run (an index within the side) |
| `honest_<side>_run_main` | Each honest run's main task, including runs in which this probe scores nothing (but not the runs in `honest_runs_without_output.csv`) |
| `<arm>_score`, `<arm>_fold`, `<arm>_fold_scores`, `<arm>_cell` | The same for each attack run's units, with the task (row of `cells.csv`) in place of a run; `<arm>` is `real_attack_run1`, `real_attack_run2` or `simulated_attack` |
| `<arm>_zero_unit_cells` | Tasks whose attack run has no scored units (never caught) |
| `cells_used` | The tasks in this probe's comparison (all three attack runs present) |

The scores are the probes' raw outputs. For the probes we trained, `main_analysis.py` standardizes them within each fold and
setting before setting thresholds (its step 1); the published probes' scores are used as they are.
[`../pipeline/score.py`](../pipeline/score.py) computes these scores from activations and [`../weights/`](../weights/).
