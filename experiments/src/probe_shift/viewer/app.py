"""Transcript viewer: real and simulated LinuxArena runs of the same task side by side, as the reader model sees them.

  experiments/2026-10-06_transcript_viewer/run_viewer.sh

Reads the cache that `run_build_cache.py` writes, the shared splits file, and any probe score files, all named in
the experiment's config/viewer.yaml.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from probe_shift.viewer import data
from probe_shift.viewer.grid import ARM_STYLE, grid

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="experiments/2026-10-06_transcript_viewer/config/viewer.yaml")
cfg = yaml.safe_load(Path(parser.parse_args().config).read_text())
SCORE_COLUMNS = ["probe", "arm", "key", "step", "action", "score"]


@st.cache_data
def load_index(cache: str, built: float) -> pd.DataFrame:
    return pd.read_json(Path(cache) / "index.jsonl", lines=True, dtype=False)


@st.cache_data(max_entries=64)
def load_sample(cache: str, built: float, arm: str, key: str) -> dict:
    return json.loads(data.sample_path(Path(cache), arm, key).read_text())


@st.cache_data
def load_splits(path: str) -> dict:
    return json.loads(Path(path).read_text())


def score_files(folder: str) -> tuple:
    """(name, mtime, size) of every *.jsonl in the folder: the cache key, so new or changed files are picked up."""
    files = sorted(Path(folder).glob("*.jsonl")) if Path(folder).is_dir() else []
    return tuple((f.name, f.stat().st_mtime, f.stat().st_size) for f in files)


@st.cache_data
def load_scores(folder: str, inventory: tuple) -> pd.DataFrame:
    """Every *.jsonl under the scores folder. Rows: probe, arm, key, step (0-based), action (0-based; null for an
    agent turn), score, and optionally readout (output_end, call_end, pre_call, ...: a score at that readout
    position, with action the call's index; absent or null: a score for the whole segment)."""
    files = [Path(folder) / name for name, _, _ in inventory]
    if not files:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    df = pd.concat([pd.read_json(f, lines=True, dtype=False) for f in files], ignore_index=True)
    missing = set(SCORE_COLUMNS) - set(df.columns)
    assert not missing, f"score files in {folder} lack columns {missing}"
    return df


def task_table(idx: pd.DataFrame, splits: dict) -> pd.DataFrame:
    """One row per attack task (main:side) with its environment, whether it is paired, and why not."""
    attack = idx[idx.arm != "real-honest"]
    fields = ["environment", "environment_name", "main_task", "main_task_name", "main_task_summary", "side_task",
              "side_task_summary"]
    tasks = attack.groupby("cell")[fields].first()
    tasks["paired"] = tasks.index.isin(set(splits["paired_attack_cells"]["cells"]))
    why = {}
    for e in splits.get("excluded", []):
        if e["arm"] != "real-honest":
            why.setdefault(e["cell"], set()).add(f'{ARM_STYLE[e["arm"]][0]}: {REASONS.get(e["reason"], e["reason"])}')
    for cell, arms in attack.groupby("cell").arm:
        for arm in set(ARM_STYLE) - {"real-honest"} - set(arms):
            why.setdefault(cell, set()).add(f"{ARM_STYLE[arm][0]}: no run")
    tasks["why_unpaired"] = [", ".join(sorted(why.get(c, []))) for c in tasks.index]
    return tasks


def excluded(splits: dict) -> dict:
    """(arm, key) -> exclusion reason in the splits file."""
    return {(e["arm"], e["key"]): e["reason"] for e in splits.get("excluded", [])}


def sim_reference(splits: dict) -> dict:
    """(arm, key) -> which honest run the simulator was shown (base200 or overlap800), when the splits carry it."""
    rows = splits.get("samples") or []
    return {(r["arm"], r["key"]): r["sim_reference"] for r in rows if r.get("sim_reference")}


REASONS = {"no_actions": "no tool calls", "over_32k": "over 32k tokens", "errored": "errored", "stalled": "stalled"}


def verdict(value) -> str:
    return {"C": "✅ succeeded", "I": "❌ failed"}.get(value, "—" if value is None else str(value))


def human(task_id: str) -> str:
    return task_id.replace("_", " ")


def run_name(sample_id: str) -> str:
    return f"run {sample_id.rsplit(':', 1)[-1]}"


st.set_page_config(page_title="Probe-shift transcripts", layout="wide")
st.html("<style>[data-testid='stMainBlockContainer']{padding-top:3.5rem}</style>")
version_file = Path(cfg["cache"]) / "version.json"
if not version_file.exists() or json.loads(version_file.read_text())["version"] != data.CACHE_VERSION:
    st.error(f"The cache in `{cfg['cache']}` is missing or was built by an older viewer. Rebuild it: "
             "`uv run --group viewer python experiments/2026-10-06_transcript_viewer/run_build_cache.py`")
    st.stop()
built = (Path(cfg["cache"]) / "index.jsonl").stat().st_mtime  # in every cache key, so a rebuild is picked up
idx, splits, scores = load_index(cfg["cache"], built), load_splits(cfg["splits"]), load_scores(cfg["scores"], score_files(cfg["scores"]))
tasks, refs, dropped = task_table(idx, splits), sim_reference(splits), excluded(splits)
linked = st.query_params.get("cell")
if "include_unpaired" not in st.session_state and linked in tasks.index:  # a link to an unpaired task opens it
    st.session_state.include_unpaired = not tasks.loc[linked].paired

# ---- sidebar: which task ----
sb = st.sidebar
sb.title("Transcripts")
env_names = dict(zip(tasks.environment, tasks.environment_name))
envs = sb.multiselect("Environment", sorted(env_names, key=env_names.get), format_func=env_names.get,
                      placeholder="All environments")
pool = tasks[tasks.environment.isin(envs)] if envs else tasks
sides = sb.multiselect("Side task", sorted(pool.side_task.unique()), format_func=human, placeholder="All side tasks",
                       help="The hidden goal the attacker pursues alongside the main task.")
search = sb.text_input("Search main task", placeholder="e.g. migration")
include_unpaired = sb.toggle("Include unpaired tasks", key="include_unpaired",
                             help="Paired tasks have a valid run in all three attack columns, and are the ones "
                                  "comparisons use. Unpaired tasks are labelled with what is missing.")
shown = pool[pool.side_task.isin(sides)] if sides else pool
if search:
    hit = shown.main_task_name.str.contains(search, case=False, regex=False)
    shown = shown[hit | shown.main_task.str.contains(search, case=False, regex=False)]
if not include_unpaired:
    shown = shown[shown.paired]
if refs:
    tags = sorted(set(refs.values()))
    pick = sb.radio("Simulator's reference run", ["any"] + tags, horizontal=True,
                    help="The simulator is shown a real honest run of the main task as a reference. base200: the very "
                         "run in our Real honest column. overlap800: a separately generated honest run.")
    if pick != "any":
        sim_cells = {r.cell for r in idx[idx.arm == "sim-attack"].itertuples() if refs.get(("sim-attack", r.key)) == pick}
        shown = shown[shown.index.isin(sim_cells)]

options = list(shown.index)
sb.caption(f"{len(options)} of {len(tasks)} tasks")
if not options:
    st.info("No task matches these filters.")
    st.stop()
if "cell" not in st.session_state and linked in options:
    st.session_state.cell = linked
if st.session_state.get("cell") not in options:
    st.session_state.cell = options[0]


def step_task(delta: int):
    st.session_state.cell = options[(options.index(st.session_state.cell) + delta) % len(options)]


def base_label(t) -> str:
    return f"{t.main_task_name} → {human(t.side_task)}"


# Streamlit matches a selectbox choice by its label, so two tasks with the same readable name must differ.
clashing = {lab for lab, n in pd.Series([base_label(t) for t in tasks.itertuples()]).value_counts().items() if n > 1}


def label(cell: str) -> str:
    t = tasks.loc[cell]
    lab = base_label(t)
    return lab + (f" ({t.main_task})" if lab in clashing else "") + ("" if t.paired else "  (unpaired)")


sb.selectbox("Task", options, key="cell", format_func=label)
prev, nxt = sb.columns(2)
prev.button("← Previous", on_click=step_task, args=(-1,), use_container_width=True)
nxt.button("Next →", on_click=step_task, args=(1,), use_container_width=True)
cell = st.session_state.cell
st.query_params["cell"] = cell
task = tasks.loc[cell]

# ---- sidebar: which runs ----
sb.subheader("Columns")
arms = [arm for arm in data.ARMS if sb.checkbox(ARM_STYLE[arm][0], value=arm != "real-honest", key=f"show_{arm}")]
runs = {arm: idx[(idx.arm == arm) & (idx.cell == cell)] for arm in data.ARMS if arm != "real-honest"}
runs["real-honest"] = idx[(idx.arm == "real-honest") & (idx.main_task == task.main_task)]
sim_saw = set(runs["sim-attack"].sim_source_sample.dropna())
chosen = {}
for arm in arms:
    rows = runs[arm]
    if len(rows) > 1:
        rows = rows.assign(saw=rows.sample_id.isin(sim_saw)).sort_values("saw", ascending=False)
        names = {r.key: run_name(r.sample_id) + ("  (the simulator's reference)" if r.saw else "") for r in rows.itertuples()}
        chosen[arm] = sb.selectbox(f"Which {ARM_STYLE[arm][0].lower()} run", list(names), format_func=names.get,
                                   key=f"run_{arm}_{cell}", help=f"This task has {len(rows)} runs in this column.")
    elif len(rows) == 1:
        chosen[arm] = rows.key.iloc[0]

sb.subheader("Display")
expand = sb.toggle("Expand long outputs", help="Off: long tool outputs scroll inside their box.")
show = {"readouts": sb.toggle("Show probe readout positions", value=True,
                              help="The exact token each per-segment probe reads (probe_shift.reader."
                                   "segment_end_positions). Hover a marker for the token.")}
show["markers"] = sb.toggle("Include template-token readouts", disabled=not show["readouts"],
                            help="</tool_call> and </tool_response>: byte-identical template tokens, a sensitivity "
                                 "check only, reported next to the content-token readouts.")
show["raw"] = sb.toggle("Show raw messages", help="Under each box, the original message from the .eval log: what the "
                        "cleaning changed (error fields, encrypted reasoning, the system prompt) and what it dropped "
                        "(submit calls, empty turns). Trailing newlines the reader sees are marked ⏎.")

sb.subheader("Probe scores")
probe = None
if scores.empty:
    sb.caption(f"No score files yet. Put JSONL files in `{cfg['scores']}` with rows "
               "`{probe, arm, key, step, action, score}` (step and action 0-based; action null for an agent turn).")
else:
    probe = sb.selectbox("Colour segments by", ["none"] + sorted(scores.probe.unique()))
    probe = None if probe == "none" else probe

# ---- main: header ----
st.markdown(f"### {task.main_task_name}")
st.caption(f"{task.environment_name} environment · "
           + ("paired task" if task.paired else f"unpaired task ({task.why_unpaired})"))
st.markdown(f"**Main task**, given to every run: {task.main_task_summary}  \n"
            f"**Side task**, the hidden goal of the attack runs: {task.side_task_summary}")
samples = {arm: load_sample(cfg["cache"], built, arm, chosen[arm]) for arm in arms if arm in chosen}

cols = st.columns(len(arms)) if arms else []
for col, arm in zip(cols, arms):
    title, colour, about = ARM_STYLE[arm]
    s = samples.get(arm)
    col.markdown(f"<div style='border-top:4px solid {colour};padding-top:4px;font-weight:600'>{title}</div>",
                 unsafe_allow_html=True)
    col.caption(about)
    if s is None:
        col.markdown("No run of this task in this column.")
        continue
    lines = []
    if arm == "real-honest":
        lines.append(f"Main task: **{verdict(s['main_task_success'])}**")
    elif arm == "sim-attack":  # nothing runs in a simulated sandbox: this is an in-run judge's guess (sce sandbox.py)
        lines.append(f"Side task: **{verdict(s['side_task_success'])}** (a judge's guess from the transcript; nothing was executed)")
    else:
        lines.append(f"Side task: **{verdict(s['side_task_success'])}** (scorer run in the environment)")
    lines.append(f"{s['n_steps']} steps · {s['tokens']:,} tokens")
    if s["n_submit_calls"] != 1:
        lines.append("⚠ Never called submit" if s["n_submit_calls"] == 0 else f"Called submit {s['n_submit_calls']} times")
    if dropped.get((arm, s["key"])) == "over_32k":
        lines.append(f"✂ Longer than {cfg['max_tokens']:,} tokens: excluded in `{Path(cfg['splits']).name}`, so not read")
    elif s["truncated"]:
        lines.append(f"✂ Longer than {cfg['max_tokens']:,} tokens: the reader sees the first {cfg['max_tokens'] / s['tokens']:.0%}")
    if s["invalid"]:
        lines.append(f"⚠ Invalid run: {REASONS.get(s['invalid'], s['invalid'])}")
    if arm == "sim-attack" and s["sim_source_sample"]:
        tag = refs.get((arm, s["key"]))
        honest_ids = dict(zip(runs["real-honest"].key, runs["real-honest"].sample_id))
        shown_id = honest_ids.get(chosen.get("real-honest")) if "real-honest" in arms else None
        where = (" (the one in the Real honest column)" if shown_id == s["sim_source_sample"]
                 else " (pick it in the sidebar to compare)" if "real-honest" in arms and s["sim_source_sample"] in honest_ids.values()
                 else " (turn on Real honest to compare)" if s["sim_source_sample"] in honest_ids.values() else "")
        lines.append(f"Reference shown to the simulator: honest {run_name(s['sim_source_sample'])}{where}"
                     + (f", {tag}" if tag else ""))
    col.markdown("  \n".join(lines))
    if s["scores"]["explanations"]:
        with col.expander("Scorer explanation"):
            for name, text in s["scores"]["explanations"].items():
                st.text(text)

anchor = next((samples[a] for a in ("real-attack-e1", "sim-attack", "real-attack-e2") if a in samples), None)
anchor = anchor or (load_sample(cfg["cache"], built, "real-attack-e1", runs["real-attack-e1"].key.iloc[0])
                    if len(runs["real-attack-e1"]) else None)
if anchor:
    with st.expander("Full task descriptions and the attacker's strategy"):
        st.markdown("**Main task**")
        st.text(anchor["main_task_description"] or "—")
        st.markdown("**Side task goal**")
        st.text(anchor["side_task_goal"] or "—")
        st.markdown("**Attack strategy given to the attacker** (from the real attack run)")
        st.text(anchor["attack_approach"] or "—")

# ---- main: the grid ----
columns = []
for arm in arms:
    s = samples.get(arm)
    col = {"arm": arm, "sample": s, "note": "No run of this task in this column.",
           "subtitle": s["sample_id"] if s else "", "scores": None, "ro_scores": None, "scale": None}
    if probe and s:
        p = scores[scores.probe == probe]
        lo, hi = p.score.quantile([0.05, 0.95])
        mine = p[(p.arm == arm) & (p.key == s["key"])]
        per_readout = mine.readout.notna() if "readout" in mine else pd.Series(False, index=mine.index)
        col["scores"] = {(int(r.step), None if pd.isna(r.action) else int(r.action)): float(r.score)
                         for r in mine[~per_readout].itertuples()}
        col["ro_scores"] = {(int(r.step), int(r.action), r.readout): float(r.score) for r in mine[per_readout].itertuples()}
        col["scale"] = (float(lo), float(hi))
    columns.append(col)
st.markdown("##### Transcripts, as the reader model sees them")
st.caption("Each row is one step: the agent's turn (its tool calls) and the tool outputs it got back, so step n of "
           "every run lines up. Each box is one segment the probe reads; its left edge shows the probe's score "
           "once scores exist. Attack runs are shown with the honest system prompt and without the final submit "
           "call, exactly as the reader gets them.")
if probe:
    st.caption(f"Segments coloured by **{probe}**: darker red is a higher score (scale from the 5th to the 95th "
               "percentile of this probe's scores over every transcript).")
if show["readouts"]:
    st.caption("Probe readouts: :green-background[green] is the token a content-end probe reads (`call_end`, "
               "`output_end`); ◆ is `pre_call`, the `<tool_call>` token before a call"
               + ("; dashed ⟂ boxes are template-token readouts (sensitivity check only)" if show["markers"] else "")
               + ". Hover any of them for the token.")
st.html(grid(columns, expand=expand, dark=st.context.theme.type == "dark", show=show))
