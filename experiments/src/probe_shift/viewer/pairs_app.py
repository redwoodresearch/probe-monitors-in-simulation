"""Bare real-vs-simulated pair viewer for phase-2 pilot runs: raw messages only, no scores, labels or summaries.

  experiments/2026-10-08_pilot_pair_viewer/serve_devbox.sh      # on the devbox that holds ~/phase2/runs

A seeded random pick chooses one simulated run and its real twin (pairs.py); each step shows the stored messages
as they are: the model's calls with raw arguments, then the tool outputs.
"""

import argparse
import json
import zipfile
from pathlib import Path

import streamlit as st
import yaml

from probe_shift.viewer.pairs import TWIN, pairs, pick, read_launch, read_sample, steps

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="experiments/2026-10-08_pilot_pair_viewer/config/pairs.yaml")
cfg = yaml.safe_load(Path(parser.parse_args().config).read_text())
ROOT = Path(cfg["runs_root"]).expanduser()


@st.cache_data(max_entries=8, ttl=60)  # runs land while the page is open
def run_folders(root: str) -> tuple[list[str], list[str]]:
    """Run folders whose launch.json has simulated arms, oldest first, and folders whose launch.json is unreadable."""
    out, bad = [], []
    for d in sorted(Path(root).iterdir()):
        if (d / "launch.json").is_file():
            try:
                runs = read_launch(d).get("runs", [])
            except (json.JSONDecodeError, OSError):
                bad.append(d.name)
                continue
            if any(r.get("arm") in TWIN for r in runs):
                out.append(d.name)
    return out, bad


@st.cache_data(max_entries=4, ttl=60)
def load_pairs(run_dir: str) -> list[dict]:
    return pairs(Path(run_dir))


@st.cache_data(max_entries=6)  # a few logs at a time: the devbox also runs the pilot's environments
def load_sample(path: str) -> dict:
    """The log's one sample, or {"unreadable": why} when a finished attempt left a log without a readable one (no
    sample, a truncated archive, undecodable JSON)."""
    try:
        return read_sample(Path(path))
    except (ValueError, zipfile.BadZipFile, EOFError) as e:
        return {"unreadable": str(e), "messages": []}


def show(m: dict):
    """One stored message, raw."""
    head = m["role"] + (f" · {m['function']}" if m.get("function") else "")
    st.markdown(f"**{head}**")
    content = m.get("content")
    if isinstance(content, str):
        if content:
            st.code(content, language=None, wrap_lines=True)
    else:
        for part in content or []:
            kind = part.get("type")
            text = part.get("text") if kind == "text" else part.get("reasoning") if kind == "reasoning" else None
            if text is not None:
                st.caption(kind)
                st.code(text, language=None, wrap_lines=True)
            else:
                st.code(json.dumps(part, indent=1, ensure_ascii=False), language="json", wrap_lines=True)
    for call in m.get("tool_calls") or []:
        st.caption(f"call · {call.get('function')}")
        st.code(json.dumps(call.get("arguments"), indent=1, ensure_ascii=False), language="json", wrap_lines=True)
    if m.get("error"):
        st.caption("error")
        st.code(json.dumps(m["error"], indent=1, ensure_ascii=False), language="json", wrap_lines=True)


st.set_page_config(page_title="Pilot pairs", layout="wide")
folders, unreadable = run_folders(str(ROOT))
for name in unreadable:
    st.sidebar.caption(f"skipped {name}: its launch.json is unreadable")
if not folders:
    st.info(f"No run folder with simulated arms under {ROOT}.")
    st.stop()
run = st.sidebar.selectbox("Run folder", folders, index=len(folders) - 1)
invalid = (cfg.get("invalid") or {}).get(run)
if invalid:
    st.sidebar.error("INVALID")
    st.error(f"INVALID: {invalid}")
P = load_pairs(str(ROOT / run))
if not P:
    st.info("This run folder has no simulated run with a log and a real twin yet.")
    st.stop()

if "seed" not in st.session_state:
    st.session_state.seed = int(cfg.get("seed", 0))


def next_pair():
    st.session_state.seed += 1


seed = int(st.sidebar.number_input("Seed", min_value=0, step=1, key="seed"))
st.sidebar.button("Next random pair", on_click=next_pair, use_container_width=True)
# The pick is remembered per run folder and seed, so a refresh that adds finished pairs cannot swap it.
picked = st.session_state.setdefault("picked", {})
ids = [q["sim"]["key"] for q in P]
if picked.get((run, seed)) not in ids:
    picked[(run, seed)] = ids[pick(P, seed)]
i = ids.index(picked[(run, seed)])
p = P[i]
st.sidebar.caption(f"seed {seed} → pair {i + 1} of {len(P)}")
show_ref = st.sidebar.toggle("Show the reference run", help="honest_a: both runs are forked from it at step 0, and the "
                             "simulator is shown its whole trajectory.")

cols = [("Real", p["real"]), ("Simulated", p["sim"])] + ([("Reference", p["ref"])] if show_ref and p["ref"] else [])
samples = [load_sample(str(log["eval"])) for _, log in cols]
split = [steps(s) for s in samples]
st.markdown(f"**{run}** · seed {seed}")
for c, (title, log), (_, st_), sample in zip(st.columns(len(cols)), cols, split, samples):
    state = "" if log.get("state") == "done" else f" · {log.get('state')}"  # e.g. stopped by the launcher
    c.markdown(f"**{title}**  \n`{log['key']}` · attempt {log['attempt']} of {log['of']} ({log['kind']}) · "
               f"`{log['eval'].name}` · {len(st_)} steps{state}")
    for field in ("unreadable", "error", "limit"):  # the sample's own failure fields, raw, so a failed run is obvious
        if sample.get(field):
            c.caption(f"sample {field}")
            c.code(json.dumps(sample[field], indent=1, ensure_ascii=False) if not isinstance(sample[field], str)
                   else sample[field], language=None, wrap_lines=True)
for c, (prompt, _) in zip(st.columns(len(cols)), split):
    with c.expander("Messages before step 1"):
        for m in prompt:
            show(m)
for k in range(max(len(s) for _, s in split)):
    st.divider()
    for c, (_, st_) in zip(st.columns(len(cols)), split):
        with c:
            if k < len(st_):
                st.markdown(f"#### Step {k + 1}")
                for m in st_[k]:
                    show(m)
