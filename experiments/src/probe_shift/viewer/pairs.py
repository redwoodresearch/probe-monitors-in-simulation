"""Real-vs-simulated pairs from a phase-2 pilot run folder (`<run>/launch.json`), and their raw messages.

A pair is one simulated run and its real twin for the same task: `sim_attack` with the real `attack` run, and
`sim_honest` with the real `honest_b` run. Both are forked from the task's `honest_a` run, which the simulator was
also shown; that run is the pair's reference. Each run is read from the last of its attempts that wrote a log.
"""

import json
import random
import struct
import time
import zipfile
from pathlib import Path

TWIN = {"sim_attack": "attack", "sim_honest": "honest_b"}  # simulated arm -> the real arm it is compared with
REFERENCE = "honest_a"  # forked from at step 0 and shown to the simulator


def _log(run_dir: Path, run: dict) -> dict | None:
    """{key, attempt, of, kind, eval} for the run's last attempt that wrote an .eval, or None."""
    attempts = run.get("attempts") or []
    for i in range(len(attempts) - 1, -1, -1):
        log_dir = run_dir / Path(attempts[i]["log_dir"]).relative_to(Path(attempts[i]["log_dir"]).parents[1])
        evals = sorted(log_dir.glob("*.eval")) if log_dir.is_dir() else []
        if evals:
            return {"key": run["key"], "attempt": i + 1, "of": len(attempts), "kind": attempts[i].get("kind"),
                    "eval": evals[-1], "state": run.get("state")}
    return None


def read_launch(run_dir: Path, tries: int = 5) -> dict:
    """launch.json, retried briefly: the launcher rewrites it in place while runs are live, so a read can catch it
    empty or half-written. The last failure is raised."""
    for i in range(tries):
        try:
            return json.loads((run_dir / "launch.json").read_text())
        except json.JSONDecodeError:
            if i == tries - 1:
                raise
            time.sleep(0.3)


def pairs(run_dir: Path) -> list[dict]:
    """Every finished simulated run with a log, with its real twin and reference: [{task, arm, simulator, sim, real,
    ref}]. Runs neither `done` nor `stopped` in launch.json are left out until they finish."""
    runs = read_launch(run_dir)["runs"]
    # Finished runs, and runs the launcher stopped after writing their log; a run still going may have no sample yet.
    runs = [r for r in runs if r.get("state") in ("done", "stopped")]
    by = {(r["arm"], r.get("task") or r["key"].split("/", 1)[1]): r for r in runs if r["arm"] in (*TWIN.values(), REFERENCE)}
    out = []
    for r in runs:
        if r["arm"] not in TWIN:
            continue
        task = r.get("task") or r["key"].split("/", 1)[1]
        sim, real = _log(run_dir, r), by.get((TWIN[r["arm"]], task))
        real = real and _log(run_dir, real)
        ref = by.get((REFERENCE, task))
        if sim and real:
            out.append({"task": task, "arm": r["arm"], "simulator": r.get("simulator"), "sim": sim, "real": real,
                        "ref": ref and _log(run_dir, ref)})
    return sorted(out, key=lambda p: (p["task"], p["arm"], p["simulator"] or ""))


def pick(items: list, seed: int):
    """The pair a seed picks: the same seed always gives the same pair."""
    return random.Random(seed).randrange(len(items))


def read_sample(path: Path) -> dict:
    """The one sample in an Inspect .eval log. Entries may be Zstandard-compressed (zip method 93), which zipfile
    reads only from Python 3.14, so they are decompressed here directly."""
    with zipfile.ZipFile(path) as z:
        names = [i for i in z.infolist() if i.filename.startswith("samples/") and i.filename.endswith(".json")]
        if len(names) != 1:
            raise ValueError(f"{path.name}: expected one sample, found {len(names)}")
        info = names[0]
        if info.compress_type != 93:
            return _inline(json.loads(z.read(info)))
    import zstandard

    with open(path, "rb") as f:
        f.seek(info.header_offset)
        header = f.read(30)
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        f.seek(info.header_offset + 30 + name_len + extra_len)
        try:
            data = zstandard.ZstdDecompressor().decompress(f.read(info.compress_size), max_output_size=info.file_size)
        except zstandard.ZstdError as e:  # a truncated archive: reported like any other unreadable log
            raise ValueError(f"{path.name}: cannot decompress its sample ({e})") from e
    return _inline(json.loads(data))


def _inline(sample: dict) -> dict:
    """The sample with each attachment://<id> in its messages (Inspect's store for long strings) replaced by the text."""
    att = sample.get("attachments") or {}

    def r(x):
        if isinstance(x, str) and x.startswith("attachment://"):
            return att.get(x.removeprefix("attachment://"), x)
        return [r(v) for v in x] if isinstance(x, list) else {k: r(v) for k, v in x.items()} if isinstance(x, dict) else x
    return sample | {"messages": r(sample.get("messages") or [])}


def steps(sample: dict) -> tuple[list[dict], list[list[dict]]]:
    """The messages before the first assistant turn, and per step the assistant turn plus the tool outputs and
    any other messages that follow it, all as stored."""
    prompt, out = [], []
    for m in sample["messages"]:
        if m["role"] == "assistant":
            out.append([m])
        elif out:
            out[-1].append(m)
        else:
            prompt.append(m)
    return prompt, out
