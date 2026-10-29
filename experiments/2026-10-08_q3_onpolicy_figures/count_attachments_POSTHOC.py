"""Attachment census (POST HOC; Aswin via the orchestrator, about 16:55 PT 2026-10-11): in the Phase 2 .eval logs that 2e's
pipelines read, how many strings are Inspect `attachment://` references (long strings stored once in the sample's
`attachments` map), and does any reported number change once they are resolved? $0, local; counts only: no command,
output or message text is printed or saved (attack transcripts are read by this script only, and only counts leave it).

1. Census, per sample (one per .eval), over launch.json's attempts in ARMS (attempts without a log counted):
   - assistant tool calls: argument values that are references (top level; nested deeper; inside JSON-string arguments),
     and bash commands that are references, with those missing from the sample's map;
   - tool messages: content that is a reference, and references inside content parts;
   - every `attachment://` occurrence in the sample JSON outside the map.
2. The replay builder (run_replay_readouts.py, the scorers' input): what its own resolution leaves unresolved, mirroring
   it: acting-model events with an x-irid; top-level argument values resolved from the map (a missing key, a nested
   reference or JSON-string arguments stay unresolved); tool outputs as text_of (parts joined, then resolved only if the
   whole text is a reference) or "Error: <message>" for a failed call.
3. With references resolved (JSON-string arguments parsed first):
   - (D) and (E): analyze_POSTHOC_streaks_errors.main run twice, its transcript reader as is and resolved, with its own
     joins and selection; the as-is run must reproduce the committed result's D and E;
   - the A2 stuck readout's runs (every run with >= 20 alarms in the A2 result, and the five unscored C runs): the
     longest same-shape bash streak, the error share and the stuck predicate, as is and resolved;
   - the t72/t92 audit's four runs: run_summary's tool calls, error share and Rscript one-liner counts, as is and resolved.

Prediction, written before running: bash command references exist (long one-liners) in a few percent of bash calls; the
replay builder leaves 0 unresolved; A2's stuck predicate changes for no run; (D)/(E) shares move by under 1 point.

  uv run --no-project --python 3.14 --with numpy --with pyyaml python count_attachments_POSTHOC.py \\
      --a2-result <A2_batches json> --a2-unscored <A2_unscored_c_runs json> --t72-launch <scale-up launch> \\
      --se-inputs <main per-call> <honest_scores.json> <acts meta root> --se-committed <streaks_errors result> <launch dir>...
"""

import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import analyze_POSTHOC_streaks_errors as SE
from analyze_honest_runs_POSTHOC_t72_t92 import TASKS, run_summary
from analyze_POSTHOC_A2_stuck import _bash_calls

HERE = Path(__file__).parent
REF = "attachment://"
ARMS = {"honest_a": "honest", "honest_b": "honest", "attack": "real attack", "sim_attack@september": "simulated attack",
        "sim_honest@september": "simulated honest"}
ACTING = "vllm/Qwen/Qwen3.5-27B"


def is_ref(v) -> bool:
    return isinstance(v, str) and v.startswith(REF)


def resolve(v, att: dict):
    if is_ref(v):
        return att.get(v.removeprefix(REF), v)
    if isinstance(v, dict):
        return {k: resolve(x, att) for k, x in v.items()}
    if isinstance(v, list):
        return [resolve(x, att) for x in v]
    return v


def deep_refs(v) -> int:
    if isinstance(v, dict):
        return sum(deep_refs(x) for x in v.values())
    if isinstance(v, list):
        return sum(deep_refs(x) for x in v)
    return int(is_ref(v))


def resolved_sample(s: dict) -> dict:
    """Every reference resolved from the sample's map; JSON-string tool-call arguments parsed first."""
    att = s.get("attachments") or {}
    s = copy.deepcopy(s)
    for m in s.get("messages") or []:
        for tc in m.get("tool_calls") or []:
            if isinstance(tc.get("arguments"), str):
                tc["arguments"] = json.loads(tc["arguments"])
    return resolve(s, att)


def census(s: dict, c: Counter) -> None:
    att = s.get("attachments") or {}
    c["samples"] += 1
    for m in s.get("messages") or []:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                args = tc.get("arguments") or {}
                if isinstance(args, str):
                    c["tool_call_json_string_arguments"] += 1
                    c["tool_call_arg_refs_in_json_strings"] += deep_refs(json.loads(args))
                    args = json.loads(args)
                c["tool_calls"] += 1
                if not isinstance(args, dict):
                    continue
                c["tool_call_arg_refs_top"] += sum(is_ref(v) for v in args.values())
                c["tool_call_arg_refs_nested"] += sum(deep_refs(v) for v in args.values() if isinstance(v, (dict, list)))
                if tc.get("function") == "bash":
                    cmd = args.get("cmd") or args.get("command")
                    c["bash_calls"] += 1
                    c["bash_cmd_refs"] += is_ref(cmd)
                    c["bash_cmd_refs_missing_from_map"] += is_ref(cmd) and cmd.removeprefix(REF) not in att
        if m["role"] == "tool":
            ct = m.get("content")
            c["tool_messages"] += 1
            c["tool_message_refs"] += is_ref(ct)
            c["tool_message_refs_in_parts"] += sum(is_ref(p.get("text")) for p in ct if isinstance(p, dict)) if isinstance(ct, list) else 0
    n = json.dumps({k: v for k, v in s.items() if k != "attachments"}).count(REF)
    c["ref_occurrences_outside_map"] += n
    c["samples_with_any_ref"] += n > 0
    c["samples_errored"] += bool(s.get("error"))
    c["refs_left_after_full_resolution"] += left(s)


def left(s: dict) -> int:
    """References still present once the sample is resolved from its own map (missing keys)."""
    return json.dumps({k: v for k, v in resolved_sample(s).items() if k != "attachments"}).count(REF)


def replay_leftovers(s: dict, c: Counter) -> None:
    """What run_replay_readouts.py's own resolution leaves: argument values and tool outputs it would hash unresolved."""
    att = s.get("attachments") or {}

    def text_of(x):  # as run_replay_readouts.text_of
        if isinstance(x, list):
            x = "".join(p.get("text", "") for p in x if p.get("type") == "text")
        return att.get(x.removeprefix(REF), x) if isinstance(x, str) and x.startswith(REF) else (x or "")
    for e in s.get("events") or []:
        if e.get("event") != "model" or e.get("model") != ACTING:
            continue
        if (((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid") is None:
            c["replay_events_without_irid (the builder refuses to stage)"] += 1
            continue
        msg = (((e.get("output") or {}).get("choices") or [{}])[0].get("message") or {})
        for tc in msg.get("tool_calls") or []:
            args = tc.get("arguments") or {}
            c["replay_calls"] += 1
            if not isinstance(args, dict):
                c["replay_json_string_arguments (unsupported by the builder)"] += 1
                continue
            out = {k: (att.get(v.removeprefix(REF), v) if is_ref(v) else v) for k, v in args.items()}
            c["replay_arg_refs_resolved"] += sum(is_ref(v) for v in args.values()) - sum(is_ref(v) for v in out.values())
            c["replay_arg_refs_left"] += deep_refs(out)
    msgs = s.get("messages") or []
    asst = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    n_acting = sum(e.get("event") == "model" and e.get("model") == ACTING for e in s.get("events") or [])
    if len(asst) != n_acting:  # the builder reads no output_end here (an errored sample's cut transcript)
        c["replay_samples_without_output_readouts"] += 1
        return
    for k in range(len(asst) - 1):  # each turn's outputs are read in the next request; the last turn's never are
        for m in msgs[asst[k] + 1: asst[k + 1]]:
            if m["role"] != "tool":
                continue
            err = m.get("error")
            txt = f"Error: {err.get('message', '')}" if isinstance(err, dict) else text_of(m.get("content"))
            c["replay_outputs"] += 1
            c["replay_output_refs_left"] += txt.count(REF)


def sample_of(ev: Path) -> dict:
    with zipfile.ZipFile(ev) as z:
        (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
        return json.loads(z.read(n))


def logs_of(d: Path, a: dict) -> list:
    return sorted((d / Path(a["log_dir"]).relative_to(Path(a["log_dir"]).parents[1])).glob("*.eval"))


def longest_bash_streak(s: dict) -> int:
    shapes = [c["shape"] for c in _bash_calls(s)]
    longest = cur = 0
    for i, sh in enumerate(shapes):
        cur = cur + 1 if i and sh == shapes[i - 1] else 1
        longest = max(longest, cur)
    return longest


def run_se(se_inputs: list, launches: list, resolved: bool) -> dict:
    """analyze_POSTHOC_streaks_errors.main, its transcript reader as is or resolved; its output kept in a scratch folder."""
    orig_here, orig_tr = SE.HERE, SE.transcript
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "results").mkdir()
        (Path(tmp) / "config").symlink_to(orig_here / "config")
        SE.HERE = Path(tmp)
        if resolved:
            def tr(path):  # SE.transcript on the resolved sample (a temporary one-sample zip)
                p = Path(tmp) / "one.eval"
                with zipfile.ZipFile(p, "w") as z:
                    z.writestr("samples/x.json", json.dumps(resolved_sample(sample_of(path))))
                return orig_tr(p)
            SE.transcript = tr
        try:
            SE.main(*se_inputs, launches)
            (f,) = list((Path(tmp) / "results").glob("*.json"))
            return json.loads(f.read_text())
        finally:
            SE.HERE, SE.transcript = orig_here, orig_tr


def main(launches, a2_result, a2_unscored, t72_launch, se_inputs, se_committed):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes: commit them first")
    subprocess.run(["git", "ls-files", "--error-unmatch", Path(__file__).name], capture_output=True, check=True, cwd=HERE)  # committed
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    dirs = {Path(d).name: Path(d) for d in launches}
    inputs = {"script": sha(__file__), **{f"{n}/launch.json": sha(d / "launch.json") for n, d in dirs.items()},
              f"{Path(t72_launch).name}/launch.json (t72)": sha(Path(t72_launch) / "launch.json"),
              **{Path(p).name: sha(p) for p in (a2_result, a2_unscored, se_committed, se_inputs[0], se_inputs[1])}}
    worlds = {}
    for name, d in dirs.items():
        for r in json.loads((d / "launch.json").read_bytes())["runs"]:
            label = r["arm"] if r.get("real", True) else f"{r['arm']}@{r['simulator']}"
            if label not in ARMS:
                continue
            c = worlds.setdefault(f"{name} | {label} ({ARMS[label]})", Counter())
            for a in r["attempts"]:
                c["attempts"] += 1
                evs = logs_of(d, a)
                c["attempts_without_a_log"] += not evs
                for ev in evs:
                    try:
                        s = sample_of(ev)
                    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as e:
                        c[f"logs unreadable ({type(e).__name__})"] += 1
                        continue
                    census(s, c)
                    replay_leftovers(s, c)
    totals = Counter()
    for c in worlds.values():
        totals.update(c)
    out = {"label": "POST HOC: attachment census of 2e's pipelines (counts only)", "git": git, "inputs": inputs,
           "worlds": {w: dict(sorted(c.items())) for w, c in sorted(worlds.items())}, "totals": dict(sorted(totals.items()))}

    # A2 stuck readout, as is and resolved
    R, U = json.loads(Path(a2_result).read_bytes()), json.loads(Path(a2_unscored).read_bytes())
    if U["inputs"].get(Path(a2_result).name) != sha(a2_result) or {a["task"] for a in U["attempts"]} != set(R["alarm_heavy"]["honC"]["c_runs_not_scored"]):
        raise SystemExit("the unscored-runs result was not made from this A2 result, or lacks one of its five runs")

    def stuck_rows(recs):
        rows = []
        for rec in recs:
            if not rec.get("assessable"):
                rows.append({"task": rec["task"], "attempt": rec.get("attempt"), "assessable": False})  # its free-text reason is not copied
                continue
            ev = dirs[rec["eval"]["launch"]] / rec["eval"]["path"]
            if sha(ev) != rec["eval"]["sha256"]:
                raise SystemExit(f"{rec['task']}: the .eval is not the one the A2 readout read")
            s = sample_of(ev)
            rs = resolved_sample(s)
            raw, res = longest_bash_streak(s), longest_bash_streak(rs)
            if left(s):
                raise SystemExit(f"{rec['task']}: references missing from the sample's map; resolved values are incomplete")
            pred = lambda st: rec["ending"] == "step cap" or st >= 20  # noqa: E731
            rows.append({"task": rec["task"], "attempt": rec.get("attempt"), "alarms": rec.get("alarms"), "ending": rec["ending"], "streak_as_reported": rec.get("longest_same_shape_bash_streak"),
                         "streak_raw": raw, "streak_resolved": res, "error_share_raw": run_summary(s)["error_share_all_calls"],
                         "error_share_resolved": run_summary(rs)["error_share_all_calls"], "stuck_predicate_raw": pred(raw), "stuck_predicate_resolved": pred(res)})
        return rows
    out["a2_stuck_readout"] = {w: stuck_rows(v["runs"]) for w, v in R["alarm_heavy"].items()} | {"honC unscored": stuck_rows(U["attempts"])}

    # t72/t92 audit, as is and resolved
    t = Path(t72_launch)
    keep = lambda x: {"tool_calls": x["tool_calls"], "error_share_all_calls": x["error_share_all_calls"],  # noqa: E731
                      **{f"rscript_{k}": x["rscript_inline"][k] for k in ("calls", "distinct_commands", "erroring", "longest_erroring_streak")}}
    out["t72_t92_audit"] = {}
    for r in json.loads((t / "launch.json").read_bytes())["runs"]:
        if r["arm"] in ("honest_a", "honest_b") and r["task"] in TASKS:
            counted = [a for a in r["attempts"] if not a.get("redrawn_because")]
            (ev,) = logs_of(t, counted[0] if r["arm"] == "honest_a" else counted[-1])
            s = sample_of(ev)
            out["t72_t92_audit"][f"{r['arm']}|{r['task']}"] = {"raw": keep(run_summary(s)), "resolved": keep(run_summary(resolved_sample(s))),
                                                               "refs_left_after_resolution": left(s)}

    # (D) and (E), as is and resolved
    C = json.loads(Path(se_committed).read_bytes())
    norm = lambda x: json.loads(json.dumps(x, default=float))  # noqa: E731  (JSON's string keys, as the committed file)
    as_is, res = norm(run_se(se_inputs, launches, False)), norm(run_se(se_inputs, launches, True))
    if (as_is["D"], as_is["E"], as_is["join"]) != (C["D"], C["E"], C["join"]):
        raise SystemExit("the as-is (D)/(E) run does not reproduce the committed result")
    out["streaks_errors"] = {"reproduced_committed": True, "join_resolved": res["join"], "D_resolved": res["D"], "E_resolved": res["E"],
                             "D_E_unchanged": (res["D"], res["E"]) == (C["D"], C["E"])}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"attachment_census_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("totals", "a2_stuck_readout", "t72_t92_audit")}, indent=1, default=float))
    print("D/E unchanged:", out["streaks_errors"]["D_E_unchanged"])
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("launches", nargs="+")
    ap.add_argument("--a2-result", required=True)
    ap.add_argument("--a2-unscored", required=True)
    ap.add_argument("--t72-launch", required=True)
    ap.add_argument("--se-inputs", nargs=3, required=True, metavar=("MAIN", "HONEST", "ROOT"))
    ap.add_argument("--se-committed", required=True)
    a = ap.parse_args()
    main(a.launches, a.a2_result, a.a2_unscored, a.t72_launch, a.se_inputs, a.se_committed)
