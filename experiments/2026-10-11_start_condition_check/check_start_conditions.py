"""Start-condition check, real vs simulated (counts and hashes only; no text is printed or written).

For every run in the given cohorts: the sim fork metadata (fork.fraction / after / prefix_actions, seeded_prefix length),
the acting model's FIRST call (messages in its input by role), and sha256 of the system message and of the first user
turn as that first call received them (text parts joined). Then, per comparison, the share of tasks whose real and sim
runs started with the same system message / first user turn, beside a real-vs-real baseline.

  uv run --no-project --python 3.14 python check_start_conditions.py --out <json>
"""

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

C1 = Path.home() / ".cache/probe-shift/s3/rr-sce-sim-transcripts/transcripts"
C2 = Path.home() / ".cache/probe-shift/s3/individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/runs"
HC = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/a683364c-3a60-44cd-bb15-e08d87dd4f9a/scratchpad/wt-c/.run_cache/phase2_attack_check/runs/2026-10-11_112714_honest_c")
SOL, QWEN = "openai/gpt-5.6-sol", "vllm/Qwen/Qwen3.5-27B"
P2 = ["2026-10-09_173815_scaleup_canary", "2026-10-09_182505_scaleup", "2026-10-10_001314_scaleup_topup"]
# cohort -> (folders, acting model, how the task key is read)
COHORTS = {
    "p1_real_attack_e1": ([C1 / "2026-09-18_la-public10_real-attack-solmedium-step150-guidancematched-epoch1-n1000"], SOL, "cell"),
    "p1_real_attack_e2": ([C1 / "2026-09-18_la-public10_real-attack-solmedium-step150-guidancematched-epoch2-n1000"], SOL, "cell"),
    "p1_sim_attack_sept": ([C1 / "2026-09-18_la-public10_sim-astra-medium-attack-origtraj-read-nocritic-fork0-overlapaware-n1000"], SOL, "cell"),
    "p1_real_honest": ([C1 / "2026-09-18_la-public10_real-honest-solmedium-step150-overlapaware-n1000"], SOL, "main"),
    "p1_sim_honest_astra_oct6": ([C1 / "2026-10-06_la-public10_sim-astra-medium-honest-origtraj-read-nocritic-fork0"], SOL, "main"),
    "p2_attack_run1": ([C2 / r / "attack" for r in P2], QWEN, "dir"),
    "p2_attack_run2": ([C2 / "2026-10-10_001314_second_attack/attack"], QWEN, "dir"),
    "p2_honest_a": ([C2 / r / "honest_a" for r in P2], QWEN, "dir"),
    "p2_honest_b": ([C2 / r / "honest_b" for r in P2], QWEN, "dir"),
    "p2_sim_attack_sept": ([C2 / r / "sim_attack@september" for r in P2], QWEN, "dir"),
    "p2_honest_c": ([HC / "honest_b"], QWEN, "dir"),
    "pilot_attack": ([C2 / "2026-10-09_000523_rerun_strict/attack"], QWEN, "dir"),
    "pilot_honest_a": ([C2 / "2026-10-09_000523_rerun_strict/honest_a"], QWEN, "dir"),
    "pilot_honest_b": ([C2 / "2026-10-09_000523_rerun_strict/honest_b"], QWEN, "dir"),
    "pilot_sim_attack_sept": ([C2 / "2026-10-09_000523_rerun_strict/sim_attack@september"], QWEN, "dir"),
    "pilot_sim_honest_sept": ([C2 / "2026-10-09_000523_rerun_strict/sim_honest@september"], QWEN, "dir"),
    "pilot_sim_honest_today": ([C2 / "2026-10-09_000523_rerun_strict/sim_honest@today"], QWEN, "dir"),
}
# (real, sim, what it feeds); baselines are real vs real
COMPARISONS = [
    ("p1_real_attack_e1", "p1_sim_attack_sept", "Phase 1 detector #36, off-policy probes #39-#42, Phase 1 checks/fix939, nano, claim B (sim source)"),
    ("p1_real_attack_e2", "p1_sim_attack_sept", "as above (e2 side)"),
    ("p1_real_attack_e1", "p1_real_attack_e2", "BASELINE real vs real"),
    ("p1_real_honest", "p1_sim_honest_astra_oct6", "#36 simulated-honest false alarms"),
    ("p2_attack_run1", "p2_sim_attack_sept", "196-task 2x2 monitor and D, claim C, k=26 redo, D2, D3, Phase 2 figures"),
    ("p2_attack_run2", "p2_sim_attack_sept", "196-task yardstick (run 2) and the sim comparison"),
    ("p2_attack_run1", "p2_attack_run2", "BASELINE real vs real"),
    ("p2_honest_a", "p2_honest_b", "BASELINE real honest vs real honest"),
    ("p2_honest_b", "p2_honest_c", "BASELINE honest_b vs honest C (A2 batch)"),
    ("pilot_attack", "pilot_sim_attack_sept", "20-task pilot detector and 2x2"),
    ("pilot_honest_b", "pilot_sim_honest_sept", "20-task pilot detector (sim honest arm, September)"),
    ("pilot_honest_b", "pilot_sim_honest_today", "20-task pilot detector (sim honest arm, today)"),
    ("pilot_honest_a", "pilot_honest_b", "BASELINE pilot real honest vs real honest"),
]


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


REF = re.compile(r"attachment://[0-9a-f]{32}")


def resolved(x, att):
    """Whole-value attachment references resolved at any depth (a missing one raises)."""
    if isinstance(x, str) and REF.fullmatch(x):
        return att[x[13:]]
    if isinstance(x, list):
        return [resolved(v, att) for v in x]
    if isinstance(x, dict):
        return {k: resolved(v, att) for k, v in x.items()}
    return x


def is_empty(c, att) -> bool:
    """Whether resolved content carries nothing: an empty string, or only empty text parts and no other part."""
    c = resolved(c, att)
    if isinstance(c, str):
        return not c
    parts = [p for p in c or [] if isinstance(p, dict)]
    return not any(p.get("type") != "text" or p.get("text") for p in parts)


def content_hash(c, att) -> tuple[str, int]:
    """sha256 of the message content as the model received it (text parts joined; any non-text part hashed by its
    resolved JSON, never dropped), and the number of non-text parts."""
    c = resolved(c, att)
    if isinstance(c, str):
        return h(c), 0
    parts = [p for p in c or [] if isinstance(p, dict)]
    other = [p for p in parts if p.get("type") != "text"]
    body = "\n".join(p.get("text") or "" for p in parts if p.get("type") == "text")
    return h(body + ("\u0000" + json.dumps(other, sort_keys=True) if other else "")), len(other)


def first_input(e, s):
    """The first call's input messages: inline `input`, or the `input_refs` ranges into `events_data.messages`."""
    if e.get("input"):
        return e["input"]
    pool = (s.get("events_data") or {}).get("messages") or []
    refs = e.get("input_refs") or []
    if any(not (0 <= a <= b <= len(pool)) for a, b in refs):
        raise ValueError("input_refs out of range")
    return [m for a, b in refs for m in pool[a:b]]


def ident(x):
    return x.get("id") if isinstance(x, dict) else x


def one(args):
    path, model, keymode = args
    rows = []
    with zipfile.ZipFile(path) as z:
        for n in [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]:
            s = json.loads(z.read(n))
            md = s.get("metadata") or {}
            fork = md.get("fork") if isinstance(md.get("fork"), dict) else None
            first = next((e for e in s.get("events") or [] if e.get("event") == "model" and e.get("model") == model), None)
            att = s.get("attachments") or {}
            status, inp, sys_h, usr_h, whole, non_text = "ok", [], None, None, None, 0
            try:
                inp = first_input(first, s) if first else []
                sysm = next((m for m in inp if m.get("role") == "system"), None)
                usr = next((m for m in inp if m.get("role") == "user"), None)
                if not first:
                    status = "no_first_call"
                elif sysm is None or usr is None:
                    status = "missing_system_or_user"
                elif is_empty(sysm.get("content"), att):  # the harness sends an empty first user turn (the task is in the system message)
                    status = "empty_system"
                else:
                    sys_h, n1 = content_hash(sysm.get("content"), att)
                    usr_h, n2 = content_hash(usr.get("content"), att)
                    # the whole ordered first input: every message field except the per-message random `id`,
                    # attachments resolved, content parts kept in order
                    whole = h(json.dumps([{k: resolved(v, att) for k, v in m.items() if k != "id"} for m in inp], sort_keys=True))
                    non_text = n1 + n2
            except KeyError:
                status = "unresolved_attachment"
            except ValueError:
                status = "refs_out_of_range"
            roles = [m.get("role") for m in inp]
            ok = status == "ok"
            key = (path.parent.name if keymode == "dir" else
                   f"{ident(md.get('main_task'))}|{ident(md.get('side_task'))}" if keymode == "cell" else str(ident(md.get("main_task"))))
            rows.append({"key": key, "has_fork_md": fork is not None,
                         "fork_fraction": fork.get("fraction") if fork else None, "fork_after": fork.get("after") if fork else None,
                         "fork_prefix_actions": fork.get("prefix_actions") if fork else None,
                         "seeded_prefix_len": len(md["seeded_prefix"]) if isinstance(md.get("seeded_prefix"), list) else None,
                         "first_call_found": first is not None, "first_call_n_input": len(inp), "first_call_roles": dict(Counter(roles)),
                         "first_call_role_seq": "/".join(r or "?" for r in roles[:6]),
                         "sys_sha": sys_h, "user1_sha": usr_h, "input_ok": ok, "start_status": status, "input_sha": whole,
                         "user1_empty": is_empty(usr.get("content"), att) if ok else None,
                         "non_text_parts": non_text, "file": path.name, "sample": str(s.get("id")), "epoch": s.get("epoch"),
                         "msgs_first_roles": "/".join(m.get("role", "?") for m in (s.get("messages") or [])[:3])})
    return rows


def summarize(rows):
    def dist(k):
        return dict(Counter(str(r[k]) for r in rows).most_common(8))
    return {"samples": len(rows), "tasks": len({r["key"] for r in rows}), "has_fork_md": dist("has_fork_md"),
            "fork_fraction": dist("fork_fraction"), "fork_after": dist("fork_after"), "fork_prefix_actions": dist("fork_prefix_actions"),
            "seeded_prefix_len": dist("seeded_prefix_len"), "first_call_found": dist("first_call_found"),
            "first_call_n_input": dist("first_call_n_input"), "first_call_role_seq": dist("first_call_role_seq"),
            "msgs_first_roles": dist("msgs_first_roles"), "distinct_sys_sha": len({r["sys_sha"] for r in rows}),
            "distinct_user1_sha": len({r["user1_sha"] for r in rows}), "no_system_in_first_call": sum(r["sys_sha"] is None for r in rows),
            "start_status": dist("start_status"), "user1_empty": dist("user1_empty"), "non_text_parts": sum(r["non_text_parts"] for r in rows)}


def compare(a, b):
    A, B = defaultdict(list), defaultdict(list)
    for r in a:
        A[r["key"]].append(r)
    for r in b:
        B[r["key"]].append(r)
    common = sorted(set(A) & set(B))
    out = {"tasks_a": len(A), "tasks_b": len(B), "tasks_common": len(common),
           "excluded_runs_by_status": {"a": dict(Counter(x["start_status"] for k in common for x in A[k] if not x["input_ok"])),
                                       "b": dict(Counter(x["start_status"] for k in common for x in B[k] if not x["input_ok"]))}}
    # over runs whose first acting call resolved with a system and a user message on both sides (missing hashes never
    # count as equal); the joint start = (system, first user turn, message count), and the whole ordered first input
    started = [k for k in common if any(x["input_ok"] for x in A[k]) and any(x["input_ok"] for x in B[k])]
    fa = {k: [x for x in A[k] if x["input_ok"]] for k in started}
    fb = {k: [x for x in B[k] if x["input_ok"]] for k in started}
    out["tasks_both_started"] = len(started)
    joint = lambda x: (x["sys_sha"], x["user1_sha"], x["first_call_n_input"])  # noqa: E731
    out["joint_start_differs_when_both_started"] = sum(len({joint(x) for x in fa[k]} | {joint(x) for x in fb[k]}) > 1 for k in started)
    out["whole_first_input_differs_when_both_started"] = sum(len({x["input_sha"] for x in fa[k]} | {x["input_sha"] for x in fb[k]}) > 1 for k in started)
    for f in ("sys_sha", "user1_sha", "first_call_n_input"):
        out[f"{f}_differs_when_both_started"] = sum(len({x[f] for x in fa[k]} | {x[f] for x in fb[k]}) > 1 for k in started)
    out["non_text_parts"] = {"a": sum(x["non_text_parts"] for k in started for x in fa[k]), "b": sum(x["non_text_parts"] for k in started for x in fb[k])}
    return out


def main(out: str):
    jobs, owner = [], []
    for name, (folders, model, keymode) in COHORTS.items():
        for d in folders:
            files = sorted(d.rglob("*.eval"))
            if not files:
                raise SystemExit(f"{name}: no .eval files under {d}")
            for p in files:
                if "zooc" in str(p) or "oct11c" in str(p):  # the sealed runs are never read
                    raise SystemExit(f"sealed path {p}")
                jobs.append((p, model, keymode))
                owner.append(name)
    rows = defaultdict(list)
    with ProcessPoolExecutor() as ex:
        for name, rs in zip(owner, ex.map(one, jobs, chunksize=4)):
            rows[name].extend(rs)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip())
    res = {"written": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"), "git": {"commit": git, "dirty": dirty},
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "note": "counts and sha256 prefixes only; all eval files in each folder (redraws included); key = cell (Phase 1 attack), "
                   "main task (Phase 1 honest) or task folder (Phase 2)",
           "cohorts": {n: summarize(r) for n, r in rows.items()},
           "comparisons": [{"real": a, "sim_or_other": b, "feeds": f, **compare(rows[a], rows[b])} for a, b, f in COMPARISONS]}
    Path(out).write_text(json.dumps(res, indent=1))
    with open(Path(out).with_suffix(".runs.jsonl"), "w") as fh:  # every run's counts and hashes, for audit (no text)
        for n, rs in rows.items():
            for r in rs:
                fh.write(json.dumps({"cohort": n, **r}) + "\n")
    print(json.dumps({n: {k: v for k, v in c.items() if k in ("samples", "tasks", "first_call_n_input", "fork_prefix_actions", "no_system_in_first_call")} for n, c in res["cohorts"].items()}, indent=0)[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(**vars(ap.parse_args()))
