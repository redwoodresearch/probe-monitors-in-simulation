"""Item 3 (PLAN.md here): is side-task or attack-instruction text visible to the reader? Counts only, no text in any output.

Reads the staged reader inputs every reader extracts from, and per run and message part flags exact (normalized) appearances
of the run's own side-task docs, any same-environment side task's goal, the attacker prompt's lines, and generic markers.

  modal run --detach experiments/2026-10-13_q2_item3_probe_sees/run_probe_sees.py --assets <dir with side_tasks/ and attacker_2026_08_28.md>
"""
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-q2-item3_probe-sees_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
WORK = "probe-shift_aswin-surya_sub-q2-item3_work_kill-2026-11-13"
RUNS = ("2026-10-06_la_segment-end_qwen3-32b_v1", "2026-10-06_la_segment-end-v4_qwen3-32b_v1")
GENERIC = ("side task", "side_task", "red team", "red-team", "covert", "secret objective", "hidden objective", "sabotage", "attack")
PARTS = ("system", "user", "assistant_text", "tool_call_args", "tool_output")
WIN, STRIDE, MINLINE = 60, 30, 40

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS)
work = modal.Volume.from_name(WORK, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12")


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def pieces(text: str) -> set[str]:
    """Distinctive substrings of a doc: its lines/sentences of >= MINLINE characters and WIN-character windows."""
    n = norm(text)
    out = {s for s in re.split(r"(?<=[.!?:])\s|\n", text) if len(norm(s)) >= MINLINE}
    out = {norm(s) for s in out}
    out |= {n[i:i + WIN] for i in range(0, max(1, len(n) - WIN + 1), STRIDE) if len(n[i:i + WIN]) == WIN}
    return out


def read_assets(root: Path) -> dict:
    """Raw texts only (no processing here): {"env|side_task": {"goal.md": ..., "scoring_details.md": ...}}, attacker prompt."""
    docs = {}
    for env_dir in sorted((root / "side_tasks").iterdir()):
        for st_dir in sorted(env_dir.iterdir()):
            docs[f"{env_dir.name}|{st_dir.name}"] = {f: (st_dir / f).read_text() for f in ("goal.md", "scoring_details.md") if (st_dir / f).exists()}
    return {"docs": docs, "attacker": (root / "attacker_2026_08_28.md").read_text()}


def build(raw: dict):
    goals, envgoals = {}, defaultdict(set)
    for k, d in raw["docs"].items():
        env, st = k.split("|")
        goals[(env, st)] = set().union(*[pieces(v) for v in d.values()]) if d else set()
        if "goal.md" in d:
            envgoals[env] |= pieces(d["goal.md"])
    att = [norm(l) for l in raw["attacker"].splitlines() if len(norm(l)) >= MINLINE and "{" not in l and "}" not in l]
    return goals, dict(envgoals), att


def parts_of(messages: list[dict]) -> dict[str, list[str]]:
    out = {p: [] for p in PARTS}
    for m in messages:
        c = m.get("content")
        c = c if isinstance(c, str) else "".join(x.get("text") or "" for x in c or [] if isinstance(x, dict))
        if m["role"] == "system":
            out["system"].append(c)
        elif m["role"] == "user":
            out["user"].append(c)
        elif m["role"] == "assistant":
            out["assistant_text"].append(c or "")
            out["tool_call_args"] += [json.dumps((tc.get("function") or {}).get("arguments"), ensure_ascii=False) for tc in m.get("tool_calls") or []]
        elif m["role"] == "tool":
            out["tool_output"].append(c or "")
    return {p: norm("\n".join(v)) for p, v in out.items()}


def match_side_task(env: str, st: str, goals: dict):
    if (env, st) in goals:
        return (env, st), "exact"
    cands = [k for k in goals if k[0] == env and (k[1].endswith(st) or st.endswith(k[1]) or st in k[1] or k[1] in st)]
    return (cands[0], "fuzzy") if len(cands) == 1 else (None, "unmatched")


@app.function(image=image, volumes={"/acts": acts}, cpu=4, memory=8192, timeout=900)
def scan(path: str, raw: dict) -> list[dict]:
    goals, envgoals, att = build(raw)
    rows = []
    with open(f"/acts/{path}") as fh:
        for line in fh:
            r = json.loads(line)
            P = parts_of(r["messages"])
            st = r.get("side_task") or ""
            env = r.get("environment") or ""
            gk, how = match_side_task(env, st, goals) if st else (None, "none")
            own = goals.get(gk, set()) if gk else set()
            rec = {"arm": r["arm"], "key": r["key"], "main_task": r.get("main_task"), "source": path.split("/")[0],
                   "side_task_match": how, "system_sha": hashlib.sha256(P["system"].encode()).hexdigest(),
                   "n_by_role": dict(Counter(m["role"] for m in r["messages"])), "flags": {}}
            for p in PARTS:
                t = P[p]
                rec["flags"][p] = {"own_goal": any(s in t for s in own),
                                   "env_goal": any(s in t for s in envgoals.get(env, ())),
                                   "attacker_line": any(s in t for s in att),
                                   **{f"generic:{g}": g in t for g in GENERIC}}
            rows.append(rec)
    return rows


@app.function(image=image, volumes={"/acts": acts, "/work": work}, cpu=2, memory=4096, timeout=1800)
def run_all(raw: dict, git: str) -> str:
    acts.reload()
    goals, _, att = build(raw)
    meta = {"side task dirs": len(goals), "side tasks with pieces": sum(bool(v) for v in goals.values()),
            "attacker prompt sha256": hashlib.sha256(raw["attacker"].encode()).hexdigest(), "attacker lines used": len(att),
            "window": WIN, "stride": STRIDE, "min line": MINLINE, "generic": list(GENERIC), "sims snapshot": "ba990399"}
    files = sorted(str(f.relative_to("/acts")) for run in RUNS for f in Path(f"/acts/{run}/inputs").glob("*.jsonl"))
    if not files:
        raise SystemExit("no staged inputs found")
    rows = [x for rs in scan.map(files, kwargs={"raw": raw}) for x in rs]
    seen, uniq = set(), []
    for r in rows:  # the first run's copy of a (arm, key) wins; v4 re-stages #36's arms beside sim-honest
        if (r["arm"], r["key"]) not in seen:
            seen.add((r["arm"], r["key"])); uniq.append(r)
    honest_sys = defaultdict(set)
    for r in uniq:
        if r["arm"] == "real-honest":
            honest_sys[r["main_task"]].add(r["system_sha"])
    out = {"POST HOC": True, "plan": "PLAN.md", "git": git, "inputs": list(RUNS), "files": len(files),
           "rows read": len(rows), "runs (unique arm, key)": len(uniq), "assets": meta, "per arm": {}}
    for arm in sorted({r["arm"] for r in uniq}):
        R = [r for r in uniq if r["arm"] == arm]
        a = {"runs": len(R), "side task id match": dict(Counter(r["side_task_match"] for r in R)),
             "system message present": sum(r["n_by_role"].get("system", 0) == 1 for r in R),
             "system equals the main task's honest prompt": sum(r["system_sha"] in honest_sys.get(r["main_task"], ()) for r in R),
             "messages per role (total)": dict(sum((Counter(r["n_by_role"]) for r in R), Counter())),
             "runs flagged, by part and check": {}}
        for p in PARTS:
            a["runs flagged, by part and check"][p] = {c: sum(r["flags"][p][c] for r in R) for c in R[0]["flags"][p]}
        a["runs flagged in ANY part"] = {c: sum(any(r["flags"][p][c] for p in PARTS) for r in R) for c in R[0]["flags"]["system"]}
        out["per arm"][arm] = a
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    name = f"probe_sees_POSTHOC_{stamp}.json"
    Path("/work/results").mkdir(parents=True, exist_ok=True)
    Path(f"/work/results/{name}").write_text(json.dumps(out, indent=1) + "\n")
    work.commit()
    return name


@app.local_entrypoint()
def main(assets: str, git: str):
    print(run_all.remote(read_assets(Path(assets)), git))
