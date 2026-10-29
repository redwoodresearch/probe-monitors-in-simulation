"""Review check for #1091 (Codex threads rrc7_, rrc8D, rrc8K): the render gate that cleared Llama's paid extraction.
analyze_render_r2j.py loops over the arms present in the report and reads most counters with .get(key, 0), so a dropped
arm or a missing counter would pass silently. This asserts, from the committed files (first copies in git, at cb0a5d7f):
the report holds exactly the five planned arms with their expected run counts; every counter the gate reads is a real key
in every arm (render-failure keys excepted: the renderer adds them only when a failure happens, and counts runs before
rendering); recomputing the gate from the report with the as-run gate code reproduces the committed gate output exactly;
and the provenance (report and gate output sha256 against manifest v1, the gate script's blob, the render's commit).
Exits nonzero on any failure.
  uv run --no-project python experiments/2026-10-12_q2_llama_reader/check_render_gate.py
"""
import hashlib, importlib.util, json, subprocess, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path

D = "experiments/2026-10-12_q2_llama_reader/"
FIRST, ASRUN, GATE_AT = "cb0a5d7f", "8cd2a6cd", "8a751b0a"
REPORT, GATE_OUT = D + "results/llama_render_r2j_2026-10-12_123421.json", D + "results/render_r2j_gate_2026-10-12_123520.json"
RUNS = "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
show = lambda c, p: subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
# the counters analyze_render_r2j.py reads (lines 26-39 at 8a751b0a); "runs" is read directly, the rest with .get(key, 0)
GET = ["outputs Qwen3-32B kept", "tool outputs", "of those, kept under C2 (R2J readout below the context length)",
       "tool outputs whose rendered span JSON-decodes exactly to the raw content", "of those, R2J at index 32,768 or beyond",
       "tool outputs with no Qwen3-32B position", "forward tokens (to the last kept readout)", "Llama tokens"] \
    + [f"slip {k}: {s}" for k in ("(a) trailing newline", "(b) str_replace gutter", "(c) insert review note", "(d) hunk header")
       for s in ("in the raw output", "in the raw output and in the rendered span")] \
    + [f"{r} readouts at C2{s}" for r in ("R0", "R1", "R2J") for s in ("", ": encodes only whitespace", ": encodes a newline")]
fails = []
rep_b, gate_b = show(FIRST, REPORT), show(FIRST, GATE_OUT)
man = json.loads(show(ASRUN, D + "results/s3_manifest_v1.json"))["files"]
for name, b in (("llama_render_r2j_2026-10-12_123421.json", rep_b), ("render_r2j_gate_2026-10-12_123520.json", gate_b)):
    ok = man[name]["sha256"] == sha(b); print(f"{name}: sha256 {sha(b)[:16]} matches manifest v1: {ok}"); fails += [] if ok else [name + " sha256"]
R = json.loads(rep_b); n = defaultdict(dict)
for r in R["rows"]:
    n[r["arm"]][r["count"]] = r["n"]
RUNS_AT, RUNS_SHA = "f0ff845f", "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"  # main when the check first ran; #36's runs.json sha256 (fix939 recorded it)
runs_bytes = subprocess.run(["git", "show", f"{RUNS_AT}:{RUNS}"], capture_output=True, check=True).stdout
assert hashlib.sha256(runs_bytes).hexdigest() == RUNS_SHA, "#36 runs.json differs from its recorded sha256"
runs36 = Counter(r["arm"] for r in json.loads(runs_bytes))
plan = show(ASRUN, D + "PLAN.md").decode()
assert "#36's 3,807 runs and fix939's 823" in plan
want = dict(runs36) | {"sim-attack-fix939": 823}
print(f"planned arms and runs (#36 runs.json at {RUNS_AT}, sha256 checked, + PLAN step A's fix939 823):", want)
print("report arms and runs:", {a: n[a].get("runs") for a in sorted(n)})
if set(n) != set(want) or any(n[a].get("runs") != want[a] for a in want):
    fails.append("arms or run counts differ from the plan")
for a in sorted(n):
    missing = [k for k in GET if k not in n[a]]
    failure_rows = [k for k in n[a] if k.startswith("render failure")]
    print(f"{a}: {len(GET)} gate counters, missing {len(missing)}; render-failure rows {len(failure_rows)}; R2J readouts at C2 {n[a].get('R2J readouts at C2')}, "
          f"whitespace-only {n[a].get('R2J readouts at C2: encodes only whitespace')}, newline {n[a].get('R2J readouts at C2: encodes a newline')}")
    if missing or not n[a].get("R2J readouts at C2"):
        fails.append(f"{a}: missing {missing[:3]} or no R2J readouts")
# recompute the gate with the as-run gate code on the committed report; it must equal the committed gate output
code = show(GATE_AT, D + "analyze_render_r2j.py")
blob = lambda c: subprocess.run(["git", "rev-parse", f"{c}:{D}analyze_render_r2j.py"], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
b_gate, b_asrun = blob(GATE_AT), blob(ASRUN)
print("gate script blob at the render's commit", GATE_AT, b_gate[:16], "| at", ASRUN, b_asrun[:16], "| identical:", b_gate == b_asrun)
fails += [] if b_gate == b_asrun else ["gate script changed after the render"]
with tempfile.TemporaryDirectory() as t:
    (Path(t) / "analyze_render_r2j.py").write_bytes(code); (Path(t) / "results").mkdir()
    rp = Path(t) / "llama_render_r2j_2026-10-12_123421.json"; rp.write_bytes(rep_b)
    spec = importlib.util.spec_from_file_location("gate", Path(t) / "analyze_render_r2j.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.HERE = Path(t); m.print = lambda *a, **k: None
    m.main(str(rp)); mine = json.loads(next((Path(t) / "results").glob("render_r2j_gate_*.json")).read_text())
same = mine == json.loads(gate_b)
print("recomputed gate output equals the committed one:", same, "| gate:", mine["gate"], "| failing:", mine["failing"])
fails += [] if same and mine["gate"] == "pass" else ["recomputed gate differs or fails"]
print("render report's own git (the renderer's commit):", R["git"][:8], "= gate script commit", GATE_AT, ":", R["git"].startswith(GATE_AT))
fails += [] if R["git"].startswith(GATE_AT) else ["the render did not run at the gate script's commit"]
print("FAIL:" if fails else "all checks pass", fails if fails else "")
sys.exit(1 if fails else 0)
