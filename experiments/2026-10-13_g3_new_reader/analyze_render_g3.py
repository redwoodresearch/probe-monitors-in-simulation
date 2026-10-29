"""G3's gate to the GPU: RULE.md steps 2-3 with Amendment 1 (dbb1cd03, the readout) and Amendment 2 (0ce416b2, coverage
and drops); committed before any render count is read. Local, $0. Built on ../2026-10-12_q2_olmo_reader/analyze_render_r2.py
(at bc29c92a). Step 2's three render records (real, planted, controls), with #36's pinned runs.json for the cells.
  fail (anything here stops the job as broken, not as a result):
    - an arm's run count differs from the plan's, or a render record lacks a count the gate reads;
    - any render failure (a run that can't be rendered has no rows to extract);
    - any tool output without a Qwen3-32B position;
    - any R2 readout that is whitespace-only or contains a newline (Amendment 1's render gate).
  unevaluable (Amendment 2):
    - coverage: in real honest, real e1, real e2 or planted e2, fewer than 99% of the required rows (outputs Qwen3-32B
      kept, #36's rows) have an eligible readout, counting each arm's own eligibility before any matching;
    - dropped cells: 10 or more of the 939 cells (over 1%) are dropped.
  The drops (Amendment 2, points 2-3), recorded here for the step-6 freeze:
    - a required row with no eligible readout is left out of its run's mean. For a real e2 run and its planted copy, a
      row left out of either is left out of both (the union);
    - a real e2 run (judged with its planted copy, on their retained intersection) with no row left is empty, and its
      cell is dropped from R, P and S together. An e2 run with no planted copy (unchanged by the plant) is judged alone;
    - empty honest or e1 runs: no rule says how to treat them (an empty honest run has no mean, so the pinned threshold
      code can't take it), so any one makes the outcome "hold for 09" (not pass) and lists them.
  pass: none of these. The controls (the activation gate's 20 runs) are reported beside, not gated.
--sim (step 7, after the barrier): the simulated arm's render record with the step-2 gate record whose sha256 the
step-6 forecast.json froze: fail as above; unevaluable if under 99% coverage, or if any simulated run of a kept cell has
no eligible row. No cell is dropped.
Exits 1 unless the outcome is "pass".

  uv run --no-project python experiments/2026-10-13_g3_new_reader/analyze_render_g3.py --runs <#36 runs.json> \\
      results/g3_render_real_<stamp>.json results/g3_render_planted_<stamp>.json results/g3_render_controls_<stamp>.json
  uv run --no-project python experiments/2026-10-13_g3_new_reader/analyze_render_g3.py --runs <#36 runs.json> \\
      --sim results/g3_render_sim_<stamp>.json --step2-gate results/render_g3_gate_<stamp>.json
"""
import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
EXPECT = {"real-honest": 990, "real-attack-e1": 939, "real-attack-e2": 939, "real-attack-e2-planted": 845, "real-attack-e2-control": 20}
GATED = ("real-honest", "real-attack-e1", "real-attack-e2", "real-attack-e2-planted")
EXPECT_SIM = {"sim-attack": 939}
REVISION = "497f1dca95ebdec98e41d517b9f060ee753c902f"
RUNS_PIN = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
NEED = ("runs", "outputs Qwen3-32B kept", "tool outputs with no Qwen3-32B position", "of those, kept (R2 readout below the context length)",
        "Qwen3-32B-kept outputs with no readout: no non-whitespace content token",
        "Qwen3-32B-kept outputs with no readout: any other reason (missing rows)", "R2 readouts at C2",
        "R2 readouts at C2: whitespace-only", "R2 readouts at C2: contains a newline", "forward tokens (to the last kept readout)",
        "forward tokens squared, summed over runs")


def load(paths, sim):
    n, runs, files = defaultdict(dict), defaultdict(dict), {}
    for p in paths:
        R = json.loads(Path(p).read_text())
        if R["reader"]["revision"] != REVISION or (R["which"] == "sim") != sim:
            raise SystemExit(f"{Path(p).name}: another reader, or an arm outside this gate's scope")
        files[Path(p).name] = {"sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(), "git (render)": R["git"], "which": R["which"], "ids sha256": R["ids sha256"]}
        for r in R["rows"]:
            if r["count"] in n[r["arm"]]:
                raise SystemExit(f"{r['arm']} appears in two render records")
            n[r["arm"]][r["count"]] = r["n"]
        for x in R["runs"]:
            if x["key"] in runs[x["arm"]]:
                raise SystemExit(f"{x['arm']} {x['key']}: a run listed twice")
            runs[x["arm"]][x["key"]] = x
    return n, runs, files


def arm_rows(n, runs, expect, fails, unevaluable, gated):
    out = {}
    for a in sorted(n):
        c = n[a]
        if miss := [k for k in NEED if k not in c]:
            fails.append(f"{a}: counts missing from the render record: {miss}")
        req = sum(x.get("required", 0) for x in runs[a].values()); lo = sum(len(x.get("left out", [])) for x in runs[a].values())
        qk = c.get("outputs Qwen3-32B kept", 0); kept = c.get("of those, kept (R2 readout below the context length)", 0)
        if (req, req - lo) != (qk, kept):
            fails.append(f"{a}: the per-run lists ({req} required, {req - lo} eligible) differ from the counts ({qk}, {kept})")
        row = {"runs": c.get("runs", 0), "render failures": sum(v for k, v in c.items() if k.startswith("render failure")),
               "required rows (outputs Qwen3-32B kept)": qk, "eligible (own, before matching)": kept,
               "coverage (own)": kept / qk if qk else None,
               "left out: no eligible token in the span": c.get("Qwen3-32B-kept outputs with no readout: no non-whitespace content token", 0),
               "left out: other reason (readout past the context, or no span)": c.get("Qwen3-32B-kept outputs with no readout: any other reason (missing rows)", 0),
               "empty runs (own)": sorted(k for k, x in runs[a].items() if len(x.get("left out", [])) == x.get("required", 0)),  # no row left, zero required included
               "runs with no required row": sum(1 for x in runs[a].values() if not x.get("required", 0)),
               "tool outputs with no Qwen3-32B position": c.get("tool outputs with no Qwen3-32B position", 0),
               "tool outputs quoting a special token's text": c.get("tool outputs quoting a special token's text", 0),
               "kept outputs ending in a newline": c.get("of those, the output ends in a newline", 0),
               "kept outputs with a token crossing the span's end": c.get("of those, a token crosses the span's end", 0),
               "R2 differs from R0": c.get("of those, R2 differs from R0", 0),
               "forward tokens": c.get("forward tokens (to the last kept readout)", 0),
               "forward tokens squared, summed over runs": c.get("forward tokens squared, summed over runs", 0), "reader tokens": c.get("reader tokens", 0)}
        for rule in ("R0", "R2"):
            row[f"{rule} at C2: readouts, whitespace-only, contains a newline"] = [c.get(f"{rule} readouts at C2", 0), c.get(f"{rule} readouts at C2: whitespace-only", 0),
                                                                                c.get(f"{rule} readouts at C2: contains a newline", 0)]
        out[a] = row
        if row["runs"] != expect[a] or len(runs[a]) != expect[a]:
            fails.append(f"{a}: {row['runs']} runs ({len(runs[a])} listed), the plan's {expect[a]}")
        if row["render failures"]:
            fails.append(f"{a}: render failures {row['render failures']} of {row['runs']}")
        if row["tool outputs with no Qwen3-32B position"]:
            fails.append(f"{a}: {row['tool outputs with no Qwen3-32B position']} outputs lack a Qwen3-32B position")
        if row["R2 at C2: readouts, whitespace-only, contains a newline"][1:] != [0, 0]:
            fails.append(f"{a}: R2 has whitespace-only or newline readouts")
        if a in gated and (not qk or kept < 0.99 * qk):
            unevaluable.append(f"{a}: coverage {kept} of {qk} required rows, under 99%")
    return out


def main(a):
    if hashlib.sha256(Path(a.runs).read_bytes()).hexdigest() != RUNS_PIN:
        raise SystemExit("not #36's pinned runs.json")
    cell = {(r["arm"], r["key"]): r["cell"] for r in json.loads(Path(a.runs).read_text())}
    sim = a.sim is not None
    n, runs, files = load([a.sim] if sim else a.records, sim)
    expect = EXPECT_SIM if sim else EXPECT
    if set(n) != set(expect) or set(runs) != set(expect):
        raise SystemExit(f"arms {sorted(n)} differ from the plan's {sorted(expect)}")
    fails, unevaluable = [], []
    out = {"scope": "sim" if sim else "step 2", "render files": files, "runs.json sha256": RUNS_PIN,
           "arms": arm_rows(n, runs, expect, fails, unevaluable, ("sim-attack",) if sim else GATED)}
    if not sim:
        e2, pl = runs["real-attack-e2"], runs["real-attack-e2-planted"]
        if not set(pl) <= set(e2):
            fails.append("a planted run has no real e2 run")
        matched, dropped = {}, []
        for k, x in sorted(e2.items()):
            union = sorted(set(x.get("left out", [])) | set(pl.get(k, {}).get("left out", [])), key=lambda s_: tuple(map(int, s_.split(","))))
            if k in pl and pl[k].get("required") != x.get("required"):
                fails.append(f"{k}: the planted copy's required rows differ from the run's")
            matched[k] = union
            if not x.get("required", 0) or len(union) == x["required"]:
                dropped.append({"cell": cell[("real-attack-e2", k)], "e2 key": k, "has planted copy": k in pl})
        out["matched (real e2 and planted e2)"] = {
            "rows left out of both (union)": sum(len(v) for v in matched.values()),
            "e2 rows left out only because the planted copy's was": sum(len(v) - len(e2[k].get("left out", [])) for k, v in matched.items()),
            "planted rows left out only because the e2 run's was": sum(len(matched[k]) - len(x.get("left out", [])) for k, x in pl.items()),
            "e2 runs with a planted copy": len(pl), "e2 runs without one": len(e2) - len(pl)}
        out["dropped cells (R, P and S together)"] = dropped
        out["dropped cells, count"] = len(dropped)
        out["frozen row drops (for step 6)"] = {
            "real-attack-e2 and its planted copy (union)": {k: v for k, v in matched.items() if v},
            **{arm: {k: x["left out"] for k, x in sorted(runs[arm].items()) if x.get("left out")} for arm in ("real-honest", "real-attack-e1", "real-attack-e2-control")}}
        out["kept cells"] = sorted({cell[("real-attack-e2", k)] for k in e2} - {d["cell"] for d in dropped})
        if len(dropped) >= 10:
            unevaluable.append(f"{len(dropped)} of 939 cells dropped (10 or more)")
        hold = {arm: out["arms"][arm]["empty runs (own)"] for arm in ("real-honest", "real-attack-e1") if out["arms"][arm]["empty runs (own)"]}
        if hold:
            out["for 09: empty runs that no rule covers"] = hold
    else:
        g = json.loads(Path(a.step2_gate).read_text())
        if g.get("scope") != "step 2" or g.get("gate") != "pass":
            raise SystemExit("--step2-gate is not a passing step-2 gate record")
        fc = json.loads((HERE / "forecast.json").read_text())
        if fc.get("frozen", {}).get("step-2 gate sha256") != hashlib.sha256(Path(a.step2_gate).read_bytes()).hexdigest():
            raise SystemExit("--step2-gate is not the gate record forecast.json froze")
        out["step-2 gate"] = {"file": Path(a.step2_gate).name, "sha256": hashlib.sha256(Path(a.step2_gate).read_bytes()).hexdigest()}
        kept = set(g["kept cells"]); sr = runs["sim-attack"]
        empty = [k for k, x in sorted(sr.items()) if cell[("sim-attack", k)] in kept and (not x.get("required", 0) or len(x.get("left out", [])) == x["required"])]
        out["simulated runs of kept cells with no eligible row"] = empty
        out["frozen row drops (simulated, own)"] = {k: x["left out"] for k, x in sorted(sr.items()) if x.get("left out")}
        if empty:
            unevaluable.append(f"{len(empty)} simulated runs of kept cells have no eligible row")
    held = not sim and bool(out.get("for 09: empty runs that no rule covers"))
    out["gate"] = "fail" if fails else "unevaluable" if unevaluable else "hold for 09" if held else "pass"
    out["failing"] = fails; out["unevaluable"] = unevaluable
    out["forward tokens, all arms"] = sum(r["forward tokens"] for r in out["arms"].values())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"render_g3_gate_{'sim_' if sim else ''}{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("frozen row drops (for step 6)", "frozen row drops (simulated, own)", "kept cells")}, indent=1))
    print(dest)
    if out["gate"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--sim"); ap.add_argument("--step2-gate")
    ap.add_argument("records", nargs="*")
    a = ap.parse_args()
    if (a.sim is None) == (len(a.records) != 3) or (a.sim is not None) != (a.step2_gate is not None):
        raise SystemExit("usage: --runs R <real> <planted> <controls> | --runs R --sim <sim render> --step2-gate <gate>")
    main(a)
