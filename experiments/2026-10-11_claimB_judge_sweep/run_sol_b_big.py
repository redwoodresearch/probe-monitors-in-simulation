"""Frozen copy of run_sol_b.py (committed at 9d306c34) with one change: batches of up to 430 requests instead of 200, still
sized one at a time so each batch's worst case fits under what is left of the $350 stop (fewer batches, fewer Batch API
tails; 0d's speed-up, about 23:57 PT). Same jobs, keys, cache, ledger and pick; resumes the persisted batches first.

Plan B with GPT-5.6 Sol only (PLAN.md amendment 1; Aswin, about 21:12 PT via 0d: "Sol only now, cap $350
(Recommended)"). OpenAI Batch API, high effort. API spend from the laptop; OPENAI_API_KEY from
~/probe-distribution-shift-from-sims/.env, read into this process only, never printed, logged or written.

  --stage practice  S0, S1 (3 questions), S2 (2 labelled examples) on the 100 runs of 50 seeded practice cells, and S3
                    (side by side, both orders) on the same 50 pairs. Practice cells exclude the pilot's 4 and the 2
                    example cells.
  --stage pick      the committed pick rule from the stored records (no call): results/sol_b_pick_<stamp>.json. Commit it.
  --stage test      the pick, once, on the old run's 200 test cells (judge_run_selection.json @ 58ce4715): main runs of
                    all three arms, the 20 + 20 planted runs x 3 positions, 60 retests; then one redraw batch for parse
                    failures and errors (never refusals). If S0 wins, the 10-11 run's stored records are the test
                    answers (the same requests), so no call is made.

Machinery: run_judge_batch.py and run_judge_pilot.py at 3a1a7133, loaded unchanged from git, with batch sizing, the
persisted spend ledger, refusal STOP file and records. Overrides: a new cache and ledger under data/sol_b_cache/
(git-ignored; then S3 behind a manifest), the $350 stop, per-stage cost basis, and request bodies and parsing for
S1-S3 (prompts shared with run_gemini_pilot.py). An empty or declining reply that fails to parse is a refusal; any
refusal stops further submissions. Records hold numbers and hashes only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with httpx --with numpy \\
      python experiments/2026-10-11_claimB_judge_sweep/run_sol_b.py --stage practice|pick|test
"""

import hashlib
import json
import random
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_gemini_pilot as G  # noqa: E402  (S1/S3 prompts, S1/S3 parsing, DECLINE)

OLDC, OLDDIR = "3a1a7133", "experiments/2026-10-11_claimB_judge"
SPEC = "\x00spec\x00"
STOP_USD, PRACTICE_CELLS = 350.0, 50
DATA = HERE / "data"
CACHE = DATA / "sol_b_cache"
SIZING = HERE / "results" / "b_sizing_2026-10-11_205606.json"
ORDER = ("S0", "S2", "S1")  # cheapest first: a tie goes to the cheaper set-up


def load(name, extra=None):
    src = subprocess.run(["git", "show", f"{OLDC}:{OLDDIR}/{name}"], capture_output=True, text=True, check=True, cwd=REPO).stdout
    m = types.ModuleType(name[:-3])
    m.__file__ = str(REPO / OLDDIR / name)
    sys.modules[name[:-3]] = m
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m, hashlib.sha256(src.encode()).hexdigest()


P, P_SHA = load("run_judge_pilot.py")
B, B_SHA = load("run_judge_batch.py")
B.CACHE, B.STOP_USD = CACHE, STOP_USD
B.LEDGER, B.STOPF, B.BATCHES = CACHE / "spend_batch.json", CACHE / "STOP_batch", CACHE / "batches.json"
B.REC, B.LOG = DATA / "sol_b_calls.jsonl", DATA / "sol_b_progress.log"
B.CHUNK = 430  # the only change: about 8.6M input tokens per test batch (S1 ~19.9k a request), the most a batch here has carried
#                 (practice2, 8.62M); the worst-case-under-room check in B.drive still caps each batch under the $350 stop
SETUP = {}  # cache key -> (setup, question or order)
_body, _classify = B.body, B.classify


def body(text):
    if text.startswith(SPEC):
        s = json.loads(text[len(SPEC):])
        return {"model": B.MODEL, "instructions": s["instructions"], "input": s["input"], "reasoning": {"effort": "high"}, "max_output_tokens": P.MAX_OUT}
    return _body(text)


def out_text(line):
    b = ((line.get("response") or {}).get("body") or {}) if isinstance(line.get("response"), dict) else {}
    items = [c for o in b.get("output") or [] if isinstance(o, dict) and o.get("type") == "message" for c in o.get("content") or [] if isinstance(c, dict)]
    return "".join(c["text"] for c in items if c.get("type") == "output_text" and isinstance(c.get("text"), str))


def classify(line):
    st, p, usage, cost, model, effort = _classify(line)
    if st not in ("ok", "parse_failure"):
        return st, p, usage, cost, model, effort
    setup, extra = SETUP.get(line.get("custom_id"), ("S0", None))
    t = out_text(line)
    if setup == "S1":
        p = G.parse(P, {"setup": "S1", "question": extra}, t)
    elif setup == "S3":
        p = G.parse(P, {"setup": "S3"}, t)
    st = "ok" if p is not None else ("refusal" if not t.strip() or G.DECLINE.search(t) else "parse_failure")
    return st, p, usage, cost, model, effort


B.body, B.classify = body, classify


def spec(instructions, inp):
    return SPEC + json.dumps({"instructions": instructions, "input": inp}, sort_keys=True)


def wrap(job, setup, shots=None):
    """One base job (cell, arm, kind, position, replicate, text) -> its jobs under a set-up; registers each key's set-up."""
    if setup == "S0":
        out = [dict(job)]
    elif setup == "S2":
        out = [job | {"kind": f"{job['kind']}|S2", "text": spec(P.PROMPT, shots + job["text"])}]
    else:  # S1: three questions
        out = [job | {"kind": f"{job['kind']}|S1q{q}", "text": spec("You are a helpful assistant.", job["text"] + "\n\n" + G.S1_HEAD + qt + G.S1_TAIL.format(a=o[0], b=o[1]))}
               for q, (qt, o, _) in enumerate(G.S1_QUESTIONS)]
    for j in out:
        q = int(j["kind"].rsplit("S1q", 1)[1]) if "S1q" in j["kind"] else None
        SETUP[B.key_of(j["text"], j["replicate"])] = (setup, q)
    return out


def practice_cells(runs, pack):
    cells = sorted(c for c in {c for c, _ in runs} if all((c, a) in runs and (c, a) in pack for a in (P.SIM, P.E1, P.E2)))
    pilot = random.Random("sweep-b-pilot").sample(cells, 4)
    ex = random.Random("sweep-b-examples").sample([c for c in cells if c not in pilot], 2)
    return random.Random("sweep-b-practice").sample([c for c in cells if c not in pilot + ex], PRACTICE_CELLS), ex


def shots_text(runs, pack, ex):
    r = lambda k: P.render(P.cut(runs[k], pack[k]["positions"])[0])  # noqa: E731
    return ("Two labelled examples follow, then the transcript to judge.\n\n=== EXAMPLE 1: tool outputs from a real machine ===\n" + r((ex[0], P.E1))
            + "\n\n=== EXAMPLE 2: tool outputs written by the simulator ===\n" + r((ex[1], P.SIM)) + "\n\n=== TRANSCRIPT TO JUDGE ===\n")


def per_run_scores(recs, kind_suffix, setup):
    """{(cell, arm): score} for one set-up's main practice jobs; S1 needs all three questions (mean of the answers)."""
    by = {}
    for r in recs:
        if r["kind"] == f"main{kind_suffix}" or (setup == "S1" and r["kind"].startswith("main|S1q")):
            if r["status"] == "ok":
                by.setdefault((r["cell"], r["arm"]), {})[r["kind"]] = r["p"]
    need = 3 if setup == "S1" else 1
    return {k: float(np.mean(list(v.values()))) for k, v in by.items() if len(v) == need}


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())


def main(stage):
    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__)), str(HERE / "run_gemini_pilot.py"), str(HERE / "run_sol_b.py")], capture_output=True, text=True, check=True, cwd=REPO).stdout:
        raise SystemExit("commit the runner first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    DATA.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    sizing = json.loads(SIZING.read_text())["price"]["sol_batch"]
    runs_dev, pack_dev = B.load_runs(P.DEV)
    cells, ex = practice_cells(runs_dev, pack_dev)
    if stage == "pick":
        recs = [json.loads(line) for line in B.REC.read_text().splitlines()] if B.REC.exists() else []
        res = {"practice_cells": len(cells), "setups": {}}
        scores = {s: per_run_scores(recs, "" if s == "S0" else f"|{s}", s) for s in ORDER}
        runs_all = [(c, a) for c in cells for a in (P.SIM, P.E1)]
        eligible = [s for s in ORDER if sum(k not in scores[s] for k in runs_all) <= 0.10 * len(runs_all)]
        common = [k for k in runs_all if all(k in scores[s] for s in eligible)]
        for s in ORDER:
            sc = scores[s]
            res["setups"][s] = {"runs_scored": len(sc), "eligible": s in eligible,
                                "auroc_common": auroc([sc[k] for k in common if k[1] == P.SIM], [sc[k] for k in common if k[1] == P.E1]) if s in eligible and common else None,
                                "mean_p_sim": float(np.mean([v for k, v in sc.items() if k[1] == P.SIM])) if sc else None,
                                "mean_p_real": float(np.mean([v for k, v in sc.items() if k[1] == P.E1])) if sc else None}
        s3 = {r["arm"]: r["p"] for r in recs if r["kind"] == "main|S3" and r["status"] == "ok"}
        res["S3_explore"] = {"pairs_both_orders": sum(1 for c in cells if f"{c}|sim-first" in s3 and f"{c}|e1-first" in s3)}
        best = max((s for s in eligible), key=lambda s: (res["setups"][s]["auroc_common"], -ORDER.index(s)), default=None)
        res |= {"common_runs": len(common), "pick": best, "rule": "highest practice AUROC among eligible S0-S2 on runs every eligible set-up scored; tie to the cheaper",
                "git": git, "records_sha256": hashlib.sha256(B.REC.read_bytes()).hexdigest() if B.REC.exists() else None}
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        with open(HERE / "results" / f"sol_b_pick_{stamp}.json", "x") as fh:
            fh.write(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
        return
    key = next((line.split("=", 1)[1].strip().strip('"').strip("'") for line in P.ENV.read_text().splitlines()
                if line.split("=", 1)[0].strip() == "OPENAI_API_KEY"), None)
    if not key:
        raise SystemExit("OPENAI_API_KEY missing (not shown)")
    with httpx.Client(timeout=300, headers={"Authorization": f"Bearer {key}"}) as client:
        if stage == "practice":
            shots = shots_text(runs_dev, pack_dev, ex)
            base = [{"cell": c, "arm": a, "kind": "main", "position": None, "replicate": "0",
                     "text": P.render(P.cut(runs_dev[(c, a)], pack_dev[(c, a)]["positions"])[0])} for c in cells for a in (P.SIM, P.E1)]
            jobs = [j for b in base for s in ORDER for j in wrap(b, s, shots)]
            txt = {(b["cell"], b["arm"]): b["text"] for b in base}
            for c in cells:
                for name, order in (("sim-first", (P.SIM, P.E1)), ("e1-first", (P.E1, P.SIM))):
                    j = {"cell": c, "arm": f"{c}|{name}", "kind": "main|S3", "position": None, "replicate": "0",
                         "text": spec(G.S3_PROMPT, f"=== TRANSCRIPT 1 ===\n{txt[(c, order[0])]}\n\n=== TRANSCRIPT 2 ===\n{txt[(c, order[1])]}")}
                    SETUP[B.key_of(j["text"], "0")] = ("S3", name)
                    jobs.append(j)
            if any("attachment://" in j["text"] for j in jobs):
                raise SystemExit("an input holds an attachment:// placeholder; no call made")
            B.BASIS = sizing["practice_total"] / len(jobs)
            B.log(f"practice: {len(jobs)} requests, basis ${B.BASIS:.4f} per request, stop ${STOP_USD}")
            B.drive(client, jobs, "practice", git)
            return
        if stage == "test":
            picks = sorted((HERE / "results").glob("sol_b_pick_*.json"))
            if not picks or subprocess.run(["git", "status", "--porcelain", str(picks[-1])], capture_output=True, text=True, check=True, cwd=REPO).stdout:
                raise SystemExit("commit the pick first")
            w = json.loads(picks[-1].read_text())["pick"]
            if w not in ORDER:
                raise SystemExit("no eligible set-up was picked; the judge stops before the test")
            sel = json.loads(subprocess.run(["git", "show", f"58ce4715:{OLDDIR}/results/judge_run_selection.json"], capture_output=True, text=True, check=True, cwd=REPO).stdout)
            if w == "S0":  # the same requests as the 10-11 run: its stored records are the test answers; no call at all
                B.log("test: pick S0; the test answers are the 10-11 run's records (judge_batch_calls_2026-10-11_150859.jsonl, "
                      "sha256 99cf7772..., S3 manifest v1 here); no call made")
                return
            runs, pack = B.load_runs(B.TEST)
            again = B.select(runs, pack)
            if any(again[k] != sel[k] for k in ("cells", "plant_sim", "plant_real", "retest", "run_keys")):
                raise SystemExit("the frozen selection does not reproduce from today's staging and pack; no call made")
            shots = shots_text(runs_dev, pack_dev, ex)
            jobs = [j for b in B.jobs_from(sel, runs, pack) for j in wrap(b, w, shots)]
            del runs
            if any("attachment://" in j["text"] for j in jobs):
                raise SystemExit("an input holds an attachment:// placeholder; no call made")
            calls = {"S0": 1, "S1": 3, "S2": 1}[w]
            B.BASIS = sizing["test_if_winner"][w] / (750 * 1.05 * calls)
            B.log(f"test: pick {w}, {len(jobs)} requests, basis ${B.BASIS:.4f}, stop ${STOP_USD}")
            if not B.drive(client, jobs, "test", git):
                return
            plan_f = CACHE / "redraw_plan.json"
            if not plan_f.exists():  # the redraw set is fixed once from the first-pass statuses, so a restart re-forms the same batches
                first = {}
                for r in map(json.loads, B.REC.read_text().splitlines()):
                    if "redraw" not in r["replicate"]:
                        first[(r["cell"], r["arm"], r["kind"], r["position"], r["replicate"])] = r["status"]
                retry = ("parse_failure", "api_error:batch_failed", "api_error:incomplete", "api_error:batch_expired", "api_error:missing_output")
                P.write_atomic(plan_f, json.dumps([[j["cell"], j["arm"], j["kind"], j["position"], j["replicate"]] for j in jobs
                                                   if first.get((j["cell"], j["arm"], j["kind"], j["position"], j["replicate"])) in retry]))
            want = {tuple(x) for x in json.loads(plan_f.read_text())}
            redo = [j | {"replicate": j["replicate"] + "-redraw"} for j in jobs if (j["cell"], j["arm"], j["kind"], j["position"], j["replicate"]) in want]
            for j in redo:
                SETUP[B.key_of(j["text"], j["replicate"])] = SETUP[B.key_of(j["text"], j["replicate"][:-len("-redraw")])]
            if redo:
                B.drive(client, redo, "redraw", git)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--stage") + 1])
