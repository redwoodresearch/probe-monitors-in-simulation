"""AI-judge run by the OpenAI Batch API (PLAN.md amendment 3): GPT-5.6 Sol at high effort, the pilot's prompt, settings,
rendering, cut, plants and parser (imported unchanged from run_judge_pilot.py). API spend; OPENAI_API_KEY from
~/probe-distribution-shift-from-sims/.env, loaded into this process only, never printed, logged or written.

  --check   the batch check: the pilot's first 6 dev main jobs as one batch; writes results/judge_batch_check_<stamp>.json
            (pass/fail on the amendment's criteria; numbers only). Stop $5 (its worst case, every reply at the 20,000-token cap, is about $3.5).
  --select  the run's seeded selection (200 test cells, plants, retest) to results/judge_run_selection.json; no call.
  --run     the run: needs a passing check result and the committed selection; chunks of at most 200 requests, one at
            a time, each sized so its worst case fits under the $107 stop, and the expected total (measured mean per call) checked too; one redraw batch at the end for parse failures
            and errors (never refusals). Any refusal stops further submissions.

Keys and numbers only leave this process: records in data/judge_batch_calls.jsonl (git-ignored; to S3 with a
manifest), progress in data/judge_batch_progress.log. Batch input files (they hold transcripts) and raw outputs stay in
the cache outside git. Failed, expired and refused requests are counted, never scored. Batch ids are persisted, so a
restart polls an existing batch instead of resubmitting.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with httpx \\
      python experiments/2026-10-11_claimB_judge/run_judge_batch.py --check|--select|--run
"""

import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_judge_pilot as P  # noqa: E402  (prompt, render, cut, plant, parse, safety, cache path, constants)

TEST = {"clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"}
MODEL, RATES, BATCH_FACTOR = "gpt-5.6-sol", (5.0, 30.0), 0.5
BASIS, STOP_USD, CHECK_STOP, CHUNK, N_CELLS, N_PLANT, N_RETEST = 0.0837, 107.0, 5.0, 200, 200, 20, 60  # check: worst case about $3.5, expected about $0.5
PARAMS = {"mode": "batch", "reasoning_effort": "high", "max_output_tokens": P.MAX_OUT}
CACHE = P.CACHE
LEDGER, STOPF, BATCHES = CACHE / "spend_batch.json", CACHE / "STOP_batch", CACHE / "batches.json"
REC, LOG = HERE / "data" / "judge_batch_calls.jsonl", HERE / "data" / "judge_batch_progress.log"
SEL = HERE / "results" / "judge_run_selection.json"
API = "https://api.openai.com/v1"


def log(msg):
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S")
    with open(LOG, "a") as fh:
        fh.write(f"{stamp} {msg}\n")
    print(f"{stamp} {msg}", flush=True)


def body(text):
    return {"model": MODEL, "instructions": P.PROMPT, "input": text, "reasoning": {"effort": "high"}, "max_output_tokens": P.MAX_OUT}


def key_of(text, tag):
    return hashlib.sha256(json.dumps({"provider": "openai", "model": MODEL, "params": PARAMS, "instructions": P.PROMPT, "input": text,
                                      "replicate": tag}, sort_keys=True).encode()).hexdigest()


def load_runs(envs):
    vol = modal.Volume.from_name(P.VOL, version=2)
    pack = {(r["cell"], r["arm"]): r for r in json.loads(b"".join(vol.read_file(f"{P.PACK}/runs.L19.json"))) if r["environment"] in envs}
    runs = {}
    for path in sorted(e.path for e in vol.listdir(f"{P.STAGED}/inputs")):
        for line in b"".join(vol.read_file(path)).splitlines():
            r = json.loads(line)
            if r["environment"] in envs and r["arm"] in (P.SIM, P.E1, P.E2):
                if (r["cell"], r["arm"]) in pack and pack[(r["cell"], r["arm"])]["key"] != r["key"]:
                    raise SystemExit("staging and pack disagree on a run key")
                runs[(r["cell"], r["arm"])] = r["messages"]
    return runs, pack


def select(runs, pack):
    cells = sorted(c for c in {c for c, _ in runs} if all((c, a) in runs and (c, a) in pack for a in (P.SIM, P.E1, P.E2)))
    tasks = sorted({c.split(":")[0] for c in cells})
    random.Random("judge-run-200").shuffle(tasks)
    pick = []
    for t in tasks:
        mine = sorted(c for c in cells if c.split(":")[0] == t)
        if len(pick) + len(mine) > N_CELLS:
            mine = random.Random("judge-run-200").sample(mine, N_CELLS - len(pick))  # the same seed, as amendment 3 says
        pick += mine
        if len(pick) == N_CELLS:
            break
    if len(pick) != N_CELLS:
        raise SystemExit(f"only {len(pick)} eligible cells")
    sims, reals = random.Random("judge-run-plant-sim").sample(pick, N_PLANT), random.Random("judge-run-plant-real").sample(pick, N_PLANT)
    base = [(c, a) for c in pick for a in (P.SIM, P.E1, P.E2)]
    retest = random.Random("judge-run-retest").sample(base, N_RETEST)
    return {"cells": pick, "plant_sim": sims, "plant_real": reals, "retest": [list(x) for x in retest],
            "run_keys": {f"{c}|{a}": pack[(c, a)]["key"] for c, a in base}, "main_tasks": len({c.split(':')[0] for c in pick})}


def jobs_from(sel, runs, pack):
    base = [(c, a) for c in sel["cells"] for a in (P.SIM, P.E1, P.E2)]
    cutm = {k: P.cut(runs[k], pack[k]["positions"]) for k in base}
    jobs = [{"cell": c, "arm": a, "kind": "main", "position": None, "replicate": "0", "text": P.render(cutm[(c, a)][0])} for c, a in base]
    for arm, chosen in ((P.SIM, sel["plant_sim"]), (P.E1, sel["plant_real"])):
        for c in chosen:
            msgs, kept_out = cutm[(c, arm)]
            for pos, w in P.POSITIONS.items():
                jobs.append({"cell": c, "arm": arm, "kind": "plant", "position": pos, "replicate": "0", "text": P.render(P.plant(msgs, kept_out, w))})
    jobs += [{"cell": c, "arm": a, "kind": "retest", "position": None, "replicate": "1", "text": P.render(cutm[(c, a)][0])} for c, a in sel["retest"]]
    return jobs


def classify(line):
    """(status, p, usage, cost, model echoed, effort echoed) for one batch output or error line. Statuses are fixed local labels."""
    resp = line.get("response") if isinstance(line.get("response"), dict) else None
    b = resp.get("body") if resp and isinstance(resp.get("body"), dict) else {}
    err = line.get("error") or (b.get("error") if b else None)
    u = b.get("usage") if isinstance(b.get("usage"), dict) else {}
    det_in = u.get("input_tokens_details") if isinstance(u.get("input_tokens_details"), dict) else {}
    usage = {"input": P.num(u.get("input_tokens")), "output": P.num(u.get("output_tokens")), "cached_input": P.num(det_in.get("cached_tokens")),
             "reasoning": P.num((u.get("output_tokens_details") or {}).get("reasoning_tokens") if isinstance(u.get("output_tokens_details"), dict) else 0)}
    # listed rates (sim.prices, the preregistered basis): uncached input 5, cached input 0.5, output 30 per million; batch half
    cost = BATCH_FACTOR * ((usage["input"] - usage["cached_input"]) * RATES[0] + usage["cached_input"] * 0.5 + usage["output"] * RATES[1]) / 1e6
    usage["missing"] = bool(resp and resp.get("status_code") == 200 and not (isinstance(u.get("input_tokens"), (int, float)) and isinstance(u.get("output_tokens"), (int, float))))
    model = str(b.get("model", ""))[:40] if b else ""
    effort = (b.get("reasoning") or {}).get("effort") if isinstance(b.get("reasoning"), dict) else None
    effort = effort if effort in ("minimal", "low", "medium", "high", "xhigh") else ("other" if effort else None)
    if P.safety(err) or (resp and resp.get("status_code") == 403):
        return "refusal", None, usage, cost, model, effort
    if err or not resp or resp.get("status_code") != 200:
        code = err.get("code") if isinstance(err, dict) else None
        return ("api_error:batch_expired" if code == "batch_expired" else "api_error:batch_failed"), None, usage, cost, model, effort
    items = [c for o in b.get("output") or [] if isinstance(o, dict) and o.get("type") == "message" for c in o.get("content") or [] if isinstance(c, dict)]
    if any(c.get("type") == "refusal" for c in items) or (b.get("incomplete_details") or {}).get("reason") == "content_filter":
        return "refusal", None, usage, cost, model, effort
    if b.get("status") != "completed":
        return "api_error:incomplete", None, usage, cost, model, effort
    p = P.parse("".join(c["text"] for c in items if c.get("type") == "output_text" and isinstance(c.get("text"), str)))
    return ("ok" if p is not None else "parse_failure"), p, usage, cost, model, effort


def ledger():
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {"spent": 0.0, "calls": 0, "counted": [], "pending": {}}


def get_json(client, url):
    """GET with a checked status; a few tries, then stop (never record a retrieval error as a result)."""
    code = None
    for attempt in range(5):
        try:
            r = client.get(url)
        except Exception as e:  # noqa: BLE001  (exception text can echo request headers; never shown)
            code = type(e).__name__
            time.sleep(15 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r
        code = r.status_code
        time.sleep(15 * (attempt + 1))
    raise SystemExit(f"retrieval failed ({code})") from None


def record(job, k, c, git, name):
    rec = {"judge": "sol", "model": MODEL, "provider": "openai", "params": PARAMS, "git": git, "batch_chunk": name,
           "cell": job["cell"], "arm": job["arm"], "kind": job["kind"], "position": job["position"], "replicate": job["replicate"],
           "status": c["status"], "p": c["p"], "usage": c["usage"], "cost_usd": c["cost"], "model_echo_ok": c["model_echo_ok"], "effort_echo": c["effort_echo"],
           "prompt_sha256": hashlib.sha256(job["text"].encode()).hexdigest(), "cache_id": k}
    with open(REC, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def recorded():
    return {(r["cache_id"], r["cell"], r["arm"], r["kind"], r["position"], r["replicate"]) for r in map(json.loads, REC.read_text().splitlines())} if REC.exists() else set()


def post(client, url, **kw):
    """POST at the network boundary: exception text (which can echo the Authorization header) is never shown."""
    try:
        return client.post(url, **kw)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"network error ({type(e).__name__})") from None


def run_batch(client, chunk, name, git, stop_usd):
    """Submit (or resume) one persisted batch, wait, record every job from the cache. chunk: jobs of this batch."""
    keyed = [(j, key_of(j["text"], j["replicate"])) for j in chunk]
    uniq = {}
    for j, k in keyed:
        uniq.setdefault(k, j)
    batches = json.loads(BATCHES.read_text()) if BATCHES.exists() else {}
    ent = batches.get(name)
    if ent is not None and "id" not in ent:
        raise SystemExit(f"{name}: a submission may exist without a recorded id; list batches and reconcile before resuming")
    if ent is None:
        todo = {k: jj for k, jj in uniq.items() if not (CACHE / f"{k}.json").exists()}
        if todo:
            led = ledger()
            reserve = sum(worst(jj["text"]) for jj in todo.values())
            if STOPF.exists() or led["spent"] + sum(led["pending"].values()) + reserve > stop_usd:
                log(f"{name}: not submitted (stop file {STOPF.exists()}, spent ${led['spent']:.2f} + pending + worst case ${reserve:.2f} vs stop ${stop_usd:.2f})")
                raise SystemExit("stopped before submission")
            ent = batches[name] = {"keys": sorted(todo), "requests": len(todo)}
            P.write_atomic(BATCHES, json.dumps(batches))  # intent first: a crash after this won't resubmit blindly
            led["pending"][name] = reserve
            P.write_atomic(LEDGER, json.dumps(led))
            (CACHE / "batch_inputs").mkdir(exist_ok=True)
            f_in = CACHE / "batch_inputs" / f"{name}.jsonl"
            f_in.write_text("".join(json.dumps({"custom_id": k, "method": "POST", "url": "/v1/responses", "body": body(todo[k]["text"])}) + "\n" for k in ent["keys"]))
            r = post(client, f"{API}/files", files={"file": (f"{name}.jsonl", f_in.read_bytes(), "application/jsonl")}, data={"purpose": "batch"})
            if r.status_code != 200:
                raise SystemExit(f"{name}: file upload failed, http {r.status_code}")
            r = post(client, f"{API}/batches", json={"input_file_id": r.json()["id"], "endpoint": "/v1/responses", "completion_window": "24h"})
            if r.status_code != 200:
                raise SystemExit(f"{name}: batch create failed, http {r.status_code}")
            ent["id"] = r.json()["id"]
            P.write_atomic(BATCHES, json.dumps(batches))
            log(f"{name}: submitted {len(todo)} requests; worst case ${reserve:.2f}; spent so far ${led['spent']:.2f}")
    if ent is not None:
        if not set(ent["keys"]) <= set(uniq):
            raise SystemExit(f"{name}: the persisted batch names requests outside this chunk; reconcile before resuming")
        todo = {k: uniq[k] for k in ent["keys"]}
        if not all((CACHE / f"{k}.json").exists() for k in todo):
            bid = ent["id"]
            while True:
                b = get_json(client, f"{API}/batches/{bid}").json()
                st, rc = b.get("status"), b.get("request_counts") or {}
                log(f"{name}: status {st if st in ('validating', 'in_progress', 'finalizing', 'completed', 'failed', 'expired', 'cancelling', 'cancelled') else 'other'} "
                    f"total {rc.get('total')} completed {rc.get('completed')} failed {rc.get('failed')}")
                if st in ("completed", "failed", "expired", "cancelled"):
                    break
                time.sleep(30)
            lines = {}
            for fid in (b.get("output_file_id"), b.get("error_file_id")):
                if fid:
                    for l in get_json(client, f"{API}/files/{fid}/content").text.splitlines():
                        d = json.loads(l)
                        cid = d.get("custom_id")
                        if cid not in todo or cid in lines:
                            raise SystemExit(f"{name}: an output line names a request not in this batch, or twice")
                        lines[cid] = d
            if (rc.get("completed") or 0) > 0 and not b.get("output_file_id"):
                raise SystemExit(f"{name}: completed requests but no output file")
            led = ledger()
            for k in sorted(todo):
                line = lines.get(k)
                if line is None:  # no line at all, even from an expired batch: billing unknown, so worst case and stop
                    status, p, usage, cost, model, effort = "api_error:missing_output", None, {"missing": True}, 0.0, "", None
                else:
                    status, p, usage, cost, model, effort = classify(line)
                if usage.get("missing"):
                    cost = worst(todo[k]["text"])  # unknown billing: charged at the worst case, and further submissions stop
                    P.write_atomic(STOPF, json.dumps({"reason": "usage_or_output_missing", "chunk": name}))
                if k not in led["counted"]:  # idempotent: a replay after a crash never counts a request twice
                    led["spent"] += cost
                    led["calls"] += 1 if (usage.get("input") or cost) else 0
                    led["counted"].append(k)
                P.write_atomic(LEDGER, json.dumps(led))  # the ledger first
                P.write_atomic(CACHE / f"{k}.json", json.dumps({"status": status, "p": p, "usage": usage, "cost": cost, "model_echo_ok": model.startswith(MODEL),
                                                             "effort_echo": effort, "raw": line, "batch": bid}))
        led = ledger()
        if led["pending"].pop(name, None) is not None:
            P.write_atomic(LEDGER, json.dumps(led))
    # every job gets its record, from the cache (so a restart reconstructs anything a crash left unrecorded)
    have, out = recorded(), []
    for j, k in keyed:
        c = json.loads((CACHE / f"{k}.json").read_text())
        if c["status"] == "refusal":
            P.write_atomic(STOPF, json.dumps({"reason": "refusal", "chunk": name}))
        if (k, j["cell"], j["arm"], j["kind"], j["position"], j["replicate"]) not in have:
            out.append(record(j, k, c, git, name))
        else:
            out.append({"status": c["status"], "cost_usd": c["cost"], "model_echo_ok": c.get("model_echo_ok"), "effort_echo": c.get("effort_echo"),
                        "prompt_sha256": hashlib.sha256(j["text"].encode()).hexdigest()})
    n = {st_: sum(1 for r in out if r["status"] == st_) for st_ in sorted({r["status"] for r in out})}
    log(f"{name}: {len(out)} jobs: {n}; billed so far ${ledger()['spent']:.2f}")
    return out


def drive(client, jobs, prefix, git):
    """Run jobs in persisted batches: resume any batch of this prefix first, then submit new ones sized so their worst case
    fits under the stop; the expected total (measured mean per call) must also fit. Stops on a stop file."""
    batches = json.loads(BATCHES.read_text()) if BATCHES.exists() else {}
    by_key = {}
    for j in jobs:
        by_key.setdefault(key_of(j["text"], j["replicate"]), []).append(j)
    for name in sorted(n for n in batches if n.startswith(prefix)):
        run_batch(client, [j for k in batches[name]["keys"] for j in by_key[k]], name, git, STOP_USD)
        if STOPF.exists():
            return False
    while True:
        batches = json.loads(BATCHES.read_text()) if BATCHES.exists() else {}
        in_batch = {k for e in batches.values() for k in e.get("keys", [])}
        left = [k for k in by_key if k not in in_batch and not (CACHE / f"{k}.json").exists()]
        if not left:
            return True
        led = ledger()
        per = max(BASIS, led["spent"] / led["calls"] if led["calls"] else 0.0)
        expected = led["spent"] + sum(led["pending"].values()) + len(left) * per
        if expected >= STOP_USD:
            log(f"{prefix}: expected total ${expected:.2f} reaches the stop ${STOP_USD:.2f}; not submitting")
            return False
        room, take = STOP_USD - led["spent"] - sum(led["pending"].values()), []
        for k in left[:CHUNK]:
            if sum(worst(by_key[x][0]["text"]) for x in take + [k]) > room:
                break
            take.append(k)
        if not take:
            log(f"{prefix}: not even one request fits under the stop at its worst case; stopping")
            return False
        n = 1 + max([int(x[len(prefix):]) for x in batches if x.startswith(prefix) and x[len(prefix):].isdigit()] or [-1])
        run_batch(client, [j for k in take for j in by_key[k]], f"{prefix}{n}", git, STOP_USD)
        if STOPF.exists():
            return False


def worst(text):
    return BATCH_FACTOR * ((len(text.encode()) + len(P.PROMPT.encode()) + 2000) * RATES[0] + P.MAX_OUT * RATES[1]) / 1e6


def code_sha():
    return hashlib.sha256((HERE / "run_judge_batch.py").read_bytes() + (HERE / "run_judge_pilot.py").read_bytes()).hexdigest()


def main(mode):
    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    top = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()).resolve()
    if CACHE == top or top in CACHE.parents:
        raise SystemExit("the cache must lie outside the repo")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    (HERE / "data").mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    key = None
    for line in P.ENV.read_text().splitlines():
        if line.split("=", 1)[0].strip() == "OPENAI_API_KEY":
            key = line.split("=", 1)[1].strip().strip('"').strip("'").strip()
    import re
    if not key or not re.fullmatch(r"[A-Za-z0-9_\-]+", key):
        raise SystemExit("OPENAI_API_KEY missing or malformed (not shown)")
    if mode == "--select":
        runs, pack = load_runs(TEST)
        sel = select(runs, pack)
        if SEL.exists():
            raise SystemExit("the selection already exists; it is never rewritten")
        SEL.write_text(json.dumps(sel | {"git": git, "seeds": ["judge-run-200 (task order and trim)", "judge-run-plant-sim", "judge-run-plant-real", "judge-run-retest"]}, indent=1))
        print(f"selected {len(sel['cells'])} cells in {sel['main_tasks']} main tasks; plants {len(sel['plant_sim'])}+{len(sel['plant_real'])}; retest {len(sel['retest'])}")
        return
    with httpx.Client(timeout=300, headers={"Authorization": f"Bearer {key}"}) as client:
        if mode == "--check":
            runs, pack = P.load_runs()
            jobs, _ = P.jobs_for(runs, pack)
            recs = run_batch(client, jobs[:6], "check", git, CHECK_STOP)
            std = {r["prompt_sha256"]: r["cost_usd"] for r in map(json.loads, (HERE / "data/judge_pilot_calls.jsonl").read_text().splitlines())
                   if r["judge"] == "sol" and r["kind"] == "main" and str(r["replicate"]) == "0" and not r["cached"]}
            per_call = sum(r["cost_usd"] for r in recs) / len(recs) if recs else float("nan")
            same = [(r["cost_usd"], std[r["prompt_sha256"]]) for r in recs if r["prompt_sha256"] in std]
            res = {"requests": len(recs), "statuses": {s: sum(1 for r in recs if r["status"] == s) for s in {r["status"] for r in recs}},
                   "model_echo_ok": all(r["model_echo_ok"] for r in recs), "effort_echo_high": all(r["effort_echo"] == "high" for r in recs),
                   "parsed": sum(1 for r in recs if r["status"] == "ok"), "refusals": sum(1 for r in recs if r["status"] == "refusal"),
                   "per_call_usd": per_call, "basis_usd": BASIS, "per_call_over_basis": per_call / BASIS,
                   "same_transcripts_batch_over_standard": sum(a for a, _ in same) / sum(b for _, b in same) if same else None, "git": git,
                   "code_sha256": code_sha(), "usage_missing_stop": STOPF.exists()}
            res["pass"] = bool(len(recs) == 6 and res["parsed"] == 6 and res["refusals"] == 0 and res["model_echo_ok"] and res["effort_echo_high"]
                               and per_call <= 1.3 * BASIS and not STOPF.exists())
            stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
            with open(HERE / "results" / f"judge_batch_check_{stamp}.json", "x") as fh:
                fh.write(json.dumps(res, indent=1))
            print(json.dumps(res, indent=1))
            return
        if mode == "--run":
            checks = sorted((HERE / "results").glob("judge_batch_check_*.json"))
            chk = json.loads(checks[-1].read_text()) if checks else {}
            if not chk.get("pass") or chk.get("code_sha256") != code_sha():
                raise SystemExit("no passing batch check for this code")
            if not SEL.exists() or subprocess.run(["git", "status", "--porcelain", str(SEL)], capture_output=True, text=True, check=True).stdout:
                raise SystemExit("commit the selection first")
            sel = json.loads(SEL.read_text())
            runs, pack = load_runs(TEST)
            again = select(runs, pack)
            if any(again[k] != sel[k] for k in ("cells", "plant_sim", "plant_real", "retest", "run_keys")):
                raise SystemExit("the committed selection does not reproduce")
            jobs = jobs_from(sel, runs, pack)
            del runs
            log(f"run: {len(jobs)} requests in chunks of {CHUNK}")
            if not drive(client, jobs, "run", git):
                log("stopped: refusal, missing usage, or the spend stop; no further submissions")
                return
            plan_f = CACHE / "redraw_plan.json"
            if not plan_f.exists():  # the redraw set is fixed once, so a restart re-forms the same batches
                final = {}
                for r in map(json.loads, REC.read_text().splitlines()):
                    final[(r["cell"], r["arm"], r["kind"], r["position"], r["replicate"].split("-")[0])] = r["status"]
                retry = ("parse_failure", "api_error:batch_failed", "api_error:incomplete", "api_error:batch_expired", "api_error:missing_output")
                P.write_atomic(plan_f, json.dumps([[j["cell"], j["arm"], j["kind"], j["position"], j["replicate"]] for j in jobs
                                                   if final.get((j["cell"], j["arm"], j["kind"], j["position"], j["replicate"])) in retry]))
            want = {tuple(x) for x in json.loads(plan_f.read_text())}
            redo = [j | {"replicate": j["replicate"] + "-redraw"} for j in jobs if (j["cell"], j["arm"], j["kind"], j["position"], j["replicate"]) in want]
            if redo and not drive(client, redo, "redraw", git):
                log("stopped during redraws")
                return
            log(f"run done; billed ${ledger()['spent']:.2f}")
            return
    raise SystemExit("mode: --check | --select | --run")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
