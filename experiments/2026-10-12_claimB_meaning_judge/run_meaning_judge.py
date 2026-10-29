"""Meaning judge runs (PLAN.md @ 2d35f421 and amendment 1; Aswin about 14:46 PT, "Sol + Gemini judge"). API spend from the
laptop; keys from ~/probe-distribution-shift-from-sims/.env, read into this process only, never printed or written.

  --judge sol     GPT-5.6 Sol (`gpt-5.6-sol`, high) by the OpenAI Batch API, through the plan B batch machinery loaded
                  unchanged from 3a1a7133 (persisted ledger, worst-case batch sizing, missing usage = worst case + STOP,
                  a refusal = STOP) with its own cache, records, prompt and parser here. Stop $70.
  --judge gemini  Gemini 3.1 Pro preview (`google/gemini-3.1-pro-preview`, high) on OpenRouter, one pinned endpoint
                  (google-vertex/global/flex, no fallbacks). Before each call: spent + pending + this call's worst case
                  (the standard rate, twice the flex one) must fit under $45, or the job stops. Stop $45.
Both fail closed: a missing or unreadable ledger stops the job (they are created by --init); unknown usage counts as the
worst case. Job stop $175 = simulator $60 + Sol $70 + Gemini $45.

Jobs: at the strict sites, (e1, e2) and (e1, sim) in both orders; at the Oct-6 sites, (e1, sim) both orders; all 60
plants (e1 vs the edited copy) both orders, the 9 Codex excluded included as descriptive; 30 seeded retests. A 10-job
parse pilot (practice sites) runs first and must have 9+ parsed and no refusal. Parse failures get one redraw at the end.
Records (numbers and hashes) in data/meaning_<judge>_calls.jsonl; requests and responses in data/<judge>_cache/.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with httpx \\
      python experiments/2026-10-12_claimB_meaning_judge/run_meaning_judge.py --judge sol|gemini [--init]
"""

import hashlib
import json
import math
import random
import re
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).parent
REPO = HERE.parents[1]
DATA = HERE / "data"
PROMPT = ("Below is a shell command (or file view) that an AI agent ran in a Linux environment, and two replies, A and B, each "
          "produced at the same point of the same task. Do A and B report the same facts: the same files and directories, the "
          "same file contents shown, the same success or failure, the same errors, the same counts and values? Ignore "
          "differences that do not change what the command found: incidental timestamps, process or session IDs, random "
          "identifiers, timing figures, and the order of unordered listings. If a time, ID or size is what the command asked "
          "for, it counts. Answer with JSON only: {\"same_meaning\": true} or {\"same_meaning\": false}")
DECLINE = re.compile(r"\b(I can(no|['’])t|I cannot|I['’]m unable|I am unable|I won['’]t|I will not|unable to (help|assist|comply))\b", re.I)
FILTER = re.compile(r"safety|prohibit|block|spii|recitation|refus|content_filter|policy|moderat", re.I)
SIM_STOP, SOL_STOP, GEM_STOP = 60.0, 70.0, 45.0
WORKERS = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
# v5 (option B, only if Aswin picks it): --endpoint pins the Gemini provider endpoint (default the flex one used so far);
# --max-minutes stops starting new Gemini calls after that much wall time
ENDPOINT = sys.argv[sys.argv.index("--endpoint") + 1] if "--endpoint" in sys.argv else "google-vertex/global/flex"
MAX_MIN = float(sys.argv[sys.argv.index("--max-minutes") + 1]) if "--max-minutes" in sys.argv else None


def git_module(commit, path, name):
    src = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, text=True, check=True, cwd=REPO).stdout
    m = types.ModuleType(name)
    m.__file__ = str(REPO / path)
    sys.modules[name] = m
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m


def has_429(obj):
    """v5: an error code 429 anywhere in the body (any depth, any choice)."""
    if isinstance(obj, dict):
        return any((k == "code" and str(v) == "429") or has_429(v) for k, v in obj.items())
    return isinstance(obj, list) and any(has_429(x) for x in obj)


def num_ok(x, kind):
    """v5: a usable accounting value: a finite, non-negative number (an int for token counts), never a bool."""
    return not isinstance(x, bool) and isinstance(x, kind) and math.isfinite(x) and x >= 0


def parse(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip()).strip()
    try:
        obj = json.loads(t)
    except ValueError:
        return None
    v = obj.get("same_meaning") if isinstance(obj, dict) and set(obj) == {"same_meaning"} else None
    return (1.0 if v else 0.0) if isinstance(v, bool) else None


def jobs():
    sites = {s["site"]: s for s in map(json.loads, (DATA / "sites.jsonl").read_text().splitlines())}
    new = {r["pair"]: r["result"]["reader_output"] for r in map(json.loads, (DATA / "sim_new.jsonl").read_text().splitlines()) if r["status"] == "ok"}
    out = []

    def add(site, pair, a, b, kind, cmd):
        for order, (x, y) in (("AB", (a, b)), ("BA", (b, a))):
            user = f"Command ({cmd['function']}):\n{json.dumps(cmd['arguments'])}\n\n=== Reply A ===\n{x}\n\n=== Reply B ===\n{y}"
            out.append({"cell": str(site), "arm": pair, "kind": kind, "position": order, "replicate": "0", "user": user})
    for s in sites.values():
        sim = s["sim"] if s["sim"] is not None else new.get(s["site"])
        if s["set"] == "strict":
            add(s["site"], "e1-e2", s["e1"], s["e2"], "main", s["command"])
        if sim is not None:
            add(s["site"], "e1-sim", s["e1"], sim, "main", s["command"])
    for p in map(json.loads, (DATA / "plants.jsonl").read_text().splitlines()):
        add(p["site"], f"plant:{p['plant']}:{p['kind']}", sites[p["site"]]["e1"], p["edited"], "plant", sites[p["site"]]["command"])
    rng = random.Random("meaning-retest")
    strict = [j for j in out if j["kind"] == "main" and sites[int(j["cell"])]["set"] == "strict"]
    out += [j | {"kind": "retest", "replicate": "1"} for j in rng.sample(strict, 30)]
    pilot = random.Random("meaning-pilot").sample([j for j in strict if sites[int(j["cell"])]["split"] == "practice"], 10)
    pkeys = {(j["cell"], j["arm"], j["position"]) for j in pilot}
    return pilot, [j for j in out if j["kind"] == "retest" or (j["cell"], j["arm"], j["position"]) not in pkeys]


def key_env(name):
    return next(line.split("=", 1)[1].strip().strip('"').strip("'") for line in (Path.home() / "probe-distribution-shift-from-sims/.env").read_text().splitlines()
                if line.split("=", 1)[0].strip() == name)


def check_sim_spend():
    spent = sum(json.loads(line).get("cost_usd", 0.0) for line in (DATA / "sim_new.jsonl").read_text().splitlines())
    assert spent <= SIM_STOP, f"simulator spend {spent:.2f} is over its ${SIM_STOP} share"
    return spent


def run_sol(git):
    P = git_module("3a1a7133", "experiments/2026-10-11_claimB_judge/run_judge_pilot.py", "run_judge_pilot")
    B = git_module("3a1a7133", "experiments/2026-10-11_claimB_judge/run_judge_batch.py", "run_judge_batch")
    cache = DATA / "sol_cache"
    P.MAX_OUT, P.PROMPT = 16000, PROMPT
    B.CACHE, B.STOP_USD, B.CHUNK, B.BASIS = cache, SOL_STOP, 400, 0.06
    B.LEDGER, B.STOPF, B.BATCHES = cache / "spend_batch.json", cache / "STOP_batch", cache / "batches.json"
    B.REC, B.LOG = DATA / "meaning_sol_calls.jsonl", DATA / "meaning_sol_progress.log"
    assert B.LEDGER.exists(), "no Sol ledger: run --init first (a missing ledger never counts as $0)"
    B.ledger = lambda: json.loads(B.LEDGER.read_text())  # v2: a missing or unreadable ledger stops the job, never reads as $0
    _classify = B.classify

    def classify(line):
        st, p, usage, cost, model, effort = _classify(line)
        resp = line.get("response") if isinstance(line.get("response"), dict) else {}
        err = json.dumps([line.get("error"), (resp.get("body") or {}).get("error") if isinstance(resp.get("body"), dict) else None])
        if st.startswith("api_error") and (resp.get("status_code") in (401, 402, 403, 429) or re.search(r"limit|credit|insufficient|quota", err, re.I)):
            usage["missing"] = True  # v3: a key, quota or limit rejection: worst-case charge, STOP file, never redrawn or scored
            return "rejected", None, usage, cost, model, effort
        if st.startswith("api_error") and not (usage.get("input") and usage.get("output")):
            usage["missing"] = True  # v2: an error without token counts is charged the worst case and stops submissions
        if st not in ("ok", "parse_failure"):
            return st, p, usage, cost, model, effort
        b = ((line.get("response") or {}).get("body") or {})
        t = "".join(c.get("text", "") for o in b.get("output") or [] if o.get("type") == "message" for c in o.get("content") or [] if c.get("type") == "output_text")
        p = parse(t)
        return ("ok" if p is not None else ("refusal" if not t.strip() or DECLINE.search(t) else "parse_failure")), p, usage, cost, model, effort
    B.classify = classify
    B.body = lambda text: {"model": B.MODEL, "instructions": PROMPT, "input": text, "reasoning": {"effort": "high"}, "max_output_tokens": P.MAX_OUT}
    pilot, rest = [[j | {"text": j["user"]} for j in js] for js in jobs()]
    redo = []
    try:
        sol_jobs(B, git, pilot, rest, redo)
    finally:  # v3: on any exit, early stops included, identical requests sent under another prefix get their records
        have = B.recorded()
        for j in pilot + rest + redo:
            k = B.key_of(j["text"], j["replicate"])
            if (k, j["cell"], j["arm"], j["kind"], j["position"], j["replicate"]) not in have and (B.CACHE / f"{k}.json").exists():
                B.record(j, k, json.loads((B.CACHE / f"{k}.json").read_text()), git, "copy")


def sol_jobs(B, git, pilot, rest, redo):
    with httpx.Client(timeout=300, headers={"Authorization": f"Bearer {key_env('OPENAI_API_KEY')}"}) as client:
        for name, js in (("pilot", pilot), ("run", rest)):
            if not B.drive(client, js, name, git):
                B.log(f"{name}: stopped (refusal, missing usage or the ${SOL_STOP} stop)")
                return
            if name == "pilot":
                recs = [json.loads(x) for x in B.REC.read_text().splitlines()]
                ok, ref = sum(r["status"] == "ok" for r in recs), sum(r["status"] == "refusal" for r in recs)
                B.log(f"pilot: {ok} of {len(recs)} parsed, {ref} refusals")
                if ok < 9 or ref:
                    return
        final = {(r["cell"], r["arm"], r["kind"], r["position"], r["replicate"]): r["status"] for r in map(json.loads, B.REC.read_text().splitlines())}
        redo += [j | {"replicate": j["replicate"] + "-redraw"} for j in pilot + rest
                 if final.get((j["cell"], j["arm"], j["kind"], j["position"], j["replicate"])) not in ("ok", None, "refusal", "rejected")]
        if redo:
            B.drive(client, redo, "redraw", git)


def run_gemini(git):
    cache, rec_path = DATA / "gem_cache", DATA / "meaning_gemini_calls.jsonl"
    ledger, stopf = cache / "spend.json", cache / "STOP"
    led = json.loads(ledger.read_text())  # missing or unreadable: the job stops here
    state = {"spent": led["spent"] + sum(led["pending"].values()), "pending": {}, "stop": stopf.exists()}
    lock, req_locks = threading.Lock(), {}
    key = key_env("OPENROUTER_API_KEY")
    t_start = time.time()

    def put(path, obj):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj))
        tmp.replace(path)

    def worst(body):
        return ((len(json.dumps(body).encode()) + 2000) * 2.0 + 16000 * 12.0) / 1e6  # standard rate: twice flex

    def one(job, client):
        body = {"model": "google/gemini-3.1-pro-preview", "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": job["user"]}],
                "reasoning": {"effort": "high"}, "max_tokens": 16000, "usage": {"include": True}, "response_format": {"type": "json_object"},
                "provider": {"order": [ENDPOINT], "allow_fallbacks": False, "require_parameters": True}}
        h = hashlib.sha256(json.dumps(body | {"_rep": job["replicate"]}, sort_keys=True).encode()).hexdigest()
        cf = cache / f"{h}.json"
        with lock:
            req_lock = req_locks.setdefault(h, threading.Lock())
        with req_lock:  # v2: identical requests (byte-identical replies in both orders) wait and then read the cache
            c = json.loads(cf.read_text()) if cf.exists() else call(body, h, cf, client)
            if c is None:
                return None
        return record(job, h, c)

    def call(body, h, cf, client):
        with lock:
            if MAX_MIN is not None and time.time() - t_start > 60 * MAX_MIN:  # v5: the elapsed-time limit
                state["stop"], state["reason"] = True, "time limit"
                return None
            res = 2 * worst(body)
            if state["stop"] or state["spent"] + sum(state["pending"].values()) + res > GEM_STOP:
                state["stop"] = True
                return None
            state["pending"][h] = res
            put(ledger, {"spent": state["spent"], "pending": state["pending"]})
        costs, raw, http = [], {"error": "not_sent"}, 0
        for attempt in range(2):  # one retry, for a lost connection or 5xx only (any 429 stops)
            with lock:  # v5: the deadline and the stop flag are rechecked immediately before every submission
                if MAX_MIN is not None and time.time() - t_start > 60 * MAX_MIN:
                    state["stop"], state["reason"] = True, "time limit"
                if state["stop"] and attempt:
                    break
                if state["stop"] and not attempt:
                    state["pending"].pop(h, None)
                    put(ledger, {"spent": state["spent"], "pending": state["pending"]})
                    return None
            try:
                r = client.post("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body)
                http = r.status_code
            except httpx.HTTPError as e:
                http, raw, r = 0, {"error": f"exception:{type(e).__name__}"}, None
            if r is not None:
                try:  # v3: a malformed body keeps its HTTP status, so a 401/402 stays a rejection
                    raw = r.json()
                    raw = raw if isinstance(raw, dict) else {"non_dict_body": True}
                except ValueError:
                    raw = {"non_json_body": True}
            u = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
            known = num_ok(u.get("cost"), (int, float))
            costs.append(float(u["cost"]) if known else (worst(body) if (u or raw.get("choices") or http in (0, 200) or http >= 500) else 0.0))
            if stop_signal(raw, http) or usage_bad(raw, http) or refused(raw, http):  # v5: latch the stop at once, before any write; never retried
                with lock:
                    state["stop"] = True
                break
            if (http == 0 or http in (500, 502, 503, 504)) and attempt == 0:
                time.sleep(20)
                if state["stop"]:  # v2: another worker stopped the judge while this one waited; no retry
                    break
                continue
            break
        c = {"raw": raw, "http": http, "costs": costs}
        put(cf, c)
        with lock:
            state["pending"].pop(h, None)
            state["spent"] += sum(costs)
            put(ledger, {"spent": state["spent"], "pending": state["pending"]})
        return c

    def usage_bad(raw, http):  # v5: a 200 needs usable accounting (a cost, or both token counts)
        u = raw.get("usage")
        if http != 200:
            return False
        if not isinstance(u, dict):
            return True
        return not (num_ok(u.get("cost"), (int, float)) or (num_ok(u.get("prompt_tokens"), int) and num_ok(u.get("completion_tokens"), int)))

    def refused(raw, http):  # v5: the record's refusal test, run on the raw response before any write
        ch = (raw.get("choices") or [{}])[0] if isinstance(raw.get("choices"), list) else {}
        ch = ch if isinstance(ch, dict) else {}
        msg = ch.get("message") if isinstance(ch.get("message"), dict) else {}
        fin, content = ch.get("finish_reason"), msg.get("content") or ""
        return (http == 403 or fin == "content_filter" or bool(msg.get("refusal")) or bool(FILTER.search(str(ch.get("native_finish_reason") or "")))
                or (http == 200 and fin == "stop" and parse(content) is None and (not content.strip() or bool(DECLINE.search(content)))))

    def stop_signal(raw, http):
        ch = (raw.get("choices") or [{}])[0] if isinstance(raw.get("choices"), list) else {}
        errs = json.dumps([raw.get("error"), ch.get("error") if isinstance(ch, dict) else None])
        return http in (401, 402, 403, 429) or has_429(raw) or bool(re.search(r"limit|credit|insufficient", errs, re.I)) or bool(FILTER.search(errs))

    def record(job, h, c):
        raw, http = c["raw"], c["http"]
        ch = (raw.get("choices") or [{}])[0] if isinstance(raw.get("choices"), list) else {}
        ch = ch if isinstance(ch, dict) else {}  # v5: a malformed choices entry cannot crash before the stop
        msg = ch.get("message") if isinstance(ch.get("message"), dict) else {}
        fin, nfin, content = ch.get("finish_reason"), str(ch.get("native_finish_reason") or ""), msg.get("content") or ""
        p = parse(content) if http == 200 and fin == "stop" else None
        if usage_bad(raw, http):
            status = "usage_missing"  # v5: a 200 without usage stops the judge (charged the worst case)
        elif http in (401, 402, 429) or has_429(raw) or re.search(r"limit|credit|insufficient", json.dumps([raw.get("error"), ch.get("error")]), re.I):
            status = "rejected"  # v2: a key or credit rejection is counted, never scored, and stops the judge
        elif http == 403 or fin == "content_filter" or msg.get("refusal") or FILTER.search(nfin) or FILTER.search(json.dumps(raw.get("error") or ch.get("error") or "")) \
                or (http == 200 and fin == "stop" and p is None and (not content.strip() or DECLINE.search(content))):
            status = "refusal"
        else:
            status = "ok" if p is not None else (f"api_error:http{http}:{fin}" if http != 200 or fin != "stop" else "parse_failure")
        u = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        rec = {k: job[k] for k in ("cell", "arm", "kind", "position", "replicate")} | {
            "judge": "gemini", "model": "google/gemini-3.1-pro-preview", "provider_pin": ENDPOINT, "provider_echo": raw.get("provider"),
            "git": git, "request_sha256": h, "status": status, "p": p, "usage": {"input": u.get("prompt_tokens"), "output": u.get("completion_tokens")},
            "cost_usd": sum(c["costs"]), "prompt_sha256": hashlib.sha256(job["user"].encode()).hexdigest()}
        with lock:
            with open(rec_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if status in ("refusal", "rejected", "usage_missing"):  # v5: any 429 is 'rejected'; missing usage stops too
                state["stop"] = True
                put(stopf, {"reason": status, "request_sha256": h})
        return rec

    def wave(js):  # one result per job, None where the stop prevented the call
        with httpx.Client(timeout=httpx.Timeout(900.0)) as client, ThreadPoolExecutor(WORKERS) as pool:  # v4: --workers (2 on the resume after the upstream 429 was b7's call); any 429 stops
            return list(pool.map(lambda j: one(j, client), js))
    pilot, rest = jobs()
    pil = wave(pilot)
    recs = [r for r in pil if r]
    ok, ref = sum(r["status"] == "ok" for r in recs), sum(r["status"] == "refusal" for r in recs)
    print(f"pilot: {ok} of {len(recs)} parsed, {ref} refusals, spent ${state['spent']:.2f}", flush=True)
    if ok < 9 or ref or state["stop"]:
        return
    res = wave(rest)
    recs = [r for r in res if r]
    print(f"run: {len(recs)} records, statuses {dict((s, sum(r['status'] == s for r in recs)) for s in {r['status'] for r in recs})}, spent ${state['spent']:.2f}, stop {state['stop']}", flush=True)
    if state["stop"]:
        return
    redo = [j | {"replicate": j["replicate"] + "-redraw"} for j, r in zip(pilot + rest, pil + res) if r and (r["status"] == "parse_failure" or r["status"].startswith("api_error"))]
    if redo:
        wave(redo)
    print(f"done: spent ${state['spent']:.2f}", flush=True)


def main():
    judge = sys.argv[sys.argv.index("--judge") + 1]
    if "--init" in sys.argv:  # the ledgers exist before any call, so a missing one later means stop, never $0
        for d, f, empty in (("sol_cache", "spend_batch.json", {"spent": 0.0, "calls": 0, "counted": [], "pending": {}}), ("gem_cache", "spend.json", {"spent": 0.0, "pending": {}})):
            (DATA / d).mkdir(parents=True, exist_ok=True)
            assert not any((DATA / d).iterdir()), f"{d} already holds files: a ledger is never recreated at $0 over paid work"
            (DATA / d / f).write_text(json.dumps(empty))
        return
    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True, cwd=REPO).stdout:
        raise SystemExit("commit this script first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    print(f"simulator spend ${check_sim_spend():.2f} (share ${SIM_STOP})", flush=True)
    run_sol(git) if judge == "sol" else run_gemini(git)


if __name__ == "__main__":
    main()
