"""Server keepalive (0d GO 04:19 via d9): every 10 min, per server of the named sets, POST <base>/completions with
{"model": <served name>, "prompt": "ok", "max_tokens": 1}, timeout 20 s. A generation increments vLLM's
request_success_total, which run_vllm_tunnel.py:198-203 reads as busy (resets last_busy); request_log_proxy.py logs only
POST /v1/chat/completions (CHAT, line 38) and passes every other path through untouched (passthrough, lines 154/163), so
these never enter the request logs. Logs each response code; an ALARM line on any non-200. A set stops when its stop file
exists (its box's queue ended) or at 07:30 PT. The key comes from the environment (rr-env), never printed.
  rr-env run --env-file env.tpl -- python3 keepalive.py <label>=<results json> ...
"""
import json, os, sys, time, urllib.request, urllib.error, datetime as dt
from zoneinfo import ZoneInfo
TZ = ZoneInfo("America/Los_Angeles")
END = dt.datetime(2026, 10, 15, 16, 30, tzinfo=TZ)  # keepalive5 (d9 10:2x): 16:30 PT
HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ["VLLM_API_KEY"]
MODEL = sys.argv[1]
SETS = dict(a.split("=", 1) for a in sys.argv[2:])
LOG = os.path.join(HERE, "keepalive.log")
def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{dt.datetime.now(TZ):%F %T} PT {msg}\n")
urls = {label: [s["base_url"] for s in json.load(open(path))["servers"]] for label, path in SETS.items()}
assert all(1 <= len(v) <= 6 and all(u.startswith("https://") for u in v) for v in urls.values()), "1-6 https URLs per set"
log(f"armed: sets {list(urls)}; model {MODEL}; every 600 s; ends 16:30 PT")
while dt.datetime.now(TZ) < END:
    for label, us in urls.items():
        if os.path.exists(os.path.join(HERE, f"stop_{label}")):
            continue
        for i, u in enumerate(us):
            body = json.dumps({"model": MODEL, "prompt": "ok", "max_tokens": 1}).encode()
            req = urllib.request.Request(u.rstrip("/") + "/completions", data=body, method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    code = r.status
            except urllib.error.HTTPError as e:
                code = e.code
            except Exception as e:  # noqa: BLE001 - logged as an alarm, never raised
                code = type(e).__name__
            log(f"{label} s{i} {code} {time.time() - t0:.1f}s" + ("" if code == 200 else "  ALARM"))
    time.sleep(600)
log("deadline 16:30")
