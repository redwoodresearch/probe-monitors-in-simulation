"""Which simulator-provider rate limit was hit (d9 ~16:05; Aswin may ask the provider for a higher limit). ON a box,
read-only, sims venv, stdout captured on the laptop:
  PYTHONDONTWRITEBYTECODE=1 ~/sims/simulated-control-evals-sept/.venv/bin/python - <run dir> <run key>... < rl_msg_census.py
For each run (exact run key, e.g. sim_honest@september/t462_file_indexing), the final .eval sample's traceback: the
UNINDENTED exception lines whose class is RateLimitError (any module prefix). From each such line only VALIDATED FIELDS are
emitted: the limit kind (an enum from the provider's "on <kind> (<ABBR>)"), the model name, the limit / used / requested
integers, the HTTP status, and the error 'type' and 'code' fields. Org or project ids are NEVER emitted: only whether the
line contains one (yes/no). No other message text. A line that does not parse is counted as unparsed (booleans only).
STOP (d9): a value-shaped credential (sk-..., AKIA/ASIA..., bearer <token>, a PEM private key) in the error message or
traceback prints only {"STOP": ..., "run": <run key>} and exits 3. Other failures: {"failed": <class>, "detail": <fixed>}, exit 2.
"""

import json
import os
import re
import sys

sys.dont_write_bytecode = True

RUN_KEY = re.compile(r"(?:sim_honest|sim_attack)@[a-z0-9_]+/(?:t|nm)[0-9]+_[a-z0-9_]+(?:_r[0-9]+)?")
RUN_DIR = re.compile(r"/home/aswin/phase2/(?:nm/)?runs/[0-9]{4}-[0-9]{2}-[0-9]{2}_[a-z0-9_]+")
RL_LINE = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*\.)*RateLimitError: (.*)")
MSG = re.compile(r"Rate limit reached for (?P<model>[A-Za-z0-9_./:-]{1,80}) in (?:organization|project) \S+ on "
                 r"(?P<kind>[a-z ]{3,40}) \((?P<abbr>[A-Z]{2,4})\): Limit (?P<limit>[0-9](?:[0-9,]*[0-9])?)"
                 r"(?:, Used (?P<used>[0-9](?:[0-9,]*[0-9])?))?(?:, Requested (?P<req>[0-9](?:[0-9,]*[0-9])?))?")
TYPE = re.compile(r"""['"]type['"]: ['"]([a-z_]{1,40})['"]""")
CODE = re.compile(r"""['"]code['"]: ['"]([a-z_]{1,40})['"]""")
HTTP = re.compile(r"Error code: ([0-9]{3})\b")
ORG = re.compile(r"\borg-[A-Za-z0-9]{6,}")
PROJ = re.compile(r"\bproj_[A-Za-z0-9]{6,}")
CRED = re.compile(r"(?i)(sk-[a-z0-9_-]{12,}|AKIA[0-9A-Z]{12,}|ASIA[0-9A-Z]{12,}|bearer\s+\S{8,}|basic\s+[A-Za-z0-9+/=]{12,}|"
                  r"-----BEGIN [A-Z ]*PRIVATE KEY-----|"
                  r"(?:api[_\s-]?key|x-api-key|authorization|access[_\s-]?token|secret)['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9._~+/=-]{12,})")
# explicit allowlists (Codex r36): anything else is suppressed, never emitted
MODELS = {"gpt-6-astra", "gpt-5.6-terra"}
TYPES = {"tokens", "requests", "tokens_usage_based", "requests_usage_based"}
CODES = {"rate_limit_exceeded", "insufficient_quota"}
KINDS = {("tokens per min", "TPM"): "tokens_per_minute", ("requests per min", "RPM"): "requests_per_minute",
         ("requests per day", "RPD"): "requests_per_day", ("tokens per day", "TPD"): "tokens_per_day"}


class Stop(Exception):
    pass


class Fail(Exception):
    pass


def num(x):
    return int(x.replace(",", "")) if x else None


def parse(line: str) -> dict:
    m = MSG.search(line)
    t, c, h = TYPE.search(line), CODE.search(line), HTTP.search(line)
    allow = lambda g, ok: (g.group(1) if g.group(1) in ok else "<other>") if g else None
    base = {"contains_org_id": bool(ORG.search(line)), "contains_project_id": bool(PROJ.search(line)),
            "http_status": int(h.group(1)) if h else None, "type_field": allow(t, TYPES), "code_field": allow(c, CODES)}
    if not m:
        return {"parsed": False, **base}
    return {"parsed": True, "limit_kind": KINDS.get((m.group("kind").strip(), m.group("abbr")), "other"),
            "model": m.group("model") if m.group("model") in MODELS else "<other>", "limit": num(m.group("limit")),
            "used": num(m.group("used")), "requested": num(m.group("req")), **base}


def one(run_dir: str, key: str) -> dict:
    from inspect_ai.log import read_eval_log

    launch = json.load(open(os.path.join(run_dir, "launch.json")))
    runs = [r for r in launch["runs"] if r.get("key") == key]
    if len(runs) != 1:
        raise Fail("run key not found exactly once")
    att = (runs[0].get("attempts") or [None])[-1]
    if not att or not att.get("log_dir"):
        raise Fail("no last attempt with a log dir")
    if not os.path.realpath(att["log_dir"]).startswith(os.path.realpath(run_dir) + "/"):
        raise Fail("log dir outside the run dir")
    evals = sorted((os.path.join(dp, f) for dp, _, fs in os.walk(att["log_dir"]) for f in fs if f.endswith(".eval")),
                   key=os.path.getmtime)
    if not evals:
        raise Fail("no .eval for the last attempt")
    log = read_eval_log(evals[-1])
    if len(log.samples or []) != 1 or log.samples[0].error is None:
        raise Fail("not exactly one sample with an error")
    msg, tb = log.samples[0].error.message or "", log.samples[0].error.traceback or ""
    if CRED.search(msg) or CRED.search(tb):
        raise Stop(key)
    lines = [m.group(1) for m in (RL_LINE.fullmatch(l) for l in tb.splitlines() if not l.startswith(" ")) if m]
    parsed = [parse(l) for l in lines]
    distinct = []
    for p in parsed:
        if p not in distinct:
            distinct.append(p)
    return {"run": key, "n_rate_limit_lines": len(lines), "distinct_parsed": distinct}


def main():
    out_fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)
    try:
        if len(sys.argv) < 3 or not RUN_DIR.fullmatch(sys.argv[1]) or not all(RUN_KEY.fullmatch(k) for k in sys.argv[2:]):
            raise Fail("usage: <run dir> <run key>... (validated)")
        res = {"run_dir": sys.argv[1], "runs": [one(sys.argv[1], k) for k in sys.argv[2:]]}
        os.write(out_fd, (json.dumps(res, indent=1) + "\n").encode())
        os._exit(0)
    except Stop as e:
        os.write(out_fd, (json.dumps({"STOP": "credential-like string in an error record", "run": e.args[0]}) + "\n").encode())
        os._exit(3)
    except BaseException as e:  # noqa: BLE001
        os.write(out_fd, (json.dumps({"failed": type(e).__name__, "detail": e.args[0] if isinstance(e, Fail) and e.args else None}) + "\n").encode())
        os._exit(2)


if __name__ == "__main__":
    main()
