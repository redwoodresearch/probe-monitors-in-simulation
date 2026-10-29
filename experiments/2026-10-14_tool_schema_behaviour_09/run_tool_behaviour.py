"""RULE.md here: does the shipped simulator's relaxed tool schema change the policy's calls? Modal CPU over the raw .eval logs;
counts and rates only. Argument VALUES are never read; argument KEYS stay inside Modal and only their counts leave.

  BEH_IN=<dir with t196.json> PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 \\
      modal run --detach experiments/2026-10-14_tool_schema_behaviour_09/run_tool_behaviour.py
"""

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
IN = Path(os.environ.get("BEH_IN", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-convaudit_toolbehaviour_kill-2026-11-14"
EV_ROOT = "/ev/evals_v1"
EPOCH_MAP = "/ev/q4b_epoch_map_v1.json"
HONEST_C = "2026-10-11_112714_honest_c/"
MAPPED = ("run1", "run2", "sim", "honest_a", "honest_b")
REAL = ("run1", "run2", "honest_a", "honest_b", "honest_c")
PIN = "55d9a8ec68f46854"
SAMPLE_RE = re.compile(r"samples/.+_epoch_(\d+)\.json")
REDRAW_RE = re.compile(r"_redraw\d+$")
PATH_RE = re.compile(r"[A-Za-z0-9_@.:/-]+\.eval")

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "zstandard==0.23.0", "numpy==2.3.3"
)
if IN.exists():
    image = image.add_local_file(IN / "t196.json", "/root/in/t196.json")
ev = modal.Volume.from_name(
    "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", version=2
).read_only()


def task_of(rel: str) -> str:
    return REDRAW_RE.sub("", Path(rel).parts[2])


def world_of_path(rel: str):
    """The world a file's folder implies (None for folders not read here)."""
    arm = Path(rel).parts[1]
    if rel.startswith(HONEST_C):
        return "honest_c"
    if arm == "attack":
        return "run2" if rel.startswith("2026-10-10_001314_second_attack/") else "run1"
    return {
        "sim_attack@september": "sim",
        "sim_honest@september": "sim_honest",
        "honest_a": "honest_a",
        "honest_b": "honest_b",
    }.get(arm)


def zstd_zip():
    """Let Python 3.12's zipfile read zstd members (zip method 93), as 2e's evalzip.py does."""
    import zipfile

    import zstandard

    if hasattr(zipfile, "ZIP_ZSTANDARD") or getattr(zipfile, "_zstd_patched", False):
        return zipfile
    check, get = zipfile._check_compression, zipfile._get_decompressor
    zipfile._check_compression = lambda ct: None if ct == 93 else check(ct)
    zipfile._get_decompressor = lambda ct: (
        zstandard.ZstdDecompressor().decompressobj() if ct == 93 else get(ct)
    )
    zipfile._zstd_patched = True
    return zipfile


def resolve(x, att: dict):
    """Resolve Inspect attachment references (attachment://<hash>) anywhere in a JSON value."""
    if isinstance(x, str) and x.startswith("attachment://"):
        return att.get(x[len("attachment://") :], x)
    if isinstance(x, list):
        return [resolve(v, att) for v in x]
    if isinstance(x, dict):
        return {k: resolve(v, att) for k, v in x.items()}
    return x


def guard(f, *a):
    try:
        return f(*a)
    except Exception as e:  # noqa: BLE001 -- the error type only, never text
        return {"fatal": type(e).__name__}


def _list() -> list:
    rels = sorted(
        str(p.relative_to(EV_ROOT))
        for p in Path(EV_ROOT).glob("*/*/*/*.eval")
        if world_of_path(str(p.relative_to(EV_ROOT))) is not None
    )
    if not rels or not all(PATH_RE.fullmatch(r) for r in rels):
        raise ValueError("unexpected eval path")
    return rels


def _scan(rels: list) -> dict:
    zipfile = zstd_zip()
    out, errors = [], {}
    for rel in rels:
        try:
            with zipfile.ZipFile(Path(EV_ROOT) / rel) as z:
                names = sorted(
                    (n for n in z.namelist() if SAMPLE_RE.fullmatch(n)),
                    key=lambda n: int(SAMPLE_RE.fullmatch(n).group(1)),
                )
                if not names:
                    raise ValueError("no samples")
                for n in names:
                    s = json.loads(z.read(n))
                    att = s.get("attachments") or {}
                    model_events = [
                        e
                        for e in (s.get("events") or [])
                        if isinstance(e, dict) and e.get("event") == "model"
                    ]
                    tools = (
                        resolve(model_events[0].get("tools") or [], att)
                        if model_events
                        else None
                    )
                    schema = None
                    schema_state = (
                        "no_model_event"
                        if not model_events
                        else (
                            "no_tools_list"
                            if not isinstance(tools, list)
                            else ("empty" if not tools else "ok")
                        )
                    )
                    if schema_state == "ok":
                        schema = {}
                        for t in tools:
                            p = t.get("parameters") or {}
                            schema[t.get("name")] = {
                                "required": sorted(p.get("required") or []),
                                "additional": p.get("additionalProperties"),
                                "props": sorted((p.get("properties") or {}).keys()),
                            }
                    calls, terr = [], {}
                    for m in s.get("messages") or []:
                        if m.get("role") == "assistant":
                            for c in m.get("tool_calls") or []:
                                args = c.get("arguments")
                                calls.append(
                                    [
                                        c.get("function"),
                                        sorted(args.keys())
                                        if isinstance(args, dict)
                                        else None,
                                        c.get("parse_error") is not None,
                                    ]
                                )
                        elif m.get("role") == "tool" and m.get("error") is not None:
                            et = (
                                (m["error"] or {}).get("type")
                                if isinstance(m["error"], dict)
                                else "unstructured"
                            )
                            terr[str(et)] = terr.get(str(et), 0) + 1
                    out.append(
                        {
                            "rel": rel,
                            "task": task_of(rel),
                            "world_path": world_of_path(rel),
                            "n_samples": len(names),
                            "schema": schema,
                            "schema_state": schema_state,
                            "calls": calls,
                            "tool_errors": terr,
                        }
                    )
        except Exception as e:  # noqa: BLE001 -- the error type only, never text; counted and gated
            errors[rel] = type(e).__name__
    return {"rows": out, "errors": errors}


def _summarize(rows: list, errors: dict) -> dict:
    from collections import Counter

    import numpy as np

    cells = json.loads(Path("/root/in/t196.json").read_text())
    import hashlib

    if hashlib.sha256(Path("/root/in/t196.json").read_bytes()).hexdigest()[:16] != PIN:
        raise ValueError("t196 pin mismatch")
    env = {c["key"]: c["env"] for c in cells["cells"]}
    emap = json.loads(Path(EPOCH_MAP).read_text())["map"]
    mapped = {k: v for k, v in emap.items() if v[0] in MAPPED}
    by_rel = {}
    for i, r in enumerate(rows):
        by_rel.setdefault(r["rel"], []).append(i)
    # gate (as approach_provenance): every mapped run found once, one sample, #e0, path world = map world, task = folder task
    gate = {w: {"mapped": 0, "ok": 0} for w in MAPPED}
    counted = {}
    for k, (w, t) in mapped.items():
        rel, _, suf = k.rpartition("#")
        gate[w]["mapped"] += 1
        ok = (
            suf == "e0"
            and rel not in errors
            and len(by_rel.get(rel, [])) == 1
            and world_of_path(rel) == w
            and task_of(rel) == t
        )
        if ok:
            gate[w]["ok"] += 1
            counted[(w, t)] = rows[by_rel[rel][0]]
    dup = sum(n > 1 for n in Counter(tuple(v) for v in mapped.values()).values())
    gate_pass = dup == 0 and all(g["ok"] == g["mapped"] > 0 for g in gate.values())
    out = {
        "gate": {
            "pass": gate_pass,
            "per_world": gate,
            "duplicate_pairs": dup,
            "errors": dict(Counter(errors.values())),
        }
    }
    if not gate_pass:
        return out
    # unmapped worlds (sim_honest, honest_c): every observation kept; each is scored with its OWN schema, then pooled per task
    obs = {}  # (world, task) -> [row]
    for (w, t), r in counted.items():
        obs[(w, t)] = [r]
    files_per_world = {}
    for r in rows:
        w = r["world_path"]
        if w in ("sim_honest", "honest_c"):
            obs.setdefault((w, r["task"]), []).append(r)
            files_per_world.setdefault(w, set()).add(r["rel"])
    out["unmapped_worlds"] = {
        "files": {w: len(v) for w, v in files_per_world.items()},
        "observations": {
            w: sum(len(v) for (ww, _), v in obs.items() if ww == w)
            for w in files_per_world
        },
        "tasks": {w: sum(1 for ww, _ in obs if ww == w) for w in files_per_world},
        "scan_errors_by_world": dict(Counter(world_of_path(rel) for rel in errors)),
    }

    def sig_key(r):
        s = r["schema"]
        if not s:
            return f"state:{r['schema_state']}"
        return json.dumps(
            {n: [v["required"], v["additional"]] for n, v in sorted(s.items())},
            sort_keys=True,
        )

    sig = {}
    for (w, _), rs in obs.items():  # every observation, before pooling
        for r in rs:
            sig.setdefault(w, Counter())[sig_key(r)] += 1
    out["schema_signatures"] = {
        w: {
            "observations": sum(c.values()),
            "all_required_empty": sum(
                n
                for k, n in c.items()
                if not k.startswith("state:")
                and all(not v[0] for v in json.loads(k).values())
            ),
            "all_additional_true": sum(
                n
                for k, n in c.items()
                if not k.startswith("state:")
                and all(v[1] is True for v in json.loads(k).values())
            ),
            "no_schema": {k: n for k, n in c.items() if k.startswith("state:")},
            "signatures": dict(c.most_common()),
        }
        for w, c in sig.items()
    }
    # the real required set per function (mapped real worlds and honest_c observations)
    req = {}
    for (w, _), rs in obs.items():
        if w in REAL:
            for r in rs:
                for fn, v in (r["schema"] or {}).items():
                    req.setdefault(fn, Counter())[tuple(v["required"])] += 1
    ambiguous = {fn: len(c) for fn, c in req.items() if len(c) > 1}
    single = {fn: next(iter(c)) for fn, c in req.items() if len(c) == 1}
    run1_req = {
        t: {fn: tuple(v["required"]) for fn, v in r["schema"].items()}
        for (w, t), r in counted.items()
        if w == "run1" and r["schema"]
    }
    out["real_required"] = {"functions": len(req), "ambiguous_functions": ambiguous}
    metrics = (
        "missing_required",
        "extra_keys",
        "parse_error",
        "tool_error",
        "tool_error_parsing",
    )
    denom = {
        "missing_required": "n_ref",
        "extra_keys": "n_own",
        "parse_error": "n",
        "tool_error": "n",
        "tool_error_parsing": "n",
    }

    def counts(t, r):
        own = r["schema"] or {}
        c = dict.fromkeys(("n", "n_ref", "n_own", *metrics), 0)
        for fn, keys, pe in r["calls"]:
            ks = set(keys or [])
            need = run1_req.get(t, {}).get(fn) if fn in ambiguous else single.get(fn)
            c["n"] += 1
            c["parse_error"] += pe
            if need is not None:
                c["n_ref"] += 1
                c["missing_required"] += bool(set(need) - ks)
            if fn in own:
                c["n_own"] += 1
                c["extra_keys"] += bool(ks - set(own[fn]["props"]))
        c["tool_error"] = sum(r["tool_errors"].values())
        c["tool_error_parsing"] = r["tool_errors"].get("parsing", 0)
        return c

    per = {}
    for (w, t), rs in obs.items():
        tot = Counter()
        for r in rs:
            tot.update(counts(t, r))
        per[(w, t)] = dict(tot)

    def rate(c, m):
        d = c.get(denom[m], 0)
        return c.get(m, 0) / d if d else None

    worlds = sorted({w for w, _ in per})
    out["per_world"] = {}
    for w in worlds:
        ks = [k for k in per if k[0] == w]
        row = {
            "tasks": len(ks),
            "zero_call_tasks": sum(1 for k in ks if not per[k].get("n")),
        }
        row |= {
            f: sum(per[k].get(f, 0) for k in ks)
            for f in ("n", "n_ref", "n_own", *metrics)
        }
        for m in metrics:
            vals = [x for x in (rate(per[k], m) for k in ks) if x is not None]
            row[f"{m}_mean_task_rate_pct"] = (
                100 * float(np.mean(vals)) if vals else None
            )
            row[f"{m}_tasks_with_denominator"] = len(vals)
        out["per_world"][w] = row
    out["tool_error_types"] = {
        w: dict(
            sum(
                (
                    Counter(r["tool_errors"])
                    for (ww, _), rs in obs.items()
                    if ww == w
                    for r in rs
                ),
                Counter(),
            )
        )
        for w in worlds
    }

    def paired(a_worlds, b_world):
        both = [t for t in env if all((w, t) in per for w in (*a_worlds, b_world))]
        res = {"tasks_with_all_worlds": len(both)}
        for m in metrics:
            tasks = sorted(
                t
                for t in both
                if all(rate(per[(w, t)], m) is not None for w in (*a_worlds, b_world))
            )
            if not tasks:
                res[m] = {"tasks": 0, "diff_pct_points": None, "ci95": None}
                continue
            g = np.array([env[t] for t in tasks])
            rng = np.random.default_rng([20261014, 1])
            Wt = np.zeros((10001, len(tasks)))
            Wt[0] = 1.0
            for e in sorted(set(g.tolist())):
                idx = np.flatnonzero(g == e)
                Wt[1:, idx] = rng.multinomial(
                    len(idx), np.full(len(idx), 1 / len(idx)), size=10000
                )
            r_ = {
                w: np.array([rate(per[(w, t)], m) for t in tasks])
                for w in (*a_worlds, b_world)
            }
            d = r_[b_world] - np.mean([r_[w] for w in a_worlds], axis=0)
            est = (Wt @ d) / Wt.sum(1)
            res[m] = {
                "tasks": len(tasks),
                "diff_pct_points": 100 * float(est[0]),
                "ci95": [100 * float(np.quantile(est[1:], q)) for q in (0.025, 0.975)],
            }
        return res

    out["paired"] = {
        "sim_minus_mean_run1_run2": paired(("run1", "run2"), "sim"),
        "yardstick_run2_minus_run1": paired(("run1",), "run2"),
        "sim_honest_minus_mean_honest_a_b": paired(
            ("honest_a", "honest_b"), "sim_honest"
        ),
    }
    return out


@app.function(
    image=image, cpu=1, memory=2048, timeout=600, retries=0, volumes={"/ev": ev}
)
def list_files():
    return guard(_list)


@app.function(
    image=image, cpu=1, memory=8192, timeout=1800, retries=0, volumes={"/ev": ev}
)
def scan(rels: list) -> dict:
    return guard(_scan, rels)


@app.function(
    image=image, cpu=2, memory=16384, timeout=1800, retries=0, volumes={"/ev": ev}
)
def summarize(rows: list, errors: dict) -> dict:
    return guard(_summarize, rows, errors)


@app.function(
    image=image, cpu=1, memory=8192, timeout=3600, retries=0, volumes={"/ev": ev}
)
def orchestrate() -> dict:
    """Everything runs inside Modal; only the summary (counts and rates) comes back, never argument keys."""
    rels = guard(_list)
    if isinstance(rels, dict):
        return rels
    parts = list(scan.map([rels[i : i + 20] for i in range(0, len(rels), 20)]))
    bad = [x["fatal"] for x in parts if "fatal" in x]
    if bad:
        return {"fatal": bad[0]}
    rows = [r for x in parts for r in x["rows"]]
    errors = {k: v for x in parts for k, v in x["errors"].items()}
    res = summarize.remote(rows, errors)
    return {
        "eval_files": len(rels),
        "samples": len(rows),
        "scan_errors": len(errors),
        "result": res,
    }


def fatal(x, where: str):
    if isinstance(x, dict) and "fatal" in x:
        raise SystemExit(f"{where}: remote error of type {x['fatal']}")
    return x


@app.local_entrypoint()
def main():
    def git(*a):
        return subprocess.run(
            ["git", *a], capture_output=True, text=True, check=True, cwd=REPO
        ).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    if not IN.exists():
        raise SystemExit("BEH_IN not set")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    o = fatal(orchestrate.remote(), "orchestrate")
    fatal(o["result"], "summarize")
    out = {
        "label": "RULE.md (tool-schema behaviour); counts and rates only",
        "git": git("rev-parse", "HEAD"),
        "stamp": stamp,
    } | o
    dest = HERE / "results" / f"tool_behaviour_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(
        json.dumps(
            out, indent=1, default=lambda o: dict(o) if hasattr(o, "items") else str(o)
        )
    )
    print(
        json.dumps(
            out, indent=1, default=lambda o: dict(o) if hasattr(o, "items") else str(o)
        ),
        flush=True,
    )
