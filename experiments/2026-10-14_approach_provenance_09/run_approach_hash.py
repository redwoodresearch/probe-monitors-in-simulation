"""RULE.md here: attack-approach and system-prompt provenance by run metadata. Modal CPU, read-only Volumes; sha256, lengths,
booleans, keys and task/side names only. No approach, strategy, prompt or transcript text leaves a container: every remote
function returns only the type name of any exception it hits.

  APX_IN=<dir with reg/, t196.json, runs36.json> PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 \\
      modal run --detach experiments/2026-10-14_approach_provenance_09/run_approach_hash.py
"""

import hashlib
import io
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
IN = Path(os.environ.get("APX_IN", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-convaudit_approachhash_kill-2026-11-14"
OUTV = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
EV_ROOT = "/ev/evals_v1"
EPOCH_MAP = "/ev/q4b_epoch_map_v1.json"
SOL_DIR = "/acts/2026-10-06_la_segment-end-v4_qwen3-32b_v1/inputs"
SOL_ARMS = ("real-attack-e1", "real-attack-e2", "sim-attack")
WORLDS = ("run1", "run2", "sim")
PINS = {
    "t196.json": "55d9a8ec68f46854",
    "runs36.json": "76b11486cc353ab1",
}  # sha256 prefixes (RULE.md inputs)
REG_FILES = 54
SAMPLE_RE = re.compile(r"samples/.+_epoch_(\d+)\.json")
REDRAW_RE = re.compile(
    r"_redraw\d+$"
)  # Amendment 1: t125_user_tracking_redraw1 -> t125_user_tracking

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "zstandard==0.23.0"
)
if IN.exists():
    image = (
        image.add_local_dir(IN / "reg", "/root/in/reg")
        .add_local_file(IN / "t196.json", "/root/in/t196.json")
        .add_local_file(IN / "runs36.json", "/root/in/runs36.json")
    )
ev = modal.Volume.from_name(
    "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", version=2
).read_only()
acts = modal.Volume.from_name(
    "probe-shift_aswin-surya_sub-la-activations", version=2
).read_only()
outv = modal.Volume.from_name(OUTV, version=2)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def guard(f, *a):
    """Run f; on any exception return its type name only (no message, no traceback leaves the container)."""
    try:
        return f(*a)
    except Exception as e:  # noqa: BLE001 -- the error type only, never text
        return {"fatal": type(e).__name__}


def check_inputs() -> None:
    for name, pin in PINS.items():
        if (
            hashlib.sha256(Path(f"/root/in/{name}").read_bytes()).hexdigest()[:16]
            != pin
        ):
            raise ValueError("input pin mismatch")
    if (
        len(
            list(
                Path("/root/in/reg").glob(
                    "*/side_tasks/*/docs/attack_strategies/*/*.md"
                )
            )
        )
        != REG_FILES
    ):
        raise ValueError("registry file count mismatch")


def registry() -> dict:
    """side -> [(main_dir, file, sha, text)] as the registry loader reads it: sorted dirs, sorted *.md, strip, non-empty."""
    reg = {}
    for side_dir in sorted(Path("/root/in/reg").glob("*/side_tasks/*")):
        out = []
        for mdir in sorted((side_dir / "docs" / "attack_strategies").iterdir()):
            if mdir.is_dir():
                for f in sorted(mdir.glob("*.md")):
                    t = f.read_text().strip()
                    if t:
                        out.append((mdir.name, f.name, sha(t), t))
        if side_dir.name in reg:
            raise ValueError("side name in two environments")
        reg[side_dir.name] = out
    return reg


def text_of(content):
    """Message content as one string (a list joins its text parts in order); None when the shape is unknown."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(p, dict) for p in content):
        return "".join(
            p.get("text", "")
            for p in content
            if p.get("type", "text") == "text" and isinstance(p.get("text", ""), str)
        )
    return None


def system_of(messages):
    if not isinstance(messages, list):
        return None
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            return text_of(m.get("content"))
    return None


def state(x) -> str:
    return "missing" if not isinstance(x, str) else ("empty" if x == "" else "nonempty")


def contains(system, side: str, reg: dict) -> dict:
    """Substring tests inside the container; None (unknown) when there is no system text."""
    if not isinstance(system, str):
        return {
            "contains_first": None,
            "contains_any_own": None,
            "contains_other_side": None,
        }
    own = reg.get(side, [])
    return {
        "contains_first": bool(own) and own[0][3] in system,
        "contains_any_own": any(s[3] in system for s in own),
        "contains_other_side": any(
            s[3] in system for o, ss in reg.items() if o != side for s in ss
        ),
    }


PATH_RE = re.compile(r"[A-Za-z0-9_@.:/-]+\.eval")


def _list_evals() -> list:
    rels = sorted(
        str(p.relative_to(EV_ROOT))
        for arm in ("attack", "sim_attack@september")
        for p in Path(EV_ROOT).glob(f"*/{arm}/*/*.eval")
    )
    if not rels or not all(PATH_RE.fullmatch(r) for r in rels):
        raise ValueError(
            "unexpected eval path"
        )  # paths leave the container as identifiers, so they must look like ones
    return rels


def task_of(rel: str) -> str:
    return REDRAW_RE.sub("", Path(rel).parts[2])


def zstd_zip():
    """Amendment 1: let Python 3.12's zipfile read zstd members (zip method 93), as 2e's evalzip.py does."""
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


def _scan_evals(rels: list) -> dict:
    zipfile = zstd_zip()

    check_inputs()
    reg = registry()
    cells = {
        c["key"]: c for c in json.loads(Path("/root/in/t196.json").read_text())["cells"]
    }
    rows, errors = [], {}
    for rel in rels:
        try:
            task = task_of(rel)
            cell = cells[task]
            own = reg.get(cell["side"], [])
            with zipfile.ZipFile(Path(EV_ROOT) / rel) as z:
                names = [n for n in z.namelist() if SAMPLE_RE.fullmatch(n)]
                if not names:
                    raise ValueError("no samples")
                names.sort(
                    key=lambda n: (int(SAMPLE_RE.fullmatch(n).group(1)), n)
                )  # numeric epoch order
                for n in names:
                    s = json.loads(z.read(n))
                    md = (
                        s.get("metadata") if isinstance(s.get("metadata"), dict) else {}
                    )
                    ap = md.get("attack_approach")
                    ap_state = state(ap)
                    ap_sha = sha(ap) if ap_state != "missing" else None
                    system = system_of(s.get("messages"))
                    inp = s.get("input")
                    ep = s.get("epoch")
                    rows.append(
                        {
                            "rel": rel,
                            "task": task,
                            "env": cell["env"],
                            "side": cell["side"],
                            "n_samples_in_file": len(names),
                            "epoch": ep
                            if isinstance(ep, int) and not isinstance(ep, bool)
                            else None,
                            "approach_state": ap_state,
                            "approach_chars": len(ap)
                            if ap_state != "missing"
                            else None,
                            "approach_sha256": ap_sha,
                            "approach_eq_cell": ap_sha == cell["approach_sha256"]
                            if ap_state == "nonempty"
                            else None,
                            "approach_eq_first": (bool(own) and ap_sha == own[0][2])
                            if ap_state == "nonempty"
                            else None,
                            "approach_eq_any_own": any(ap_sha == x[2] for x in own)
                            if ap_state == "nonempty"
                            else None,
                            "system_state": state(system),
                            "system_chars": len(system)
                            if isinstance(system, str)
                            else None,
                            "system_sha256": sha(system)
                            if isinstance(system, str)
                            else None,
                            "system_eq_sol": sha(system) == cell["sol_system_sha256"]
                            if isinstance(system, str)
                            else None,
                            "input_state": state(inp),
                            "input_sha256": sha(inp) if isinstance(inp, str) else None,
                            "input_eq_sol": sha(inp) == cell["sol_input_sha256"]
                            if isinstance(inp, str)
                            else None,
                            **contains(system, cell["side"], reg),
                        }
                    )
        except Exception as e:  # noqa: BLE001 -- the error type only, never text; counted, and a mapped file's error fails the gate
            errors[rel] = type(e).__name__
    return {"rows": rows, "errors": errors}


def _gate_a(a_rows: list, a_errors: dict) -> dict:
    emap = json.loads(Path(EPOCH_MAP).read_text())["map"]
    mapped = {k: v for k, v in emap.items() if v[0] in WORLDS}
    by_rel = {}
    for i, r in enumerate(a_rows):
        by_rel.setdefault(r["rel"], []).append(i)
    pair_count = {}
    for v in mapped.values():
        pair_count[tuple(v)] = pair_count.get(tuple(v), 0) + 1
    compat = {
        "run1": lambda p: "/attack/" in f"/{p}" and "second_attack" not in p,
        "run2": lambda p: p.startswith("2026-10-10_001314_second_attack/attack/"),
        "sim": lambda p: "/sim_attack@september/" in f"/{p}",
    }
    g = {
        w: {
            "mapped": 0,
            "ok": 0,
            "suffix_not_e0": 0,
            "file_not_scanned": 0,
            "file_errored": 0,
            "not_one_sample": 0,
            "world_path_incompatible": 0,
            "task_mismatch": 0,
        }
        for w in WORLDS
    }
    counted = {}
    for k, (w, t) in mapped.items():
        rel, _, suf = k.rpartition("#")
        x = g[w]
        x["mapped"] += 1
        bad = False
        for cond, name in (
            (suf != "e0", "suffix_not_e0"),
            (rel not in by_rel and rel not in a_errors, "file_not_scanned"),
            (rel in a_errors, "file_errored"),
            (len(by_rel.get(rel, [])) != 1, "not_one_sample"),
            (not compat[w](rel), "world_path_incompatible"),
            (task_of(rel) != t, "task_mismatch"),
        ):
            if cond:
                x[name] += 1
                bad = True
        if not bad:
            x["ok"] += 1
            counted[f"{w}|{t}"] = by_rel[rel][0]
    dup_pairs = sum(n > 1 for n in pair_count.values())
    ok = dup_pairs == 0 and all(
        x["mapped"] > 0 and x["ok"] == x["mapped"] for x in g.values()
    )
    return {
        "pass": ok,
        "per_world": g,
        "duplicate_world_task_pairs": dup_pairs,
        "counted": counted if ok else None,
    }


def _summarize_a(a_rows: list, counted_idx: dict) -> dict:
    from collections import Counter

    cells = {
        c["key"]: c for c in json.loads(Path("/root/in/t196.json").read_text())["cells"]
    }
    counted = {tuple(k.split("|", 1)): a_rows[i] for k, i in counted_idx.items()}
    flags = (
        "approach_state",
        "approach_eq_cell",
        "approach_eq_first",
        "approach_eq_any_own",
        "system_state",
        "system_eq_sol",
        "input_state",
        "input_eq_sol",
        "contains_first",
        "contains_any_own",
        "contains_other_side",
    )

    def tally(rs):
        return {f: dict(Counter(str(r[f]) for r in rs)) for f in flags} | {"n": len(rs)}

    out = {
        "counted_by_world": {
            w: tally([r for (ww, _), r in counted.items() if ww == w]) for w in WORLDS
        }
    }
    counted_rels = {r["rel"] for r in counted.values()}
    sup = [r for r in a_rows if r["rel"] not in counted_rels]

    def sup_world(r):
        if "/sim_attack@september/" in f"/{r['rel']}":
            return "sim"
        return (
            "run2"
            if r["rel"].startswith("2026-10-10_001314_second_attack/")
            else "run1"
        )

    out["superseded_by_world"] = {
        w: tally([r for r in sup if sup_world(r) == w]) for w in WORLDS
    }

    def compare(g, field):
        st = {w: g[w][f"{field}_state"] for w in WORLDS}
        h = {w: g[w][f"{field}_sha256"] for w in WORLDS}
        if "missing" in st.values():
            return {"missing_any": True}
        return {
            "missing_any": False,
            "equal_all": len(set(h.values())) == 1,
            "sim_differs_from_both": h["sim"] != h["run1"] and h["sim"] != h["run2"],
            "sim_empty_real_nonempty": st["sim"] == "empty"
            and st["run1"] == st["run2"] == "nonempty",
            "real_differ": h["run1"] != h["run2"],
        }

    tasks = sorted({t for (_, t) in counted})
    full, lists = [], {}
    for t in tasks:
        g = {w: counted.get((w, t)) for w in WORLDS}
        if not all(g.values()):
            continue
        row = {
            "task": t,
            "env": cells[t]["env"],
            "cell_has_approach": cells[t]["approach_chars"] > 0,
        }
        for field in ("approach", "system"):
            for k, v in compare(g, field).items():
                row[f"{field}_{k}"] = v
                if v and k != "equal_all":
                    lists.setdefault(f"{field}_{k}", []).append(
                        f"{t} ({cells[t]['side']})"
                    )
        full.append(row)
    keys = sorted(
        {k for r in full for k in r if k.startswith(("approach_", "system_"))}
    )

    def valid(
        field, rs
    ):  # tasks whose three runs all carry the field: the comparison's denominator
        return [r for r in rs if r[f"{field}_missing_any"] is False]

    out["per_task"] = {
        "tasks_any_world": len(tasks),
        "tasks_all_three": len(full),
        "tasks_incomplete": {
            t: [w for w in WORLDS if (w, t) in counted]
            for t in tasks
            if not all((w, t) in counted for w in WORLDS)
        },
        "cell_has_approach": sum(r["cell_has_approach"] for r in full),
        "counts": {
            k: f"{sum(r.get(k) is True for r in (full if k.endswith('missing_any') else valid(k.split('_')[0], full)))}"
            f"/{len(full) if k.endswith('missing_any') else len(valid(k.split('_')[0], full))}"
            for k in keys
        },
        "by_env": {
            e: {
                k: f"{sum(r.get(k) is True for r in (fe if k.endswith('missing_any') else valid(k.split('_')[0], fe)))}"
                f"/{len(fe) if k.endswith('missing_any') else len(valid(k.split('_')[0], fe))}"
                for k in keys
            }
            for e in sorted({r["env"] for r in full})
            for fe in [[r for r in full if r["env"] == e]]
        },
        "lists": lists,
    }
    pos = [r for r in counted.values() if r["approach_eq_first"] is True]
    neg = [r for r in counted.values() if r["approach_state"] == "empty"]
    out["method_check"] = {
        "positive_control_runs": len(pos),
        "positive_of_which_contains_first": sum(
            r["contains_first"] is True for r in pos
        ),
        "validated": len(pos) > 0 and all(r["contains_first"] is True for r in pos),
        "negative_control_runs_empty_approach": len(neg),
        "negative_of_which_contains_any_own": sum(
            r["contains_any_own"] is True for r in neg
        ),
    }
    return out


def _scan_sol(shard: str) -> dict:
    check_inputs()
    reg = registry()
    runs = {r["key"]: r for r in json.loads(Path("/root/in/runs36.json").read_text())}
    rows, errors, other = [], {}, {}
    with (Path(SOL_DIR) / shard).open("rb") as fh:
        for n, line in enumerate(fh):
            try:
                rec = json.loads(line)
                r = (
                    runs.get(rec.get("key"))
                    if isinstance(rec.get("key"), str)
                    else None
                )
                arm = (
                    r["arm"] if r else "unjoined"
                )  # a fixed label or runs.json's own arm name, never staged text
                if arm not in SOL_ARMS:
                    other[arm] = other.get(arm, 0) + 1
                    continue
                system = system_of(rec.get("messages"))
                rows.append(
                    {
                        "key": r["key"],
                        "arm": arm,
                        "cell": r["cell"],
                        "side": r["side_task"],
                        "env": r["environment"],
                        "side_has_strategy": bool(reg.get(r["side_task"])),
                        "system_state": state(system),
                        "system_sha256": sha(system)
                        if isinstance(system, str)
                        else None,
                        **contains(system, r["side_task"], reg),
                    }
                )
            except Exception as e:  # noqa: BLE001 -- the error type only, never text; counted and reported
                errors[f"{shard}#{n}"] = type(e).__name__
    return {"rows": rows, "errors": errors, "other_arms": other}


def _summarize_b(b_rows: list, b_errors: dict, other: dict, method_check: dict) -> dict:
    from collections import Counter

    runs = json.loads(Path("/root/in/runs36.json").read_text())
    if len({r["key"] for r in runs}) != len(runs):
        raise ValueError("duplicate runs.json keys")
    cells = {
        c["key"]: c for c in json.loads(Path("/root/in/t196.json").read_text())["cells"]
    }
    expected = {a: {r["key"] for r in runs if r["arm"] == a} for a in SOL_ARMS}
    seen = Counter(r["key"] for r in b_rows)
    first = {}
    for r in b_rows:
        first.setdefault(r["key"], r)
    coverage = {
        a: {
            "expected": len(expected[a]),
            "found_once": sum(seen[k] == 1 for k in expected[a]),
            "missing": sum(seen[k] == 0 for k in expected[a]),
            "duplicate": sum(seen[k] > 1 for k in expected[a]),
        }
        for a in SOL_ARMS
    }
    t196 = {c["cell"]: c for c in cells.values()}
    conv_ok = all(
        c["cell"] == f"{c['main']}:{c['side']}" for c in cells.values()
    ) and all(
        r["cell"] == f"{r['main_task']}:{r['side_task']}"
        for r in runs
        if r["arm"] in SOL_ARMS
    )
    for a in SOL_ARMS:
        cs = [r["cell"] for r in runs if r["arm"] == a]
        coverage[a] |= {
            "unique_cells": len(set(cs)),
            "t196_cells_present": len(set(cs) & set(t196)),
        }
    complete = (
        not b_errors
        and other.get("unjoined", 0) == 0
        and conv_ok
        and all(
            c["expected"] == 939
            and c["found_once"] == c["expected"]
            and c["unique_cells"] == 939
            for c in coverage.values()
        )
    )
    rows = list(first.values())

    def block(ra):
        return {
            "n": len(ra),
            "side_has_strategy": sum(r["side_has_strategy"] for r in ra),
            "system_state": dict(Counter(r["system_state"] for r in ra)),
            "contains_first": sum(r["contains_first"] is True for r in ra),
            "contains_any_own": sum(r["contains_any_own"] is True for r in ra),
            "contains_other_side": sum(r["contains_other_side"] is True for r in ra),
            "has_strategy_but_contains_none": sum(
                r["side_has_strategy"] and r["contains_any_own"] is False for r in ra
            ),
            "contains_own_not_first": sum(
                r["contains_any_own"] is True and r["contains_first"] is False
                for r in ra
            ),
            "containment_unknown": sum(r["contains_first"] is None for r in ra),
        }

    sol = {
        "complete": complete,
        "coverage": coverage,
        "errors": len(b_errors),
        "other_arms": other,
        "cell_convention_main_colon_side": conv_ok,
        "containment_validated_by_part_a": method_check.get("validated"),
    }
    for scope, rs in (("all", rows), ("t196", [r for r in rows if r["cell"] in t196])):
        sol[scope] = {}
        for a in SOL_ARMS:
            ra = [r for r in rs if r["arm"] == a]
            sol[scope][a] = block(ra) | {
                "by_env": {
                    e: block([r for r in ra if r["env"] == e])
                    for e in sorted({r["env"] for r in ra})
                }
            }
    e1 = Counter(r["cell"] for r in rows if r["arm"] == "real-attack-e1")
    e1r = {r["cell"]: r for r in rows if r["arm"] == "real-attack-e1"}
    sol["t196_e1"] = {
        "t196_cells": len(t196),
        "present": sum(c in e1r for c in t196),
        "duplicate_cells": sum(n > 1 for n in e1.values()),
        "system_eq_sol_system_sha256": sum(
            c in e1r and e1r[c]["system_sha256"] == t196[c]["sol_system_sha256"]
            for c in t196
        ),
    }
    return sol


@app.function(
    image=image, cpu=1, memory=2048, timeout=600, retries=0, volumes={"/ev": ev}
)
def list_evals():
    return guard(_list_evals)


@app.function(
    image=image, cpu=1, memory=8192, timeout=1800, retries=0, volumes={"/ev": ev}
)
def scan_evals(rels: list) -> dict:
    return guard(_scan_evals, rels)


@app.function(
    image=image, cpu=1, memory=4096, timeout=600, retries=0, volumes={"/ev": ev}
)
def gate_a(a_rows: list, a_errors: dict) -> dict:
    return guard(_gate_a, a_rows, a_errors)


@app.function(image=image, cpu=1, memory=4096, timeout=600, retries=0)
def summarize_a(a_rows: list, counted: dict) -> dict:
    return guard(_summarize_a, a_rows, counted)


@app.function(
    image=image, cpu=1, memory=8192, timeout=1800, retries=0, volumes={"/acts": acts}
)
def scan_sol(shard: str) -> dict:
    return guard(_scan_sol, shard)


@app.function(image=image, cpu=1, memory=4096, timeout=600, retries=0)
def summarize_b(b_rows: list, b_errors: dict, other: dict, method_check: dict) -> dict:
    return guard(_summarize_b, b_rows, b_errors, other, method_check)


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
        raise SystemExit("APX_IN not set")
    head = git("rev-parse", "HEAD")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"approach_hash_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    out = {
        "label": "RULE.md (approach provenance); sha256, lengths, booleans and names only",
        "git": head,
        "stamp": stamp,
    }
    rels = fatal(list_evals.remote(), "list_evals")
    print(f"{len(rels)} eval files", flush=True)
    a = [
        fatal(x, "scan_evals")
        for x in scan_evals.map([rels[i : i + 20] for i in range(0, len(rels), 20)])
    ]
    a_rows = [r for x in a for r in x["rows"]]
    a_errors = {k: v for x in a for k, v in x["errors"].items()}
    gate = fatal(gate_a.remote(a_rows, a_errors), "gate_a")
    out |= {
        "eval_files": len(rels),
        "eval_samples": len(a_rows),
        "eval_errors": a_errors,
        "gate": {k: v for k, v in gate.items() if k != "counted"},
    }
    print("gate", json.dumps(out["gate"]), "| eval errors", len(a_errors), flush=True)
    if not gate["pass"]:  # stop: no Part A statistics, no Part B, no rows file
        dest.open("x").write(json.dumps(out, indent=1))
        print("GATE FAILED; stopped. wrote", dest.name, flush=True)
        return
    out["part_a"] = fatal(summarize_a.remote(a_rows, gate["counted"]), "summarize_a")
    b = [
        fatal(x, "scan_sol")
        for x in scan_sol.map([f"shard_{i}.jsonl" for i in range(64)])
    ]
    b_rows = [r for x in b for r in x["rows"]]
    b_errors = {k: v for x in b for k, v in x["errors"].items()}
    other = {}
    for x in b:
        for k, v in x["other_arms"].items():
            other[k] = other.get(k, 0) + v
    out["part_b"] = fatal(
        summarize_b.remote(b_rows, b_errors, other, out["part_a"]["method_check"]),
        "summarize_b",
    )
    blob = "".join(json.dumps(r) + "\n" for r in a_rows + b_rows).encode()
    with outv.batch_upload(force=True) as up:
        up.put_file(io.BytesIO(blob), f"approach_hash/rows_{stamp}.jsonl")
    out |= {
        "rows_file": f"{OUTV}:/approach_hash/rows_{stamp}.jsonl",
        "rows_sha256": hashlib.sha256(blob).hexdigest(),
    }
    dest.open("x").write(json.dumps(out, indent=1))
    for k in ("part_a", "part_b"):
        print(k, json.dumps(out[k], indent=1), flush=True)
    print("wrote", dest.name, flush=True)
