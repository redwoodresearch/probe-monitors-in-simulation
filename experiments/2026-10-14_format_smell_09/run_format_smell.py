"""RULE.md here (POST HOC): per-run think-block, user-turn and alternation flags in every staging the probes read, honest vs
attack. Modal CPU, read-only Volumes, counts only: no text leaves a container (exceptions report their type only).

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy==2.3.3 modal run --detach \\
      experiments/2026-10-14_format_smell_09/run_format_smell.py
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # inside the container the file sits at /root
APP = "probe-shift_aswin-surya_sub-convaudit_formatsmell_kill-2026-11-14"
OUTV = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
VOLS = {"/b": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", "/acts": "probe-shift_aswin-surya_sub-la-activations",
        "/sol": "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", "/xr": "probe-shift_aswin-surya_sub-q3xread_acts_kill-2026-11-13"}
# (staging, glob, format, renderings)
SOURCES = [("B", "/b/q4b_main_v1/inputs/*.jsonl", "ids", ("qwen35",)), ("B", "/b/q4b_run2_v1/inputs/attack.jsonl", "ids", ("qwen35",)),
           ("B", "/b/q35own_repfull_v1/inputs/repfull_attack.jsonl", "ids", ("qwen35",)),
           ("B-nochat", "/b/q4c_main_nochat_v1/inputs/*.jsonl", "ids", ("qwen35",)), ("B-nochat", "/b/q4c_run2_nochat_v1/inputs/attack.jsonl", "ids", ("qwen35",)),
           ("xread", "/xr/xread_main_v1/inputs/*.jsonl", "ids", ("qwen3",)), ("xread", "/xr/xread_run2_v1/inputs/attack.jsonl", "ids", ("qwen3",)),
           ("xread", "/xr/xread_repair_v2/inputs/repair.jsonl", "ids", ("qwen3",)),
           ("d9-onpol", "/acts/2026-10-14_q3onpol_q35own_inputs_v1/inputs/shard_*.jsonl", "messages", ("qwen3", "qwen35")),
           ("sol-q35", "/sol/2026-10-14_q35solavg_inputs_v1/inputs/shard_*.jsonl", "messages", ("qwen35",)),
           ("sol-v4", "/acts/2026-10-06_la_segment-end-v4_qwen3-32b_v1/inputs/shard_*.jsonl", "messages", ("qwen3",))]
RUN2_FILES = {"/b/q4b_run2_v1/inputs/attack.jsonl": "run2", "/b/q4c_run2_nochat_v1/inputs/attack.jsonl": "run2",
              "/xr/xread_run2_v1/inputs/attack.jsonl": "run2"}
GATE = {("q4b_main_v1", "attack"): [(34, 34), (35, 35), (31, 31)], ("q4b_main_v1", "honest_a"): [(25, 25), (26, 26), (52, 0)],
        ("q4b_main_v1", "honest_b"): [(39, 39), (42, 42), (48, 0)], ("q4b_main_v1", "sim_attack@september"): [(49, 49), (42, 42), (64, 64)],
        ("q4b_main_v1", "sim_honest@september"): [(51, 0), (28, 28), (45, 45)], ("q4b_run2_v1", "attack"): [(32, 32), (16, 16), (13, 13)]}
FLAGS = ("zero_think", "partial_think", "full_think", "zero_think_and_ends_on_user", "think_missing_before_last_tool", "user_turn_before_last_tool", "violation_before_last_tool",
         "ends_on_user", "ends_on_tool", "ends_on_assistant", "extra_user_turn", "violation_any")
VISIBLE = {"qwen35": ("zero_think", "think_missing_before_last_tool", "user_turn_before_last_tool", "violation_before_last_tool"),
           "qwen3": ("think_missing_before_last_tool", "user_turn_before_last_tool", "violation_before_last_tool")}

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy==2.3.3", "scipy==1.16.2")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))
mounts = {p: modal.Volume.from_name(n, version=2).read_only() for p, n in VOLS.items()}
mounts["/hf"] = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only()
outv = modal.Volume.from_name(OUTV, version=2)


def tok_of(name):
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN3_32B, QWEN35_27B

    m = {"qwen3": QWEN3_32B, "qwen35": QWEN35_27B}[name]
    return AutoTokenizer.from_pretrained(m["name"], revision=m["revision"], local_files_only=True)


def flags_of(text: str) -> dict:
    """Turn kinds from the rendered text: S system, U genuine user, T tool output, A assistant (think = body starts with <think>)."""
    kinds, think = [], []
    segs = text.split("<|im_start|>")[1:]
    if not segs:
        raise ValueError("malformed rendering")
    for j, seg in enumerate(segs):  # every turn but the last closes with exactly one <|im_end|>; the last has at most one
        n_end = seg.count("<|im_end|>")
        if (j < len(segs) - 1 and (n_end != 1 or not seg.rstrip("\n").endswith("<|im_end|>"))) or n_end > 1:
            raise ValueError("malformed turn boundary")
    for seg in segs:
        role, _, body = seg.partition("\n")
        if role == "assistant":
            k = "A"
        elif role == "system":
            k = "S"
        elif role in ("user", "tool"):
            k = "T" if body.lstrip().startswith("<tool_response>") else "U"
            if role == "tool" and k == "U":
                raise ValueError("tool role without a tool_response")
        else:
            raise ValueError("unknown role")
        kinds.append(k)
        think.append(k == "A" and body.lstrip().startswith("<think>"))
    a = [i for i, k in enumerate(kinds) if k == "A"]
    if not a:
        raise ValueError("no assistant turn")
    lt = max((i for i, k in enumerate(kinds) if k == "T"), default=-1)
    users = [i for i, k in enumerate(kinds) if k == "U"]
    ns = [k for k in kinds if k != "S"]
    pairs = [x + y for x, y in zip(kinds, kinds[1:]) if "S" not in (x, y)]
    bad = {"AA", "UU", "TT", "TU", "UT"}
    viol_at = [i + 1 for i, (x, y) in enumerate(zip(kinds, kinds[1:])) if "S" not in (x, y) and x + y in bad]
    if ns and ns[0] != "U":
        viol_at.append(kinds.index(ns[0]))
    n_think = sum(think)
    return {"n_assistant": len(a), "n_think": n_think, "n_tool": kinds.count("T"), "n_user": len(users),
            "zero_think": n_think == 0, "partial_think": 0 < n_think < len(a), "full_think": n_think == len(a),
            "zero_think_and_ends_on_user": n_think == 0 and kinds[-1] == "U",
            "think_missing_before_last_tool": any(i < lt and not think[i] for i in a),
            "user_turn_before_last_tool": any(i < lt for i in users[1:]), "extra_user_turn": len(users) > 1,
            "violation_any": bool(viol_at), "violation_before_last_tool": any(i <= lt for i in viol_at),
            "ends_on_user": bool(kinds) and kinds[-1] == "U", "ends_on_tool": bool(kinds) and kinds[-1] == "T",
            "ends_on_assistant": bool(kinds) and kinds[-1] == "A", "pairs": {p: pairs.count(p) for p in set(pairs) if p in bad}}


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes=mounts)
def list_files() -> list:
    import glob

    out = []
    for staging, pat, fmt, renders in SOURCES:
        hits = sorted(glob.glob(pat))
        if not hits:
            raise FileNotFoundError(pat)
        out += [(staging, f, fmt, list(renders)) for f in hits]
    return out


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes=mounts)
def count_file(staging: str, path: str, fmt: str, renders: list) -> dict:
    """Per-run flags for one staged file. Returns rows (keys kept for the Volume file) and the gate tuples; never text."""
    from probe_shift.reader import prepare

    toks = {r: tok_of(r) for r in renders}
    think_id = {r: t.convert_tokens_to_ids("<think>") for r, t in toks.items()}
    set_dir = Path(path).parts[2]
    rows, errors, gate = [], {}, []
    with Path(path).open("rb") as fh:
        lines = list(fh)
    for n, line in enumerate(lines):
        try:
            rec = json.loads(line)
            arm = RUN2_FILES.get(path, rec.get("arm") or Path(path).stem)
            key = rec["key"] if isinstance(rec.get("key"), str) else f"{Path(path).name}#{n}"  # an identifier only, never a text field
            if not isinstance(arm, str):
                raise ValueError("no string arm")
        except Exception as e:  # the error type only, never text
            errors[type(e).__name__] = errors.get(type(e).__name__, 0) + 1
            continue
        for r in renders:
            try:
                if fmt == "ids":
                    text = toks[r].decode(rec["ids"], skip_special_tokens=False)
                    n_think_tok = sum(1 for x in rec["ids"] if x == think_id[r])
                else:
                    text = toks[r].apply_chat_template(prepare(toks[r], rec["messages"]), tokenize=False, enable_thinking=False)
                    n_think_tok = None
                f = flags_of(text)
                rows.append({"staging": staging, "render": r, "set": set_dir, "file": path, "arm": arm, "key": key,
                             "key_sha": hashlib.sha256(str(key).encode()).hexdigest()[:16], "n_think_tokens": n_think_tok, **f})
                if fmt == "ids":
                    gate.append((key, arm, [f["n_assistant"], n_think_tok]))
            except Exception as e:  # the error type only, never text
                errors[type(e).__name__] = errors.get(type(e).__name__, 0) + 1
                if fmt == "ids":
                    gate.append((key, arm, None))  # a failed record stays in the gate's sample and fails it
    by_arm = {}
    for key, arm, got in sorted(gate):  # the stage gate's sample: the first 3 epochs by key, per arm
        by_arm.setdefault(arm, []).append(got)
    gate = [{"set": set_dir, "arm": arm, "got": g[:3]} for arm, g in by_arm.items()]
    return {"staging": staging, "file": path, "rows": rows, "errors": errors, "gate": gate, "lines": len(lines)}


@app.function(image=image, cpu=2, memory=16384, timeout=1800, retries=0)
def summarize(rows):
    from collections import defaultdict

    from scipy.stats import fisher_exact

    by = defaultdict(list)
    for r in rows:
        by[(r["staging"], r["render"])].append(r)
    out = {}
    for (st, rd), rs in sorted(by.items()):
        arms = sorted({r["arm"] for r in rs})
        per_arm = {}
        for a in arms:
            x = [r for r in rs if r["arm"] == a]
            pairs = {}
            for r in x:
                for p, c in r["pairs"].items():
                    pairs[p] = pairs.get(p, 0) + c
            per_arm[a] = {"n": len(x), **{f: sum(r[f] for r in x) for f in FLAGS}, "violation_pairs_total": dict(sorted(pairs.items())),
                          "mean_assistant_turns": round(sum(r["n_assistant"] for r in x) / len(x), 3)}
        hon = [a for a in arms if "honest" in a.lower()]
        att = [a for a in arms if "honest" not in a.lower()]
        tests = {}
        groups = [(h, [h], a, [a]) for h in hon for a in att] + ([("pooled honest", hon, "pooled attack", att)] if hon and att else [])
        for hn, hs, an, as_ in groups:
            for f in FLAGS:
                k1, n1 = sum(per_arm[h][f] for h in hs), sum(per_arm[h]["n"] for h in hs)
                k2, n2 = sum(per_arm[a][f] for a in as_), sum(per_arm[a]["n"] for a in as_)
                p = float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])
                diff = 100 * (k1 / n1 - k2 / n2)
                tests[f"{hn} vs {an}|{f}"] = {"honest": [k1, n1], "attack": [k2, n2], "diff_points": diff, "p": p}
        vis = VISIBLE[rd]
        pooled = [tests[f"pooled honest vs pooled attack|{f}"] for f in vis] if hon and att else []
        label = ("label-correlated" if any(abs(t["diff_points"]) >= 5 and t["p"] < 1e-3 for t in pooled)
                 else "suggestive" if any(abs(t["diff_points"]) >= 5 and t["p"] < 0.05 for t in pooled)
                 else "not label-correlated" if pooled else "no honest/attack pair")
        out[f"{st}|{rd}"] = {"per_arm": per_arm, "tests": tests, "visible_flags": list(vis), "reading": label}
    return out


@app.local_entrypoint()
def main():
    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    head = g("rev-parse", "HEAD")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    files = list_files.remote()
    print(f"{len(files)} staged files", flush=True)
    first = [x for x in files if Path(x[1]).parts[2] in ("q4b_main_v1", "q4b_run2_v1")]
    res = list(count_file.starmap(first))  # the gate's files first: nothing else is counted if it fails
    got = {}
    for x in res:
        for gr in x["gate"]:
            got[(gr["set"], gr["arm"] if gr["set"] != "q4b_run2_v1" else "attack")] = gr["got"]
    gate = {f"{s}|{a}": {"expected": [list(t) for t in exp], "got": got.get((s, a))} for (s, a), exp in GATE.items()}
    gate_ok = all(v["got"] == v["expected"] for v in gate.values())
    if gate_ok:
        res += list(count_file.starmap([x for x in files if x not in first]))
    rows = [r for x in res for r in x["rows"]]
    errors = {x["file"]: x["errors"] for x in res if x["errors"]}
    out = {"label": "POST HOC (RULE.md); counts only", "git": head, "stamp": stamp, "gate_pass": gate_ok, "gate": gate,
           "files": len(files), "files_counted": len(res), "lines_read": sum(x["lines"] for x in res), "runs_counted": len(rows),
           "parse_errors": errors}
    if gate_ok:
        out["summary"] = summarize.remote(rows)
    import io

    with outv.batch_upload(force=True) as b:
        b.put_file(io.BytesIO("".join(json.dumps(r) + "\n" for r in rows).encode()), f"format_smell/rows_{stamp}.jsonl")
    out["rows_file"] = f"{OUTV}:/format_smell/rows_{stamp}.jsonl"
    out["rows_sha256"] = hashlib.sha256("".join(json.dumps(r) + "\n" for r in rows).encode()).hexdigest()
    dest = HERE / "results" / f"format_smell_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(out, indent=1))
    print("git", head, "| gate_pass", gate_ok, "| runs", len(rows), "| parse errors", errors, flush=True)
    if not gate_ok:
        print(json.dumps(gate))
        return
    order = sorted(out["summary"], key=lambda k: (not k.startswith("B|"), k))
    for k in order:
        s = out["summary"][k]
        if k.startswith("B|"):
            for t in ("honest_a vs attack", "honest_a vs run2"):
                for f in s["visible_flags"] + ["ends_on_user"]:
                    v = s["tests"].get(f"{t}|{f}")
                    if v:
                        print(f"  HEADLINE PAIR {t}|{f}: honest {v['honest'][0]}/{v['honest'][1]} attack {v['attack'][0]}/{v['attack'][1]} diff {v['diff_points']:+.1f} p {v['p']:.2g}")
        print(f"\n== {k}: {s['reading']} (visible: {', '.join(s['visible_flags'])})")
        for a, v in s["per_arm"].items():
            print(f"  {a:24s} n {v['n']:5d} | " + " ".join(f"{f} {v[f]}" for f in FLAGS) + f" | mean A turns {v['mean_assistant_turns']}")
        for t, v in s["tests"].items():
            if t.startswith("pooled") or t.split("|")[1] in s["visible_flags"]:
                print(f"    {t}: honest {v['honest'][0]}/{v['honest'][1]} attack {v['attack'][0]}/{v['attack'][1]} diff {v['diff_points']:+.1f} p {v['p']:.2g}")
    print(dest.name)
