"""Writing-session example index (rule: experiments/2026-10-14_p2_simh_scores/results/writing_examples_rule_2026-10-15_193657.json @ 160620a04).
Index only: keys, step and call indices, character lengths and sha256s of command bytes. NO command text is printed, logged, returned or
written: text is measured and hashed inside the container only. One Modal CPU app; every Volume is mounted read-only.

  --mode inspect  field names, types and counts only (no values from any transcript), to fix the layouts before the run mode
  --mode run      index A (seeded fit-sample draws: own_frac100_L48 and Sol (a), attack and honest) and index B (the most suspicious
                  command of eight pertask455v2 runs); every input is read from git by commit and checked against its sha256
  uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-15_writing_examples/run_examples_index.py --mode inspect|run \\
      [--sol-fit <sol fit-sample json> --final-spec <09 FINAL spec json> --out <json>]   (inspect only)
"""

import hashlib
import json
import random
import re
import struct
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent

APP = "probe-shift_aswin-surya_sub-p2probes_examples_kill-2026-11-13"
VOLS = {
    "/q4": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13",
    "/p2": "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13",
    "/sol": "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13",
    "/conv": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14",
    "/acts": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14",
}
FAMEXP_SPEC = "/p2/famexp/2026-10-14_163027/spec_2026-10-14_163027.json"
PINS = {  # name -> (commit, path, sha256), read from git and refused on any byte difference
    "rule": (
        "160620a04",
        "experiments/2026-10-14_p2_simh_scores/results/writing_examples_rule_2026-10-15_193657.json",
        "b941a9115118bdfb46185655468e8c7a5c1d722ea5ac0cde1d17dc49fded4611",
    ),
    "rule_amendment1": (
        "ee20abeca",
        "experiments/2026-10-14_p2_simh_scores/results/writing_examples_rule_amendment1_2026-10-15_221039.json",
        "9906f49df7e14df93c9f1258d329ff95b9744c8ce2f66a688444b4dedc1cc8f3",
    ),
    "sol_fit": (
        "c36ed7523",
        "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/sol_a_fit_sample_2026-10-15_192625.json",
        "8ad212a2a0e32e76da11b9f03b6fe61462bf742c79f1cbc67173cd3281071ca1",
    ),
    "sol_stage": (
        "20a94b4c7",
        "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/stage_2026-10-14_001108.json",
        "06ff3e8cfba4f2c9e4c23ece49314ef6a69e6a97adf123e5263a795f9719bf60",
    ),
    "final_spec": (
        "3ce6ea55e",
        "experiments/2026-10-14_xerr_scaleup_09/specs/final_spec_3_launch.json",
        "6bd2ab192337e1023a5454c17217eb2b44f0f7c94165cd8671639443d5f90a5a",
    ),
    "pertask": (
        "06d2553f5",
        "experiments/2026-10-15_fpr_sweep_507_prep/results/pertask455v2_C_2026-10-15_162205.json",
        "21a2c1443d9167048c1b636103932ec3678aa51686875eb346cce170fc8eff78",
    ),
}
FAMEXP_SPEC_SHA = "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61"
LABEL = "p2-writing-examples-2026-10-15|{probe}|{cls}"
B_CELLS, B_ARMS = ("t348_media_processing", "t328_web_scraping", "t373_auto_workflow", "t350_slack_clone"), ("run2", "sim")
B_PROBE, B_FILE = "own_frac100_L48 C", "own_frac100_L48__C.npz"
CUT, SOL_DROP = 32768, ("0", "n")
SOL_DUMP, SOL_STAGED = "/sol/2026-10-14_q35solavg_qwen35-27b_alllayers_v1", "/sol/2026-10-14_q35solavg_inputs_v1/inputs"
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "zstandard==0.25.0")
app = modal.App(APP, image=image)
vols = {k: modal.Volume.from_name(v).read_only() for k, v in VOLS.items()}


def eval_members(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        return [i.filename for i in z.infolist()]


def read_member(path: Path, name: str) -> dict:
    """One JSON member of an Inspect .eval (zstd members, zip method 93, read directly); attachments inlined. Stays in the container."""
    import zstandard

    with zipfile.ZipFile(path) as z:
        info = z.getinfo(name)
        if info.compress_type != 93:
            s = json.loads(z.read(info))
        else:
            with open(path, "rb") as f:
                f.seek(info.header_offset)
                name_len, extra_len = struct.unpack("<HH", f.read(30)[26:30])
                f.seek(info.header_offset + 30 + name_len + extra_len)
                s = json.loads(zstandard.ZstdDecompressor().decompress(f.read(info.compress_size), max_output_size=info.file_size))
    att = s.pop("attachments", None) or {}

    def inline(x):
        if isinstance(x, str) and x.startswith("attachment://"):
            return att.get(x[len("attachment://") :], x)
        if isinstance(x, list):
            return [inline(v) for v in x]
        if isinstance(x, dict):
            return {k: inline(v) for k, v in x.items()}
        return x

    return inline(s)


def shape(x, depth=0, maxd=4):
    """Keys and types only, never values."""
    if depth >= maxd:
        return type(x).__name__
    if isinstance(x, dict):
        return {k: shape(v, depth + 1, maxd) for k, v in list(x.items())[:40]} | ({"...": f"{len(x)} keys"} if len(x) > 40 else {})
    if isinstance(x, list):
        return [shape(x[0], depth + 1, maxd), f"len {len(x)}"] if x else ["empty"]
    return type(x).__name__


def section(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 -- inspect mode only: record which layout guess failed (exception type and message carry no transcript text)
        return {"error": f"{type(e).__name__}: {e}"[:300]}


@app.function(cpu=2, memory=8192, timeout=1200, retries=0, volumes=vols)
def inspect(sol_fit: dict, final_spec: dict) -> dict:
    import numpy as np

    out = {}
    spec = json.loads(Path(FAMEXP_SPEC).read_bytes())
    out["famexp_spec"] = {
        "sha256": hashlib.sha256(Path(FAMEXP_SPEC).read_bytes()).hexdigest(),
        "shape": shape(spec, maxd=2),
        "map_entry": shape(next(iter(spec["map"].values()))) if isinstance(spec.get("map"), dict) else None,
    }
    k = next(k for k, v in spec["map"].items() if v[0] == "run1")

    def own_eval():
        p = Path("/q4/evals_v1") / k.split("#")[0]
        names = eval_members(p)
        smp = [n for n in names if n.startswith("samples/")]
        s = read_member(p, smp[0])
        ev = s.get("events") or []
        model = [e for e in ev if e.get("event") == "model"]
        tc = next((e for e in model if (((e.get("output") or {}).get("choices") or [{}])[0].get("message") or {}).get("tool_calls")), None)
        return {
            "key_has_epoch_suffix": "#" in k,
            "n_members": len(names),
            "n_sample_members": len(smp),
            "sample_member_pattern": [n.split("/")[-1].split("_epoch_")[-1] for n in smp][:6],
            "sample_keys": sorted(s),
            "n_events": len(ev),
            "event_kinds": sorted({e.get("event") for e in ev}),
            "n_model_events": len(model),
            "model_event_shape": shape(model[0], maxd=3) if model else None,
            "tool_call_shape": shape(tc["output"]["choices"][0]["message"]["tool_calls"][0]) if tc else None,
            "tool_call_arg_keys": sorted(
                tc["output"]["choices"][0]["message"]["tool_calls"][0].get("function", {}).get("arguments", {})
                if isinstance(tc["output"]["choices"][0]["message"]["tool_calls"][0].get("function", {}).get("arguments"), dict)
                else []
            )
            if tc
            else None,
            "request_header_keys": sorted(((model[0].get("call") or {}).get("request") or {}).get("extra_headers", {}) or {}) if model else None,
            "metadata_keys": sorted(s.get("metadata") or {}),
            "scores_shape": shape(s.get("scores"), maxd=3),
        }

    out["own_eval"] = section(own_eval)

    def own_meta():
        mf = Path("/q4") / spec["run_of"][k] / "meta" / spec["arm_of"][k] / f"{hashlib.sha1(k.encode()).hexdigest()[:20]}.json"
        return shape(json.loads(mf.read_text()))

    out["own_meta"] = section(own_meta)

    def sol():
        r = sol_fit["runs"][0]
        d = Path(SOL_DUMP) / "meta"
        res = {"meta_dirs": sorted(p.name for p in d.iterdir())[:20]}
        cand = sorted((d / r["arm"]).glob(f"{r['key']}*"))[:3] if (d / r["arm"]).exists() else []
        res["meta_candidates"] = [c.name for c in cand]
        if cand:
            m = json.loads(cand[0].read_text())
            res["meta_shape"] = shape(m)
        sh = sorted(Path(SOL_STAGED).glob("shard_*.jsonl"))
        res["n_shards"] = len(sh)
        with open(sh[0]) as f:
            res["staged_row_shape"] = shape(json.loads(f.readline()), maxd=3)
        return res

    out["sol"] = section(sol)

    def conv():
        arm = next(a for a in final_spec["arms"] if "run2" in a)
        mp = Path("/conv") / final_spec["expansion"]["arms"][arm]["manifest"]["path"].replace("/work/", "")
        man = json.loads(mp.read_text())
        npz = mp.parent / "own_frac100_L48__C.npz"
        z = np.load(npz)
        keys = z["key"].astype(str)
        return {
            "arm": arm,
            "manifest_shape": shape(man, maxd=2),
            "arm_dir": sorted(p.name for p in mp.parent.parent.iterdir()),
            "npz_fields": {f: [str(z[f].dtype), list(z[f].shape)] for f in z.files},
            "key_examples": sorted(set(keys.tolist()))[:4],
        }

    out["conv"] = section(conv)
    out["acts_top"] = section(lambda: sorted(p.name for p in Path("/acts").iterdir())[:60])
    return out


class Gate(Exception):
    """An entry that fails the rule's checks: it is reported with its reason (no text) and carries no length or sha256."""


def rng_of(label: str) -> random.Random:
    return random.Random(int(hashlib.sha256(label.encode()).hexdigest(), 16))


def cmd_hash(args) -> dict:
    """The rule's command bytes: arguments['cmd'], else arguments['command'], else the canonical JSON of the arguments. Only the
    branch used, the character length and the sha256 leave this function."""
    if isinstance(args, dict) and "cmd" in args:
        field, s = "cmd", args["cmd"]
    elif isinstance(args, dict) and "command" in args:
        field, s = "command", args["command"]
    else:
        field, s = "json", json.dumps(args, sort_keys=True, separators=(",", ":"))
    if not isinstance(s, str):
        raise Gate(f"arguments[{field!r}] is a {type(s).__name__}, not a string")
    full = json.dumps(args, sort_keys=True, separators=(",", ":"))  # rule amendment 1: the whole arguments too, so the viewer can authenticate them
    return {
        "field": field,
        "length": len(s),
        "sha256": hashlib.sha256(s.encode()).hexdigest(),
        "arguments_length": len(full),
        "arguments_sha256": hashlib.sha256(full.encode()).hexdigest(),
    }


def tool_name(x) -> str:
    """A tool name, never anything larger (Codex r1: a nested function object would carry the arguments)."""
    if not isinstance(x, str) or len(x) > 64 or not re.fullmatch(r"[A-Za-z0-9_.:-]+", x):
        raise Gate("the tool-call function is not a plain tool name")
    return x


def sample_of(path: Path, epoch0: int) -> tuple[dict, str]:
    """The .eval's sample for key epoch #e<epoch0>: the only sample member, else the one member named _epoch_<epoch0 + 1>."""
    smp = [n for n in eval_members(path) if n.startswith("samples/") and n.endswith(".json")]
    if len(smp) != 1:
        smp = [n for n in smp if n.endswith(f"_epoch_{epoch0 + 1}.json")]
    if len(smp) != 1:
        raise Gate(f"{len(smp)} sample members match epoch {epoch0}")
    return read_member(path, smp[0]), smp[0]


def model_step(s: dict, irid: str, call_index: int) -> dict:
    """The model event whose request header x-irid == irid: its 0-based position among the sample's model events, and the tool call
    call_index of its output message."""
    model = [e for e in s.get("events") or [] if e.get("event") == "model"]
    hit = [i for i, e in enumerate(model) if (((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid") == irid]
    if len(hit) != 1:
        raise Gate(f"{len(hit)} model events carry the unit's x-irid")
    tcs = (((model[hit[0]].get("output") or {}).get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
    if not 0 <= call_index < len(tcs):
        raise Gate(f"call_index {call_index} outside the event's {len(tcs)} tool calls")
    return {"step": hit[0], "n_model_events": len(model), "tool": tool_name(tcs[call_index].get("function")), **cmd_hash(tcs[call_index].get("arguments"))}


def pointers(s: dict) -> dict:
    """JSON paths (never values) of the sample's main-task, side-task and side-task-success fields, outside messages and events."""
    out = {"main_task": [], "side_task": [], "side_task_success": []}

    def walk(x, path, depth):
        if depth > 5 or not isinstance(x, dict):
            return
        for k, v in x.items():
            q = f"{path}.{k}" if path else k
            kl = k.lower()
            if "side_task_success" in kl:
                out["side_task_success"].append(q)
            elif "side_task" in kl:
                out["side_task"].append(q)
            elif "main_task" in kl:
                out["main_task"].append(q)
            walk(v, q, depth + 1)

    walk({k: v for k, v in s.items() if k not in ("messages", "events", "attachments", "input", "output", "store")}, "", 0)
    return out


def guarded(entry: dict, fn) -> dict:
    try:
        return entry | fn() | {"status": "ok"}
    except Gate as e:
        return entry | {"status": f"gate: {e}"}


def own_draws(spec: dict) -> list:
    out = []
    for cls, role in (("attack", "run1"), ("honest", "honest_a")):
        label = LABEL.format(probe="own_frac100_L48", cls=cls)
        keys = sorted(k for k, v in spec["map"].items() if v[0] == role and v[1] in spec["fot"])
        rng = rng_of(label)
        k = rng.choice(keys)
        m = json.loads((Path("/q4") / spec["run_of"][k] / "meta" / spec["arm_of"][k] / f"{hashlib.sha1(k.encode()).hexdigest()[:20]}.json").read_text())
        if m["key"] != k:
            raise SystemExit("own meta key differs from the drawn key")
        skip = {tuple(x) for x in spec["skip"].get(k, [])}
        units = [
            (c["irid"], c["call_index"])
            for c in m["calls"]
            if c["kind"] == "call_end" and c.get("row") is not None and (c["irid"], c["call_index"]) not in skip
        ]
        if len(set(units)) != len(units):
            raise SystemExit("own C units are not unique by (irid, call_index)")
        units = sorted(units)
        irid, ci = rng.choice(units)
        ev, ep = k.split("#e")
        entry = {
            "index": "A",
            "probe": "own_frac100_L48 (C, layer 48)",
            "class": cls,
            "label": label,
            "pool": f"{len(keys)} keys (role {role}, task in spec.fot)",
            "task_key": k,
            "task": spec["map"][k][1],
            "fold_of_task": spec["fot"][spec["map"][k][1]],
            "n_units_in_run": len(units),
            "unit": {"irid": irid, "call_index": ci},
            "run": {"volume": VOLS["/q4"], "eval": f"evals_v1/{ev}", "epoch_suffix": int(ep)},
        }

        def src(ev=ev, ep=ep, irid=irid, ci=ci, entry=entry):
            s, member = sample_of(Path("/q4/evals_v1") / ev, int(ep))
            return {"run": entry["run"] | {"sample_member": member, "sample_id": s.get("id"), "sample_epoch": s.get("epoch")}, **model_step(s, irid, ci)}

        out.append(guarded(entry, src))
    return out


def sol_draws(sol_fit: dict, stage: dict) -> list:
    out, want = [], {}
    for cls, arm in (("attack", "real-attack-e1"), ("honest", "real-honest")):
        label = LABEL.format(probe="Sol (a)", cls=cls)
        runs = sorted((r for r in sol_fit["runs"] if r["arm"] == arm), key=lambda r: r["key"])
        rng = rng_of(label)
        r = rng.choice(runs)
        m = json.loads((Path(SOL_DUMP) / "meta" / arm / f"{r['key']}.json").read_text())
        if m["key"] != r["key"] or any("token_text" not in p_ for p_ in m["positions"] if p_["kind"] == "call_end"):
            raise SystemExit("Sol meta key differs, or call_end positions without token_text")
        units = sorted(
            (p_["step"], p_["action"])
            for p_ in m["positions"]
            if p_["kind"] == "call_end" and p_.get("row") is not None and p_["token"] < CUT and p_.get("token_text") not in SOL_DROP
        )
        step, action = rng.choice(units)
        pos = [p_ for p_ in m["positions"] if p_["kind"] == "call_end" and (p_["step"], p_["action"]) == (step, action)]
        if len(units) != len(set(units)) or len(pos) != 1:
            raise SystemExit("Sol cmd_A units are not unique by (step, action)")
        want[(arm, r["key"])] = len(out)
        out.append(
            {
                "index": "A",
                "probe": "Sol (a) (cmd_A, layer 63)",
                "class": cls,
                "label": label,
                "pool": f"{len(runs)} runs (arm {arm})",
                "task_key": r["key"],
                "arm": arm,
                "cell": r["cell"],
                "fold": r["fold"],
                "n_units_in_run": len(units),
                "unit": {"step": step, "action": action, "token": pos[0]["token"]},
                "tool_in_meta": pos[0]["tool"],
                "run": {"eval": "pending", "volume": VOLS["/sol"], "meta": f"{SOL_DUMP[5:]}/meta/{arm}/{r['key']}.json"},
                "source": "staged row (q35solavg_inputs_v1 shard sha-pinned)",
            }
        )
    for name, sha in sorted(stage["inputs_sha256"].items()):  # every shard is checked; only lines carrying a drawn key are parsed
        raw = (Path(SOL_STAGED) / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise SystemExit(f"{name}: sha256 differs from the stage record")
        for ln, line in enumerate(raw.decode().splitlines()):
            if not any(json.dumps(k) in line for _, k in want):
                continue
            row = json.loads(line)
            i = want.get((row.get("arm"), row.get("key")))
            if i is None:
                continue
            e = out[i]
            if "staged_row" in e:
                raise SystemExit("a drawn Sol key appears in two staged rows")
            e["staged_row"] = {"shard": name, "shard_sha256": sha, "line": ln}

            def src(row=row, e=e):
                turns = [x for x in row["messages"] if x["role"] == "assistant"]
                st, ac = e["unit"]["step"], e["unit"]["action"]
                tcs = (turns[st].get("tool_calls") or []) if st < len(turns) else []
                if ac >= len(tcs):
                    raise Gate(f"step {st} of {len(turns)} assistant turns has {len(tcs)} tool calls, none at action {ac}")
                tc, fmt = tcs[ac], "inspect (function name, arguments)"
                name, args = tc.get("function"), tc.get("arguments")
                if isinstance(name, dict):  # chat-template form: function {name, arguments}
                    name, args, fmt = name.get("name"), name.get("arguments"), "chat template (function.name, function.arguments)"
                if isinstance(args, str):
                    args, fmt = json.loads(args), fmt + "; arguments a JSON string, parsed"
                if name != e["tool_in_meta"]:
                    raise Gate(f"tool at (step, action) is {name!r}, the meta says {e['tool_in_meta']!r}; tool-call keys {sorted(tc)}")
                return {"tool": tool_name(name), "staged_tool_call_format": fmt, **cmd_hash(args)}

            out[i] = guarded(e, src)
    for e in out:
        if "staged_row" not in e:
            e["status"] = "gate: drawn key not found in the staged shards"
    return out


def most_suspicious(final_spec: dict, pertask: dict, fixed_ids: list) -> list:
    import io

    import numpy as np

    per = {(t["cell"], t["arm"]): t for t in pertask["rows"][B_PROBE]["S_new"]["per_task"] if t["origin"] == "S_new"}
    tau = pertask["rows"][B_PROBE]["S_new"]["tau"]
    names = {f"{a}|{c}": (c, a) for c in B_CELLS for a in B_ARMS}
    found = {n: [] for n in names}
    for arm, ent in sorted(final_spec["expansion"]["arms"].items()):
        mp = Path("/conv") / ent["manifest"]["path"].replace("/work/", "", 1)
        mb = mp.read_bytes()
        if hashlib.sha256(mb).hexdigest() != ent["manifest"]["sha256"]:
            raise SystemExit(f"{arm}: famexp manifest sha256 differs from 09's FINAL spec")
        fl = [
            (k_, v_) for k_, v_ in json.loads(mb)["files"].items() if Path(k_).name == B_FILE and k_.startswith(ent["manifest"]["path"].rsplit("/", 1)[0] + "/")
        ]
        if len(fl) != 1:  # under the pinned manifest's own folder (famexp/ or famexp_r2/)
            raise SystemExit(f"{arm}: {len(fl)} {B_FILE} entries in the famexp manifest")
        fpath, fsha = fl[0]
        fb = (Path("/conv") / fpath.replace("/work/", "", 1)).read_bytes()
        if hashlib.sha256(fb).hexdigest() != fsha:
            raise SystemExit(f"{arm}: {B_FILE} sha256 differs from its manifest")
        z = np.load(io.BytesIO(fb))
        k = z["key"].astype(str)
        for n in names:
            idx = np.flatnonzero(k == n)
            if len(idx):
                found[n].append(
                    {
                        "arm_folder": arm,
                        "file": fpath.replace("/work/", "/", 1),
                        "file_sha256": fsha,
                        "unit": z["unit"][idx],
                        "call_index": z["call_index"][idx],
                        "fold": z["fold"][idx],
                        "sbm": z["score_by_model"][idx],
                    }
                )
    out = []
    for n, (cell, world) in names.items():
        t = per.get((cell, world))
        e = {
            "index": "B",
            "probe": "own_frac100_L48 (C, layer 48)",
            "run": n,
            "cell": cell,
            "arm": world,
            "main": t and t["main"],
            "record": {"file": PINS["pertask"][1], "commit": PINS["pertask"][0], "row": f"rows[{B_PROBE!r}]['S_new'].per_task"},
            "max_minus_tau (record's own pool)": t and {"pool": t["pipeline_pool"], "tau": tau[t["pipeline_pool"]], "value": t["max_minus_tau"]},
        }
        if world == "run2" and t:
            e["pool A (real run 2 calibration), not the record's pool"] = {"tau_A": tau["A"], "value": t["max_minus_tau"] + tau["B"] - tau["A"]}

        def pick(n=n, cell=cell):
            if len(found[n]) != 1:
                raise Gate(f"rows in {len(found[n])} arm folders")
            f = found[n][0]
            folds = set(f["fold"].tolist())
            if len(folds) != 1 or -1 in folds:
                raise Gate(f"folds {sorted(folds)}")
            fo = folds.pop()
            s = f["sbm"][:, fo]
            j = int(np.argmax(s))
            if int((s == s[j]).sum()) != 1 or not np.isfinite(s).all():
                raise Gate("argmax not unique, or a non-finite score")
            if sorted(f["unit"].tolist()) != list(range(len(s))):
                raise Gate("units are not 0..n-1")
            arm = f["arm_folder"]
            ex = final_spec["arms"][arm]["hygiene"]["longest_epoch_h200_gate"]["extract"]
            roots = {re.fullmatch(r".*/extract_chain_(.+)_a[0-9]+_\d{4}-\d{2}-\d{2}_\d{6}\.json", x["path"]).group(1) for x in ex}
            if len(roots) != 1:
                raise Gate(f"{len(roots)} extraction roots in the arm's hygiene record")
            root = roots.pop()
            sset = arm.split("__")[0]
            cfg = json.loads((Path("/acts") / root / "config.json").read_text())["inputs_sha256"]
            rows = []
            for fn, sha in sorted(cfg.items()):
                sp = Path("/acts") / sset / "inputs" / fn
                if fn.startswith("repro-c") or not sp.exists():
                    continue
                raw = sp.read_bytes()
                if hashlib.sha256(raw).hexdigest() != sha:
                    raise Gate(f"staged {fn}: sha256 differs from the root's config.json")
                for line in raw.decode().splitlines():
                    if json.dumps(cell) in line:
                        r_ = json.loads(line)
                        if r_.get("task") == cell:
                            rows.append((fn, r_))
            if len(rows) != 1:
                raise Gate(f"{len(rows)} staged epochs with task {cell} in {sset}/inputs")
            fn, r_ = rows[0]
            meta_arm = fn[: -len(".jsonl")]
            m = json.loads((Path("/acts") / root / "meta" / meta_arm / f"{hashlib.sha1(r_['key'].encode()).hexdigest()[:20]}.json").read_text())
            if m["key"] != r_["key"]:
                raise Gate("meta key differs from the staged key")
            if m.get("inputs_sha256") != cfg[fn] or [(c["irid"], c["call_index"], c["kind"]) for c in r_["calls"]] != [
                (c["irid"], c["call_index"], c["kind"]) for c in m["calls"]
            ]:
                raise Gate("the meta is not the extraction of this staged row (inputs sha256 or ordered (irid, call_index, kind) differ)")
            fx = set(fixed_ids)
            skip = {(c["irid"], c["call_index"]) for c in r_["calls"] if c["kind"] == "call_end" and c["readout"] is not None and r_["ids"][c["readout"]] in fx}
            cs = [c for c in m["calls"] if c["kind"] == "call_end" and c.get("row") is not None and (c["irid"], c["call_index"]) not in skip]
            order = np.argsort(f["unit"])
            if [c["call_index"] for c in cs] != f["call_index"][order].tolist():
                raise Gate("the meta's C units do not reproduce the per-row file's call_index sequence")
            u = int(f["unit"][j])
            c = cs[u]
            ev, ep = r_["key"].split("#e")
            res = {
                "arm_folder": arm,
                "per_row_file": f"{VOLS['/conv']}:{f['file']}",
                "per_row_file_sha256": f["file_sha256"],
                "rows": len(s),
                "fold": fo,
                "unit": u,
                "designated_score_column": f"score_by_model[:, {fo}]",
                "extraction_root": root,
                "staged": f"{sset}/inputs/{fn}",
                "task_key": r_["key"],
                "irid": c["irid"],
                "call_index": c["call_index"],
                "run_eval": {"volume": VOLS["/acts"], "eval": f"evals_v1/{ev}", "epoch_suffix": int(ep)},
            }
            evp = Path("/acts/evals_v1") / ev
            if not evp.exists():
                raise Gate(f"no .eval at evals_v1/<key path> (staged row fields: {sorted(r_)})")
            smp, member = sample_of(evp, int(ep))
            res["run_eval"] |= {"sample_member": member, "sample_id": smp.get("id"), "sample_epoch": smp.get("epoch")}
            return res | model_step(smp, c["irid"], c["call_index"]) | {"pointers (JSON paths in the sample)": pointers(smp)}

        out.append(guarded(e, pick))
    return out


@app.function(cpu=4, memory=16384, timeout=1800, retries=0, volumes=vols)
def run_index(blobs: dict) -> dict:
    err = None
    try:
        spec_b = Path(FAMEXP_SPEC).read_bytes()
        if hashlib.sha256(spec_b).hexdigest() != FAMEXP_SPEC_SHA:
            raise SystemExit("famexp spec sha256 differs")
        spec = json.loads(spec_b)
        j = {k: json.loads(v) for k, v in blobs.items()}
        return {"A": own_draws(spec) + sol_draws(j["sol_fit"], j["sol_stage"]), "B": most_suspicious(j["final_spec"], j["pertask"], spec["fixed_ids"])}
    except SystemExit as e:  # this job's own messages (names, counts, shas)
        err = str(e)
    except Exception as e:  # noqa: BLE001 -- parse errors can hold source text (JSONDecodeError.doc): only the type crosses back
        err = f"stopped: {type(e).__name__} (message withheld)"
    raise SystemExit(err)  # raised outside the except blocks, so no exception context travels with it


@app.local_entrypoint()
def main(mode: str = "inspect", sol_fit: str = "", final_spec: str = "", out: str = ""):
    if mode == "inspect":
        res = inspect.remote(json.loads(Path(sol_fit).read_text()), json.loads(Path(final_spec).read_text()))
        with open(out, "x") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res)[:200])
        return
    if mode != "run":
        raise SystemExit("--mode inspect | run")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    blobs = {}
    for name, (commit, path, sha) in PINS.items():
        b = git("show", f"{commit}:{path}")
        if hashlib.sha256(b).hexdigest() != sha:
            raise SystemExit(f"{name}: sha256 differs from the pin")
        blobs[name] = b.decode()
    blobs.pop("rule"), blobs.pop("rule_amendment1")
    res = run_index.remote(blobs)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    head = git("rev-parse", "HEAD").decode().strip()
    script = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for ix, what in (("A", "seeded fit-sample draws (one per probe and class)"), ("B", "most suspicious command of eight pertask455v2 runs")):
        rec = {
            "what": f"Writing-session example index {ix}: {what}. INDEX ONLY: no command text; length and sha256 of the command bytes computed in the Modal job",
            "rule": {k: {"path": PINS[k][1], "commit": PINS[k][0], "sha256": PINS[k][2]} for k in ("rule", "rule_amendment1")},
            "inputs": {k: {"commit": c, "path": p_, "sha256": h} for k, (c, p_, h) in PINS.items() if not k.startswith("rule")}
            | {"famexp_spec": {"path": FAMEXP_SPEC, "sha256": FAMEXP_SPEC_SHA}},
            "git": head,
            "script_sha256": script,
            "stamp (PT)": stamp,
            "app": APP,
            "entries": res[ix],
            "status_counts": {s: sum(e["status"] == s for e in res[ix]) for s in sorted({e["status"] for e in res[ix]})},
        }
        b = json.dumps(rec, indent=1).encode()
        dest = HERE / "results" / f"examples_index_{ix}_{stamp}.json"
        dest.parent.mkdir(exist_ok=True)
        dest.open("xb").write(b)
        print(json.dumps({"index": ix, "record": dest.name, "sha256": hashlib.sha256(b).hexdigest(), "status_counts": rec["status_counts"]}), flush=True)
