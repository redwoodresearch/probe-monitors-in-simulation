"""v81 outcome table add-on (the writing session via q2, 10-16 ~15:0x): for the 4 drawn tasks of the per-set outcome table, per arm (real run 1,
sim 1): the side-task success score as the run's own scorer / judge recorded it, plus the main and side task ids and the task metadata's
one-line descriptions. PRELIMINARY; default the per-set draw (record @ c4e57c2efbd8), or --tables <record>@<commit> --tables-sha for another.

  --mode inspect  key names, types and string lengths under each sample's metadata (two levels) and scores (three levels); no values
  --mode run      per arm: the success values at --success-path (short codes / booleans only) and the main / side ids at the id paths, to a
                  keys-only record (exclusive create); the one-liner fields at --main-text / --side-text are returned by the Modal function and
                  written ONLY to --text-out under this session's scratchpad (exclusive create), never printed, logged or committed (q2's one-time
                  exception for these tables)

Every .eval is located by its key under the evals roots and checked against the pinned inventory's (or routing manifest's) sha256 before parsing.
No transcript message, event or command is read: only the sample's `metadata` and `scores` members' fields named on the command line.

Route C (10-17; values off the laptop, 0d via 2e): the tables record (it holds z, T and z - T) is hashed on the laptop and parsed only in the
container (arms_from_record), which also builds the record and the text file; the laptop writes both byte for byte after a sha256 check and
prints paths and shas only (no count: the number of drawn cells is a Table 3 number, 0d 10-17). The laptop parses only K files:
INPUTS_FULL (--inputs-full, default 110447), its cohort inventories and their routing records / .eval manifests (inventory_rows), each
hashed or commit-pinned first. Every drawn outcome cell gives exactly two arms (an outcome with no cell, "none", gives none). --text-out
stays directly in q2's SCRATCH (0d: the text exception keeps its 10-16 scope).

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach experiments/2026-10-16_examples_v81/run_outcome_meta_v81.py --mode inspect
"""

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
sys.path.insert(0, str(HERE))
import run_examples_v81 as X  # noqa: E402  (pinned, SCRATCH)

APP = "probe-shift_aswin-surya_sub-routec-v81_outcome_kill-2026-11-14"  # route C's own description (was q2examples_v81_kill-2026-11-15)
VOLS = {"/q4": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", "/acts": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14",
        "/q3": "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"}
EVAL_ROOTS = ("/q4/evals_v1", "/acts/evals_v1", "/q3/simh_evals_v1")
TABLES = ("c4e57c2efbd8", "experiments/2026-10-16_examples_v81/results/examples_v81_tables_per_origin_2026-10-16_145536.json",
          "f3df4fee4d879042066b19ce3c176ec9dd6e63ed3cc4d4219a23ba88768e7cf3")
INVENTORIES = ("S_new", "S_old", "S_newmains", "S_paper493", "S_paper493_w23")  # INPUTS_FULL's inventories that hold the cohort's run1 / sim arms
ARM_WORLD = {"real (run 1)": "run1", "sim (sim 1)": "sim"}
SHORT = re.compile(r"[A-Za-z0-9_:.\-]{1,80}")  # an id must be a plain identifier (format check only)
MAX_TEXT = 600  # a one-liner longer than this is not returned (its length is)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "zstandard==0.25.0").add_local_file(
    HERE / "run_examples_v81.py", "/root/run_examples_v81.py")
app = modal.App(APP, image=image)
vols = {k: modal.Volume.from_name(v).read_only() for k, v in VOLS.items()}


def shape(x, depth, maxd):
    """Key names, types and string lengths only, never values."""
    if isinstance(x, dict):
        if depth >= maxd:
            return f"dict({len(x)} keys)"
        return {k: shape(v, depth + 1, maxd) for k, v in list(x.items())[:40]}
    if isinstance(x, list):
        return f"list(len {len(x)})"
    if isinstance(x, str):
        return f"str(len {len(x)})"
    return type(x).__name__


def at(d, path: str):
    """The value at a dotted path, or None."""
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def meta_scores(path: Path, epoch0: int) -> dict:
    """Codex m1 #2: a restricted reader. The key's sample member (the only samples/*.json member, else the one named _epoch_<n+1>.json) is
    decompressed and parsed in memory, and ONLY its epoch, metadata and scores are kept; attachment references are resolved inside those two
    fields alone (messages, events and every other field are dropped unread)."""
    import struct
    import zipfile

    import zstandard

    with zipfile.ZipFile(path) as z:
        smp = [i for i in z.infolist() if i.filename.startswith("samples/") and i.filename.endswith(".json")]
        if len(smp) != 1:
            smp = [i for i in smp if i.filename.endswith(f"_epoch_{epoch0 + 1}.json")]
        if len(smp) != 1:
            raise SystemExit(f"{len(smp)} sample members match epoch {epoch0}")
        info = smp[0]
        if info.compress_type != 93:
            raw = z.read(info)
        else:
            with open(path, "rb") as f:
                f.seek(info.header_offset)
                nl, el = struct.unpack("<HH", f.read(30)[26:30])
                f.seek(info.header_offset + 30 + nl + el)
                raw = zstandard.ZstdDecompressor().decompress(f.read(info.compress_size), max_output_size=info.file_size)
    s = json.loads(raw)
    keep = {"epoch": s.get("epoch"), "metadata": s.get("metadata") or {}, "scores": s.get("scores") or {}}
    att = s.get("attachments") or {}
    del s, raw

    def inline(x):
        if isinstance(x, str) and x.startswith("attachment://"):
            return att.get(x[len("attachment://"):], x)
        if isinstance(x, list):
            return [inline(v) for v in x]
        if isinstance(x, dict):
            return {k: inline(v) for k, v in x.items()}
        return x

    out = {"metadata": inline(keep["metadata"]), "scores": inline(keep["scores"])}
    if keep["epoch"] != epoch0 + 1:
        raise SystemExit("the sample's epoch is not the key's")
    return out


def load_sample(e: dict) -> dict:
    """The run's sample: the .eval under EVAL_ROOTS by its key path (with and without runs/), exactly one distinct file with the pinned sha256,
    read from a private snapshot of the verified bytes; the sample member of the key's epoch, its epoch checked."""
    base = e["key"].split("#e")[0].removeprefix("runs/")
    hits = []
    for r in EVAL_ROOTS:
        for rel in (base, "runs/" + base):
            p = Path(r) / rel
            if p.exists():
                b = p.read_bytes()
                if hashlib.sha256(b).hexdigest() == e["eval_sha256"]:
                    hits.append((str(p), b))
    if len({h[0] for h in hits}) != 1:
        raise SystemExit(f"{e['where']}: {len(hits)} .eval files with the pinned sha256")
    snap = Path("/tmp") / f"verified_{e['eval_sha256']}.eval"
    snap.write_bytes(hits[0][1])
    return meta_scores(snap, int(e["key"].rsplit("#e", 1)[1]))


def arms_from_record(tb: bytes, sha: str, rows: list) -> tuple:
    """Container side (the tables record holds z, T and z - T, so it is parsed only here): the arms (each drawn cell x real run 1 / sim 1)
    of the tables record, each resolved to ONE physical key by the laptop's inventory rows (task + world + origin; the record's own physical
    key where it has one, which must agree) and ONE .eval sha256 (every source the row carries must agree). Exactly two arms per drawn outcome
    cell; at least one cell drawn. -> (arms, the record's label and run_label)."""
    if hashlib.sha256(tb).hexdigest() != sha:
        raise SystemExit("the tables record's bytes differ from --tables-sha (container re-check)")
    rec = json.loads(tb)
    if not str(rec.get("what", "")).startswith("v81 threshold tables"):
        raise SystemExit("the tables record was not written by run_tables_v81.py's tables mode")
    out, n_drawn = [], 0
    for oc, d in rec["outcome_table"]["outcomes"].items():
        if d["drawn"] == "none":
            continue
        n_drawn += 1
        origin, cell = d["drawn"]["cell"].split("|", 1)
        for side, world in ARM_WORLD.items():
            match = [r for r in rows if r["task"] == cell and r["world"] == world and (r["origin_any"] or r["origin"] == origin)]
            keys = {r["key"] for r in match}  # Codex m1 #3: every matching row is kept; exactly one physical key; every hash source agrees
            if len(keys) != 1:
                raise SystemExit(f"{oc} {cell} {side}: {len(keys)} physical keys among the inventory runs (task, world, origin)")
            key = keys.pop()
            v = d["drawn"][side]
            if isinstance(v, dict) and v.get("run_key") and v["run_key"] != key:
                raise SystemExit(f"{oc} {cell} {side}: the inventory's physical key differs from the record's")
            have = {h for r in match for h in r["eval_sha256s"]}
            if len(have) != 1:
                raise SystemExit(f"{oc} {cell} {side}: {len(have)} distinct .eval sha256s from the inventory / routing manifest")
            out.append({"where": f"{oc} | {d['drawn']['cell']} | {side}", "key": key, "eval_sha256": have.pop(), "world": world,
                        "shown": "no command" if v == "no command" else "command"})
    if n_drawn == 0 or len(out) != 2 * n_drawn:
        raise SystemExit(f"{len(out)} arms for {n_drawn} drawn outcome cells (two per drawn cell, at least one cell)")
    return out, {"label": rec.get("label"), "run_label": rec.get("run_label")}


def _run(mode: str, p: dict) -> dict:
    arms, lab = arms_from_record(p["tables_bytes"], p["tables"]["sha256"], p["rows"])
    paths, m = p["paths"], p["meta"]
    out = []
    for e in arms:
        s = load_sample(e)
        if mode == "inspect":
            out.append({"where": e["where"], "metadata": shape(s["metadata"], 0, 2), "scores": shape(s["scores"], 0, 3)})
            continue
        succ = at(s, paths["success"])
        ok_s = succ is None or isinstance(succ, bool) or (isinstance(succ, str) and len(succ) <= 3) or isinstance(succ, int | float)
        ids = {k: at(s, paths[k]) for k in ("main_id", "side_id")}
        rec = {"where": e["where"], "key": e["key"], "eval_sha256": e["eval_sha256"],
               "side_task_success": succ if ok_s else "refused: not a short score code", "side_task_success_field": paths["success"]}
        for k, v in ids.items():
            rec[k] = v if isinstance(v, str) and SHORT.fullmatch(v) else ("absent" if v is None else "refused: not a plain identifier")
            rec[k + "_field"] = paths[k]
        texts = {}
        for k in ("main_text", "side_text"):
            v = at(s, paths[k])
            texts[k] = (v if isinstance(v, str) and len(v) <= MAX_TEXT else
                        {"not returned": "absent" if v is None else (f"{len(v)} chars > {MAX_TEXT}" if isinstance(v, str) else type(v).__name__)})
        out.append({"keys_only": rec, "text": texts | {"where": e["where"], "fields": {k: paths[k] for k in ("main_text", "side_text")}}})
    tb_ = p["tables"]
    if mode == "inspect":
        rec = {"what": "key names, types and string lengths only (no values)", "tables": [tb_["commit"], tb_["path"], tb_["sha256"]], "arms": arms,
               "git": m["git"], "inputs_full": m["inputs_full"], "tables_label": lab, "rows": out}
        text = None
    else:
        rec = {"what": "v81 outcome table add-on: side-task success per arm and the task ids (keys only; no text), for the tables record named below. Preliminary.",
               "tables": tb_, "tables_label": lab, "git": m["git"], "script_sha256": m["script_sha256"], "stamp (PT)": m["stamp"], "app": APP,
               "inputs_full": m["inputs_full"], "fields": paths, "rows": [r["keys_only"] for r in out]}
        text = json.dumps([r["text"] for r in out], indent=1)
    b = json.dumps(json.loads(json.dumps(rec, allow_nan=False)), indent=1)
    return {"record_text": b, "sha256": hashlib.sha256(b.encode()).hexdigest(), "text": text,
            "text_sha256": None if text is None else hashlib.sha256(text.encode()).hexdigest()}  # no count leaves the container (0d, 10-17)


@app.function(cpu=1, memory=8192, timeout=900, retries=0, volumes=vols)
def job(mode: str, payload: dict) -> dict:
    err = None
    try:
        return _run(mode, payload)
    except SystemExit as x:
        err = str(x)
    except Exception as x:  # noqa: BLE001 -- parse errors can hold source text: only the type crosses back
        err = f"stopped: {type(x).__name__} (message withheld)"
    raise SystemExit(err)


def git(*a) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout


def inventory_rows(inputs_full: tuple) -> list:
    """Laptop side, K files only: the run 1 / sim rows of INPUTS_FULL's cohort inventories (key, task, world, origin) with every .eval sha256
    source each row has (the inventory's own; the routing record's .eval manifest, by key). INPUTS_FULL and each inventory are hashed against
    their pins before the parse; routing records and manifests are read at their pinned commits."""
    inf = json.loads(X.pinned_at(*inputs_full, name="inputs_full"))
    rows = []
    for k in INVENTORIES:
        e = inf["inventories"][k]
        b = git("show", f"{e['commit']}:{e['path']}")
        if hashlib.sha256(b).hexdigest() != e["sha256"]:
            raise SystemExit(f"inventory {k}: sha256 differs from INPUTS_FULL")
        rows += [r for r in json.loads(b).get("runs") or [] if isinstance(r, dict) and r.get("world") in ARM_WORLD.values()]
    man = {}

    def manifest_sha(r):
        rt = (r.get("uncut_planned") or {}).get("routing") or ""
        if " @ " not in rt:
            return None
        if rt not in man:
            rp, rc = rt.split(" @ ")
            mm = json.loads(git("show", f"{rc}:{rp}")).get("manifest") or ""
            mp, mc = mm.split(" @ ") if " @ " in mm else ("", "")
            man[rt] = {f["path"].removeprefix("runs/"): f["sha256"] for f in json.loads(git("show", f"{mc}:{mp}")).get("files", [])} if mp else {}
        return man[rt].get(r["key"].split("#e")[0].removeprefix("runs/"))

    return [{"key": r["key"], "task": r.get("task"), "world": r["world"], "origin": r.get("origin"), "origin_any": "origin" not in r,
             "eval_sha256s": sorted({h for h in (r.get("eval_sha256"), manifest_sha(r)) if h})} for r in rows]


def pin_arg(arg: str, sha: str, default: tuple) -> tuple:
    """<path>@<commit> + its sha256 -> (commit, path, sha256); neither -> default; one without the other STOPs."""
    if bool(arg) != bool(sha):
        raise SystemExit("a <path>@<commit> pin and its sha256 go together")
    return (arg.rsplit("@", 1)[1], arg.rsplit("@", 1)[0], sha) if arg else default


@app.local_entrypoint()
def main(mode: str = "", success_path: str = "", main_id: str = "", side_id: str = "", main_text: str = "", side_text: str = "", text_out: str = "",
         tables: str = "", tables_sha: str = "", inputs_full: str = "", inputs_full_sha: str = ""):
    """--tables <path>@<commit> --tables-sha <sha256>: another committed tables record of run_tables_v81.py (e.g. the pooled draw); default the
    per-set record pinned in TABLES. --inputs-full <path>@<commit> --inputs-full-sha <sha256>: route C's FROZEN file; default 110447."""
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    if mode not in ("inspect", "run"):
        raise SystemExit("--mode inspect | run")
    tb_ = pin_arg(tables, tables_sha, TABLES)
    if_ = pin_arg(inputs_full, inputs_full_sha, X.PINS["inputs_full"])
    tb_bytes = X.pinned_at(*tb_, name="tables")  # hashed only; arms_from_record parses it in the container
    paths = {}
    if mode == "run":
        paths = {"success": success_path, "main_id": main_id, "side_id": side_id, "main_text": main_text, "side_text": side_text}
        if not all(re.fullmatch(r"(metadata|scores)(\.[A-Za-z0-9_]+){1,4}", v or "") for v in paths.values()):
            raise SystemExit("every field path must be metadata.<...> or scores.<...> (1-4 dotted names)")
        out_ = Path(text_out).resolve()
        sd_ = X.SCRATCH.resolve()
        chk = subprocess.run(["git", "-C", str(sd_), "rev-parse", "--git-dir"], capture_output=True, text=True)
        not_repo = chk.returncode == 128 and "not a git repository" in chk.stderr
        if out_.parent != sd_ or not not_repo or out_.exists():  # Codex m1 #1 / m2: directly in the scratchpad root, confirmed not a git repository
            raise SystemExit("--text-out must be a new file directly in this session's scratchpad root, which must not be a git repository")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / (f"outcome_meta_inspect_{stamp}.json" if mode == "inspect" else f"outcome_meta_{stamp}.json")
    if dest.exists():
        raise SystemExit(f"{dest.name} exists")
    meta = {"git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "stamp": stamp,
            "inputs_full": {"commit": if_[0], "path": if_[1], "sha256": if_[2]}}
    got = job.remote(mode, {"tables_bytes": tb_bytes, "tables": {"commit": tb_[0], "path": tb_[1], "sha256": tb_[2]}, "rows": inventory_rows(if_),
                            "paths": paths, "meta": meta})
    b = got["record_text"].encode()
    if hashlib.sha256(b).hexdigest() != got["sha256"] or (got["text"] is not None and hashlib.sha256(got["text"].encode()).hexdigest() != got["text_sha256"]):
        raise SystemExit("the returned bytes differ from the container's sha256: nothing written")
    if mode == "run":
        with out_.open("x") as fh:  # the only place the one-liners are written; never printed
            fh.write(got["text"])
    with dest.open("xb") as fh:  # the container's record bytes, never parsed here
        fh.write(b)
    print(json.dumps({"record": str(dest.relative_to(REPO)), "sha256": got["sha256"]} | ({"text_file": str(out_)} if mode == "run" else {})), flush=True)
