"""Route C slot readout (0d via 2e, 10-17): ONE Modal CPU function reads pooled records BY SHA and returns writing's slot strings by key
(slot_table.build), plus the input shas and the code identity. No raw arrays leave the container, and nothing on the laptop parses a
record: the laptop sends each input's pointer (a Volume path, or a git blob's bytes unparsed), receives {slot_key: str} and writes it to an
exclusive-create JSON file. It prints keys and counts only.

Inputs (each optional; slot_table.build lists what an absent role leaves uncovered):
  --a               pooled Stage A record                         vol:<sol|figs>:<path>@<sha256>  or  git:<commit>:<path>@<sha256>
  --a-group-catch   run_group_catch.py's record on that A         (tied to A's sha in the container)
  --a-group-excess  run_group_excess.py's record on that A        (tied to A's sha in the container)
  --b               pooled B (sweep15) record
  --r37             pooled R37 record
  --kappa, --kappa-extra, --detdrop, --anchor, --plant   the downstream records (10-17; slots 21-25)
  --labels-171pool  the committed 171-pool registry (09's one label rule, 10-17): a record whose sha is in it takes its label
  --<role>-label    a route C label sidecar {record, record_sha256, label} for a record not in the registry (any record role);
                    a record in neither STOPs (slot_table.record_label)
  --control         a CONTROL run (Aswin 10-17): only then may an appendix-bound record (kappa, kappa extra, detector_vs_drop, anchor,
                    PLANT, B, R37) carry a 171-pool registry label; --out must contain CONTROL_ONLY, and only then
A path ending in .gz is decompressed in the container and its sha256 is the DECOMPRESSED bytes' (as the committed records cite).

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-17_routec_slot_readout/run_slot_readout.py \
      --a vol:sol:<path>@<sha256> [--a-group-catch ...] [--a-group-excess ...] [--b ...] [--r37 ...] --out <new .json path>
"""

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import modal

import slot_table as ST  # beside this file locally, at /root in the container (the image adds it there)

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-routec-slots_read_kill-2026-11-14"
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").add_local_file(HERE / "slot_table.py", "/root/slot_table.py")
VOLS = {"sol": modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2),
        "figs": modal.Volume.from_name("probe-shift_aswin-surya_sub-render_final-figs_kill-2026-11-13")}
ROLES = {"a": "A", "a_group_catch": "A_group_catch", "a_group_excess": "A_group_excess", "b": "B", "r37": "R37",
         "kappa": "KAPPA", "kappa_extra": "KAPPA_EXTRA", "detdrop": "DETDROP", "anchor": "ANCHOR", "plant": "PLANT",  # downstream (10-17)
         "labels_171pool": "LABELS_171",  # the committed 171-pool registry (09's one label rule, 10-17)
         **{f"{k}_label": f"{v}_label" for k, v in (("a", "A"), ("a_group_catch", "A_group_catch"), ("a_group_excess", "A_group_excess"),
            ("b", "B"), ("r37", "R37"), ("kappa", "KAPPA"), ("kappa_extra", "KAPPA_EXTRA"), ("detdrop", "DETDROP"), ("anchor", "ANCHOR"),
            ("plant", "PLANT"))}}  # route C label sidecars (09)
CODE = ("run_slot_readout.py", "slot_table.py")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def check_code(code, read):
    """The code that runs is the code the laptop pinned (rule 27: both files at /root); read(name) -> bytes. STOP on any difference."""
    for n_ in CODE:
        if sha(read(n_)) != code["sha256"][n_]:
            raise SystemExit(f"{n_}: the container's copy is not the pinned code (STOP)")


def load_inputs(inputs, read):
    """Every input's sha256 is checked BEFORE it is parsed (fail closed); read(pointer) -> bytes. -> (records, input records)."""
    recs, ins = {}, {}
    for role, x in inputs.items():
        b = read(x)
        if x["path"].endswith(".gz"):
            b = gzip.decompress(b)
        if sha(b) != x["sha256"]:
            raise SystemExit(f"{role}: sha256 differs from its pin (STOP)")
        recs[role] = json.loads(b)
        ins[role] = {k_: x[k_] for k_ in ("src", "vol", "commit", "path", "sha256") if k_ in x} | {"bytes": len(b)}
    return recs, ins


OUT_KEYS = {"slots", "full", "missing", "not_carried", "inputs", "code"}


def assemble(built, ins, code):
    """The container's reply: EXACTLY build()'s four string maps plus the input records and the code identity; any other key STOPs
    (a63478 REQUIRED-1: no parsed record or raw value leaves the container)."""
    res = built | {"inputs": ins, "code": code | {"checked_in_container": list(CODE)}}
    if set(res) != OUT_KEYS:
        raise SystemExit(f"the reply's keys are {sorted(res)}, not {sorted(OUT_KEYS)} (STOP)")
    return res


@app.function(image=image, cpu=2, memory=(4096, 16384), timeout=900, retries=0, volumes={"/sol": VOLS["sol"], "/figs": VOLS["figs"]})
def readout(inputs: dict, code: dict, control: bool) -> dict:
    check_code(code, lambda n_: Path("/root", n_).read_bytes())
    recs, ins = load_inputs(inputs, lambda x: Path("/" + x["vol"], x["path"]).read_bytes() if x["src"] == "vol" else x["bytes"])
    return assemble(ST.build(recs, {r: ins[r]["sha256"] for r in ins}, {r: ins[r]["path"] for r in ins}, control), ins, code)


def check_out(out, control):
    """Aswin's ruling (10-17, via 0d and 2e): a CONTROL run's output file name carries CONTROL_ONLY, and only a CONTROL run's does."""
    if control != ("CONTROL_ONLY" in Path(out).name):
        sys.exit("STOP: --control and an output file name containing CONTROL_ONLY go together")


def check_mode(res, control):
    """a63478 O1: the reply's mode, in slots and in full, is the one this run asked for; else STOP before anything is written."""
    want = ST.CONTROL_MODE if control else "writing"
    if res["slots"].get("mode") != want or res["full"].get("mode") != want:
        sys.exit(f"STOP: the reply's mode is not the {'CONTROL_ONLY' if control else 'writing'} mode this run asked for")


def git(*a):
    return subprocess.run(["git", *a], cwd=HERE, capture_output=True, check=True).stdout


def code_pins(git_):
    """Laptop side: the code sent is the committed code. For each file, git's HEAD blob must equal hash-object of the working copy, else
    STOP; -> {"git": HEAD, "sha256": {...}, "blob": {...}}. git_(*args) -> bytes (a fake in the test)."""
    code = {"git": git_("rev-parse", "HEAD").decode().strip(), "sha256": {}, "blob": {}}
    for n_ in CODE:
        blob = git_("rev-parse", f"HEAD:./{n_}").decode().strip()  # relative to this folder (git's cwd)
        if git_("hash-object", str(HERE / n_)).decode().strip() != blob:
            raise SystemExit(f"STOP: {n_} differs from HEAD")
        code["sha256"][n_], code["blob"][n_] = sha((HERE / n_).read_bytes()), blob
    return code


def pointer(arg):
    """vol:<sol|figs>:<path>@<sha256> -> read in the container; git:<commit>:<path>@<sha256> -> the blob's bytes, sent unparsed."""
    spec, _, h = arg.rpartition("@")
    src, _, rest = spec.partition(":")
    if len(h) != 64 or src not in ("vol", "git"):
        sys.exit(f"STOP: {arg!r} is not vol:<sol|figs>:<path>@<sha256> or git:<commit>:<path>@<sha256>")
    loc, _, path = rest.partition(":")
    if src == "vol":
        if loc not in VOLS or not path:
            sys.exit(f"STOP: {arg!r}: unknown Volume")
        return {"src": "vol", "vol": loc, "path": path, "sha256": h}
    return {"src": "git", "commit": loc, "path": path, "sha256": h, "bytes": git("show", f"{loc}:{path}")}


@app.local_entrypoint()
def main(a: str = "", a_group_catch: str = "", a_group_excess: str = "", b: str = "", r37: str = "", kappa: str = "", kappa_extra: str = "",
         detdrop: str = "", anchor: str = "", plant: str = "", labels_171pool: str = "", a_label: str = "", a_group_catch_label: str = "",
         a_group_excess_label: str = "", b_label: str = "", r37_label: str = "", kappa_label: str = "", kappa_extra_label: str = "",
         detdrop_label: str = "", anchor_label: str = "", plant_label: str = "", control: bool = False, out: str = ""):
    if git("status", "--porcelain", "--untracked-files=no").strip():
        sys.exit("STOP: uncommitted tracked changes")
    code = code_pins(git)  # the files sent are the committed ones
    args = {"a": a, "a_group_catch": a_group_catch, "a_group_excess": a_group_excess, "b": b, "r37": r37, "kappa": kappa, "kappa_extra": kappa_extra,
            "detdrop": detdrop, "anchor": anchor, "plant": plant, "labels_171pool": labels_171pool, "a_label": a_label,
            "a_group_catch_label": a_group_catch_label, "a_group_excess_label": a_group_excess_label, "b_label": b_label, "r37_label": r37_label,
            "kappa_label": kappa_label, "kappa_extra_label": kappa_extra_label, "detdrop_label": detdrop_label, "anchor_label": anchor_label,
            "plant_label": plant_label}
    inputs = {ROLES[k]: pointer(v) for k, v in args.items() if v}
    if not inputs or not out or Path(out).exists():
        sys.exit("STOP: give at least one input and a new --out path")
    check_out(out, control)
    Path(out).parent.mkdir(parents=True, exist_ok=True)  # a0b05 R-2: the output's folder exists before the paid call, never after
    res = readout.remote(inputs, code, control)
    check_mode(res, control)
    blob_ = json.dumps(res, indent=1, ensure_ascii=False).encode()
    with open(out, "xb") as f:
        f.write(blob_)
    print(json.dumps({"out": out, "sha256": sha(blob_), "slots": len(res["slots"]), "missing": sorted(res["missing"]),
                      "not_carried": sorted(res["not_carried"]), "control": control, "mode": res["slots"]["mode"], "inputs": {r: v["sha256"] for r, v in res["inputs"].items()}, "code": code}))
    if res["missing"]:
        sys.exit(f"STOP: {len(res['missing'])} slots missing (listed above by key; the file holds where each path broke)")
