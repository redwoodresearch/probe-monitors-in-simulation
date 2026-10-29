"""nm172 PROVENANCE (09's GO item 1, via 2e 17:4x-17:5x; CUT glance R1-R5 taken 18:3x; NOT RUN until CUT's re-glance): which transcript produced
the probe units that the real honest_a_e2|nm172_shared_dotfiles and the sim simh_nm360|nm172_shared_dotfiles share (the pair check found their
.eval messages differ, cause (ii)), and by what mechanism. KEYS, IDENTIFIERS, COUNTS, BOOLEANS AND HASHES ONLY: no score, transcript, message, tool
or attack text is printed or written; every staged or meta content field is reported as its type, length and sha256.

Scope: every nm172_shared_dotfiles run (honest_a / honest_b epochs 1 and 2, sim honest, attack run1 / run2 / sim), both lines (32k and uncut).
1. Score files: every .npz on the conv, p2 and solavg Volumes (activation dirs skipped) whose `key` array holds a row of cell nm172_shared_dotfiles
   (tags _r<k> / _e2a / _e2b / _redraw<k> allowed). Per (file, key): the file sha256, units, the sha256 of the key's rows of every per-row array
   (canonical bytes as pw_canon) and of `score` alone; two labels: the folder-regex one (a path naming ep2 -> <world>_e2) and the builder's (the
   E12 block's folder -> world_e2, by folder name, its 32k original name, or file path), the builder's used when it exists (R5).
2. Each such folder's manifest*.json: provenance leaves only (staged files and shas, extraction run, volume, sources, worlds), its original name
   and its inventory entries for nm172 (eval key -> folder, original folder, count).
3. Extraction: every dir holding meta/ at depth 1-3 on both activation Volumes (R3: nested run dirs included; the cutfull routing records'
   extraction roots are cross-checked against the walk): every meta/<arm>/*.json whose bytes name nm172, with its identifier fields, its mtime (PT),
   the length and sha256 of its `calls`, and the sha256 of its activation rows at L48 (by_layer/L48/<chunk>.npy[row_start:row_start+n_rows]); the
   root's run-level jsons, identifier leaves only. Control per root: two other runs' rows must hash differently from each other.
4. Staging: the staged root beside each such extraction root (_extract dropped) and every staged file a folder manifest names: the file sha256 and
   each nm172 line's identifier fields plus the type, length and sha256 of every other field (`ids`: the rendered-input hash). A meta's OWN staged
   row is the line with its key in the staged file whose sha256 is its inputs_sha256.
5. Source: each staged / meta key "<launch>/<rel>#e<k>" -> the sha256 of evals_v1/<launch>/<rel> on the 32k Volume, against the frozen inventory
   (507-prep inventory_b_fix_v1, sha256 4a5d0fb3...) and the E12 block (sha256 683d4de1...).
Then: groups sharing units, activation rows or staged fields; per run label a clean flag; and 09's A / B / C facts. The SOURCE of rows that a real-key
and a sim-key meta share byte-for-byte within one line (R4) is named only if ALL hold: exactly one of the two runs' own extractions in the OTHER line
is close to the shared rows (every row within 5%); in that other line each run's own meta n_tokens equals its own staged row's `ids` length and its
n_rows or `calls` length equals its own staged row's `calls` length; and the two-other-runs control holds in every root involved. Else
"undetermined". Per key, whether its staged `ids` hash is equal across the two lines ("close not expected (different rendered inputs)" if not: the
uncut re-stage used the fixed chat template and the 32k line is cut). Within the shared line, which run's staged row each shared meta's counts
match is reported beside, as evidence only. Unreadable inputs (a score file with nm172 rows, an nm172 meta, an activation pair) are listed and
push the reading to "undetermined" (CUT R2): the source rule, the A / B / C reading (checked first) and the clean flag of any label they may name.
units_sha256 hashes all numeric per-row arrays of the RAW file, not gate 2's post-loader s / f / sbm; per-array hashes sit beside it (CUT R1).
CPU only, every function <= 2,400 s; expected ~10-20 min and ~$1-3.
Usage: modal run --detach check_nm172_provenance.py --inventory <json> --e12 <json> --routing <dir of routing jsons> --out <new local json>"""
import hashlib
import json
import re

import modal

VOLS = {"/conv": ("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", None), "/p2": ("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", 2),
        "/sol": ("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", 2), "/a32": ("probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", None),
        "/aU": ("probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14", None)}
MOUNT_OF = {v[0]: m for m, v in VOLS.items()}
LINE = {"/a32": "32k", "/aU": "uncut"}
INV_SHA = "4a5d0fb3cad32740f849df5def3c7a03a40a18aa4c42bc360709f0c740973d03"
E12_SHA = "683d4de1040f489a768a44fe5ad53f787d35d8ca22117b897a42a75880e1f4dc"
REAL, SIM = "honest_a_e2|nm172_shared_dotfiles", "simh_nm360|nm172_shared_dotfiles"
NEEDLE = b"nm172_shared_dotfiles"
CELLRX = re.compile(r"^nm172_shared_dotfiles(_r\d+|_e2[ab]|_redraw\d*)*$")
EP2RX = re.compile(r"(^|[/_-])ep2([/_-]|$)")
SKIP_DIRS = {"by_layer", "chunks", "parts", "meta", "shard_results", "evals_v1", "inputs"}
IDF = {"key", "run", "task", "epoch", "arm", "meta_arm", "world", "launch", "attempt_index", "attempt_kind", "redrawn_because", "policy_called", "chunk",
       "row_start", "n_rows", "inputs_sha256", "eval", "eval_rel", "eval_sha256", "sample_id", "sample_index", "id", "uuid", "source", "n_tokens", "cut",
       "full_len", "tokens_before", "launch_task", "arm_dir", "origin", "world_e2", "app", "app_id", "call_id", "function_call_id", "started", "finished",
       "created", "timestamp", "time", "git", "commit", "model", "revision", "layer", "layers", "dtype", "status", "attempt", "tokens_total",
       "max_position", "padded_tokens", "truncated"}
IDV = re.compile(r"[A-Za-z0-9_.:/@#|=+-]{0,300}")
# a folder manifest's provenance leaves (paths): staged files and shas, extraction run, volume, sources, worlds; never counts, gates or values
PROVP = re.compile(r"^(app|folder|corrected_b_manifest|files\.|blobs_at_launch\.|arm\.(worlds|epochs_per_run|n_epochs)|dump\.(run|volume|world|world_map|"
                   r"d9_arm_to_world|folders|dump_manifest_sha256|out_dir|line|mode)\b)|staged|extract|routing|source|original|volume|inputs", re.I)
TAGS = re.compile(r"(_e2[ab]|_redraw[0-9]+)+$")  # the builder's epoch-2 renaming strips these from the cell
EVKEY = re.compile(rb"[\w.@:-]+/[\w.@:-]+/nm172_shared_dotfiles[\w-]*/[\w.:-]+\.eval#e\d+")  # an nm172 eval key inside an unreadable meta's bytes
UNITS_SCOPE = "all numeric per-row arrays of the raw file (not gate 2's post-loader s/f/sbm); per-array hashes in array_sha256"
T = 2400  # R1: every function, the driver included, ends by 40 min
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.1.3")
app = modal.App("probe-shift_aswin-surya_sub-e2esim_nm172prov_kill-2026-11-16")
mounts = {m: (modal.Volume.from_name(n, version=v) if v else modal.Volume.from_name(n)).read_only() for m, (n, v) in VOLS.items()}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def idlike(v):
    return v is None or isinstance(v, (bool, int, float)) or (isinstance(v, str) and bool(IDV.fullmatch(v)))


def canon(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def summ(d):  # -> identifier fields kept (allowlisted names with identifier-like values), every other field as type, length and sha256 only
    ids, other = {}, {}
    for k, v in (d.items() if isinstance(d, dict) else []):
        if k in IDF and idlike(v):
            ids[k] = v
        else:
            other[k] = {"type": type(v).__name__, "len": len(v) if hasattr(v, "__len__") else None, "sha256": sha(canon(v).encode())}
    return ids, other


def id_leaves(x, path="", out=None):  # identifier-like leaves (strings without whitespace, ints) of a json; counts/gates/values never
    out = {} if out is None else out
    if isinstance(x, dict):
        for k, v in x.items():
            if str(k).startswith(("counts", "gate", "run_means", "auroc", "score", "thresh", "tau", "mean", "stats")):
                continue
            id_leaves(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(x, list):
        for i, v in enumerate(x[:200]):
            id_leaves(v, f"{path}[{i}]", out)
    elif isinstance(x, (str, int)) and not isinstance(x, bool) and idlike(x):
        out[path] = x
    return out


def unit_bytes(arrs):  # pw_canon: floats float64, one NaN, -0 as 0; integers int64; each array prefixed by its shape
    import numpy as np

    parts = []
    for a_ in arrs:
        a_ = np.asarray(a_)
        if a_.dtype.kind == "f":
            a_ = a_.astype(np.float64)
            a_ = np.where(np.isnan(a_), np.nan, a_ + 0.0)
        elif a_.dtype.kind in "iub":
            a_ = a_.astype(np.int64)
        else:
            raise AssertionError(f"unexpected dtype {a_.dtype}")
        parts.append(repr(a_.shape).encode() + b":" + np.ascontiguousarray(a_).tobytes())
    return b"|".join(parts)


def counts_match(m, row):  # a meta's counts against a staged row: n_tokens = len(ids); n_rows or len(calls) = the row's len(calls)
    if row is None:
        return None
    ids_, calls_ = row["other_fields"].get("ids"), row["other_fields"].get("calls")
    return bool(ids_ and calls_ and m.get("n_tokens") == ids_["len"] and calls_["len"] in (m.get("n_rows"), m.get("calls_len")))


def abc_of(src, score_files_equal, n_lost, n_meta_err, n_cp_err):  # 09's reading; unreadable inputs come first (CUT R2b)
    if n_lost or n_meta_err or n_cp_err:
        return (f"undetermined (unreadable inputs: {n_lost} score files with lost nm172 rows, {n_meta_err} nm172 metas unreadable, "
                f"{n_cp_err} activation pairs unreadable)")
    srcs = {v["source"] for v in src.values()}
    if len(srcs) == 1 and "undetermined" not in srcs:
        return "A: the shared rows are the " + next(iter(srcs)) + " run's own; the other key carries a copy"
    return ("B: cannot tell which transcript produced the shared units" if score_files_equal else
            "no shared units in the score files on the Volumes: the identity arises downstream (the build)")


def unclean_labels(labels, lab_keys, lost, meta_errs):
    """CUT R2c: labels that a lost score-file hit or an unreadable nm172 meta may name. lost: [{path, lost_hit_keys}] (raw <world>|<cell> keys);
    meta_errs: [{names_nm172, keys_in_bytes}]; lab_keys: label -> eval keys. A meta error naming nm172 with no key readable marks every label."""
    out = set()
    for e in lost:
        for x in e.get("lost_hit_keys") or []:
            w, c = x.split("|", 1)
            out |= {lab for lab in labels if lab.split("|", 1)[1] == TAGS.sub("", c) and lab.split("|", 1)[0].removesuffix("_e2") == w.removesuffix("_e2")}
    for e in meta_errs:
        if e.get("names_nm172"):
            ks = set(e.get("keys_in_bytes") or [])
            out |= set(labels) if not ks else {lab for lab in labels if ks & set(lab_keys.get(lab, ()))}
    return out


def source_rule(am, cp, rk, sk, own_row, ctl, n_nm172_meta_err=0):
    """R4. am: nm172 metas {mount, root, meta, key, n_rows, n_tokens, calls_len, inputs_sha256}; cp: close_pairs rows; rk / sk: the real / sim eval
    keys; own_row: meta -> its own staged row (or None); ctl: root -> the two-other-runs control; n_nm172_meta_err: unreadable metas naming nm172 on
    either line (CUT R2a: any such, or an unreadable pair touching the metas involved, leaves the source undetermined). -> (lines with byte-equal
    real/sim rows, close (real line, sim line) pairs, per shared line the facts and the source)."""
    by = {m["meta"]: m for m in am}

    def rel(kind):
        out = set()
        for x in cp:
            a, b = by[x["a"]], by[x["b"]]
            if x.get(kind) and a["key"] in rk and b["key"] in sk:
                out.add((LINE[a["mount"]], LINE[b["mount"]]))
            elif x.get(kind) and b["key"] in rk and a["key"] in sk:
                out.add((LINE[b["mount"]], LINE[a["mount"]]))
        return sorted(out)

    shared_lines, src = sorted({a for a, b in rel("exact") if a == b}), {}
    for ln in shared_lines:
        oth = [k_ for k_, v_ in LINE.items() if v_ != ln][0]
        shared = {m["meta"] for m in am if LINE[m["mount"]] == ln and m["key"] in rk | sk}
        own = {who: {m["meta"] for m in am if m["mount"] == oth and m["key"] in ks} for who, ks in (("real", rk), ("sim", sk))}
        cl = {who: any(x.get("close") for x in cp if ({x["a"], x["b"]} & shared) and ({x["a"], x["b"]} & o_)) for who, o_ in own.items()}
        own_ok = {who: bool(o_) and all(counts_match(by[x], own_row.get(x)) for x in o_) for who, o_ in own.items()}
        roots = {by[x]["mount"] + ":" + by[x]["root"] for x in shared | own["real"] | own["sim"]}
        ctl_ok = all(ctl.get(r) is True for r in roots)
        ids_eq = {}
        for who, ks in (("real", rk), ("sim", sk)):
            h_ = {(LINE[by[x]["mount"]], (own_row.get(x) or {}).get("other_fields", {}).get("ids", {}).get("sha256")) for x in shared | own[who] if by[x]["key"] in ks}
            l_ = {}
            for line_, h in h_:
                l_.setdefault(line_, set()).add(h)
            eq = None if len(l_) < 2 or None in set().union(*l_.values()) else len(set().union(*l_.values())) == 1
            ids_eq[who] = {"equal": eq, "note": "close not expected (different rendered inputs)" if eq is False else None}
        one = [w for w in ("real", "sim") if cl[w]]
        involved = shared | own["real"] | own["sim"]
        pair_err = any("error" in x for x in cp if {x["a"], x["b"]} & involved)
        ok = len(one) == 1 and own_ok["real"] and own_ok["sim"] and ctl_ok and not pair_err and not n_nm172_meta_err
        within = {by[x]["key"] + " @ " + x: {who: counts_match(by[x], next((own_row.get(y) for y in shared if by[y]["key"] in ks), None))
                                             for who, ks in (("real", rk), ("sim", sk))} for x in sorted(shared)}
        src[ln] = {"other_line": LINE[oth], "own_metas_other_line": {k_: sorted(v_) for k_, v_ in own.items()}, "close_to_shared": cl,
                   "own_counts_match_own_staged_row": own_ok, "controls_hold_in_every_root": ctl_ok, "roots": sorted(roots),
                   "unreadable_pair_touching_these_metas": pair_err, "nm172_metas_unreadable_on_either_line": n_nm172_meta_err,
                   "staged_ids_equal_across_lines": ids_eq, "source": one[0] if ok else "undetermined",
                   "within_line_counts_match (evidence only)": within}
    return shared_lines, rel("close"), src


@app.function(image=image, cpu=1, memory=2048, timeout=T, retries=0, volumes=mounts)
def walk(mount: str, top: str) -> dict:
    import os

    npz, mans, stack = [], [], [f"{mount}/{top}"]
    while stack:
        d = stack.pop()
        for e in os.scandir(d):
            if e.is_dir(follow_symlinks=False):
                if e.name not in SKIP_DIRS and "alllayers" not in e.name:
                    stack.append(e.path)
            elif e.name.endswith(".npz"):
                npz.append(e.path)
            elif e.name.startswith("manifest") and e.name.endswith(".json"):
                mans.append(e.path)
    return {"mount": mount, "top": top, "npz": npz, "manifests": mans}


@app.function(image=image, cpu=1, memory=2048, timeout=T, retries=0, volumes=mounts)
def find_roots(mount: str) -> dict:
    """R3: every dir at depth 1-3 holding meta/ (an extraction root) or inputs/ (a staged root); SKIP_DIRS are never descended."""
    import os

    ext, stg, frontier = [], [], [""]
    for _ in range(3):
        nxt = []
        for rel in frontier:
            for e in os.scandir(f"{mount}/{rel}" if rel else mount):
                if not e.is_dir(follow_symlinks=False) or e.name in SKIP_DIRS:
                    continue
                r = f"{rel}/{e.name}" if rel else e.name
                names = {c.name for c in os.scandir(e.path) if c.is_dir(follow_symlinks=False)}
                if "meta" in names:
                    ext.append(r)
                if "inputs" in names:
                    stg.append(r)
                nxt.append(r)
        frontier = nxt
    return {"mount": mount, "extraction_roots": sorted(ext), "staged_roots": sorted(stg)}


@app.function(image=image, cpu=1, memory=4096, timeout=T, retries=0, volumes=mounts)
def scan_npz(paths: list) -> dict:
    import os

    import numpy as np

    hits, nokey, errs = [], 0, []
    for p in paths:
        try:
            z = np.load(p, allow_pickle=False)
            if "key" not in z.files:
                nokey += 1
                continue
            k = z["key"].astype(str)
        except Exception as ex:  # counted and listed by path, never scored
            errs.append({"path": p, "error": type(ex).__name__})
            continue
        want = sorted(x for x in set(k.tolist()) if "|" in x and CELLRX.match(x.split("|", 1)[1]))
        if not want:
            continue
        try:  # a file holding nm172 rows that cannot be read is listed with the keys it held (a lost hit), never dropped silently
            rows_ = sorted(a for a in z.files if a != "key" and z[a].ndim >= 1 and z[a].shape[0] == len(k))
            per_row = [a for a in rows_ if z[a].dtype.kind in "fiub"]  # the units (pw_canon: numbers only); per-row strings hashed apart
            str_row = [a for a in rows_ if a not in per_row]
            folder = os.path.dirname(p)
            fs = fsha(p)
            new = []
            for x in want:
                idx = np.flatnonzero(k == x)
                w, c = x.split("|", 1)
                new.append({"path": p, "file": os.path.basename(p), "folder": folder, "sha256": fs, "raw_key": x,
                            "label_regex": f"{w}_e2|{TAGS.sub('', c)}" if EP2RX.search(folder) and not w.endswith("_e2") else x, "units": int(len(idx)),
                            "arrays": per_row, "string_arrays": str_row, "other_arrays": sorted(set(z.files) - set(rows_) - {"key"}),
                            "string_arrays_sha256": sha(canon([z[a][idx].astype(str).tolist() for a in str_row]).encode()) if str_row else None,
                            "units_sha256": sha(unit_bytes([z[a][idx] for a in per_row])),  # UNITS_SCOPE
                            "array_sha256": {a: sha(unit_bytes([z[a][idx]])) for a in per_row},  # CUT R1: comparable to gate 2's s / f / sbm
                            "score_sha256": sha(unit_bytes([z["score"][idx]])) if "score" in per_row else None})
            hits += new
        except Exception as ex:
            errs.append({"path": p, "error": type(ex).__name__, "lost_hit_keys": want})
    return {"hits": hits, "files": len(paths), "no_key_array": nokey, "errors": errs}


@app.function(image=image, cpu=1, memory=4096, timeout=T, retries=0, volumes=mounts)
def meta_sweep(mount: str, root: str) -> dict:
    import datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo

    import numpy as np

    base = Path(mount) / root

    def acts(d):  # -> sha256, shape and dtype of this run's activation rows at L48, or the reason there are none
        ch, rs, n = d.get("chunk"), d.get("row_start"), d.get("n_rows")
        if ch is None or rs is None or n is None:
            return {"none": "meta without chunk / row_start / n_rows"}
        f = base / "by_layer" / "L48" / f"{Path(str(ch)).stem}.npy"
        if not f.exists():
            return {"none": f"no by_layer/L48/{f.name}"}
        a = np.load(f, mmap_mode="r")
        r = np.ascontiguousarray(a[int(rs):int(rs) + int(n)])
        return {"sha256": sha(repr(r.shape).encode() + str(r.dtype).encode() + r.tobytes()), "shape": list(r.shape), "dtype": str(r.dtype),
                "chunk_file": f"{root}/by_layer/L48/{f.name}"}

    hits, others, n, errs = [], [], 0, []
    for f in sorted(base.glob("meta/*/*.json")):
        n += 1
        b = f.read_bytes()
        try:  # an unreadable meta is listed (and whether it named nm172), never dropped silently
            if NEEDLE not in b:
                if len(others) < 2:
                    others.append(json.loads(b))
                continue
            d = json.loads(b)
            ids, other = summ(d)
            hits.append({"meta": f"{root}/meta/{f.parent.name}/{f.name}", "meta_sha256": sha(b), "fields": ids, "other_fields": other,
                         "mtime_pt": datetime.datetime.fromtimestamp(f.stat().st_mtime, ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
                         "acts_L48": acts(d)})
        except Exception as ex:
            errs.append({"meta": f"{root}/meta/{f.parent.name}/{f.name}", "error": type(ex).__name__, "names_nm172": NEEDLE in b,
                         "keys_in_bytes": sorted(set(x.decode() for x in EVKEY.findall(b)))})
    out = {"mount": mount, "root": root, "metas": n, "hits": hits, "errors": errs}
    if hits:
        ctl = [acts(d).get("sha256") for d in others]
        out["control_two_other_runs_distinct"] = len(ctl) == 2 and None not in ctl and ctl[0] != ctl[1]
        out["root_jsons"] = {p.name: id_leaves(json.loads(p.read_bytes())) for p in sorted(base.glob("*.json"))}
    return out


@app.function(image=image, cpu=2, memory=8192, timeout=T, retries=0, volumes=mounts)
def staged_scan(mount: str, root: str) -> dict:
    from pathlib import Path

    out = {"mount": mount, "root": root, "files": []}
    for f in sorted((Path(mount) / root).glob("inputs/*.jsonl")):
        rows = []
        with f.open("rb") as fh:
            for i, line in enumerate(fh):
                if NEEDLE in line:
                    try:  # an unparseable nm172 line is listed, never dropped silently
                        ids, other = summ(json.loads(line))
                    except Exception as ex:
                        ids, other = {"parse_error": type(ex).__name__}, {}
                    rows.append({"line": i, "line_sha256": sha(line.rstrip(b"\n")), "fields": ids, "other_fields": other})
        out["files"].append({"file": f"{root}/inputs/{f.name}", "sha256": fsha(f), "nm172_rows": rows})
    out["stage_jsons"] = {p.name: id_leaves(json.loads(p.read_bytes())) for p in sorted((Path(mount) / root).glob("stage_*.json"))}
    return out


@app.function(image=image, cpu=4, memory=16384, timeout=T, retries=0, volumes=mounts)
def close_pairs(metas: list) -> list:
    """For every two nm172 runs' L48 rows of one shape: exact (byte-equal), the number of rows within 5% (|a_r - b_r| <= 0.05 max(|a_r|, |b_r|),
    Euclidean per row) and close (every row within 5%); booleans and counts only."""
    from pathlib import Path

    import numpy as np

    rows = []
    for m in metas:
        f = Path(m["mount"]) / m["root"] / "by_layer" / "L48" / f"{Path(str(m['chunk'])).stem}.npy"
        try:  # a run whose rows cannot be read gets an error on each of its pairs, never a silent "not close"
            r = np.load(f, mmap_mode="r")[int(m["row_start"]):int(m["row_start"]) + int(m["n_rows"])]
            rows.append(np.asarray(r, dtype=np.float32).reshape(len(r), -1))
        except Exception as ex:
            rows.append(type(ex).__name__)
    out = []
    for i in range(len(metas)):
        for j in range(i + 1, len(metas)):
            a, b = rows[i], rows[j]
            if isinstance(a, str) or isinstance(b, str):
                out.append({"a": metas[i]["meta"], "b": metas[j]["meta"], "error": a if isinstance(a, str) else b})
                continue
            if a.shape != b.shape:
                out.append({"a": metas[i]["meta"], "b": metas[j]["meta"], "same_shape": False})
                continue
            e = np.linalg.norm(a - b, axis=1) / np.maximum(np.maximum(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)), 1e-12)
            n5 = int((e <= 0.05).sum())
            out.append({"a": metas[i]["meta"], "b": metas[j]["meta"], "same_shape": True, "rows": len(a), "exact": bool(np.array_equal(a, b)),
                        "rows_within_5pct": n5, "close": len(a) > 0 and n5 == len(a)})
    return out


@app.function(image=image, cpu=1, memory=2048, timeout=T, retries=0, volumes=mounts)
def eval_sha(rel: str) -> dict:
    from pathlib import Path

    p = Path("/a32/evals_v1") / rel
    return {"rel": rel, "sha256": fsha(p) if p.exists() else None}


@app.function(image=image, cpu=1, memory=4096, timeout=T, retries=0, volumes=mounts)
def drive(inv: dict, e12: dict, routing: dict) -> dict:
    import os
    from collections import defaultdict

    # 1. score files
    tops = [(m, t) for m in ("/conv", "/p2", "/sol") for t in sorted(os.listdir(m)) if os.path.isdir(f"{m}/{t}") and t not in SKIP_DIRS
            and "alllayers" not in t]
    ws = list(walk.starmap(tops))
    npz = sorted(p for w in ws for p in w["npz"])
    sc = list(scan_npz.map([npz[i:i + 400] for i in range(0, len(npz), 400)]))
    hits = [h for s in sc for h in s["hits"]]
    man_of = {os.path.dirname(p): p for w in ws for p in w["manifests"]}
    folders = {}
    for fo in sorted({h["folder"] for h in hits}):
        mp = man_of.get(fo)
        if not mp:
            folders[fo] = {"manifest": None, "orig_name": None}
            continue
        mj = json.loads(open(mp, "rb").read())
        dump_ = mj.get("dump") or {}
        inv_nm = {k: v for k, v in (dump_.get("cutfull_inventory") or {}).items() if "nm172" in k}
        orig_ = (dump_.get("cutfull_orig_names") or {}).get(os.path.basename(fo))
        dump_.pop("cutfull_inventory", None)
        folders[fo] = {"manifest": mp, "manifest_sha256": fsha(mp), "provenance": {k: v for k, v in id_leaves(mj).items() if PROVP.search(k)},
                       "orig_name": orig_ if isinstance(orig_, str) else None, "inventory_nm172": inv_nm}
    # R5: the builder's epoch-2 label (the E12 block's folder -> world_e2, by folder name, its original 32k name, or file path) beside the regex one
    e12f = e12.get("folders") or {}
    e12_path = {f_["path"]: v["world_e2"] for v in e12f.values() for f_ in (v.get("files") or {}).values() if isinstance(f_, dict) and f_.get("path")}
    for h in hits:
        b_ = os.path.basename(h["folder"])
        cand = [b_, folders[h["folder"]].get("orig_name"), b_.replace("_cutfull_v1", "_v1")]
        w2 = next((e12f[c]["world_e2"] for c in cand if c in e12f), None) or next((w for p_, w in e12_path.items() if h["path"].endswith("/" + p_)), None)
        h["label_builder"] = f"{w2}|{TAGS.sub('', h['raw_key'].split('|', 1)[1])}" if w2 else None
        h["label"] = h["label_builder"] or h["label_regex"]
        h["labels_disagree"] = bool(h["label_builder"]) and h["label_builder"] != h["label_regex"]
        h["line"] = "uncut" if "/cutfull" in h["path"] else "contfull" if "/contfull" in h["path"] else "32k"
    # 3. extraction metas (R3: the walk, cross-checked against the routing records), 4. staged inputs
    fr = {r["mount"]: r for r in find_roots.map(["/a32", "/aU"])}
    roots = [(m, r) for m in ("/a32", "/aU") for r in fr[m]["extraction_roots"]]
    walked = {r for _, r in roots}  # the routing records' uncut extraction_root and 32k original_extraction_root, on either Volume
    rr_ = {str(v.get(f)).strip("/") for v in routing.values() for f in ("extraction_root", "original_extraction_root") if v.get(f)}
    routing_roots_missed = sorted(r for r in rr_ if r not in walked)
    msall = list(meta_sweep.starmap(roots))
    ms = [r for r in msall if r["hits"]]
    meta_errors = [e | {"mount": r["mount"]} for r in msall for e in r["errors"]]
    ctl = {f"{r['mount']}:{r['root']}": r.get("control_two_other_runs_distinct") for r in ms}
    sroots = set()
    for r in ms:  # the staged root beside each extraction root with a hit
        par, nm = os.path.split(r["root"])
        s_ = re.sub(r"_extract(?=_v\d|$)", "", nm)
        if s_ and s_ != nm and os.path.join(par, s_) in fr[r["mount"]]["staged_roots"]:
            sroots.add((r["mount"], os.path.join(par, s_)))
    for fo in folders.values():  # and every staged file a folder manifest names
        for v in (fo.get("provenance") or {}).values():
            mm = re.match(r"^/?(.+)/inputs/[^/]+\.jsonl$", str(v))
            if mm:
                vol_ = fo["provenance"].get("dump.volume")
                for m in ([MOUNT_OF[vol_]] if vol_ in MOUNT_OF else ["/aU", "/a32"]):
                    if mm.group(1) in fr[m]["staged_roots"]:
                        sroots.add((m, mm.group(1)))
    st = list(staged_scan.starmap(sorted(sroots)))
    row_by = {(f["sha256"], row["fields"].get("key")): row for s in st for f in s["files"] for row in f["nm172_rows"]}
    # 5. source .eval files
    keys = sorted({h["fields"].get("key") for r in ms for h in r["hits"] if isinstance(h["fields"].get("key"), str)}
                  | {row["fields"].get("key") for s in st for f in s["files"] for row in f["nm172_rows"] if isinstance(row["fields"].get("key"), str)}
                  | {k for fo in folders.values() for k in fo.get("inventory_nm172", {})})
    rels = sorted({k.rsplit("#", 1)[0] for k in keys if "#" in k})
    es = {r["rel"]: r["sha256"] for r in eval_sha.map(rels)}
    inv_by_key = {r["key"]: r for r in inv["runs"]}
    e12_by_key = {e["inventory_key"]: e for p in e12["pools"].values() for g in p.values() for e in (g.get("e2") or [])}
    evals = {k: {"eval_sha256": es.get(k.rsplit("#", 1)[0]), "inventory": {f: inv_by_key[k].get(f) for f in ("world", "arm", "launch", "task", "eval_sha256")}
                 if k in inv_by_key else None, "e12": {f: e12_by_key[k].get(f) for f in ("key_e2", "key", "eval_sha256")} if k in e12_by_key else None}
             for k in keys}
    for k, e in evals.items():
        want = [x["eval_sha256"] for x in (e["inventory"], e["e12"]) if x]
        e["matches_frozen"] = None if not want else all(w == e["eval_sha256"] for w in want)
    # run label -> eval keys: the inventory (sim_honest is the simh_nm360 world), the E12 block's epoch-2 keys, each folder's own inventory entries
    lab2key, wmap = defaultdict(set), {"sim_honest": "simh_nm360"}
    for k, r in inv_by_key.items():
        if r.get("task") == "nm172_shared_dotfiles" or "/nm172_shared_dotfiles" in k:
            lab2key[f"{wmap.get(r['world'], r['world'])}|nm172_shared_dotfiles"].add(k)
    for k, e in e12_by_key.items():
        if "nm172" in k:
            lab2key[e["key_e2"]].add(k)
    for h in hits:
        b_ = os.path.basename(h["folder"])
        h["folder_inventory_keys"] = sorted(k for k, v in folders[h["folder"]].get("inventory_nm172", {}).items() if isinstance(v, list) and v and v[0] == b_)
    # groups: labels sharing a units hash within one <line>:<file name> or anywhere; metas sharing activation rows; staged rows sharing a field hash
    by_file, by_units, rk_multi = defaultdict(lambda: defaultdict(set)), defaultdict(set), defaultdict(list)
    for h in hits:
        by_file[f"{h['line']}:{h['file']}"][h["units_sha256"]].add(h["label"])
        by_units[h["units_sha256"]].add((h["label"], h["path"]))
        rk_multi[(h["line"], h["raw_key"])].append({"folder": h["folder"], "file": h["file"], "units_sha256": h["units_sha256"]})
    raw_key_multi_folder = {ln: {k: v for (l_, k), v in sorted(rk_multi.items()) if l_ == ln and len({e["folder"] for e in v}) > 1}
                            for ln in sorted({h["line"] for h in hits})}
    unit_groups = [{"file": f, "units_sha256": u, "labels": sorted(ls)} for f, d in sorted(by_file.items()) for u, ls in sorted(d.items()) if len(ls) > 1]
    unit_groups_any_file = [{"units_sha256": u, "label_paths": sorted(v)} for u, v in sorted(by_units.items()) if len({lab for lab, _ in v}) > 1]
    ag = defaultdict(list)
    for r in ms:
        for h in r["hits"]:
            if h["acts_L48"].get("sha256"):
                ag[h["acts_L48"]["sha256"]].append({"mount": r["mount"], "meta": h["meta"], "key": h["fields"].get("key")})
    act_groups = [{"sha256": s_, "metas": v} for s_, v in ag.items() if len(v) > 1]
    sg = defaultdict(list)
    for s in st:
        for f in s["files"]:
            for row in f["nm172_rows"]:
                for fn, o in row["other_fields"].items():
                    sg[(fn, o["sha256"])].append((f["file"], row["fields"].get("key")))
    staged_groups = [{"field": fn, "sha256": s_, "rows": sorted(set(v))} for (fn, s_), v in sorted(sg.items()) if len({k for _, k in v}) > 1]
    # the pair: per <line>:<score-file name>, equal (a units hash in common) / different / one side missing; a label with two hashes is listed
    pair, two_copies = {}, {}
    for f, d in sorted(by_file.items()):
        u = defaultdict(set)
        for s_, ls in d.items():
            for lab in ls:
                u[lab].add(s_)
        two_copies.update({f"{f}|{lab}": len(v) for lab, v in u.items() if len(v) > 1})
        pair[f] = "equal" if u[REAL] & u[SIM] else "different" if u[REAL] and u[SIM] else f"missing: {[x for x in (REAL, SIM) if not u[x]]}"
    # activations: every two nm172 runs' L48 rows, exact and close; then R4's source rule
    am = [{"mount": r["mount"], "root": r["root"], "meta": h["meta"], "key": h["fields"]["key"], "n_tokens": h["fields"].get("n_tokens"),
           "calls_len": (h["other_fields"].get("calls") or {}).get("len"), "inputs_sha256": h["fields"].get("inputs_sha256"),
           **{f: h["fields"][f] for f in ("chunk", "row_start", "n_rows")}}
          for r in ms for h in r["hits"] if all(h["fields"].get(f) is not None for f in ("key", "chunk", "row_start", "n_rows"))]
    own_row = {m["meta"]: row_by.get((m["inputs_sha256"], m["key"])) for m in am}
    cp = close_pairs.remote(am) if len(am) > 1 else []
    rk, sk = set(lab2key.get(REAL, ())), set(lab2key.get(SIM, ()))
    lost = [e for s in sc for e in s["errors"] if e.get("lost_hit_keys")]
    nm_meta_err = [e for e in meta_errors if e["names_nm172"]]
    cp_err = [x for x in cp if "error" in x]
    shared_lines, close_lines, src = source_rule(am, cp, rk, sk, own_row, ctl, len(nm_meta_err))
    per_line = {lab: {ln: sum(1 for m in am if LINE[m["mount"]] == ln and m["key"] in ks) for ln in LINE.values()} for lab, ks in ((REAL, rk), (SIM, sk))}
    multi = defaultdict(set)  # an eval key staged in more than one root of one line (a re-staged run: which copy does a scorer read?)
    for s in st:
        for f in s["files"]:
            for row in f["nm172_rows"]:
                multi[(LINE.get(s["mount"]), row["fields"].get("key"))].add(f["file"])
    reading = {"score_files_equal": [f for f, v in pair.items() if v == "equal"], "score_files_different": [f for f, v in pair.items() if v == "different"],
               "score_files_one_side_missing": [f for f, v in pair.items() if v.startswith("missing")],
               "equal_score_file_lines": sorted({f.split(":", 1)[0] for f, v in pair.items() if v == "equal"}),
               "labels_with_two_units_hashes_in_one_line_file": two_copies, "raw_key_multi_folder": raw_key_multi_folder,
               "labels_disagree": sorted({(h["path"], h["label_regex"], h["label_builder"]) for h in hits if h["labels_disagree"]}),
               "pair_metas_per_line": per_line, "not_found": [f"{lab} in {ln}: not found" for lab, d in per_line.items() for ln, n in d.items() if not n],
               "metas_without_own_staged_row": sorted(x for x, v in own_row.items() if v is None),
               "acts_exact_equal_lines": shared_lines, "acts_close_lines": close_lines, "source_by_line": src,
               "real_eval_keys": sorted(rk), "sim_eval_keys": sorted(sk),
               "eval_keys_staged_in_more_than_one_root": {f"{ln}|{k}": sorted(v) for (ln, k), v in multi.items() if len(v) > 1},
               "key_formats": {"staged_and_meta": "<launch>/<arm_dir>/<task_dir>/<file>.eval#e<sample> (launch and file name: unique per .eval)",
                               "score_npz": "<world>|<cell> (no launch, no epoch: epoch 2 is named by its folder)"}}
    # per run label (attack arms included): its score files, the labels sharing its units or its activation rows, its source .eval check -> clean
    lab_of = {k: lab for lab, ks in lab2key.items() for k in ks}
    by_meta = {m["meta"]: m for m in am}
    act_lab = defaultdict(set)
    for x in cp:
        if x.get("exact"):
            la, lb = lab_of.get(by_meta[x["a"]]["key"]), lab_of.get(by_meta[x["b"]]["key"])
            if la != lb:
                act_lab[la].add(lb), act_lab[lb].add(la)
    runs, all_labs = {}, sorted(set(lab2key) | {h["label"] for h in hits})
    lab_keys = {lab: set(lab2key.get(lab, ())) | {k for h in hits if h["label"] == lab for k in h["folder_inventory_keys"]} for lab in all_labs}
    unclean = unclean_labels(all_labs, lab_keys, lost, nm_meta_err)
    for lab in all_labs:
        hs = [h for h in hits if h["label"] == lab]
        su = sorted({o for h in hs for o, _ in by_units[h["units_sha256"]] if o != lab})
        ek = set(lab2key.get(lab, ())) | {k for h in hs for k in h["folder_inventory_keys"]}
        ev = {k: evals.get(k, {}).get("matches_frozen") for k in sorted(ek)}
        runs[lab] = {"score_files": len(hs), "lines": sorted({h["line"] for h in hs}), "eval_keys_matching_frozen": ev,
                     "metas": sorted(m["meta"] for m in am if m["key"] in ek), "shares_units_with": su,
                     "shares_acts_with": sorted(str(x) for x in act_lab.get(lab, ())),
                     "named_by_unreadable_input": lab in unclean,
                     "clean": bool(hs) and not su and not act_lab.get(lab) and bool(ev) and all(v is True for v in ev.values()) and lab not in unclean}
    reading["runs"] = runs
    reading["unreadable_inputs"] = {"score_files_with_lost_nm172_rows": len(lost), "nm172_metas_unreadable": len(nm_meta_err),
                                    "activation_pairs_unreadable": len(cp_err)}
    reading["abc"] = abc_of(src, reading["score_files_equal"], len(lost), len(nm_meta_err), len(cp_err))
    return {"walk": {"tops": len(tops), "npz_files": len(npz), "no_key_array": sum(s["no_key_array"] for s in sc),
                     "npz_errors": [e for s in sc for e in s["errors"]]},
            "score_hits": hits, "folders": folders, "extraction_roots_scanned": {m: len(fr[m]["extraction_roots"]) for m in fr},
            "metas_read": sum(r["metas"] for r in msall), "meta_errors": meta_errors,
            "routing_roots_missed_by_walk": routing_roots_missed, "extraction_hits": ms, "staged": st, "evals": evals,
            "label_to_eval_keys": {k: sorted(v) for k, v in sorted(lab2key.items())},
            "groups": {"units": unit_groups, "units_any_file": unit_groups_any_file, "activations_L48": act_groups, "staged_fields": staged_groups},
            "pair_units_by_file": pair, "pair": [REAL, SIM], "units_sha256_scope": UNITS_SCOPE, "acts_pairs_L48": cp, "reading": reading}


@app.local_entrypoint()
def main(inventory: str, e12: str, routing: str, out: str):
    import os

    ib, eb = open(inventory, "rb").read(), open(e12, "rb").read()
    assert sha(ib) == INV_SHA and sha(eb) == E12_SHA, "STOP: the inventory or the E12 block differs from the frozen digests"
    rt = {f: json.loads(open(os.path.join(routing, f), "rb").read()) for f in sorted(os.listdir(routing)) if f.endswith(".json") and "_attest" not in f}
    assert rt, "STOP: no routing records"
    r = drive.remote(json.loads(ib), json.loads(eb), rt)
    with open(out, "x") as fh:
        json.dump(r, fh, indent=1, default=str)
    rd = r["reading"]
    print(json.dumps({"npz_files": r["walk"]["npz_files"], "npz_errors": len(r["walk"]["npz_errors"]),
                      "npz_lost_hit_files": sum(1 for e in r["walk"]["npz_errors"] if e.get("lost_hit_keys")),
                      "score_hits": len(r["score_hits"]), "folders": len(r["folders"]),
                      "extraction_roots": r["extraction_roots_scanned"], "routing_roots_missed_by_walk": len(r["routing_roots_missed_by_walk"]),
                      "metas_read": r["metas_read"], "meta_errors": len(r["meta_errors"]),
                      "meta_errors_naming_nm172": sum(1 for e in r["meta_errors"] if e["names_nm172"]),
                      "close_pair_errors": sum(1 for x in r["acts_pairs_L48"] if "error" in x),
                      "extraction_hits": sum(len(x["hits"]) for x in r["extraction_hits"]),
                      "staged_rows": sum(len(f["nm172_rows"]) for s in r["staged"] for f in s["files"]),
                      "unit_groups": len(r["groups"]["units"]), "unit_groups_any_file": len(r["groups"]["units_any_file"]),
                      "activation_groups": len(r["groups"]["activations_L48"]), "staged_groups": len(r["groups"]["staged_fields"]),
                      "pair_units_by_file": r["pair_units_by_file"], "pair_metas_per_line": rd["pair_metas_per_line"], "not_found": rd["not_found"],
                      "acts_exact_equal_lines": rd["acts_exact_equal_lines"], "acts_close_lines": rd["acts_close_lines"],
                      "sources": {k: v["source"] for k, v in rd["source_by_line"].items()}, "unreadable_inputs": rd["unreadable_inputs"], "abc": rd["abc"],
                      "clean": {k: v["clean"] for k, v in rd["runs"].items()}}))
