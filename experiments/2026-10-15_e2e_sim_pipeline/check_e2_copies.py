"""Census of epoch-2 score rows that are COPIES of epoch-1 rows (09 via 2e 19:4x; NOT RUN until CUT's glance). KEYS, COUNTS, POSITIONS AND HASHES ONLY.
1. Every .npz on the conv, p2 and solavg Volumes (activation dirs skipped): for every honest row key (raw world honest_a / honest_b / honest_c),
   gate 2's content hash (score, fold, score_by_model or score_by_fold, as unit_bytes) and the unit count. A file whose path names ep2 holds epoch-2
   rows; every other file epoch-1 rows (the npz key carries no epoch: the folder does).
2. An epoch-2 row is IDENTICAL to epoch 1 when some epoch-1 file of the same line (uncut: under /cutfull; else 32k) and the same name (a trailing
   __L<k> layer tag ignored) holds the same raw key with the same content hash. Counts per line x family (the folder name above the file, e.g.
   p2, famexp_r2, own_all_rs_uncut) x file name: epoch-2 rows, rows identical to epoch 1.
3. A SEEDED sample (seed: int(sha256("E2E e2-copy census 2026-10-16")) mod 2**32; up to 3 identical rows per line x family x name, sorted first) is
   classified by the token-prefix rule (09; check_token_prefix.py @ b8281a4f8 classify(), copied): the two runs' STAGED ids (32k: the routing record's
   original_staged_set; uncut: its staged set; the arm from original_folders or the world; the row whose key's task dir is the cell), the common
   prefix d, every unit's need (token + 1; a 32k file without positions: its call's staged readouts, coarse), the routing (fold) -> natural or
   MIX-UP (a copy). Unresolvable runs are counted, never guessed.
4. 09's additions (2e 20:0x-20:1x): (a) the SWAP test: epoch-1 rows byte-equal to an epoch-2 row of the same key / line / scorer / name, by the
   file's directory; and each row's call / unit STRUCTURE against each run's reference structure (its p2 files, when unique): a row carrying the
   OTHER epoch's structure (and not its own) holds the other epoch's run; (b) per identical row, both runs' eval references (E12 inventory_key;
   frozen inventories); (c) per group, which epoch-1 directories the identical epoch-2 rows equal (e.g. layers under pass2_... vs famexp_r2), with
   every identical epoch-1 path per row; (d) the nm305 ROW check: the staged file sha256, ids length and ids sha256 (int64 bytes) of honest_b /
   honest_a_e2 / honest_b_e2 nm305 in both lines, PASS / FAIL against the token-prefix record (c7b0da2f...).
5. 507-prep review (51319cf99): B1 keys normalised on both sides as the loader does (raw keys kept; tagged keys counted); B2 coverage of the
   E12 block's 392 epoch-2 files and 1,475 runs per line, npz without key, groups without a hashed epoch-1 counterpart (INCOMPLETE, never clean);
   B3 the positive control (the 32k famexp nm172 honest_a / honest_b copies must be found); B4 every hashed candidate compared; B5 provenance
   (script sha256, git + dirty, input shas, routing shas); B6 the verdict (STOP on scan errors, row-check FAILs, coverage, the control; a
   non-zero exit); B7 the swap test by readout type, p2 rows only as the reference, "matches neither" apart from "no ref"; B8 file_units,
   classify and common_prefix are check_token_prefix.py @ b8281a4f8's, checked by source at launch; B9 the E12 cross-checks (per-pool counts,
   e2_shas, inventory_ep2_keys) and conflicting duplicates STOP.
Usage: modal run --detach check_e2_copies.py --routing <dir of routing jsons> --out <new local json>"""
import hashlib
import json
import re

import modal

VOLS = {"/conv": ("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", None), "/p2": ("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", 2),
        "/sol": ("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", 2), "/a32": ("probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", None),
        "/aU": ("probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14", None)}
MOUNT_OF = {v[0]: m for m, v in VOLS.items()}
HONEST = {"honest_a", "honest_b", "honest_c"}
EP2RX = re.compile(r"(^|[/_-])ep2([/_-]|$)")
FOLDER = re.compile(r"^[\w.@-]+__[\w.@-]+$")  # a run folder: <staged set>__<arm>
SKIP_DIRS = {"by_layer", "chunks", "parts", "meta", "shard_results", "evals_v1", "inputs"}
SEED = int(hashlib.sha256(b"E2E e2-copy census 2026-10-16").hexdigest(), 16) % 2**32
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.1.3")
app = modal.App("probe-shift_aswin-surya_sub-e2esim_e2copies_kill-2026-11-16")
mounts = {m: (modal.Volume.from_name(n, version=v) if v else modal.Volume.from_name(n)).read_only() for m, (n, v) in VOLS.items()}
line_of = lambda p: "uncut" if "/cutfull" in p else "32k"  # noqa: E731
norm = lambda p: re.sub(r"__L\d+(?=\.npz$)", "", p.rsplit("/", 1)[-1])  # noqa: E731
TAGS = re.compile(r"(_e2[ab]|_redraw[0-9]+)+$")  # the product loader's normalisation (check_stagea_alltasks.py l.1053), both sides (review B1)
nkey = lambda k: k.split("|", 1)[0] + "|" + TAGS.sub("", k.split("|", 1)[1])  # noqa: E731
TYPES = [("C", re.compile(r"(__C(__L\d+)?|__command|__cmd_A)\.npz$")), ("R", re.compile(r"(__R(__L\d+)?|__reply_mean|__reply_last|__avg_top2)\.npz$")),
         ("CE", re.compile(r"__call_end_content\.npz$"))]
rtype = lambda name: next((t for t, rx in TYPES if rx.search(name)), "other")  # noqa: E731


def sha(b):
    return hashlib.sha256(b).hexdigest()


def family_of(p):  # the SCORER: the folder above the file, skipping run folders (<set>__<arm>), layers / pass* / timestamp dirs; a trailing _r<k> dropped
    parts = p.split("/")[:-1]
    while parts and (FOLDER.match(parts[-1]) or parts[-1] == "layers" or parts[-1].startswith("pass") or re.match(r"^\d{4}-\d{2}-\d{2}_\d+$", parts[-1])):
        parts.pop()
    return re.sub(r"_r\d+$", "", parts[-1]) if parts else ""


def unit_bytes(a_):
    import numpy as np

    a_ = np.asarray(a_)
    if a_.dtype.kind == "f":
        a_ = np.where(np.isnan(a_.astype(np.float64)), np.nan, a_.astype(np.float64) + 0.0)
    elif a_.dtype.kind in "iub":
        a_ = a_.astype(np.int64)
    return repr(a_.shape).encode() + b":" + np.ascontiguousarray(a_).tobytes()


def common_prefix(a, b):  # copied from check_token_prefix.py @ b8281a4f8
    import numpy as np

    n = min(len(a), len(b))
    ne = np.flatnonzero(np.asarray(a[:n]) != np.asarray(b[:n]))
    return int(ne[0]) if len(ne) else n


def file_units(path, raw_key, calls):
    """One score file's units of one run: needed prefix per unit (+ coarse flag), the routing, and gate 2's s / f / sbm bytes."""
    import numpy as np

    z = np.load(path, allow_pickle=False)
    k = z["key"].astype(str)
    idx = np.flatnonzero(k == raw_key)
    if not len(idx):
        return None
    g = lambda a: z[a][idx] if a in z.files and z[a].ndim >= 1 and z[a].shape[0] == len(k) else None  # noqa: E731
    tok, se, ci = g("token"), g("span_end"), g("call_index")
    coarse, span_form = tok is None, None
    if not coarse:  # CUT R1: one causal forward, so the row at p depends on ids[0..p]: need = token + 1; span_end only checked
        need = tok.astype(np.int64) + 1
        if se is not None:  # the extractor's output_end span is [start, end) with end - 1 == readout; also accept the inclusive form; else STOP
            se_ = se.astype(np.int64)
            span_form = "exclusive (end = token + 1)" if np.array_equal(se_, need) else "inclusive (end = token)" if np.array_equal(se_, tok.astype(np.int64)) \
                else "STOP: span_end is neither token nor token + 1"
        pos = [tok.astype(np.int64).tolist(), None if se is None else se.astype(np.int64).tolist()]
    elif ci is not None:  # 32k: the unit's call -> the staged calls' readouts of that call (every kind; coarse)
        by = {}
        for c in calls:
            if isinstance(c, dict) and isinstance(c.get("readout"), int):
                by.setdefault(c.get("call_index"), []).append(c["readout"])
        need = np.array([max(by.get(int(x), [10**12])) + 1 for x in ci], np.int64)
        pos = [ci.astype(np.int64).tolist(), None]
    else:
        need, pos = None, None
    fold, f36 = g("fold"), g("fold_r36")
    return {"n_units": int(len(idx)), "coarse": coarse, "span_form": span_form, "need": None if need is None else need.tolist(), "pos": pos,
            "routing": "single model" if fold is None else [sorted(set(fold.astype(np.int64).tolist())), None if f36 is None else sorted(set(f36.astype(np.int64).tolist()))],
            "fold_bytes": None if fold is None else sha(unit_bytes(fold) + (b"" if f36 is None else unit_bytes(f36))),
            "s_f_sbm": sha(b"|".join(unit_bytes(x) for x in (g("score"), fold, g("score_by_model") if g("score_by_model") is not None else g("score_by_fold")) if x is not None))}


def classify(d, need_x, need_y, pos_equal, routing_equal, observed_identical):
    """09's generalized gate 2 on one compared file: d = common-prefix length of the two staged id arrays; need_* = the needed prefix per unit (last
    readout position + 1). -> (predicted identical, outcome)."""
    inside =bool(need_x) and bool(need_y) and max(need_x) <= d and max(need_y) <= d
    pred = bool(pos_equal and routing_equal and inside)
    out = ("natural" if pred else "MIX-UP") if observed_identical else ("UNEXPECTED DIFFERENCE" if pred else "different")
    return pred, out


@app.function(image=image, cpu=1, memory=2048, timeout=2400, retries=0, volumes=mounts)
def walk(mount: str, top: str) -> list:
    import os

    npz, stack = [], [f"{mount}/{top}"]
    while stack:
        d = stack.pop()
        for e in os.scandir(d):
            if e.is_dir(follow_symlinks=False):
                if e.name not in SKIP_DIRS and "alllayers" not in e.name:
                    stack.append(e.path)
            elif e.name.endswith(".npz"):
                npz.append(e.path)
    return npz


@app.function(image=image, cpu=1, memory=4096, timeout=2400, retries=0, volumes=mounts)
def scan(paths: list) -> dict:
    """-> {"rows": [(path index, raw key, content hash[:24] or None, units)], "arrays": {path index: array names}, "errors": [...]}: honest rows only.
    The content hash covers score, fold, score_by_model, score_by_fold where present; a file with none of score / score_by_model / score_by_fold
    gets None ("no score array", 2e / CUT (b)): its rows are never called identical."""
    import numpy as np

    rows, errs, names, nokey = [], [], {}, []
    for pi, p in enumerate(paths):
        try:
            z = np.load(p, allow_pickle=False)
            if "key" not in z.files:
                nokey.append(pi)
                continue
            k = z["key"].astype(str)
            hon = sorted(x for x in set(k.tolist()) if x.split("|", 1)[0] in HONEST)
            if not hon:
                continue
            names[pi] = sorted(z.files)
            per_row = {a for a in z.files if a != "key" and z[a].ndim >= 1 and z[a].shape[0] == len(k)}
            arrs = [z[a] for a in ("score", "fold", "score_by_model", "score_by_fold") if a in per_row]
            scored = bool(per_row & {"score", "score_by_model", "score_by_fold"})
            order = np.argsort(k, kind="stable")
            ks = k[order]
            st = [z[a] for a in ("call_index", "unit") if a in per_row]  # the row's STRUCTURE (which calls and units), for the swap test
            for x in hon:
                lo, hi = np.searchsorted(ks, x, "left"), np.searchsorted(ks, x, "right")
                idx = np.sort(order[lo:hi])
                rows.append((pi, x, hashlib.sha256(b"|".join(unit_bytes(a[idx]) for a in arrs)).hexdigest()[:24] if scored else None, int(hi - lo),
                             hashlib.sha256(b"|".join(unit_bytes(a[idx]) for a in st)).hexdigest()[:16] if st else None))
        except Exception as ex:  # listed by path, never dropped silently
            errs.append({"path": p, "error": type(ex).__name__})
    return {"rows": rows, "arrays": names, "errors": errs, "no_key": nokey}


@app.function(image=image, cpu=4, memory=16384, timeout=2400, retries=0, volumes=mounts)
def classify_pairs(samples: list, routing: dict, folder_of_run: dict) -> list:
    """samples: [{"line", "e2": {path, key}, "e1": {path, key}}] -> per sample: the two staged rows, d, needs, routing, outcome. folder_of_run:
    "<e1|e2> <world>|<cell>" -> the run's 32k folder (frozen inventories' original_folder; E12 entries' folder), for multi-run files."""
    import os

    import numpy as np

    fol = {}  # 32k folder -> (record, meta_arm)
    for recs in routing.values():
        for r in recs:
            for arm, f in (r.get("original_folders") or {}).items():
                fol.setdefault(f, (r, arm))
    by_batch = {r["batch"]: r for recs in routing.values() for r in recs}

    def where(path, key, line, ep):  # -> ((mount, staged file), cell) or (None, reason)
        w, c = key.split("|", 1)
        if line == "uncut":
            m = re.search(r"/cutfull(?:_ep2)?/([^/]+)/", path)
            r = by_batch.get(m.group(1)) if m else None
            arms = [a for a in (r or {}).get("arms", []) if w in (a.get("world") or [a.get("meta_arm")])]
            if not r or len(arms) != 1:
                return None, "uncut: no routing record or arm"
            mnt, f = MOUNT_OF.get(r["volume"]), arms[0]["staged_inputs_file"]
        else:
            fo = next((x for x in path.split("/") if FOLDER.match(x) and x in fol), None) or folder_of_run.get(f"{ep} {key}")
            if fo not in fol:
                return None, "32k: no run folder (a multi-run file without a unique inventory folder)"
            r, arm = fol[fo]
            mnt, f = "/a32", f"{r['original_staged_set']}/inputs/{arm}.jsonl"
        if not mnt or not os.path.exists(f"{mnt}/{f}"):
            return None, "staged file missing"
        return (mnt, f), c

    loc = [(where(s["e2"]["path"], s["e2"]["key"], s["line"], "e2"), where(s["e1"]["path"], s["e1"]["key"], s["line"], "e1")) for s in samples]
    need_cells = {}
    for pair_ in loc:
        for fl, c in pair_:
            if fl is not None:
                need_cells.setdefault(fl, set()).update({c, TAGS.sub("", c)})  # review B1: a tagged cell may be staged under its tagged or plain name
    found = {}  # (file, cell) -> staged rows; every staged file read once
    for (mnt, f), cells in need_cells.items():
        with open(f"{mnt}/{f}", "rb") as fh:
            for ln in fh:
                for c in cells:
                    if f"/{c}/".encode() in ln:
                        d = json.loads(ln)
                        if d.get("key", "").split("/")[2:3] == [c]:
                            found.setdefault(((mnt, f), c), []).append({"ids": np.asarray(d["ids"], np.int64), "calls": d.get("calls") or []})

    def staged_of(fl_c):
        fl, c = fl_c
        if fl is None:
            return None, c
        rs = found.get((fl, c)) or found.get((fl, TAGS.sub("", c)), [])
        return (rs[0], f"{fl[0]}:{fl[1]}") if len(rs) == 1 else (None, f"{len(rs)} staged rows for the cell")

    out = []
    for s, (l2, l1) in zip(samples, loc):
        a, ha = staged_of(l2)
        b, hb = staged_of(l1)
        rec = {"line": s["line"], "e2": s["e2"], "e1": s["e1"], "staged": [ha, hb]}
        if a is None or b is None:
            out.append(rec | {"outcome": "unresolved"})
            continue
        d = common_prefix(a["ids"], b["ids"])
        ua, ub = file_units(s["e2"]["path"], s["e2"]["key"], a["calls"]), file_units(s["e1"]["path"], s["e1"]["key"], b["calls"])  # review B8
        if ua is None or ub is None or ua["need"] is None or ub["need"] is None:
            out.append(rec | {"d": d, "outcome": "no positions"})
            continue
        if any((u_["span_form"] or "").startswith("STOP") for u_ in (ua, ub)):
            out.append(rec | {"d": d, "outcome": "STOP: span_end is neither token nor token + 1"})
            continue
        _, oc = classify(d, ua["need"], ub["need"], ua["pos"] == ub["pos"], ua["routing"] == ub["routing"] and ua["fold_bytes"] == ub["fold_bytes"], True)
        out.append(rec | {"n_tokens": [int(len(a["ids"])), int(len(b["ids"]))], "d": d, "max_need": [max(ua["need"]), max(ub["need"])],
                          "min_need": [min(ua["need"]), min(ub["need"])], "routing": [ua["routing"], ub["routing"]], "coarse": bool(ua["coarse"] or ub["coarse"]),
                          "outcome": oc})
    return out


@app.function(image=image, cpu=2, memory=32768, timeout=2400, retries=0, volumes=mounts)
def drive(routing: dict, folder_of_run: dict, eval_of_run: dict, row_items: list, e12_files: list, e2_keys: list) -> dict:
    """e12_files: the E12 block's 392 pinned epoch-2 files ("/conv/<path>"); e2_keys: its 1,475 epoch-2 runs as "<world>|<cell>" (epoch-1 key form)."""
    import os
    import random

    tops = [(m, t) for m in ("/conv", "/p2", "/sol") for t in sorted(os.listdir(m)) if os.path.isdir(f"{m}/{t}") and t not in SKIP_DIRS and "alllayers" not in t]
    npz = sorted(p for w in walk.starmap(tops) for p in w)
    chunks = [npz[i:i + 400] for i in range(0, len(npz), 400)]
    e2_idx, e1_idx, errs, n_rows, arrays_of, nokey, tagged, scanned = {}, {}, [], 0, {}, [], {"e1": 0, "e2": 0}, set()
    for ch, res in zip(chunks, scan.map(chunks)):
        errs += res["errors"]
        nokey += [ch[i] for i in res["no_key"]]
        arrays_of |= {ch[int(pi)]: v for pi, v in res["arrays"].items()}
        for pi, key, h, n, sh in res["rows"]:
            p = ch[pi]
            n_rows += 1
            scanned.add(p)
            ep = "e2" if EP2RX.search(p) else "e1"
            tagged[ep] += nkey(key) != key
            # review B1: both sides keyed by the product loader's normalised key; the raw key is kept for the staged lookup and file_units
            (e2_idx if ep == "e2" else e1_idx).setdefault((line_of(p), family_of(p), norm(p), nkey(key)), []).append((p, h, n, sh, key))
    edir = lambda p: next((x for x in reversed(p.split("/")[:-1]) if not FOLDER.match(x)), "")  # noqa: E731  (09 (c): famexp_r2 vs layers, ...)
    groups, hits = {}, {}
    for (ln, fam, nm, key), lst in e2_idx.items():
        e1 = e1_idx.get((ln, fam, nm, key), [])
        e1h = [c for c in e1 if c[1] is not None]  # review B4: every HASHED e1 candidate is compared; unhashed candidates are counted apart
        for p, h, n, _, raw in lst:
            g = (ln, fam, nm)
            st = groups.setdefault(g, {"e2_rows": 0, "e2_rows_with_e1_row": 0, "e2_rows_with_hashed_e1_row": 0, "identical_to_e1": 0,
                                       "e2_no_score_array": 0, "e1_candidates_without_score_array": 0, "array_names": set(), "identical_e1_dirs": {}})
            st["e2_rows"] += 1
            st["e2_rows_with_e1_row"] += bool(e1)
            st["e2_rows_with_hashed_e1_row"] += bool(e1h)
            st["e1_candidates_without_score_array"] += len(e1) - len(e1h)
            st["array_names"] |= set(arrays_of.get(p, [])) | {x for c in e1[:3] for x in arrays_of.get(c[0], [])}
            if h is None:
                st["e2_no_score_array"] += 1
                continue
            same = [c for c in e1h if c[1] == h and n > 0]
            if same:
                st["identical_to_e1"] += 1
                for d_ in {edir(c[0]) for c in same}:
                    st["identical_e1_dirs"][d_] = st["identical_e1_dirs"].get(d_, 0) + 1
                hits.setdefault(g, []).append({"e2": {"path": p, "key": raw, "eval_ref": eval_of_run.get(f"e2 {key}")},
                                               "e1": {"path": same[0][0], "key": same[0][4], "eval_ref": eval_of_run.get(f"e1 {key}")},
                                               "e1_identical_paths": sorted(c[0] for c in same)})
    for st in groups.values():
        st["array_names"] = sorted(st["array_names"])
    incomplete = sorted(" | ".join(g) for g, st in groups.items() if not st["e2_rows_with_hashed_e1_row"])  # review B2: never read as clean
    # review B2: coverage of the E12 block's pinned epoch-2 files, and of its 1,475 epoch-2 runs in each line
    err_paths = {e["path"] for e in errs}
    cov = {"e12_files": len(e12_files), "not_walked": sorted(set(e12_files) - set(npz)), "not_classified_e2": sorted(f for f in e12_files if not EP2RX.search(f)),
           "scan_errors": sorted(set(e12_files) & err_paths), "no_honest_rows": sorted(f for f in e12_files if f in set(npz) and f not in scanned and f not in err_paths)}
    found = {}
    for (ln, fam, nm, key), lst in e2_idx.items():
        found.setdefault(ln, {}).setdefault(key, 0)
        found[ln][key] += len(lst)
    cov["e2_runs_with_rows"] = {ln: sum(1 for k in e2_keys if found.get(ln, {}).get(nkey(k))) for ln in ("32k", "uncut")}
    cov["e2_runs_missing"] = {ln: sorted(k for k in e2_keys if not found.get(ln, {}).get(nkey(k)))[:50] for ln in ("32k", "uncut")}
    cov["e2_runs_missing_n"] = {ln: sum(1 for k in e2_keys if not found.get(ln, {}).get(nkey(k))) for ln in ("32k", "uncut")}
    # review B3: the POSITIVE CONTROL, the known 32k famexp_r2 copies of nm172 honest_a_e2 and honest_b_e2
    pc_ = {w: sum(1 for g, hs in hits.items() if g[0] == "32k" and g[1] == "famexp" for x in hs if nkey(x["e2"]["key"]) == f"{w}|nm172_shared_dotfiles")
           for w in ("honest_a", "honest_b")}
    # review B7, the SWAP test: per line x scorer x directory x epoch: rows byte-equal to an other-epoch row; STRUCTURE against each run's reference
    # structure, built from the p2 files of the SAME readout type (C / R / CE), never from the rows being classified (p2 rows are the reference)
    ref = {}
    for idx_, ep in ((e1_idx, "e1"), (e2_idx, "e2")):
        for (ln, fam, nm, key), lst in idx_.items():
            if fam == "p2":
                for c in lst:
                    if c[3]:
                        ref.setdefault((ln, key, ep, rtype(nm)), set()).add(c[3])
    ref = {k: next(iter(v)) for k, v in ref.items() if len(v) == 1}
    swap = {}
    for idx_, ep, oth, other_idx in ((e1_idx, "e1", "e2", e2_idx), (e2_idx, "e2", "e1", e1_idx)):
        for (ln, fam, nm, key), lst in idx_.items():
            for p, h, n, sh, _ in lst:
                e = swap.setdefault(f"{ln} | {fam} | {edir(p)} | {ep} rows", {"rows": 0, f"equal_to_an_{oth}_row": 0, "structure_own": 0, f"structure_{oth}": 0,
                                                                              "structure_matches_neither": 0, "structure_uninformative": 0,
                                                                              "structure_no_ref": 0, "p2_reference_rows": 0})
                e["rows"] += 1
                e[f"equal_to_an_{oth}_row"] += bool(h is not None and any(c[1] == h and n > 0 for c in other_idx.get((ln, fam, nm, key), [])))
                if fam == "p2":
                    e["p2_reference_rows"] += 1
                    continue
                ro, rx = ref.get((ln, key, ep, rtype(nm))), ref.get((ln, key, oth, rtype(nm)))
                e["structure_no_ref" if not sh or ro is None or rx is None else "structure_uninformative" if ro == rx else "structure_own" if sh == ro
                  else f"structure_{oth}" if sh == rx else "structure_matches_neither"] += 1
    rng, samples = random.Random(SEED), []
    for g in sorted(hits):
        hs = sorted(hits[g], key=lambda x: (x["e2"]["path"], x["e2"]["key"], x["e1"]["path"]))
        samples += [{"line": g[0], "group": list(g), **x} for x in (rng.sample(hs, 3) if len(hs) > 3 else hs)]
    cls = classify_pairs.remote(samples, routing, folder_of_run) if samples else []
    for s, c in zip(samples, cls):
        c["group"] = s["group"]
    summ = {}
    for c in cls:
        e = summ.setdefault(" | ".join(c["group"]), {})
        k_ = c["outcome"] + (" (coarse)" if c.get("coarse") else "")
        e[k_] = e.get(k_, 0) + 1
    return {"npz_files": len(npz), "npz_without_key": len(nokey), "npz_without_key_paths": nokey[:50], "honest_rows": n_rows, "tagged_keys": tagged,
            "scan_errors": errs, "seed": SEED, "coverage": cov, "positive_control_famexp_nm172": pc_, "incomplete_groups": incomplete,
            "groups": [{"line": g[0], "family": g[1], "file": g[2], **v} for g, v in sorted(groups.items())],
            "groups_with_copies_candidates": len(hits), "hits": {" | ".join(g): v for g, v in sorted(hits.items())},
            "sample_classification": cls, "sample_summary": summ, "swap_test": swap, "structure_refs": len(ref), "row_check": row_check.remote(row_items)}


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes=mounts)
def row_check(items: list) -> list:
    """09 (2e 20:1x): per item {run, line, file "<mount>:<path>", eval_ref, want_n_tokens, want_ids_sha256} -> the staged file's sha256, the row's ids
    length and ids sha256 (int64 bytes, the token-prefix record's convention), and PASS / FAIL against the token-prefix record."""
    import numpy as np

    out = []
    for it in items:
        mnt, f = it["file"].split(":", 1)
        h, row = hashlib.sha256(), None
        with open(f"{mnt}/{f}", "rb") as fh:
            for li, ln in enumerate(fh):
                h.update(ln)
                if it["eval_ref"].encode() in ln:
                    d = json.loads(ln)
                    if d.get("key") == it["eval_ref"]:
                        assert row is None, ("two staged lines", it["run"])
                        ids = np.asarray(d["ids"], np.int64)
                        row = {"line_index": li, "n_tokens": int(len(ids)), "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()}
        r = {"run": it["run"], "line": it["line"], "file": it["file"], "file_sha256": h.hexdigest()} | (row or {"row": "not found"})
        r["pass"] = bool(row) and row["n_tokens"] == it["want_n_tokens"] and row["ids_sha256"] == it["want_ids_sha256"]
        out.append(r)
    return out


def folder_map():
    """Laptop, keys only, every file sha256-checked: "<e1|e2> <world>|<cell>" -> the run's 32k folder and eval reference (e1 from the locked
    INPUTS_FULL's set inventories, only when unique; e2 from its E12 block, where a conflict STOPs: review B9), the E12 block's 392 pinned epoch-2
    files and its 1,475 epoch-2 runs, cross-checked against the block's own per-pool counts and its e2_shas (review B9), and the input shas."""
    import subprocess

    shas = {}

    def gj(c, p, s):
        b = subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True).stdout
        if hashlib.sha256(b).hexdigest() != s:
            raise SystemExit(f"STOP: {p} @ {c} differs from {s[:16]}")
        shas[f"{p} @ {c}"] = s
        return json.loads(b)

    inf = gj("927fabb0d", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_2026-10-16_110447.json",
             "d6923a75aa8f914cd3b2e92d8403f125dd095219997aebea8b1d6e4e2b67f77c")
    seen, evals, amb = {}, {}, 0
    for s in ("S_old", "S_new", "S_newmains", "S_paper493"):
        p = inf["inventories"][s]
        for r in gj(p["commit"], p["path"], p["sha256"])["runs"]:
            if r.get("world") in HONEST and r.get("original_folder"):
                seen.setdefault(f"e1 {r['world']}|{r['task']}", set()).add(r["original_folder"])
                evals.setdefault(f"e1 {r['world']}|{r['task']}", set()).add(r["key"])
    e12 = gj(inf["e12"]["commit"], inf["e12"]["path"], inf["e12"]["sha256"])
    ents, per_pool = [], {}

    def walk_e2(x, path):  # every "e2" list at ANY depth (2e: S_new's H_A nests them one level deeper, under each family group)
        if isinstance(x, dict):
            if isinstance(x.get("e2"), list):
                ents.extend(x["e2"])
                per_pool[path] = len(x["e2"])
            for k, v in x.items():
                walk_e2(v, f"{path}.{k}" if path else k)
        elif isinstance(x, list):
            for v in x:
                walk_e2(v, path)

    walk_e2(e12["pools"], "")
    keys2 = {e["key_e2"] for e in ents}
    n_by = {w: sum(1 for k in keys2 if k.startswith(w + "|")) for w in ("honest_a_e2", "honest_b_e2")}
    if len(keys2) != 1475 or n_by != {"honest_a_e2": 639, "honest_b_e2": 836}:
        raise SystemExit(f"STOP: e2 runs {len(keys2)} {n_by}, not 1,475 (639 + 836)")
    want_pool = {}  # the block's own counts, flattened to the walk's paths
    for s, d in e12["counts"].items():
        for pool, v in d.items():
            if isinstance(v, int):
                want_pool[f"{s}.{pool}"] = v
            elif isinstance(v, dict):
                want_pool |= {f"{s}.{pool}.{g}": n for g, n in v.items() if isinstance(n, int)}
    if any(per_pool.get(k, 0) != v for k, v in want_pool.items()):
        raise SystemExit(f"STOP: E12 per-pool e2 counts differ from the block's counts: {[(k, per_pool.get(k), v) for k, v in want_pool.items() if per_pool.get(k, 0) != v][:4]}")
    e2_shas = {e["eval_sha256"] for e in ents}
    if len(e2_shas) != e12["identity_asserts"]["ii"]["e2_shas"]:
        raise SystemExit(f"STOP: {len(e2_shas)} distinct e2 eval sha256, the block's e2_shas says {e12['identity_asserts']['ii']['e2_shas']}")
    if not {e["inventory_key"] for e in ents} <= set(e12["identity_sets"]["inventory_ep2_keys"]):
        raise SystemExit("STOP: an E12 entry's inventory_key is not in the block's inventory_ep2_keys")
    for e in ents:
        seen.setdefault(f"e2 {e['key']}", set()).add(e["folder"])
        evals.setdefault(f"e2 {e['key']}", set()).add(e["inventory_key"])
    conf = sorted(k for k in seen if k.startswith("e2 ") and (len(seen[k]) > 1 or len(evals[k]) > 1))
    if conf:  # review B9: conflicting duplicates STOP (never dropped)
        raise SystemExit(f"STOP: {len(conf)} e2 runs with two folders or eval references in the E12 block, e.g. {conf[:3]}")
    amb = sum(1 for k, v in seen.items() if k.startswith("e1 ") and len(v) > 1)
    one = lambda m: {k: next(iter(v)) for k, v in m.items() if len(v) == 1}  # noqa: E731
    files = sorted({"/" + f["vol"] + "/" + f["path"] for v in e12["folders"].values() for f in (v.get("files") or {}).values() if isinstance(f, dict)})
    if len(files) != 392:
        raise SystemExit(f"STOP: the E12 block pins {len(files)} epoch-2 files, not 392")
    e2_keys = sorted({e["key"] for e in ents})
    return one(seen), one(evals), files, e2_keys, shas, {"e1_runs_ambiguous": amb, "per_pool": per_pool}


TOKEN_PREFIX = ("experiments/2026-10-15_e2e_sim_pipeline/results/check_token_prefix_2026-10-16_192643.json",
                "c7b0da2f42cfece95ce1675983173d2c6383c80c92325d152fa10e0dbe06f7a6")
ROW_RUNS = ("honest_b|nm305_user_tracking", "honest_a_e2|nm305_user_tracking", "honest_b_e2|nm305_user_tracking")
REUSED = ("b8281a4f8", "experiments/2026-10-15_e2e_sim_pipeline/check_token_prefix.py", ("file_units", "classify", "common_prefix"))


@app.local_entrypoint()
def main(routing: str, out: str):
    import ast
    import os
    import subprocess

    here = os.path.abspath(__file__)
    src = open(here, "rb").read()
    tp_src = subprocess.run(["git", "show", f"{REUSED[0]}:{REUSED[1]}"], capture_output=True, check=True, text=True).stdout
    defs = lambda s: {n.name: ast.unparse(n) for n in ast.parse(s).body if isinstance(n, ast.FunctionDef)}  # noqa: E731
    a_, b_ = defs(src.decode()), defs(tp_src)
    if any(a_.get(f) != b_.get(f) for f in REUSED[2]):  # review B8: the reused functions are the reviewed ones, byte for byte (as parsed)
        raise SystemExit(f"STOP: {[f for f in REUSED[2] if a_.get(f) != b_.get(f)]} differ from {REUSED[1]} @ {REUSED[0]}")
    rt, rsh = {}, {}
    for f in sorted(os.listdir(routing)):
        if f.endswith(".json") and "_attest" not in f:
            rb = open(os.path.join(routing, f), "rb").read()
            rsh[f] = sha(rb)
            r = json.loads(rb)
            rt.setdefault(r["batch"], []).append({k: r.get(k) for k in ("batch", "volume", "staged_set", "original_staged_set", "original_folders", "arms")})
    tb = open(TOKEN_PREFIX[0], "rb").read()  # 09's nm305 row check: the token-prefix record's staged rows (keys and counts only), sha256-checked
    if sha(tb) != TOKEN_PREFIX[1]:
        raise SystemExit("STOP: the token-prefix record differs")
    tx = json.loads(tb)["runs_export"]
    items = [{"run": k, "line": ln, "file": s["file"], "eval_ref": tx[k]["eval_ref"], "want_n_tokens": s["n_tokens"], "want_ids_sha256": s["ids_sha256"]}
             for k in ROW_RUNS for ln, s in sorted(tx[k]["staged"].items()) if s]
    if len(items) != 6:
        raise SystemExit("STOP: the token-prefix record lacks an nm305 staged row")
    fo, ev, e12_files, e2_keys, in_shas, fm_info = folder_map()
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script_sha256": sha(src), "git": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", here)),  # review B5
            "inputs": in_shas | {f"{TOKEN_PREFIX[0]} (worktree)": TOKEN_PREFIX[1]}, "routing_files": len(rsh),
            "routing_sha256": sha(json.dumps(rsh, sort_keys=True).encode()), "folder_map": fm_info}
    r = drive.remote(rt, fo, ev, items, e12_files, e2_keys) | {"provenance": prov}
    # review B6: the verdict. Any of these is a STOP with a non-zero exit (the record is still written); "0 rows" never reads as clean
    stops = [*(["scan errors"] if r["scan_errors"] else []), *(["row check FAIL"] if not all(x["pass"] for x in r["row_check"]) else []),
             *(["no epoch-2 rows"] if not any(g["e2_rows"] for g in r["groups"]) else []),
             *[f"E12 files: {k}" for k in ("not_walked", "not_classified_e2", "scan_errors", "no_honest_rows") if r["coverage"][k]],
             *(["positive control: no famexp e2 = e1 copy for nm172 honest_a / honest_b"] if not all(r["positive_control_famexp_nm172"].values()) else []),
             *(["a sampled pair STOPs on its span_end"] if any(c["outcome"].startswith("STOP") for c in r["sample_classification"]) else [])]
    r["verdict"] = ("STOP: " + "; ".join(stops)) if stops else (f"INCOMPLETE: {len(r['incomplete_groups'])} groups without a hashed epoch-1 counterpart"
                                                                if r["incomplete_groups"] else "COMPLETE")
    with open(out, "x") as fh:
        json.dump(r, fh, indent=1)
    print(json.dumps({"verdict": r["verdict"], "npz_files": r["npz_files"], "npz_without_key": r["npz_without_key"], "honest_rows": r["honest_rows"],
                      "tagged_keys": r["tagged_keys"], "scan_errors": len(r["scan_errors"]),
                      "coverage": {k: (len(v) if isinstance(v, list) else v) for k, v in r["coverage"].items() if k != "e2_runs_missing"},
                      "positive_control": r["positive_control_famexp_nm172"], "incomplete_groups": len(r["incomplete_groups"]),
                      "groups_with_identical_e2_rows": r["groups_with_copies_candidates"],
                      "identical": [{"g": f"{g['line']} | {g['family']} | {g['file']}", "e2": g["e2_rows"], "same": g["identical_to_e1"],
                                     "e1_dirs": g["identical_e1_dirs"], "score_arrays": [a for a in ("score", "score_by_model", "score_by_fold") if a in g["array_names"]]}
                                    for g in r["groups"] if g["identical_to_e1"]],
                      "swap_test_signals": {k: v for k, v in r["swap_test"].items() if any(v[x] for x in v if x.startswith(("equal_to", "structure_e", "structure_m")))},
                      "row_check": [{k: x.get(k) for k in ("run", "line", "n_tokens", "pass")} for x in r["row_check"]],
                      "sample_summary": r["sample_summary"]}))
    if not r["verdict"].startswith("COMPLETE"):
        raise SystemExit(f"{r['verdict']} (record written: {out})")
