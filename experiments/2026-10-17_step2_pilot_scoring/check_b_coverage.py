"""Keys-only (B) coverage, BEFORE any spend (CUT / 2e / 507-prep / Astra, 10-17). For one STEP-2 regression label: the stored production files that
(B) compares NEW against are cited EXACTLY as the runner's _prod cites them: one well-formed entry (string path, lowercase 64-hex string sha256,
and its manifest's string path and 64-hex sha256) for every variant x readout (R, C) x S_old A25 folder in FOLDERS (b1, b2), and no b1 / b2 entry
under any other key of the label's subtree (_prod would cite it too, and (B)'s set(b_out) == new_files would then fail after the spend). Entries
for folders OUTSIDE FOLDERS (e.g. t196fill, sp493_g4 under the same keys) are ignored, exactly as _prod ignores them; every element is still
validated as an object with a string folder (_prod reads x["folder"] of each). Paths are usable the way the runner uses them. Variants come from
the scorer's own constants (git show HEAD; each needed name bound exactly once, by a module-level assignment without calls; no repeats).
Sources are committed and read with git show, each sha-checked against its pin, parsed with duplicate JSON keys refused:
  - own_all_rs_uncut, own_all_rs, famexp (the runner's prod "inputs_full"): INPUTS_FULL 2026-10-16_110447 @ 927fabb0d -> its S_old|R and S_old|C
    overrides (commit, path, sha256 from INPUTS_FULL) -> every simh["<sub>/..."] key, its b1 / b2 entries (as _prod); each path RELATIVE, no ".."
    component, in <...>/<folder>/<variant>__<rd>.npz, its manifest in the same directory;
  - own_all_rs_simtrained_uncut, own_all_loo3_uncut, own_all_loo4_uncut (prod "record": not pinned in INPUTS_FULL): the probe's committed score
    record (the runner's LABELS[label]["orig"]; its folder entry selected exactly as the runner's _prod does, path == /work/cutfull/sold/<sub>/<folder>,
    exactly ONE, its manifest_sha256 a 64-hex string) and the probe's latest committed addendum pins (score_files["<sub>/<folder>"]: manifest path
    == /cutfull/sold/<sub>/<folder>/manifest_own_all.json with the record's sha256, the score record named, files exactly {R, C}, each path ==
    /cutfull/sold/<sub>/<folder>/<variant>__<rd>.npz with a 64-hex sha256).
Run each label on ITS package branch (SIMT_VARS exists only in the simtrained scorer, LOOU only in loo3's; elsewhere the check STOPs).
KEYS-ONLY means: whole JSON documents are parsed, but only key names, paths, commits, shas, counts and folder / variant names are accessed or
printed; no measured field (score, statistic, threshold, catch) is ever accessed. Output: stdout only (no file is written). A malformed source
structure STOPs with a reason. Exit 1 on any gap. No Modal.
  uv run --no-project --python 3.12 python -I check_b_coverage.py <label>      (labels: LABELS below)
  uv run --no-project --python 3.12 python -I check_b_coverage.py --selftest   (synthetic objects only; reads nothing)"""

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
P2 = "experiments/2026-10-14_p2_family_expansion"
RES = f"{P2}/results"
FOLDERS = ("simh_q4f_v1_full_b1_cutfull_v1__sim_honest", "simh_q4f_v1_full_b2_cutfull_v1__sim_honest")  # S_old A25 (the runners' FOLDERS)
READOUTS = ("R", "C")
INPUTS_FULL = ("927fabb0d", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_2026-10-16_110447.json",
               "d6923a75aa8f914cd3b2e92d8403f125dd095219997aebea8b1d6e4e2b67f77c")  # the FROZEN inputs (the runners' INPUTS_FULL)
# label -> (scorer, variant expression in the scorer, prod kind, sub / tree, score record pin, addendum pins pin); record pins: the commit that added it
LABELS = {
    "own_all_rs_uncut": (f"{P2}/run_own_all.py", "list(UNCUT_VARS)", "inputs_full", "own_all_rs_uncut", None, None),
    "own_all_rs": (f"{P2}/run_own_all.py", "list(RS_VARS)", "inputs_full", "own_all_rs", None, None),
    "famexp": (f"{P2}/run_famexp_cutfull.py", "list(OWN_FRAC)", "inputs_full", "famexp", None, None),
    "own_all_rs_simtrained_uncut": (f"{P2}/run_own_all.py", "list(SIMT_VARS)", "record", "own_all_rs_simtrained_uncut",
                                    ("3e2aaae62", f"{RES}/own_all_rs_simtrained_uncut_cutfull_score_2026-10-16_154236.json",
                                     "3b6d0811e08bf98d59e81e222df1ba6b8b388f981840a1c477e2c33a3ec7bb4e"),
                                    ("cead5bc6f", f"{RES}/own_all_rs_simtrained_uncut_addendum_pins_2026-10-16_163049.json",
                                     "d266beec04f040d31839216fe539eb011eb76ba202f14ae05003249e84ee99cc")),
    "own_all_loo3_uncut": (f"{P2}/run_own_all.py", "[LOOU[False][0]]", "record", "own_all_loo3_uncut",
                           ("e5a00e652", f"{RES}/own_all_loo3_uncut_cutfull_score_2026-10-16_034101.json",
                            "56aecb10ca16198422a9ac775d330e1e8f59f190c5e8c0ba7a8885f4f007f8d1"),
                           ("98a0cd49b", f"{RES}/own_all_loo3_uncut_addendum_pins_2026-10-16_171312.json",
                            "ec74f97bb225e71b933393dc95526b123cf31c407170ca8980fc6c49e340b63d")),
    "own_all_loo4_uncut": (f"{P2}/run_own_all.py", "[LOOU[True][0]]", "record", "own_all_loo4_uncut",
                           ("ef198839f", f"{RES}/own_all_loo4_uncut_cutfull_score_2026-10-16_041727.json",
                            "98e109b0af35487376498d0f47b7552bfdc6cd461e189302a5ac9d8f82c0f632"),
                           ("e422298cf", f"{RES}/own_all_loo4_uncut_addendum_pins_2026-10-16_101448.json",
                            "6055514950120f3c49dbbfe3d4b156fdd4d0c8aee2eba0a5b661bbbcd02b4f94")),
}
HEX64 = re.compile(r"[0-9a-f]{64}")


def _str(x):
    return isinstance(x, str) and x != ""


def _hex(x):
    return isinstance(x, str) and HEX64.fullmatch(x) is not None


def _no_dup_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"STOP: a JSON object repeats a key: {sorted({k for k in keys if keys.count(k) > 1})[:3]}")
    return dict(pairs)


def _loads(b):
    """json.loads refusing duplicate keys (json would silently keep the last one)."""
    return json.loads(b, object_pairs_hook=_no_dup_keys)


def show(commit, path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True, check=True).stdout


def pinned(pin):
    b = show(pin[0], pin[1])
    if hashlib.sha256(b).hexdigest() != pin[2]:
        raise SystemExit(f"STOP: {pin[1]} @ {pin[0]} differs from its pinned sha256")
    return _loads(b)


def _bound_names(node):
    """Names a module-level statement binds (assignment targets incl. tuples, for / with targets, imports, defs), not looking into def / class /
    lambda bodies."""
    out = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [(a.asname or a.name).split(".")[0] for a in node.names]
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)) else \
        [i.optional_vars for i in node.items if i.optional_vars] if isinstance(node, (ast.With, ast.AsyncWith)) else []
    for t in targets:
        out += [x.id for x in ast.walk(t) if isinstance(x, ast.Name)]
    for f in ("body", "orelse", "finalbody"):
        for x in getattr(node, f, []):
            out += _bound_names(x)
    for h in getattr(node, "handlers", []):
        for x in h.body:
            out += _bound_names(x)
    return out


def variants(scorer_src, expr):
    """The scorer's variant names: expr over its module-level constant assignments WITHOUT any call (string / dict / tuple literals, f-string
    comprehensions), exec'd in an empty namespace. Every name expr needs (and, transitively, every name those assignments read) must be bound
    EXACTLY ONCE at module level, by an assignment without any call (a call, a def or an import binding it: STOP). Code only, never a record."""
    body = ast.parse(scorer_src).body
    counts = {}
    for n in body:
        for nm in _bound_names(n):
            counts[nm] = counts.get(nm, 0) + 1
    assigns = {}
    for n in body:
        if isinstance(n, ast.Assign) and all(isinstance(t, (ast.Name, ast.Tuple)) for t in n.targets) and not any(isinstance(x, ast.Call) for x in ast.walk(n.value)):
            for nm in _bound_names(n):
                assigns.setdefault(nm, n)
    need, todo = set(), {x.id for x in ast.walk(ast.parse(expr, mode="eval")) if isinstance(x, ast.Name)} - {"list"}
    while todo:
        nm = todo.pop()
        need.add(nm)
        if nm in assigns:
            comp = {t.id for x in ast.walk(assigns[nm].value) if isinstance(x, ast.comprehension) for t in ast.walk(x.target) if isinstance(t, ast.Name)}
            todo |= {x.id for x in ast.walk(assigns[nm].value) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)} - need - comp
    bad = sorted(f"{nm} bound {counts.get(nm, 0)} times" for nm in need if counts.get(nm, 0) != 1)
    bad += sorted(f"{nm} not bound by a call-free assignment" for nm in need if counts.get(nm, 0) == 1 and nm not in assigns)
    if bad:
        raise SystemExit(f"STOP: the scorer's variant constants are not bound exactly once by call-free assignments: {bad}")
    ns = {}
    for n in body:
        if n in {assigns[nm] for nm in need}:  # only the call-free assignments of the needed names, in file order
            exec(compile(ast.Module([n], []), "<scorer constants>", "exec"), {"__builtins__": {}}, ns)  # noqa: S102
    return eval(expr, {"__builtins__": {"list": list}}, ns)  # noqa: S307


def _distinct(vs):
    if not vs or not all(_str(v) for v in vs) or len(set(vs)) != len(vs):
        raise SystemExit(f"STOP: variant names must be distinct non-empty strings ({len(vs)} given, {len(set(map(str, vs)))} distinct)")


def cover_inputs_full(sub, vs, overrides):
    """overrides: {rd: the S_old|<rd> override JSON}. Aligned with the runner's _prod: every simh key under "<sub>/" is read, each a LIST whose
    every element is validated (an object with a string folder); only its b1 / b2 (FOLDERS) entries count, entries for other folders are ignored.
    Each variant x readout key needs exactly one entry per folder in FOLDERS; any other "<sub>/" key carrying a b1 / b2 entry is extra.
    Returns (expected, cited, missing, malformed, extra)."""
    _distinct(vs)
    exp, cited, missing, bad, extra = 0, 0, [], [], []
    for rd in READOUTS:
        simh = overrides[rd].get("simh") if isinstance(overrides[rd], dict) else None
        if not isinstance(simh, dict):
            bad.append(f"S_old|{rd}: simh is not an object")
            continue
        want = {f"{sub}/{v}__{rd}.npz": v for v in vs}
        for key in sorted({k for k in simh if k.split("/", 1)[0] == sub} | set(want)):
            ents, cand = simh.get(key), {fo: [] for fo in FOLDERS}
            if ents is not None and not isinstance(ents, list):
                bad.append(f"{key}: not a list of entries")
                ents = []
            for i, e in enumerate(ents or []):  # EVERY element validated before any folder is matched (_prod reads x["folder"] of each)
                if not isinstance(e, dict) or not _str(e.get("folder")):
                    bad.append(f"{key}[{i}]: not an object with a string folder")
                elif e["folder"] in FOLDERS:  # entries for other folders (t196fill, sp493_g4, ...) are ignored, as _prod ignores them
                    cand[e["folder"]].append(e)
            if key not in want:
                if any(cand.values()):
                    extra.append(f"{key} (S_old|{rd}: a b1 / b2 entry under a key outside variant x {rd})")
                continue
            v = want[key]
            for fo in FOLDERS:
                exp += 1
                if len(cand[fo]) != 1:
                    missing.append(f"{key} @ {fo} ({len(cand[fo])} entries)")
                    continue
                e = cand[fo][0]
                m = e.get("manifest") if isinstance(e.get("manifest"), dict) else {}
                p, mp = e.get("path"), m.get("path")
                pp, mm = (PurePosixPath(p) if _str(p) else None), (PurePosixPath(mp) if _str(mp) else None)
                why = [w for w, ok in (("path not a non-empty string", pp is not None), ("sha256 not a 64-hex string", _hex(e.get("sha256"))),
                                        ("manifest path not a non-empty string", mm is not None), ("manifest sha256 not a 64-hex string", _hex(m.get("sha256"))),
                                        ("path absolute or with '..', or not .../<folder>/<variant>__<rd>.npz",
                                         pp is not None and not pp.is_absolute() and ".." not in pp.parts and pp.name == f"{v}__{rd}.npz" and pp.parent.name == fo),
                                        ("manifest absolute, with '..', or not in the file's directory",
                                         mm is not None and pp is not None and not mm.is_absolute() and ".." not in mm.parts and mm.parent == pp.parent)) if not ok]
                cited += not why
                if why:
                    bad.append(f"{key} @ {fo}: {'; '.join(why)}")
    return exp, cited, missing, bad, extra


def cover_record(sub, vs, record, pins, record_path):
    """record: the probe's committed score record, its folder entries selected EXACTLY as the runner's _prod does (fo in FOLDERS and path ==
    /work/cutfull/sold/<sub>/<fo>), exactly ONE per folder (several, even with equal hashes, FAIL); pins: its addendum pins (see the module
    docstring). Same returns."""
    _distinct(vs)
    exp, cited, missing, bad, extra = 0, 0, [], [], []
    results = record.get("results") if isinstance(record, dict) else None
    if not isinstance(results, dict) or not all(isinstance(r, dict) and isinstance(r.get("folders", {}), dict) for r in results.values()):
        return len(vs) * len(READOUTS) * len(FOLDERS), 0, [], ["score record: results / folders not objects"], []
    sel = {fo: [x for r in results.values() for f_, x in r.get("folders", {}).items() if f_ == fo and isinstance(x, dict)
                and x.get("path") == f"/work/cutfull/sold/{sub}/{fo}"] for fo in FOLDERS}
    sfs = pins.get("score_files") if isinstance(pins, dict) else None
    for fo in FOLDERS:
        sf = sfs.get(f"{sub}/{fo}") if isinstance(sfs, dict) else None
        sf = sf if isinstance(sf, dict) else {}
        files = sf.get("files") if isinstance(sf.get("files"), dict) else {}
        extra += sorted(f"{sub}/{fo} files.{k} (not R / C)" for k in files if k not in READOUTS)
        man = sf.get("manifest") if isinstance(sf.get("manifest"), dict) else {}
        rman = sel[fo][0].get("manifest_sha256") if len(sel[fo]) == 1 else None
        fwhy = [w for w, ok in ((f"score record entries for {fo} at the runner's path: {len(sel[fo])} (want exactly 1)", len(sel[fo]) == 1),
                                 ("score record manifest_sha256 not a 64-hex string", _hex(rman)),
                                 ("pins manifest path not /cutfull/sold/<sub>/<folder>/manifest_own_all.json", man.get("path") == f"/cutfull/sold/{sub}/{fo}/manifest_own_all.json"),
                                 ("pins manifest sha256 not a 64-hex string equal to the score record's", _hex(man.get("sha256")) and man.get("sha256") == rman),
                                 ("pins name another score record", sf.get("score_record") == record_path)) if not ok]
        for v in vs:
            for rd in READOUTS:
                exp += 1
                f = files.get(rd)
                if not isinstance(f, dict):
                    missing.append(f"{sub}/{fo} {rd}")
                    continue
                why = fwhy + [w for w, ok in (("sha256 not a 64-hex string", _hex(f.get("sha256"))),
                                               ("path not /cutfull/sold/<sub>/<folder>/<variant>__<rd>.npz", f.get("path") == f"/cutfull/sold/{sub}/{fo}/{v}__{rd}.npz")) if not ok]
                cited += not why
                if why:
                    bad.append(f"{sub}/{fo} {v}__{rd}: {'; '.join(why)}")
    return exp, cited, missing, bad, extra


def check(label):
    scorer, expr, kind, sub, rec_pin, pins_pin = LABELS[label]
    try:
        vs = variants(show("HEAD", scorer).decode(), expr)  # the committed scorer at HEAD (code)
        if not isinstance(vs, list):
            raise SystemExit(f"STOP: {expr} did not give a list")
        _distinct(vs)
        if kind == "inputs_full":
            ov = pinned(INPUTS_FULL)["overrides"]
            pins_ = {rd: (ov[f"S_old|{rd}"]["commit"], ov[f"S_old|{rd}"]["path"], ov[f"S_old|{rd}"]["sha256"]) for rd in READOUTS}
            if not all(_str(c) and _str(p) and _hex(h) for c, p, h in pins_.values()):
                raise SystemExit("STOP: an S_old override pin in INPUTS_FULL is not (string commit, string path, 64-hex sha256)")
            res = cover_inputs_full(sub, vs, {rd: pinned(pins_[rd]) for rd in READOUTS})
            sources = {"inputs_full": list(INPUTS_FULL), "overrides": {rd: list(pins_[rd]) for rd in READOUTS}}
        else:
            res = cover_record(sub, vs, pinned(rec_pin), pinned(pins_pin), rec_pin[1])
            sources = {"score_record": list(rec_pin), "addendum_pins": list(pins_pin)}
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as e:  # a malformed source STOPs with a reason (a key name, never a value)
        raise SystemExit(f"STOP: malformed source structure ({type(e).__name__}{': missing key ' + repr(e.args[0]) if isinstance(e, KeyError) else ''})") from None
    exp, cited, missing, bad, extra = res
    out = {"label": label, "prod": kind, "variants": vs, "folders": list(FOLDERS), "readouts": list(READOUTS), "expected": exp, "cited_exactly": cited,
           "missing": missing, "malformed": bad, "extra": extra, "sources": sources,
           "pass": exp == cited == len(vs) * len(READOUTS) * len(FOLDERS) and not missing and not bad and not extra}
    print(json.dumps(out, indent=1))
    return out["pass"]


def selftest():
    """Synthetic objects only: a full citation set passes; each listed defect fails (or STOPs) with its key named."""
    h, cp = "a" * 64, lambda o: json.loads(json.dumps(o))  # noqa: E731
    vs = ["v1", "v2"]
    ovr = {rd: {"simh": {f"s/{v}__{rd}.npz": [{"folder": fo, "path": f"cutfull/sold/s/{fo}/{v}__{rd}.npz", "sha256": h,
                                                "manifest": {"path": f"cutfull/sold/s/{fo}/manifest.json", "sha256": h}} for fo in FOLDERS] for v in vs}
                        | {"other/x__R.npz": []}} for rd in READOUTS}
    res = {"inputs_full: full set passes (another subtree ignored)": cover_inputs_full("s", vs, ovr)[1:] == (8, [], [], [])}

    def mod(f):
        t = cp(ovr)
        f(t)
        return cover_inputs_full("s", vs, t)
    res["dropped variant x readout"] = mod(lambda t: t["C"]["simh"].pop("s/v2__C.npz"))[2] == [f"s/v2__C.npz @ {fo} (0 entries)" for fo in FOLDERS]
    res["dropped folder"] = mod(lambda t: t["R"]["simh"]["s/v1__R.npz"].pop())[2] == [f"s/v1__R.npz @ {FOLDERS[1]} (0 entries)"]
    res["(A) extra key WITH a b1 / b2 entry fails"] = mod(lambda t: t["R"]["simh"].__setitem__("s/v9__R.npz", [dict(t["R"]["simh"]["s/v1__R.npz"][0])]))[4] == \
        ["s/v9__R.npz (S_old|R: a b1 / b2 entry under a key outside variant x R)"]
    res["(A) extra key with ONLY other-folder entries passes"] = mod(lambda t: t["R"]["simh"].__setitem__("s/v9__R.npz", [dict(t["R"]["simh"]["s/v1__R.npz"][0], folder="t196fill")]))[1:] == \
        (8, [], [], [])
    res["(A) extra empty key passes"] = mod(lambda t: t["R"]["simh"].__setitem__("s/v9__R.npz", []))[1:] == (8, [], [], [])
    res["(A) another folder's entry under an expected key passes"] = mod(lambda t: t["C"]["simh"]["s/v1__C.npz"].append(
        dict(t["C"]["simh"]["s/v1__C.npz"][0], folder="other")))[1:] == (8, [], [], [])
    four = ("simh_t196fill_cutfull_v1__sim_honest", "sp493_g4_simh_cutfull_v1__sim_honest")
    real = {rd: {"simh": {f"s/{v}__{rd}.npz": [{"folder": fo, "path": f"cutfull/sold/s/{fo}/{v}__{rd}.npz", "sha256": h,
                                                 "manifest": {"path": f"cutfull/sold/s/{fo}/manifest.json", "sha256": h}} for fo in FOLDERS + four] for v in vs}}
            for rd in READOUTS}
    res["(2e) real-shaped keys with b1, b2, t196fill and sp493_g4 entries pass"] = cover_inputs_full("s", vs, real)[1:] == (8, [], [], [])
    nob1 = cp(real)
    nob1["R"]["simh"]["s/v1__R.npz"] = [e for e in nob1["R"]["simh"]["s/v1__R.npz"] if e["folder"] != FOLDERS[0]]
    r_ = cover_inputs_full("s", vs, nob1)
    res["(2e) real-shaped key missing its b1 entry fails"] = r_[1] == 7 and r_[2] == [f"s/v1__R.npz @ {FOLDERS[0]} (0 entries)"]
    res["(4) a non-object element in a selected list"] = mod(lambda t: t["R"]["simh"]["s/v1__R.npz"].append("not an entry"))[3] == \
        ["s/v1__R.npz[2]: not an object with a string folder"]
    res["(4) an empty object in a selected list"] = mod(lambda t: t["R"]["simh"]["s/v1__R.npz"].append({}))[3] == ["s/v1__R.npz[2]: not an object with a string folder"]
    res["(4) the entry list is not a list"] = mod(lambda t: t["R"]["simh"].__setitem__("s/v1__R.npz", {"folder": FOLDERS[0]}))[3][:1] == ["s/v1__R.npz: not a list of entries"]
    res["two entries for one folder"] = mod(lambda t: t["R"]["simh"]["s/v1__R.npz"].append(t["R"]["simh"]["s/v1__R.npz"][0]))[2] == [f"s/v1__R.npz @ {FOLDERS[0]} (2 entries)"]
    for nm, field, val in (("(b) absolute path", "path", f"/cutfull/sold/s/{FOLDERS[0]}/v1__R.npz"),
                           ("(b) '..' in the path", "path", f"cutfull/../s/{FOLDERS[0]}/v1__R.npz"),
                           ("(b) manifest in another directory", "manifest", {"path": "cutfull/sold/elsewhere/manifest.json", "sha256": h}),
                           ("(1) manifest without a path", "manifest", {"sha256": h}),
                           ("(d) bad manifest sha256", "manifest", {"path": f"cutfull/sold/s/{FOLDERS[0]}/manifest.json", "sha256": "x"}),
                           ("(2) manifest sha256 an int", "manifest", {"path": f"cutfull/sold/s/{FOLDERS[0]}/manifest.json", "sha256": 7}),
                           ("(2) path an int", "path", 123), ("(2) file sha256 an int", "sha256", int("1" * 20)), ("bad file sha256", "sha256", "A" * 64)):
        r_ = mod(lambda t, field=field, val=val: t["R"]["simh"]["s/v1__R.npz"][0].__setitem__(field, val))
        res[nm] = r_[1] == 7 and len(r_[3]) == 1 and r_[3][0].startswith(f"s/v1__R.npz @ {FOLDERS[0]}")
    rec = {"results": {"simh|m": {"folders": {fo: {"manifest_sha256": h, "path": f"/work/cutfull/sold/t/{fo}"} for fo in FOLDERS}}}}
    pins = {"score_files": {f"t/{fo}": {"manifest": {"path": f"/cutfull/sold/t/{fo}/manifest_own_all.json", "sha256": h}, "score_record": "r.json",
                                        "files": {rd: {"path": f"/cutfull/sold/t/{fo}/v1__{rd}.npz", "sha256": h} for rd in READOUTS}} for fo in FOLDERS}}
    res["record: full set passes"] = cover_record("t", ["v1"], rec, pins, "r.json")[1:] == (4, [], [], [])

    def modr(fp=None, fr=None):
        p_, r_ = cp(pins), cp(rec)
        if fp:
            fp(p_)
        if fr:
            fr(r_)
        return cover_record("t", ["v1"], r_, p_, "r.json")
    f0 = f"t/{FOLDERS[0]}"
    res["record: manifest differs from the score record's"] = len(modr(lambda p: p["score_files"][f0]["manifest"].__setitem__("sha256", "b" * 64))[3]) == 2
    res["(1) record: pins manifest without a path"] = len(modr(lambda p: p["score_files"][f0]["manifest"].pop("path"))[3]) == 2
    res["(2) record: file sha256 an int"] = len(modr(lambda p: p["score_files"][f0]["files"]["R"].__setitem__("sha256", 5))[3]) == 1
    res["(2) record: score record manifest_sha256 an int"] = len(modr(fr=lambda r: r["results"]["simh|m"]["folders"][FOLDERS[0]].__setitem__("manifest_sha256", 5))[3]) == 2
    res["record: dropped readout"] = modr(lambda p: p["score_files"][f0]["files"].pop("C"))[2] == [f"{f0} C"]
    res["(a) record: extra file key"] = modr(lambda p: p["score_files"][f0]["files"].__setitem__("X", {"path": "p", "sha256": h}))[4] == [f"{f0} files.X (not R / C)"]
    res["record: file path elsewhere"] = len(modr(lambda p: p["score_files"][f0]["files"]["R"].__setitem__("path", f"/cutfull/sold/x/{FOLDERS[0]}/v1__R.npz"))[3]) == 1
    res["(c) record: zero entries at the runner's path"] = len(modr(fr=lambda r: r["results"]["simh|m"]["folders"][FOLDERS[0]].__setitem__("path", "/work/x"))[3]) == 2
    res["(c / 5) record: two entries, conflicting hashes"] = len(modr(fr=lambda r: r["results"].__setitem__("simh|n", {"folders": {
        FOLDERS[0]: {"manifest_sha256": "c" * 64, "path": f"/work/cutfull/sold/t/{FOLDERS[0]}"}}}))[3]) == 2
    res["(c) record: two entries, equal hashes"] = len(modr(fr=lambda r: r["results"].__setitem__("simh|n", cp(r["results"]["simh|m"])))[3]) == 4
    for nm, f_ in (("(3) duplicate JSON key", lambda: _loads(b'{"a": 1, "a": 2}')), ("(6) repeated variants (inputs_full)", lambda: cover_inputs_full("s", ["v1", "v1"], ovr)),
                   ("(6) repeated variants (record)", lambda: cover_record("t", ["v1", "v1"], rec, pins, "r.json"))):
        try:
            f_()
            res[nm] = False
        except SystemExit as e:
            res[nm] = str(e).startswith("STOP")
    res["no duplicate: parses"] = _loads(b'{"a": 1, "b": {"a": 2}}') == {"a": 1, "b": {"a": 2}}
    res["variants from scorer constants (no calls run)"] = variants('A = {"x_L48": 1.0, "y_L48": 0.5}\nB = [f"f{p}" for p in (1, 2)]\nC = open("z")\n',
                                                                    "list(A) + list(B)") == ["x_L48", "y_L48", "f1", "f2"]
    res["variants through a tuple-bound name"] = variants('X, Y = "sv_L48", "d"\nS = {X: 1.0}\n', "list(S)") == ["sv_L48"]
    for nm, src_ in (("needed binding with a call STOPs", 'A = dict(x=1.0)\n'), ("needed binding by a def STOPs", 'def A():\n    return 1\n'),
                     ("a name it reads bound with a call STOPs", 'X = str("a")\nS = {X: 1.0}\n')):
        try:
            variants(src_, "list(S)" if "S =" in src_ else "list(A)")
            res[nm] = False
        except SystemExit as e:
            res[nm] = str(e).startswith("STOP")
    for nm, src_, ex in (("(e) duplicate binding", 'A = {"x": 1.0}\nA = {"y": 1.0}\n', "list(A)"),
                         ("(e) duplicate via an if block", 'A = {"x": 1.0}\nif True:\n    A = {"y": 1}\n', "list(A)"),
                         ("(e) duplicate of a name it reads", 'X = "a"\nS = {X: 1.0}\nX = "b"\n', "list(S)"), ("(e) unbound name", 'B = {"x": 1.0}\n', "list(A)")):
        try:
            variants(src_, ex)
            res[nm] = False
        except SystemExit as e:
            res[nm] = str(e).startswith("STOP")
    print(json.dumps(res, indent=1))
    return all(res.values())


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in (*LABELS, "--selftest"):
        raise SystemExit(f"usage: check_b_coverage.py <{' | '.join(LABELS)}> | --selftest")
    ok = selftest() if sys.argv[1] == "--selftest" else check(sys.argv[1])
    raise SystemExit(0 if ok else 1)
