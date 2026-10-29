"""Token-prefix check (09 / 0d via 2e 19:1x; NOT RUN until CUT's glance). Are identical probe units between two runs NATURAL (the two runs' STAGED token
ids are identical through every compared readout, and each compared unit has the same scoring model + fold) or an extraction-level MIX-UP?
09's rule, fixed in advance: (1) every readout of both runs inside the common prefix of the two staged id arrays, with the same routing -> identity
natural, no mix-up; (2) any readout outside it (or routing differing) where the units are identical -> extraction-level mix-up. This is also the
generalized gate 2 (09): classify() is the reusable core. KEYS, COUNTS, POSITIONS AND HASHES ONLY: token ids are never decoded or printed.

Per pair and line (32k: the q3su311 acts Volume; uncut: q3cutfull), from the inputs file (make_token_prefix_inputs.py):
- each run's staged row (the line whose key is its eval key): n_tokens, the sha256 of its ids, its staged calls' readout positions by kind;
- the common-prefix length d of the two id arrays and the first differing index (d, if d < both lengths), and the tails beyond d;
- every score file of the pair (matched by name within the line, a trailing __L<k> layer tag ignored): per unit the readout position (token; a
  32k file without positions uses its units' call_index -> the staged calls' readouts, marked coarse), the needed prefix token + 1 (CUT R1: one
  causal forward; span_end is only checked to be token + 1 (the extractor's [start, end)) or token, else STOP), the routing (fold per unit; fold_r36 when present; "single model" when no fold), the observed identity (score, fold and
  score_by_model byte-equal, gate 2's s / f / sbm), the prediction (positions equal, every readout inside d, routing equal) and the outcome:
  "natural" (identical, predicted), "MIX-UP" (identical, not predicted), "different" (not identical, not predicted), or "UNEXPECTED DIFFERENCE"
  (predicted identical, observed different: not a mix-up, listed apart);
- by readout type (C: __C / __command / __cmd_A; R: __R / __reply_mean / __reply_last / __avg_top2; CE: __call_end_content) and family.
Also: the sim nm172 run's .eval metadata "fork" (identifier-like values only); controls: (i) a run against itself, identical -> natural; (ii) a
PLANTED change of one token at the first readout and one just before it, units "identical" -> both must be MIX-UP; (iii) the real long e2/e1 pair
(nm204), which must be INFORMATIVE (its d below its first readout's need, CUT (b)) and must not come out natural. An export per run (eval key, staged row, readout positions by type, routing per unit)
for refresh2's pooled gate 2. Hash maps are written as [name, sha256] pairs. CPU only, one container, 2,400 s.
Usage: modal run --detach check_token_prefix.py --inputs <json> --out <new local json>"""
import hashlib
import json
import re

import modal

VOLS = {"/conv": ("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", None), "/p2": ("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", 2),
        "/sol": ("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", 2), "/a32": ("probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", None), "/aU": ("probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14", None)}
LINE_OF_FILE = lambda p: "uncut" if "/cutfull" in p else "32k"  # noqa: E731
TYPES = [("C", re.compile(r"(__C(__L\d+)?|__command|__cmd_A)\.npz$")), ("R", re.compile(r"(__R(__L\d+)?|__reply_mean|__reply_last|__avg_top2)\.npz$")),
         ("CE", re.compile(r"__call_end_content\.npz$"))]
SIM_EVAL = "2026-10-15_simh_nm_s1/sim_honest@september/nm172_shared_dotfiles/2026-10-15T19-11-26-00-00_la-shared-dotfiles-sim-honest-1_kqJxHhCUmqLXYxy5NAvzE8.eval"
IDV = re.compile(r"[A-Za-z0-9_.:/@#|=+-]{0,300}")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.1.3", "zipfile-zstd==0.0.4")
app = modal.App("probe-shift_aswin-surya_sub-e2esim_tokprefix_kill-2026-11-16")
mounts = {m: (modal.Volume.from_name(n, version=v) if v else modal.Volume.from_name(n)).read_only() for m, (n, v) in VOLS.items()}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def scorer_of(p):  # CUT (a): the directory above the file, skipping run folders (<set>__<arm>), layers / pass* / timestamp dirs; a trailing _r<k> dropped
    parts = p.split("/")[:-1]
    while parts and (re.match(r"^[\w.@-]+__[\w.@-]+$", parts[-1]) or parts[-1] == "layers" or parts[-1].startswith("pass")
                     or re.match(r"^\d{4}-\d{2}-\d{2}_\d+$", parts[-1])):
        parts.pop()
    return re.sub(r"_r\d+$", "", parts[-1]) if parts else ""


def rtype(name):
    return next((t for t, rx in TYPES if rx.search(name)), "other")


def common_prefix(a, b):  # -> the length of the common prefix of two int sequences
    import numpy as np

    n = min(len(a), len(b))
    ne = np.flatnonzero(np.asarray(a[:n]) != np.asarray(b[:n]))
    return int(ne[0]) if len(ne) else n


def classify(d, need_x, need_y, pos_equal, routing_equal, observed_identical):
    """09's generalized gate 2 on one compared file: d = common-prefix length of the two staged id arrays; need_* = the needed prefix per unit (last
    readout position + 1). -> (predicted identical, outcome)."""
    inside = bool(need_x) and bool(need_y) and max(need_x) <= d and max(need_y) <= d
    pred = bool(pos_equal and routing_equal and inside)
    out = ("natural" if pred else "MIX-UP") if observed_identical else ("UNEXPECTED DIFFERENCE" if pred else "different")
    return pred, out


def unit_bytes(a_):  # pw_canon on one array
    import numpy as np

    a_ = np.asarray(a_)
    if a_.dtype.kind == "f":
        a_ = np.where(np.isnan(a_.astype(np.float64)), np.nan, a_.astype(np.float64) + 0.0)
    elif a_.dtype.kind in "iub":
        a_ = a_.astype(np.int64)
    return repr(a_.shape).encode() + b":" + np.ascontiguousarray(a_).tobytes()


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


@app.function(image=image, cpu=4, memory=16384, timeout=2400, retries=0, volumes=mounts)
def check(inp: dict) -> dict:
    import os
    import zipfile
    from pathlib import Path

    import numpy as np
    import zipfile_zstd  # noqa: F401

    runs, out = inp["runs"], {"pairs": [], "runs_export": {}, "controls": {}}
    want = {}  # (mount, staged file) -> eval keys wanted
    for lab, r in runs.items():
        for line, (m, f) in r["staged"].items():
            want.setdefault((m, f), set()).add(r["eval_ref"])
    staged = {}  # (line, eval key) -> {ids, calls, file, line_index}
    for (m, f), keys in sorted(want.items()):
        line = "32k" if m == "/a32" else "uncut"
        with open(Path(m) / f, "rb") as fh:
            for i, ln in enumerate(fh):
                for ek in keys:
                    if ek.encode() in ln:
                        d = json.loads(ln)
                        if d.get("key") == ek:
                            assert (line, ek) not in staged, ("one key on two staged lines", ek)
                            staged[(line, ek)] = {"ids": np.asarray(d["ids"], np.int64), "calls": d.get("calls") or [], "file": f"{m}:{f}", "line_index": i}
    for lab, r in runs.items():
        out["runs_export"][lab] = {"eval_ref": r["eval_ref"], "npz_row": r["npz_row"], "main_sha256_12": r.get("main_sha256_12"), "staged": {}, "files": []}
        for line in ("32k", "uncut"):
            s = staged.get((line, r["eval_ref"]))
            out["runs_export"][lab]["staged"][line] = None if s is None else {
                "file": s["file"], "line_index": s["line_index"], "n_tokens": int(len(s["ids"])), "ids_sha256": sha(s["ids"].tobytes()),
                "calls_by_kind": {k_: sorted(c["readout"] for c in s["calls"] if isinstance(c, dict) and c.get("kind") == k_ and isinstance(c.get("readout"), int))
                                  for k_ in sorted({c.get("kind") for c in s["calls"] if isinstance(c, dict)}, key=str)}}
    fu = {}  # (label, path) -> units
    for lab, r in runs.items():
        for _, p in r["files"]:
            line = LINE_OF_FILE(p)
            s = staged.get((line, r["eval_ref"]))
            u = file_units(p, r["npz_row"], s["calls"] if s else []) if os.path.exists(p) else {"missing": True}
            fu[(lab, p)] = u
            if u and "missing" not in u:
                out["runs_export"][lab]["files"].append({"path": p, "line": line, "type": rtype(p), "n_units": u["n_units"], "coarse": u["coarse"],
                                                         "span_form": u["span_form"], "need": u["need"], "routing": u["routing"]})
    out["span_forms"] = {}
    for (lab, p), u in fu.items():
        if u and "missing" not in u and u["span_form"]:
            out["span_forms"][u["span_form"]] = out["span_forms"].get(u["span_form"], 0) + 1
    out["span_violations"] = sorted(p for (lab, p), u in fu.items() if u and "missing" not in u and (u["span_form"] or "").startswith("STOP"))
    norm = lambda p: re.sub(r"__L\d+(?=\.npz$)", "", os.path.basename(p))  # noqa: E731
    for x, y, kind in inp["pairs"]:
        rec = {"pair": [x, y], "kind": kind, "lines": {}}
        for line in ("32k", "uncut"):
            sx, sy = staged.get((line, runs[x]["eval_ref"])), staged.get((line, runs[y]["eval_ref"]))
            if sx is None or sy is None:
                rec["lines"][line] = {"staged_missing": [n for n, s in ((x, sx), (y, sy)) if s is None]}
                continue
            d = common_prefix(sx["ids"], sy["ids"])
            nx, ny = len(sx["ids"]), len(sy["ids"])
            kinds = {}
            for who, s in ((x, sx), (y, sy)):
                for c in s["calls"]:
                    if isinstance(c, dict) and isinstance(c.get("readout"), int):
                        e = kinds.setdefault(who, {}).setdefault(str(c.get("kind")), {"n": 0, "inside": 0, "outside": 0, "max_pos": -1})
                        e["n"] += 1
                        e["inside" if c["readout"] + 1 <= d else "outside"] += 1
                        e["max_pos"] = max(e["max_pos"], c["readout"])
            fx = {p: fu[(x, p)] for _, p in runs[x]["files"] if LINE_OF_FILE(p) == line and fu.get((x, p)) and "missing" not in fu[(x, p)]}
            fy = {p: fu[(y, p)] for _, p in runs[y]["files"] if LINE_OF_FILE(p) == line and fu.get((y, p)) and "missing" not in fu[(y, p)]}
            files, summ = [], {}
            for px, ux in sorted(fx.items()):
                for py, uy in sorted(fy.items()):
                    if norm(px) != norm(py) or scorer_of(px) != scorer_of(py):  # CUT (a): pairs only within one scorer
                        continue
                    obs = ux["s_f_sbm"] == uy["s_f_sbm"]
                    pos_eq = ux["pos"] == uy["pos"] and ux["n_units"] == uy["n_units"]
                    r_eq = ux["routing"] == uy["routing"] and ux["fold_bytes"] == uy["fold_bytes"]
                    if ux["need"] is None or uy["need"] is None:
                        pred, oc = None, "no positions"
                    else:
                        pred, oc = classify(d, ux["need"], uy["need"], pos_eq, r_eq, obs)
                        if ux["coarse"] or uy["coarse"]:  # call-level readouts (the max over the call's kinds): "natural" holds, "MIX-UP" is not conclusive
                            oc += " (coarse)"
                    t = rtype(px)
                    files.append({"file_x": px, "file_y": py, "scorer": scorer_of(px), "type": t, "units": [ux["n_units"], uy["n_units"]],
                                  "coarse": ux["coarse"] or uy["coarse"],
                                  "max_need": [max(ux["need"]) if ux["need"] else None, max(uy["need"]) if uy["need"] else None],
                                  "positions_equal": pos_eq, "routing": [ux["routing"], uy["routing"]], "routing_equal": r_eq, "observed_identical": obs,
                                  "predicted_identical": pred, "outcome": oc})
                    summ.setdefault(t, {}).setdefault(oc, 0)
                    summ[t][oc] += 1
            rec["lines"][line] = {"n_tokens": [nx, ny], "common_prefix": d, "first_differing_index": d if d < min(nx, ny) else None,
                                  "tails_beyond_prefix": [nx - d, ny - d], "staged_calls_inside_outside": kinds, "files": files, "by_type": summ,
                                  "files_unpaired": [len(set(map(norm, fx)) - set(map(norm, fy))), len(set(map(norm, fy)) - set(map(norm, fx)))]}
        oc_all = [f["outcome"] for ln in rec["lines"].values() for f in ln.get("files", [])]  # the verdict reads exact positions; coarse ones beside
        rec["verdict"] = ("(2) extraction-level mix-up: identical units with a readout outside the common prefix or a different routing" if "MIX-UP" in oc_all
                          else "(1) natural: every identical file's readouts sit inside the common prefix, with the same routing" if "natural" in oc_all
                          else "no identical files with exact positions")
        rec["coarse_outcomes"] = {o: oc_all.count(o) for o in sorted(set(oc_all)) if o.endswith("(coarse)")}
        out["pairs"].append(rec)
    # controls: (i) self; (ii) a planted change before the first readout; (iii) is the real long pair in "pairs"
    x0 = inp["pairs"][0][0]
    s0 = staged.get(("uncut", runs[x0]["eval_ref"]))
    u0 = next((u for (lab, p), u in fu.items() if lab == x0 and LINE_OF_FILE(p) == "uncut" and u and "missing" not in u and u["need"] and not u["coarse"]), None)
    if s0 is not None and u0 is not None:
        d_self = common_prefix(s0["ids"], s0["ids"])
        p0 = min(u0["need"]) - 1  # the first readout position
        out["controls"] = {"self": classify(d_self, u0["need"], u0["need"], True, True, True)[1], "planted": []}
        for at_ in (p0, p0 - 1):  # CUT (a): at the first readout and just before it; units "identical" -> both must be MIX-UP
            if at_ < 0:
                continue
            planted = s0["ids"].copy()
            planted[at_] = planted[at_] + 1
            d_pl = common_prefix(s0["ids"], planted)
            out["controls"]["planted"].append({"changed_index": int(at_), "common_prefix": d_pl,
                                               "outcome": classify(d_pl, u0["need"], u0["need"], True, True, True)[1]})
    out["controls"]["long_pairs"] = []
    for r in out["pairs"]:  # CUT (b): informative only if the pair truly diverges before a readout (d < its first readout's need, exact files)
        if r["kind"].startswith("CONTROL"):
            for line, ln in r["lines"].items():
                needs = [n for f in ln.get("files", []) if not f["coarse"] for n in f["max_need"] if n is not None]
                mins = [min(fu[(lab, f[k])]["need"]) for f in ln.get("files", []) if not f["coarse"] for lab, k in ((r["pair"][0], "file_x"), (r["pair"][1], "file_y"))]
                out["controls"]["long_pairs"].append({"pair": r["pair"], "line": line, "common_prefix": ln.get("common_prefix"),
                                                      "first_readout_need": min(mins) if mins else None, "exact_files": len(needs) // 2,
                                                      "informative": bool(mins) and ln.get("common_prefix") is not None and ln["common_prefix"] < min(mins),
                                                      "verdict": r["verdict"]})
    c_ = out["controls"]
    c_["pass"] = (c_.get("self") == "natural" and len(c_.get("planted", [])) >= 1 and all(x["outcome"] == "MIX-UP" for x in c_["planted"])
                  and any(x["informative"] for x in c_["long_pairs"]) and all(not x["verdict"].startswith("(1)") for x in c_["long_pairs"]))
    with zipfile.ZipFile(Path("/a32/evals_v1") / SIM_EVAL) as z:  # the sim run's fork metadata: identifier-like values only
        sams = sorted(n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json"))
        md = (json.loads(z.read(sams[0])).get("metadata") or {}) if len(sams) == 1 else {}
    fk = md.get("fork") if isinstance(md.get("fork"), dict) else {}
    out["sim_fork_metadata"] = {k: (v if isinstance(v, (int, float, bool)) or v is None or (isinstance(v, str) and IDV.fullmatch(v))
                                    else {"type": type(v).__name__, "len": len(v) if hasattr(v, "__len__") else None}) for k, v in fk.items()}
    return out


@app.local_entrypoint()
def main(inputs: str, out: str):
    ib = open(inputs, "rb").read()
    r = check.remote(json.loads(ib)) | {"inputs": inputs, "inputs_sha256": sha(ib)}
    with open(out, "x") as fh:
        json.dump(r, fh, indent=1)
    print(json.dumps({"controls": r["controls"], "span_forms": r["span_forms"], "span_violations": len(r["span_violations"]),
                      "sim_fork_metadata": r["sim_fork_metadata"],
                      "pairs": [{"pair": p["pair"], "verdict": p["verdict"],
                                 "lines": {ln: {k: v.get(k) for k in ("n_tokens", "common_prefix", "first_differing_index", "by_type", "staged_missing")}
                                           for ln, v in p["lines"].items()}} for p in r["pairs"]]}))
    if r["span_violations"]:
        raise SystemExit(f"STOP: {len(r['span_violations'])} score files with span_end neither token nor token + 1")
    if not r["controls"].get("pass"):
        raise SystemExit("STOP: a control failed")
