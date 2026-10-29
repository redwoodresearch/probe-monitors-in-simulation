"""sol106 DIAG_AMEND3: 09's decisive data check. Where do the nano (and terra) model events sit in the event stream?

For every sha-matched copy of a population path that carries nano or terra in header.json model_usage, read each
samples/*.json entry and keep ONLY: each event's `event` type, the `model` name of model events, their `span_id`, and
the span structure of span_begin events (id, parent_id, type, name). Content is never kept, stored or printed.
Same population / join / copies / gates / worker isolation as diag_sol106_ids.py. The laptop gets names and
{gt0, eq_all} pairs only, plus the outcome letter. Spec: PRESTATED_DIAG_AMEND3_<stamp>.md.

Outcomes (09, pre-stated): (A) nano-in-solvers gt0 false AND nano-in-scorers eq_all true (and every nano copy read
cleanly with spans and nano events) -> DONE; (B) nano-in-solvers gt0 true -> STOP, span NAMES printed; (C) anything
else, or an unreadable event stream -> STOP.
DIAG_AMEND5: zstd member support is explicit (gate G_ZSTD_ZIP), the toys are written as Inspect writes (zstd members,
Inspect-shaped headers and events, one header from Inspect's own writer), and a pinned real copy must read before the
scan; if it does not, the run STOPs as PLUMBING (no outcome letter).

Modal (named binary, from the committed blob in the worktree):
  "$M" run diag3_sol106_events.py --mode test-neg      # toy, predicted FAIL (exit 1)
  "$M" run diag3_sol106_events.py --mode test          # toy, predicted PASS (exit 0)
  "$M" run --detach diag3_sol106_events.py --mode real # the check
Local toy only (no Modal, no data): uv run --no-project --python 3.12 --with modal==1.6.1
  python diag3_sol106_events.py toy | toy-neg
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
import zlib
from collections import Counter
from pathlib import Path

# DIAG_AMEND5 (the named cause of the 15:56 outcome C): Inspect writes every .eval member with zstd (ZIP method 93,
# inspect_ai/_util/zipfile.py `zipfile_compress_kwargs`). Stdlib zipfile < 3.14 cannot decompress it and raises
# NotImplementedError, which raw_presence recorded for every copy. diag v2 read the same members only because its toy
# imported Inspect, whose zipfile module imports zipfile_zstd and patches zstd into zipfile. Here the patch is explicit.
if sys.version_info < (3, 14):
    try:
        import zipfile_zstd  # noqa: F401  (an inspect-ai dependency on Python < 3.14)
    except ImportError:  # only the laptop's local entrypoint can lack it, and it never opens a zip; the worker gates
        pass  # on G_ZSTD_ZIP before anything else
try:  # Inspect's own zip helper (Codex r26 FOLLOW-UP): imports zipfile_zstd and, at import, installs its multi-frame
    import inspect_ai._util.zipfile  # noqa: F401  zstd decoder, so members decode exactly as Inspect's reader does
except ImportError:  # laptop only, as above; G_ZSTD_ZIP requires the patch in the worker
    pass
ZIP_ZSTD = getattr(zipfile, "ZIP_ZSTANDARD", None)

_WORKER = os.environ.get("SOL106_WORKER") == "1"  # the isolated data worker never imports modal

APP_NAME = "probe-shift_aswin-surya_sub-rerun_sol106evt_kill-2026-11-19"
OUT_VOL_NAME = "probe-shift_aswin-surya_sub-rerun_sol106_kill-2026-11-19"  # the item's output Volume
OUT_DIR = "/out/sol106_events"
VOLS = {
    "chain": ("probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", "/chain", "/chain/evals_v1"),
    "q4own": ("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", "/q4own", "/q4own/evals_v1"),
    "pilots": ("probe-shift_aswin-surya_sub-q3pilots_runs_kill-2026-11-13", "/pilots", "/pilots"),
    "q2fresh": ("probe-shift_aswin-surya_sub-q2-fresh_work_kill-2026-11-13", "/q2fresh", "/q2fresh"),
}
CELLS = ("command|L48", "command|L63", "reply|L48", "reply|L63")
SIM_YAML_L19 = "  PT_SIM_MODEL: openai/gpt-6-astra"
SIM_YAML_L20 = "  PT_SIM_FALLBACK_MODEL: openai/gpt-5.6-sol"
SIM_YAML_L21 = "  PT_SIM_REFUSAL_JUDGE_MODEL: openai/gpt-5.6-terra"
SIM_YAML_L22 = "  PT_SIM_ROUTER_JUDGE_MODEL: openai/gpt-5.6-terra"
NANO, TERRA = "openai/gpt-5-nano-2025-08-07", "openai/gpt-5.6-terra"
WATCH = {"nano": NANO, "terra": TERRA}
SCORER_TYPES = {"scorers", "scorer"}  # span types of the scoring phase
SOLVER_TYPES = {"solvers", "solver"}  # span types of the solving phase
AGENT_TYPES = {"agent"}  # NOT a phase (it nests in either); counts as solvers only with no phase ancestor (AMEND4)
NAME_FIELDS = ("solver_span_names", "scorer_span_names", "ambiguous_span_names")
SHA_RE = re.compile(r"[0-9a-f]{64}")
ID_RE = re.compile(r"[A-Za-z0-9._/:-]{1,80}")
NON_ID = "<non-id string>"
MAX_PRINTED = 10
PINS = {
    "refit3": ("80d75b27e", "experiments/2026-10-16_attack_refit3/results/attack_all_refit3_2026-10-16_141813.json",
               "df7171a98d82d28ae86653480be922d978fc36104c63f622b3a69fb43a074eac"),
    "sim_yaml": ("8d7d6a192", "experiments/2026-10-07_phase2_attack_check/config/sim.yaml",
                 "03340860a688a04ea6cb017b46067e037748f5864108b79a4ff84d02245f4612"),
}
# DIAG_AMEND5 real-copy positive control (pre-stated, accepted by 09): the first sorted population key; its one
# manifest row is manifest_nm360_s5_sim_s1.json @ 1df288e61. The earlier diagnostic read every population copy's
# header (n_copies_raw_not_ok {false, false}) and found nano in every readable copy. Only its header's model_usage
# KEYS are read (raw_presence); nothing else of this attack run.
CONTROL_KEY = "sim|nm000_auto_workflow"
CONTROL_PATH = ("2026-10-14_newmains_s5_fu/sim_attack@september/nm000_auto_workflow/"
                "2026-10-15T07-54-10-00-00_la-auto-workflow-sim-attack-1_FnjkqEukeqBUUiFU74Wfwi.eval")
CONTROL_SHA = "b84d7d9b4a85cabe85153e1d6c5e742620cce4ea64007f53f8e2f87c5c4ba369"
MANIFEST_COMMIT = "1df288e61"
MANIFEST_DIR = "experiments/2026-10-14_q3_overnight_chain/config/"
MANIFEST_N = 293
MANIFEST_DIGEST = "bce4051be4bf26b9bb855b300604732799df88fc8c03c86f23dcbbb0c7425ab1"


class GateError(RuntimeError):
    """Raised with a gate name only."""


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def manifest_digest(items: list[tuple[str, bytes]]) -> str:
    return sha256("".join(f"{p}\t{sha256(b)}\n" for p, b in sorted(items)).encode())


def as_int(x) -> int | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return int(x) if float(x).is_integer() else None


def guard(s) -> str:
    return s if isinstance(s, str) and ID_RE.fullmatch(s) else NON_ID


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_presence(path: str) -> dict:
    """header.json stats.model_usage KEYS only (which models were used). Strict: types are checked BEFORE any default;
    an ABSENT stats / model_usage key means no usage, but a PRESENT non-object value, including null and falsy values
    such as [] / "" / 0 / False, makes the header unreadable (Codex r18 #4, r19 #4)."""
    try:
        with zipfile.ZipFile(path) as z:
            if "header.json" not in z.namelist():
                return {"ok": False, "err": None, "ids": []}
            h = json.loads(z.read("header.json"))
        if not isinstance(h, dict):
            raise TypeError("header is not an object")
        st = {}
        if "stats" in h:
            st = h["stats"]
            if not isinstance(st, dict):
                raise TypeError("stats is present but not an object")
        mu = {}
        if "model_usage" in st:
            mu = st["model_usage"]
            if not isinstance(mu, dict):
                raise TypeError("model_usage is present but not an object")
        out = {"ok": True, "err": None, "ids": sorted(str(m) for m in mu)}
        del h, st, mu
        return out
    except Exception as e:
        return {"ok": False, "err": type(e).__name__, "ids": []}


def zstd_selftest() -> bool:
    """G_ZSTD_ZIP (AMEND5): Inspect's zip patches are installed (zstd plus its multi-frame decoder) and zipfile can
    write AND read back a zstd member (Inspect's member format) in this process."""
    if ZIP_ZSTD is None or getattr(zipfile, "_inspect_ai_multiframe_installed", False) is not True:
        return False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p = f"{tmp}/z.zip"
            with zipfile.ZipFile(p, "w", compression=ZIP_ZSTD) as z:
                z.writestr("m.json", b'{"k": [1, 2]}')
            with zipfile.ZipFile(p) as z:
                return z.getinfo("m.json").compress_type == ZIP_ZSTD and z.read("m.json") == b'{"k": [1, 2]}'
    except Exception:
        return False


def control_check(refit3: dict, manifests: list, roots: dict, key: str, path: str, sha: str) -> dict:
    """AMEND5 real-copy positive control, before any scan: the pinned copy must be the population key's joined path
    and sha, exist as exactly one sha-matched file on the roots, and its header must read with nano among the
    model_usage keys. Only raw_presence touches the file. Booleans and an exception type name only."""
    pop, _, path_of, exp_sha, _ = population_and_join(refit3, manifests)
    pinned = key in pop and path_of.get(key) == path and exp_sha.get(key) == sha
    matched, sha_err = [], None
    for r in roots.values():  # Codex r26 FOLLOW-UP: a filesystem error is a PLUMBING result, not a worker crash
        full = os.path.join(r, path)
        try:
            if os.path.isfile(full) and file_sha256(full) == sha:
                matched.append(full)
        except Exception as e:
            sha_err = type(e).__name__
    pres = (raw_presence(matched[0]) if len(matched) == 1 and sha_err is None
            else {"ok": False, "err": sha_err, "ids": []})
    out = {"pinned": pinned, "one_copy": len(matched) == 1 and sha_err is None, "header_ok": pres["ok"],
           "nano_listed": NANO in pres["ids"], "err": guard(pres["err"]) if pres["err"] is not None else None}
    out["ok"] = all(out[k] for k in ("pinned", "one_copy", "header_ok", "nano_listed"))
    return out


def _span_chain(sid, spans: dict) -> list:
    """(type, name) of the span `sid` and its ancestors, innermost first (cycle-safe)."""
    chain, seen = [], set()
    while sid is not None and sid in spans and sid not in seen:
        seen.add(sid)
        parent, typ, name = spans[sid]
        chain.append((typ, name))
        sid = parent
    return chain


def _new_counts() -> dict:
    return {k: {"n": 0, "scorers": 0, "solvers": 0, "ambiguous": 0, "outside": 0, **{f: set() for f in NAME_FIELDS}}
            for k in WATCH}


def _classify(proj: list, spans: dict, res: dict) -> None:
    """Classify each projected watched model event by its span ancestry into `res` (AMEND4 region rule): ambiguous
    only if BOTH phases (scorers/scorer and solvers/solver) are ancestors; else scorers if any scorers/scorer ancestor,
    even under an agent span; else solvers if a solvers/solver ancestor, or an agent ancestor with no phase ancestor
    (conservative); else outside (including no span)."""
    by_model = {m: k for k, m in WATCH.items()}
    for model, sid in proj:
        r = res[by_model[model]]
        r["n"] += 1
        chain = _span_chain(sid, spans)
        types = {t for t, _ in chain}
        in_sc, in_so = bool(types & SCORER_TYPES), bool(types & SOLVER_TYPES)
        names = {n for _, n in chain if n is not None}
        if in_sc and in_so:
            r["ambiguous"] += 1
            r["ambiguous_span_names"] |= names
        elif in_sc:
            r["scorers"] += 1
            r["scorer_span_names"] |= names
        elif in_so or types & AGENT_TYPES:
            r["solvers"] += 1
            r["solver_span_names"] |= names
        else:
            r["outside"] += 1


def event_structure(path: str) -> dict:
    """For each samples/*.json entry (each ZipInfo read directly), the WHOLE member operation (read, project,
    classify) runs inside its own error boundary into member-local counts, merged only on success (Codex r19 #1): a
    bad member is a member error and never erases or stops the other members. Contents are never kept."""
    res = _new_counts()
    n_samples = n_no_events = n_no_spans = n_member_errors = 0
    try:
        with zipfile.ZipFile(path) as z:
            infos = [i for i in z.infolist() if i.filename.startswith("samples/") and i.filename.endswith(".json")]
            for info in infos:
                n_samples += 1
                try:
                    proj, spans, has_events = _project(z.read(info))
                    local = _new_counts()
                    if has_events:
                        _classify(proj, spans, local)
                    del proj
                except Exception:
                    n_member_errors += 1
                    continue
                if not has_events:
                    n_no_events += 1
                    continue
                if not spans:
                    n_no_spans += 1
                for k, v in local.items():  # merge the member's counts
                    for f in ("n", "scorers", "solvers", "ambiguous", "outside"):
                        res[k][f] += v[f]
                    for f in NAME_FIELDS:
                        res[k][f] |= v[f]
                del spans, local
        err = None
    except Exception as e:  # the archive itself could not be opened / listed
        err = type(e).__name__
    return {"ok": err is None and n_member_errors == 0, "err": err, "n_samples": n_samples,
            "n_no_events": n_no_events, "n_no_spans": n_no_spans, "n_member_errors": n_member_errors,
            "by_id": {k: {**v, **{f: sorted(v[f]) for f in NAME_FIELDS}} for k, v in res.items()}}


def _str_or_none(x, what: str):
    if x is not None and not isinstance(x, str):
        raise TypeError(f"{what} is not a string")
    return x


def _project(raw_bytes: bytes) -> tuple[list, dict, bool]:
    """Parse one sample member and keep ONLY (model, span_id) of watched model events and span_begin
    (id -> (parent_id, type, name)), each validated as a string (or None where allowed), so no raw nested object can
    survive (Codex r19 #2). Raw objects are dropped before returning."""
    watched = set(WATCH.values())
    s = json.loads(raw_bytes)
    evs = s.get("events") if isinstance(s, dict) else None
    del s
    if not isinstance(evs, list) or not evs:
        return [], {}, False
    proj, spans = [], {}
    for e in evs:
        if not isinstance(e, dict):
            raise TypeError("event is not an object")
        t = e.get("event")
        if t == "span_begin":
            sid = e.get("id")
            if not isinstance(sid, str):
                raise TypeError("span id is not a string")
            spans[sid] = (_str_or_none(e.get("parent_id"), "parent_id"), _str_or_none(e.get("type"), "span type"),
                          _str_or_none(e.get("name"), "span name"))
        elif t == "model" and e.get("model") in watched:
            proj.append((str(e.get("model")), _str_or_none(e.get("span_id"), "span_id")))
    del evs, e
    return proj, spans, True


async def scan_path(key, path, exp_sha, roots, sem) -> dict:
    copies = []
    for vol, root in roots.items():
        full = os.path.join(root, path)
        async with sem:
            if not await asyncio.to_thread(os.path.isfile, full):
                continue
            try:
                got = await asyncio.to_thread(file_sha256, full)
            except Exception as e:
                copies.append({"vol": vol, "sha_ok": False, "sha_err": type(e).__name__})
                continue
            if got != exp_sha:
                copies.append({"vol": vol, "sha_ok": False, "sha_err": None})
                continue
            pres = await asyncio.to_thread(raw_presence, full)
            watched = bool(set(pres["ids"]) & set(WATCH.values()))
            ev = await asyncio.to_thread(event_structure, full) if watched else None
        copies.append({"vol": vol, "sha_ok": True, "pres": pres, "ev": ev})
    return {"key": key, "path": path, "copies": copies}


async def scan_all(jobs, roots, conc=32):
    sem = asyncio.Semaphore(conc)
    return await asyncio.gather(*(scan_path(k, p, s, roots, sem) for k, p, s in jobs))


def population_and_join(refit3: dict, manifests: list[tuple[str, dict]]):
    """Exactly run_sol106's gap-1 population and gap-2 join, with its blocking gates."""
    rk = refit3["run_keys"]
    sim = {k: v for k, v in rk.items() if k.startswith("sim|")}
    g = {"G_RUNKEYS_SUFFIX_E0": all(isinstance(v, str) and v.endswith("#e0") for v in sim.values())}
    part = True
    for c in CELLS:
        rpr, rwr = refit3["cells"][c]["rows_per_run"], refit3["cells"][c]["runs_without_rows"]
        zr = [r.get("name") for r in refit3["zero_row_runs"][c]["runs"]]
        part &= (sorted(list(rpr) + list(rwr)) == sorted(rk) and not set(rpr) & set(rwr)
                 and all(as_int(n) is not None and n > 0 for n in rpr.values()) and set(zr) == set(rwr))
    g["G_CELLS_PARTITION"] = part
    cell_pop = {c: sorted(k for k in refit3["cells"][c]["rows_per_run"] if k in sim) for c in CELLS}
    pop = sorted(set().union(*cell_pop.values()))
    path_of = {k: sim[k].rsplit("#", 1)[0] for k in pop}
    g["G_PATHS_INJECTIVE"] = len(set(path_of.values())) == len(pop)
    by_path: dict = {}
    for _, m in manifests:
        for r in ((m.get("files") or []) if isinstance(m, dict) else []):
            if isinstance(r, dict) and str(r.get("arm_dir") or "").startswith("sim_attack"):
                by_path.setdefault(r.get("path"), []).append(r)
    exp_sha, ok_all = {}, True
    for k in pop:
        rs = by_path.get(path_of[k], [])
        shas = {r.get("sha256") for r in rs}
        ok = (bool(rs) and len(shas) == 1 and all(isinstance(s, str) and SHA_RE.fullmatch(s) for s in shas)
              and all(r.get("task") == k.split("|", 1)[1] for r in rs))
        ok_all &= ok
        if ok:
            exp_sha[k] = next(iter(shas))
    g["G_MANIFEST_JOIN"] = ok_all
    for name, v in g.items():
        if not v:
            raise GateError(name)
    return pop, cell_pop, path_of, exp_sha, len(sim)


def diagnose_events(refit3: dict, manifests: list, roots: dict) -> dict:
    """Per watched id: copy-level counts over the copies whose header lists the id; event-level counts over EVERY
    scanned copy (any copy listing nano or terra), including copies whose header does not list that id; read errors
    over all scanned copies; the span names; and the pre-stated outcome for nano."""
    pop, _, path_of, exp_sha, _ = population_and_join(refit3, manifests)
    per_path = asyncio.run(scan_all([(k, path_of[k], exp_sha[k]) for k in pop], roots))
    copies = [c for r in per_path for c in r["copies"] if c["sha_ok"]]
    scanned = [c for c in copies if c["ev"] is not None]  # every copy whose events were read (nano OR terra listed)
    out = {"n_copies": len(copies), "n_copies_header_unreadable": sum(1 for c in copies if not c["pres"]["ok"]),
           "n_scanned_copies": len(scanned),
           # Codex r19 #3: completeness over ALL scanned copies, not only header-listed ones
           "n_scanned_copies_read_error": sum(1 for c in scanned if not c["ev"]["ok"]),
           "per_id": {}}
    for k, m in WATCH.items():
        cs = [c for c in copies if m in c["pres"]["ids"]]  # copy-level denominators: copies whose header lists the id
        cb = [c["ev"]["by_id"][k] for c in cs]
        # event level (Codex r18 #1): EVERY observed event of the id in ANY scanned copy, including copies whose
        # header does not list it and copies with a member read error (their other members still count)
        eb = [c["ev"]["by_id"][k] for c in scanned]
        unlisted = [c["ev"]["by_id"][k] for c in scanned if m not in c["pres"]["ids"]]
        out["per_id"][k] = {
            "id": m,
            "n_copies": len(cs),
            "n_copies_read_error": sum(1 for c in cs if not c["ev"]["ok"]),
            "n_copies_no_events": sum(1 for c in cs if c["ev"]["n_no_events"] > 0),
            "n_copies_no_spans": sum(1 for c in cs if c["ev"]["n_no_spans"] > 0),
            "n_copies_no_samples": sum(1 for c in cs if c["ev"]["n_samples"] == 0),
            "n_copies_with_id_events": sum(1 for b in cb if b["n"] > 0),
            "n_copies_with_scorer_event": sum(1 for b in cb if b["scorers"] > 0),
            "n_copies_with_solver_event": sum(1 for b in cb if b["solvers"] > 0),
            "n_events": sum(b["n"] for b in eb),
            "n_events_in_scorers": sum(b["scorers"] for b in eb),
            "n_events_in_solvers": sum(b["solvers"] for b in eb),
            "n_events_ambiguous": sum(b["ambiguous"] for b in eb),
            "n_events_outside": sum(b["outside"] for b in eb),
            "n_events_in_unlisted_copies": sum(b["n"] for b in unlisted),
            **{f: Counter(n for b in eb for n in b[f]) for f in NAME_FIELDS},
        }
    out["outcome"] = nano_outcome(out["per_id"]["nano"], out["n_copies_header_unreadable"],
                                  out["n_scanned_copies_read_error"])
    out["_per_path"] = per_path  # Volume record only
    return out


def nano_outcome(v: dict, n_header_unreadable: int, n_scanned_read_error: int) -> str:
    """09's pre-stated mapping: (B) any observed nano event in solvers (any scanned copy); (A) none in solvers AND
    every observed nano event in scorers, with every nano copy read cleanly (events, spans, samples, no member error)
    and carrying nano events, no unreadable header anywhere, and no read error in ANY scanned copy (either could hide
    nano; AMEND3 fail-closed); (C) anything else."""
    if v["n_events_in_solvers"] > 0:
        return "B"
    clean = (v["n_copies"] > 0 and v["n_copies_read_error"] == 0 and v["n_copies_no_events"] == 0
             and v["n_copies_no_spans"] == 0 and v["n_copies_no_samples"] == 0
             and v["n_copies_with_id_events"] == v["n_copies"] and n_header_unreadable == 0
             and n_scanned_read_error == 0)
    if clean and v["n_events"] > 0 and v["n_events_in_scorers"] == v["n_events"]:
        return "A"
    return "C"


def _pair(n: int, total: int) -> dict:
    return {"gt0": n > 0, "eq_all": n == total}


def _names(c: Counter) -> dict:
    """Guarded NAMES only (at most MAX_PRINTED, most frequent first); counts stay on the Volume."""
    g = Counter()
    for k, v in c.items():
        g[guard(k)] += v
    return {"names": sorted(k for k, _ in sorted(g.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_PRINTED])}


def print_set(out: dict) -> dict:
    """The ONLY data that reaches the laptop: names and {gt0, eq_all} pairs, plus the outcome letter. No integer."""
    per = {}
    for k, v in out["per_id"].items():
        nc, ne = v["n_copies"], v["n_events"]
        per[k] = {"id": v["id"],
                  "copies_present": _pair(nc, out["n_copies"]),
                  "copies_read_error": _pair(v["n_copies_read_error"], nc),
                  "copies_no_events": _pair(v["n_copies_no_events"], nc),
                  "copies_no_spans": _pair(v["n_copies_no_spans"], nc),
                  "copies_no_samples": _pair(v["n_copies_no_samples"], nc),
                  "copies_with_id_events": _pair(v["n_copies_with_id_events"], nc),
                  "copies_with_scorer_event": _pair(v["n_copies_with_scorer_event"], nc),
                  "copies_with_solver_event": _pair(v["n_copies_with_solver_event"], nc),
                  "events_in_scorers": _pair(v["n_events_in_scorers"], ne),
                  "events_in_solvers": _pair(v["n_events_in_solvers"], ne),
                  "events_ambiguous": _pair(v["n_events_ambiguous"], ne),
                  "events_outside": _pair(v["n_events_outside"], ne),
                  "events_in_unlisted_copies": _pair(v["n_events_in_unlisted_copies"], ne),
                  **{f: _names(v[f]) for f in NAME_FIELDS}}
    return {"per_id": per, "copies_header_unreadable": _pair(out["n_copies_header_unreadable"], out["n_copies"]),
            "scanned_copies_read_error": _pair(out["n_scanned_copies_read_error"], out["n_scanned_copies"]),
            "outcome": out["outcome"]}


def _int_leaves(x) -> int:
    if isinstance(x, dict):
        return sum(_int_leaves(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(_int_leaves(v) for v in x)
    return int(isinstance(x, (int, float)) and not isinstance(x, bool))


# ---------------------------------------------------------------- toy (hand-derived expectations)
_A, _Q, _S = "openai/gpt-6-astra", "vllm/Qwen/Qwen3.5-27B", "openai/gpt-5.6-sol"


def _sb(i, parent, typ, name):
    return {"event": "span_begin", "id": i, "parent_id": parent, "type": typ, "name": name}


def _me(model, sid):
    return {"event": "model", "model": model, "span_id": sid}


# run -> (model_usage ids, sample events (None = no samples entry; "bad" = malformed events), cells with rows)
TOY = {
    "e01": ([_A, _Q, TERRA, NANO], [_sb("s0", None, "solvers", "solvers"), _sb("s1", "s0", "agent", "react"),
                                    _me(TERRA, "s1"), _me(_Q, "s1"), {"event": "span_end", "id": "s1"},
                                    _sb("c0", None, "scorers", "scorers"), _sb("c1", "c0", "scorer", "side_task"),
                                    _me(NANO, "c1"), {"event": "score", "span_id": "c1"}], None),
    "e02": ([_A, _Q, NANO], [_sb("s0", None, "solvers", "solvers"), _sb("s1", "s0", "agent", "react"),
                             _me(NANO, "s1")], None),  # nano INSIDE solvers -> (B)
    "e03": ([_A, _Q, NANO], [_me(NANO, None), _me(_Q, None)], None),  # no spans -> (C)
    "e04": ([_A, _Q, TERRA, NANO], [_sb("s0", None, "solvers", "solvers"), _me(TERRA, "s0"),
                                    _sb("c0", None, "scorers", "scorers"), _sb("c1", "c0", "scorer", "side_task"),
                                    _me(NANO, "c1"), _me(TERRA, "c1")], None),  # terra also in scorers (info only)
    "e05": ([_A, _Q, NANO], [], None),  # nano in usage, empty event stream -> (C)
    "e06": ([_A, _Q, NANO], [_sb("i0", None, "init", "init"), _me(NANO, "i0")], None),  # nano outside both -> (C)
    "e07": ([_A, _Q, NANO], "bad", None),  # malformed stream (non-object events) -> read error -> (C)
    "e08": ([_A, _Q], [_sb("s0", None, "solvers", "solvers"), _me(_Q, "s0")], None),  # neither id: not read
}
TOY_SUBSETS = {  # subset -> hand-derived expectations
    "A": (("e01", "e04"), {"outcome": "A", "nano_events_in_scorers": {"gt0": True, "eq_all": True},
                           "nano_events_in_solvers": {"gt0": False, "eq_all": False},
                           "terra_events_in_scorers": {"gt0": True, "eq_all": False},
                           "terra_events_in_solvers": {"gt0": True, "eq_all": False},
                           "nano_scorer_span_names": ["scorers", "side_task"]}),
    "B_solvers": (("e01", "e02"), {"outcome": "B", "nano_events_in_solvers": {"gt0": True, "eq_all": False},
                                   "nano_solver_span_names": ["react", "solvers"]}),
    "C_no_spans": (("e01", "e03"), {"outcome": "C", "nano_copies_no_spans": {"gt0": True, "eq_all": False}}),
    "C_no_events": (("e01", "e05"), {"outcome": "C", "nano_copies_no_events": {"gt0": True, "eq_all": False}}),
    "C_outside": (("e01", "e06"), {"outcome": "C", "nano_events_outside": {"gt0": True, "eq_all": False}}),
    "C_malformed": (("e01", "e07"), {"outcome": "C", "nano_copies_read_error": {"gt0": True, "eq_all": False}}),
    # Codex r18 #1: a nano event in solvers inside a copy whose header lists only terra -> still (B)
    "B_unlisted": (("e01", "e04", "e09"), {"outcome": "B", "nano_events_in_solvers": {"gt0": True, "eq_all": False},
                                           "nano_events_unlisted": {"gt0": True, "eq_all": False},
                                           "nano_solver_span_names": ["react", "solvers"]}),
    # Codex r18 #2: a valid nano-in-solvers sample followed by a malformed one -> the evidence survives -> (B)
    "B_then_malformed": (("e01", "e10"), {"outcome": "B", "nano_events_in_solvers": {"gt0": True, "eq_all": False},
                                          "nano_copies_read_error": {"gt0": True, "eq_all": False}}),
    # Codex r18 #4: a falsy non-object model_usage ([]) is an unreadable header -> (C) (it could hide nano)
    "C_header_falsy": (("e01", "e11"), {"outcome": "C", "copies_header_unreadable": {"gt0": True, "eq_all": False}}),
    # Codex r19 #1/#2: a member whose span name is an object is a member error; the LATER valid member still counts
    "B_after_bad_member": (("e01", "e12"), {"outcome": "B", "nano_events_in_solvers": {"gt0": True, "eq_all": False},
                                            "nano_copies_read_error": {"gt0": True, "eq_all": False}}),
    # Codex r19 #3: a terra-only-listed copy with nano evidence and a bad member blocks (A) -> (C)
    "C_unlisted_unreadable": (("e01", "e14"), {"outcome": "C",
                                               "scanned_copies_read_error": {"gt0": True, "eq_all": False},
                                               "nano_events_in_scorers": {"gt0": True, "eq_all": True},
                                               "nano_events_in_solvers": {"gt0": False, "eq_all": False}}),
    # Codex r19 #4: explicit nulls are unreadable headers -> (C)
    "C_stats_null": (("e01", "e15"), {"outcome": "C", "copies_header_unreadable": {"gt0": True, "eq_all": False}}),
    "C_usage_null": (("e01", "e16"), {"outcome": "C", "copies_header_unreadable": {"gt0": True, "eq_all": False}}),
    # AMEND4 (reviewer): nano under scorers -> scorer -> agent counts as scorers -> (A)
    "A_scorer_agent": (("e01", "e17"), {"outcome": "A", "nano_events_in_scorers": {"gt0": True, "eq_all": True},
                                        "nano_events_in_solvers": {"gt0": False, "eq_all": False},
                                        "nano_events_ambiguous": {"gt0": False, "eq_all": False},
                                        "nano_scorer_span_names": ["judge", "scorers", "side_task"]}),
    # AMEND4 (09): nano under solvers -> solver -> agent -> (B)
    "B_solver_agent": (("e01", "e18"), {"outcome": "B", "nano_events_in_solvers": {"gt0": True, "eq_all": False},
                                        "nano_events_ambiguous": {"gt0": False, "eq_all": False},
                                        "nano_solver_span_names": ["main", "react", "solvers"]}),
    # AMEND4: both phases in the ancestry -> ambiguous -> (C), with the guarded ambiguous span names printed
    "C_ambiguous": (("e01", "e19"), {"outcome": "C", "nano_events_ambiguous": {"gt0": True, "eq_all": False},
                                     "nano_events_in_solvers": {"gt0": False, "eq_all": False},
                                     "nano_ambiguous_span_names": ["side_task", "solvers"]}),
    # AMEND4: an agent span with no phase ancestor stays solvers (conservative) -> (B)
    "B_agent_no_phase": (("e01", "e20"), {"outcome": "B", "nano_events_in_solvers": {"gt0": True, "eq_all": False},
                                          "nano_events_outside": {"gt0": False, "eq_all": False},
                                          "nano_solver_span_names": ["react"]}),
}
TOY_EXTRA = {  # runs used only by the subsets above (the main toy e01-e08 is unchanged)
    "e09": ([_A, _Q, TERRA], [_sb("s0", None, "solvers", "solvers"), _sb("s1", "s0", "agent", "react"),
                              _me(NANO, "s1"), _me(TERRA, "s1")], None),  # header lists terra only
    "e10": ([_A, _Q, NANO], ("multi", [[_sb("s0", None, "solvers", "solvers"), _sb("s1", "s0", "agent", "react"),
                                        _me(NANO, "s1")], "bad"]), None),  # 2 samples: valid (solvers), then malformed
    "e11": ("falsy", None, None),  # header.json model_usage is [] (falsy, not an object)
    "e12": ([_A, _Q, NANO], ("multi", [[_sb("s0", None, "solvers", {"x": 1}), _me(NANO, "s0")],  # object span name
                                       [_sb("s0", None, "solvers", "solvers"), _sb("s1", "s0", "agent", "react"),
                                        _me(NANO, "s1")]]), None),
    "e14": ([_A, _Q, TERRA], ("multi", [[_sb("c0", None, "scorers", "scorers"), _sb("c1", "c0", "scorer", "side_task"),
                                         _me(NANO, "c1")], "bad"]), None),  # terra-only header; nano evidence + bad
    "e15": ("stats_null", None, None),  # header.json "stats": null
    "e16": ("usage_null", None, None),  # header.json "stats": {"model_usage": null}
    "e17": ([_A, _Q, NANO], [_sb("c0", None, "scorers", "scorers"), _sb("c1", "c0", "scorer", "side_task"),
                             _sb("c2", "c1", "agent", "judge"), _me(NANO, "c2")], None),  # scorers > scorer > agent
    "e18": ([_A, _Q, NANO], [_sb("s0", None, "solvers", "solvers"), _sb("s1", "s0", "solver", "main"),
                             _sb("s2", "s1", "agent", "react"), _me(NANO, "s2")], None),  # solvers > solver > agent
    "e19": ([_A, _Q, NANO], [_sb("s0", None, "solvers", "solvers"), _sb("c1", "s0", "scorer", "side_task"),
                             _me(NANO, "c1")], None),  # both phases in the ancestry
    "e20": ([_A, _Q, NANO], [_sb("a0", None, "agent", "react"), _me(NANO, "a0")], None),  # agent, no phase ancestor
}
# main toy (e01-e08): 8 copies; nano in 7 (all but e08); terra in 2 (e01, e04); e02 puts nano in solvers -> (B)
TOY_MAIN = {"outcome": "B", "nano_copies_present": {"gt0": True, "eq_all": False},
            "terra_copies_present": {"gt0": True, "eq_all": False},
            "nano_n_copies": 7, "nano_n_events": 5, "nano_in_scorers": 2, "nano_in_solvers": 1, "nano_outside": 2,
            "nano_read_error": 1, "nano_no_spans": 1, "nano_no_events": 1, "terra_n_events": 3,
            "terra_in_solvers": 2, "terra_in_scorers": 1}


_TS = "2026-10-20T00:00:00+00:00"


def _inspect_header(ids: list) -> dict:
    """AMEND5: a toy header.json with the key/type structure Inspect 0.3.276's writer produces (top level: version int,
    status str, eval dict, plan dict, stats dict, invalidated bool, tags list, metadata dict; stats: started_at,
    completed_at, model_usage dict, role_usage dict, connection_limit_history list)."""
    return {"version": 2, "status": "success",
            "eval": {"created": _TS, "task": "toy", "dataset": {}, "model": "none/none", "config": {}},
            "plan": {"name": "plan", "steps": []},
            "stats": {"started_at": _TS, "completed_at": _TS,
                      "model_usage": {m: {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2} for m in ids},
                      "role_usage": {}, "connection_limit_history": []},
            "invalidated": False, "tags": [], "metadata": {}}


def _inspect_event(e):
    """AMEND5: give a toy event Inspect's serialized fields (BaseEvent: uuid, span_id, timestamp, working_start,
    metadata, pending; ModelEvent: role, input, tools, tool_choice, config, output). The projection must ignore them."""
    if not isinstance(e, dict):
        return e
    base = {"uuid": None, "timestamp": _TS, "working_start": 0.0, "metadata": None, "pending": None}
    if e.get("event") == "model":
        base.update({"role": None, "input": [], "tools": [], "tool_choice": "none", "config": {}, "output": {}})
    if e.get("event") == "span_begin":
        base["span_id"] = e.get("parent_id")  # a span_begin sits in its parent span
    return {**base, **e}


def _toy_out(toy: dict, perturb: bool, controls: tuple = ()) -> dict:
    """Builds the toy Volumes (every member zstd-compressed, as Inspect writes them; Inspect-shaped headers and events),
    runs diagnose_events, and runs control_check for each (key run, path run, sha mode) in `controls` (sha mode
    "manifest" = the path run's manifest sha, or "wrong") into out["_controls"]."""
    with tempfile.TemporaryDirectory() as tmp:
        roots = {"chain": f"{tmp}/chain/evals_v1", "q4own": f"{tmp}/q4own/evals_v1",
                 "pilots": f"{tmp}/pilots", "q2fresh": f"{tmp}/q2fresh"}
        run_keys, mrows = {}, []
        for t, (ids, evs, _) in toy.items():
            rel = f"L/sim_attack@september/{t}/x_{t}.eval"
            run_keys[f"sim|{t}"] = rel + "#e0"
            run_keys[f"run1|{t}"] = f"L/attack/{t}/y_{t}.eval#e0"
            full = os.path.join(roots["chain"], rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            if perturb and t == "e01":  # negative control: move nano from the scorer span into the agent span
                evs = [dict(e, span_id="s1") if e.get("model") == NANO else e for e in evs]
            header = ({"stats": {"model_usage": []}} if ids == "falsy" else {"stats": None} if ids == "stats_null"
                      else {"stats": {"model_usage": None}} if ids == "usage_null"
                      else _inspect_header(ids))
            with zipfile.ZipFile(full, "w", compression=ZIP_ZSTD) as z:
                z.writestr("header.json", json.dumps(header))
                if evs is not None:
                    samples = evs[1] if isinstance(evs, tuple) and evs[0] == "multi" else [evs]
                    for j, sevs in enumerate(samples, 1):
                        body = ({"events": [1, 2]} if sevs == "bad" else
                                {"id": j, "epoch": 1, "input": "", "target": "", "messages": [], "output": {},
                                 "scores": {}, "metadata": {}, "events": [_inspect_event(e) for e in sevs]})
                        z.writestr(f"samples/{j}_epoch_1.json", json.dumps(body))
            mrows.append({"path": rel, "sha256": file_sha256(full), "task": t, "arm_dir": "sim_attack@september"})
        cells, zero = {}, {}
        for c in CELLS:
            rpr = {k: 3 for k in run_keys}
            cells[c] = {"rows_per_run": rpr, "runs_without_rows": []}
            zero[c] = {"n": 0, "runs": []}
        refit3, manifests = {"run_keys": run_keys, "cells": cells, "zero_row_runs": zero}, [("m1.json", {"files": mrows})]
        out = diagnose_events(refit3, manifests, roots)
        sha_of = {r["task"]: r["sha256"] for r in mrows}
        out["_controls"] = [control_check(refit3, manifests, roots, f"sim|{tk}", f"L/sim_attack@september/{tp}/x_{tp}.eval",
                                          sha_of[tp] if mode == "manifest" else "0" * 64) for tk, tp, mode in controls]
        return out


def _flat(out: dict) -> dict:
    """Printed pairs / names and raw counts flattened under the keys used by the toy expectations."""
    ps = print_set(out)
    f = {"outcome": out["outcome"], "copies_header_unreadable": ps["copies_header_unreadable"],
         "scanned_copies_read_error": ps["scanned_copies_read_error"]}
    for k in WATCH:
        f[f"{k}_events_unlisted"] = ps["per_id"][k]["events_in_unlisted_copies"]
        p, v = ps["per_id"][k], out["per_id"][k]
        f.update({f"{k}_events_in_scorers": p["events_in_scorers"], f"{k}_events_in_solvers": p["events_in_solvers"],
                  f"{k}_events_outside": p["events_outside"], f"{k}_copies_no_spans": p["copies_no_spans"],
                  f"{k}_copies_no_events": p["copies_no_events"], f"{k}_copies_read_error": p["copies_read_error"],
                  f"{k}_copies_present": p["copies_present"], f"{k}_solver_span_names": p["solver_span_names"]["names"],
                  f"{k}_scorer_span_names": p["scorer_span_names"]["names"],
                  f"{k}_ambiguous_span_names": p["ambiguous_span_names"]["names"],
                  f"{k}_events_ambiguous": p["events_ambiguous"], f"{k}_n_copies": v["n_copies"],
                  f"{k}_n_events": v["n_events"], f"{k}_in_scorers": v["n_events_in_scorers"],
                  f"{k}_in_solvers": v["n_events_in_solvers"], f"{k}_outside": v["n_events_outside"],
                  f"{k}_read_error": v["n_copies_read_error"], f"{k}_no_spans": v["n_copies_no_spans"],
                  f"{k}_no_events": v["n_copies_no_events"]})
    return f


def _run_worker(cmd: list[str], sink, timeout: int = 1700) -> int:
    env = {**os.environ, "SOL106_WORKER": "1"}
    return subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=sink, stderr=sink, env=env, check=False,
                          timeout=timeout).returncode


def _noise_main() -> None:
    print("s1", end="")
    sys.stderr.write("s2")
    os.write(1, b" s3\n")
    os.write(2, b" s4\n")
    subprocess.run(["sh", "-c", "printf ' s5'; printf ' s6' 1>&2"], check=False)
    raise SystemExit(3)


def _worker_selftest() -> bool:
    with tempfile.TemporaryFile() as sink:
        rc = _run_worker([sys.executable, os.path.abspath(__file__), "--noise"], sink, timeout=120)
        sink.seek(0)
        text = sink.read().decode(errors="replace")
    return rc == 3 and all(f"s{i}" in text for i in range(1, 7))


def run_toy(perturb: bool) -> dict:
    out = _toy_out(TOY, perturb)
    f = _flat(out)
    checks = {f"main.{k}": f[k] == v for k, v in TOY_MAIN.items()}
    no_ints = _int_leaves(print_set(out)) == 0
    for name, (sub, exp) in TOY_SUBSETS.items():
        o = _toy_out({k: {**TOY, **TOY_EXTRA}[k] for k in sub}, perturb)
        fo = _flat(o)
        no_ints &= _int_leaves(print_set(o)) == 0
        checks.update({f"sub_{name}.{k}": fo[k] == v for k, v in exp.items()})
    checks["print_no_integers"] = (no_ints and _int_leaves({"a": [1, True]}) == 1
                                   and _int_leaves({"x": True, "y": "3"}) == 0)
    checks["span_chain_cycle_safe"] = _span_chain("a", {"a": ("b", "t", "x"), "b": ("a", "t", "y")}) == [("t", "x"),
                                                                                                         ("t", "y")]
    checks["guard"] = guard("a b") == NON_ID and guard("x" * 81) == NON_ID and guard(NANO) == NANO
    checks["worker_output_isolated"] = _worker_selftest()
    # AMEND5: zstd support, a header written by Inspect's own writer, and the real-copy control's decision
    checks["zstd_roundtrip"] = zstd_selftest()
    checks["toy_members_zstd"] = _toy_members_zstd()
    checks["inspect_written_header"] = _inspect_written_unit()
    ctl = _toy_out({k: {**TOY, **TOY_EXTRA}[k] for k in ("e01", "e08", "e11")}, perturb, CONTROL_UNITS)["_controls"]
    checks.update({f"control_{i}": c == e for i, (c, e) in enumerate(zip(ctl, CONTROL_EXPECT, strict=True))})
    return {"pass": all(checks.values()), "checks": checks}


# AMEND5 control units on the toy {e01, e08, e11}: (key run, path run, sha mode) -> hand-derived result
CONTROL_UNITS = (("e01", "e01", "manifest"), ("e01", "e01", "wrong"), ("e08", "e08", "manifest"),
                 ("e11", "e11", "manifest"), ("e01", "e08", "manifest"))
CONTROL_EXPECT = (
    {"pinned": True, "one_copy": True, "header_ok": True, "nano_listed": True, "err": None, "ok": True},
    {"pinned": False, "one_copy": False, "header_ok": False, "nano_listed": False, "err": None, "ok": False},
    {"pinned": True, "one_copy": True, "header_ok": True, "nano_listed": False, "err": None, "ok": False},
    {"pinned": True, "one_copy": True, "header_ok": False, "nano_listed": False, "err": "TypeError", "ok": False},
    {"pinned": False, "one_copy": True, "header_ok": True, "nano_listed": False, "err": None, "ok": False},
)


def _toy_members_zstd() -> bool:
    """AMEND5: the toy builder writes zstd members, so the toys exercise the members' real compression."""
    with tempfile.TemporaryDirectory() as tmp:
        p = f"{tmp}/t.eval"
        with zipfile.ZipFile(p, "w", compression=ZIP_ZSTD) as z:
            z.writestr("header.json", json.dumps(_inspect_header([NANO])))
        with zipfile.ZipFile(p) as z:
            return z.getinfo("header.json").compress_type == ZIP_ZSTD and raw_presence(p)["ids"] == [NANO]


def _inspect_written_unit() -> bool:
    """AMEND5: a header written by Inspect 0.3.276's own writer (same constructor as run_sol106's toy) is a zstd
    member, and raw_presence reads it with every model_usage key."""
    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSpec, EvalStats, write_eval_log
    from inspect_ai.model import ModelUsage

    u = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    with tempfile.TemporaryDirectory() as tmp:
        p = f"{tmp}/w.eval"
        write_eval_log(EvalLog(status="success", eval=EvalSpec(created=_TS, task="toy", dataset=EvalDataset(),
                                                               model="none/none", config=EvalConfig()),
                               stats=EvalStats(model_usage={m: ModelUsage(**u) for m in (_A, _Q, NANO)})),
                       p, format="eval")
        with zipfile.ZipFile(p) as z:
            zstd = "header.json" in z.namelist() and z.getinfo("header.json").compress_type == ZIP_ZSTD
        pres = raw_presence(p)
    return zstd and pres["ok"] and pres["ids"] == sorted([_A, _Q, NANO])


# ---------------------------------------------------------------- run (worker side, then the Modal driver)
def _summary_line(d: dict) -> str:
    return "EVENTS106_RESULT " + json.dumps(d, sort_keys=True)


def _write_out(name: str, record: dict) -> tuple[str, str]:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/{name}"
    data = json.dumps(record, sort_keys=True, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)
                      ).encode()
    with open(path, "xb") as f:  # append-only; the driver commits after the worker exits
        f.write(data)
    return path, sha256(data)


def _drive(mode: str, blobs: dict, manifests: list, meta: dict, stage: list) -> dict:
    stage[0] = "zstd"  # AMEND5: without zstd support no real member can be read; fail loudly in every mode
    if not zstd_selftest():
        raise GateError("G_ZSTD_ZIP")
    stage[0] = "toy"
    toy_pos, toy_neg = run_toy(False), run_toy(True)
    gates = {"G_ZSTD_ZIP": True, "G_TOY_POS_PASS": toy_pos["pass"],
             "G_TOY_NEG_FAILS": (not toy_neg["pass"]) and not toy_neg["checks"]["sub_A.outcome"]}
    if mode in ("test", "test-neg"):
        res = toy_neg if mode == "test-neg" else toy_pos
        stage[0] = "write"
        path, rsha = _write_out(f"events_{mode}_{meta['stamp']}.json", {"mode": mode, "meta": meta, "toy": res,
                                                                         "gates": gates})
        return {"mode": mode, "status": "PASS" if res["pass"] else "MISMATCH", "gates": gates,
                "checks_all_true": all(res["checks"].values()), "results_path": path, "results_sha256": rsha}
    for name in ("G_TOY_POS_PASS", "G_TOY_NEG_FAILS"):
        if not gates[name]:
            raise GateError(name)
    stage[0] = "inputs"
    raw = {k: zlib.decompress(v) for k, v in blobs.items()}
    gates["G_INPUT_SHAS"] = set(raw) == set(PINS) and all(sha256(raw[k]) == PINS[k][2] for k in PINS)
    mf = [(p, zlib.decompress(b)) for p, b in manifests]
    gates["G_MANIFEST_DIGEST"] = len(mf) == MANIFEST_N and manifest_digest(mf) == MANIFEST_DIGEST
    lines = raw["sim_yaml"].decode().splitlines() if "sim_yaml" in raw else []
    gates["G_SIM_YAML_L19_L22"] = (len(lines) >= 22 and lines[18] == SIM_YAML_L19 and lines[19] == SIM_YAML_L20
                                   and lines[20] == SIM_YAML_L21 and lines[21] == SIM_YAML_L22)
    roots = {lab: root for lab, (_, _, root) in VOLS.items()}
    gates["G_ROOTS_EXIST"] = all(os.path.isdir(r) for r in roots.values())
    for name in ("G_INPUT_SHAS", "G_MANIFEST_DIGEST", "G_SIM_YAML_L19_L22", "G_ROOTS_EXIST"):
        if not gates[name]:
            raise GateError(name)
    refit3, mfj = json.loads(raw["refit3"]), [(p, json.loads(b)) for p, b in mf]
    stage[0] = "control"  # AMEND5 real-copy positive control, before the scan
    ctl = control_check(refit3, mfj, roots, CONTROL_KEY, CONTROL_PATH, CONTROL_SHA)
    gates["G_CONTROL_PINNED"] = ctl["pinned"]
    if not gates["G_CONTROL_PINNED"]:
        raise GateError("G_CONTROL_PINNED")
    if not ctl["ok"]:  # the reader fails on a copy the earlier diagnostic read: PLUMBING, never an outcome letter
        stage[0] = "write"
        path, rsha = _write_out(f"events_results_{meta['stamp']}.json", {
            "mode": mode, "status": "STOP", "stop_reasons": ["PLUMBING"], "outcome": "PLUMBING", "meta": meta,
            "pins": PINS, "gates": gates, "control": ctl, "toy": {"pos": toy_pos, "neg": toy_neg}})
        return {"mode": mode, "status": "STOP", "stop_reasons": ["PLUMBING"], "outcome": "PLUMBING", "gates": gates,
                "control": ctl, "results_path": path, "results_sha256": rsha}
    stage[0] = "events"
    out = diagnose_events(refit3, mfj, roots)
    status = "DONE" if out["outcome"] == "A" else "STOP"
    stop_reasons = [] if status == "DONE" else [f"OUTCOME_{out['outcome']}"]
    full = {k: v for k, v in out.items() if not k.startswith("_")}  # all counts: Volume record only
    stage[0] = "write"
    path, rsha = _write_out(f"events_results_{meta['stamp']}.json", {
        "mode": mode, "status": status, "stop_reasons": stop_reasons, "meta": meta, "pins": PINS, "gates": gates,
        "control": ctl, "toy": {"pos": toy_pos, "neg": toy_neg}, "full_counts": full, "per_path": out["_per_path"]})
    return {"mode": mode, "status": status, "stop_reasons": stop_reasons, "gates": gates, "control": ctl,
            **print_set(out), "results_path": path, "results_sha256": rsha}


def _worker_main(td: str) -> None:
    import logging
    import pickle
    import warnings

    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    stage = ["start"]
    try:
        with open(f"{td}/in.pkl", "rb") as f:
            a = pickle.load(f)
        res = {"out": _drive(a["mode"], a["blobs"], a["manifests"], a["meta"], stage)}
    except GateError as e:
        res = {"err": f"gate {e} at stage {stage[0]}"}
    except Exception as e:
        res = {"err": f"{type(e).__name__} at stage {stage[0]}"}
    with open(f"{td}/summary.json", "w") as f:
        json.dump(res, f)


def _git(*args: str, cwd: Path) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True).stdout


if not _WORKER:  # ------------------------------------------------ Modal (driver and local entrypoint)
    import modal

    image = modal.Image.debian_slim(python_version="3.12").pip_install("inspect-ai==0.3.276")
    app = modal.App(APP_NAME)
    _vols = {mnt: modal.Volume.from_name(name).read_only() for name, mnt, _ in VOLS.values()}
    out_vol = modal.Volume.from_name(OUT_VOL_NAME, create_if_missing=True)

    @app.function(image=image, volumes={**_vols, "/out": out_vol}, cpu=(2.0, 2.0), memory=(4096, 4096), timeout=1800,
                  startup_timeout=120, scaledown_window=60, retries=0, max_containers=1)
    def drive(mode: str, blobs: dict, manifests: list, meta: dict) -> dict:
        """Driver: never parses data; isolated worker; commit in finally; sanitized errors only."""
        import pickle

        res, err = None, None
        try:
            try:
                with tempfile.TemporaryDirectory() as td:
                    with open(f"{td}/in.pkl", "wb") as f:
                        pickle.dump({"mode": mode, "blobs": blobs, "manifests": manifests, "meta": meta}, f)
                    rc = _run_worker([sys.executable, os.path.abspath(__file__), "--worker", td], subprocess.DEVNULL)
                    sp = f"{td}/summary.json"
                    res = (json.loads(Path(sp).read_text()) if os.path.exists(sp)
                           else {"err": f"worker exit {rc}, no summary"})
            finally:
                out_vol.commit()
        except Exception as e:
            err = f"{type(e).__name__} in driver"
        if err is None and "err" in res:
            err = str(res["err"])
        if err is not None:
            raise RuntimeError("events106 failed: " + err)
        return res["out"]

    @app.local_entrypoint()
    def main(mode: str = "real"):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        if mode not in ("real", "test", "test-neg"):
            raise SystemExit("mode must be real, test or test-neg")
        here = Path(__file__).resolve()
        meta = {"script_sha256": sha256(here.read_bytes()), "tz": "America/Los_Angeles",
                "stamp": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")}
        blobs, manifests = {}, []
        if mode == "real":
            top = Path(_git("rev-parse", "--show-toplevel", cwd=here.parent).decode().strip())
            rel = here.relative_to(top).as_posix()
            blob_here = _git("hash-object", str(here), cwd=top).decode().strip()
            committed = subprocess.run(["git", "rev-parse", f"HEAD:{rel}"], cwd=top,
                                       capture_output=True).stdout.decode().strip()
            meta.update({"git_head": _git("rev-parse", "HEAD", cwd=top).decode().strip(), "script_rel": rel,
                         "script_blob": blob_here})
            if _git("status", "--porcelain", cwd=top).strip():
                raise SystemExit("refusing: worktree not clean")
            if blob_here != committed:
                raise SystemExit("refusing: script is not the committed blob at HEAD")
            for name, (commit, path, pin) in PINS.items():
                b = _git("show", f"{commit}:{path}", cwd=top)
                if sha256(b) != pin:
                    raise SystemExit(f"refusing: input sha differs: {name}")
                blobs[name] = zlib.compress(b, 6)
            ls = _git("ls-tree", "-r", "--full-tree", MANIFEST_COMMIT, "--", MANIFEST_DIR, cwd=top).decode().splitlines()
            items = []
            for line in ls:
                info, path = line.split("\t", 1)
                if (path.rsplit("/", 1)[-1].startswith("manifest_") and path.endswith(".json")
                        and "/" not in path[len(MANIFEST_DIR):]):
                    items.append((path, _git("cat-file", "blob", info.split()[2], cwd=top)))
            if len(items) != MANIFEST_N or manifest_digest(items) != MANIFEST_DIGEST:
                raise SystemExit("refusing: manifest digest differs")
            manifests = [(p, zlib.compress(b, 6)) for p, b in items]
        out = drive.remote(mode, blobs, manifests, meta)
        print(_summary_line(out))
        if out["status"] not in ("PASS", "DONE"):
            raise SystemExit(1)


if __name__ == "__main__":  # worker / noise (internal), or the local toy
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2])
        raise SystemExit(0)
    if len(sys.argv) == 2 and sys.argv[1] == "--noise":
        _noise_main()
    if len(sys.argv) != 2 or sys.argv[1] not in ("toy", "toy-neg"):
        raise SystemExit("usage: diag3_sol106_events.py toy|toy-neg")
    res = run_toy(sys.argv[1] == "toy-neg")
    print(_summary_line({"mode": sys.argv[1], "status": "PASS" if res["pass"] else "MISMATCH",
                         "checks_false": sorted(k for k, v in res["checks"].items() if not v)}))
    raise SystemExit(0 if res["pass"] else 1)
