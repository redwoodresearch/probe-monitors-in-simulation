"""PATHGATE by code (09's item 1 after the census side a STOP, 10-17): check every file-system site a records-wrapper step of
run_peraction311 can reach, from a committed table derived from the code (fs_sites.py), against the step spec and the K configs at H.

  table: uv run --no-project --python 3.12 python -I tools_routec/pathgate_code.py table <commit> <annotations.json> > <table.json>
  check: uv run --no-project --python 3.12 python -I tools_routec/pathgate_code.py check --commit <H> --spec <repo path> \
             --table <repo path> --allowlist <repo path>
  cfgs:  uv run --no-project --python 3.12 python -I tools_routec/pathgate_code.py cfgs --commit <H> --spec <repo path>
         (PATHGATE3's config list for the K allowlist match: git-form and xfer/ flag values checked, not listed; see cfgs())

check (laptop; reads code, the step spec, the table and the allowlist at H by git show; parses a config only after its (path, sha256 at H)
matched a class-K allowlist entry; never a record outside the allowlist; prints paths, keys and site ids only):
1. the table is current: its module blobs (the 11 modules and every subprocess target) equal theirs at H, the in-step sites fs_sites.py
   finds at H are exactly the table's annotated sites, and no reachable import is outside the stdlib, the analysed modules, modal and
   numpy (a code change STOPs until the table is redone and reviewed);
2. the spec is a records-wrapper step of the runner: script run_peraction311.py, entry.attr main, alltasks true, exactly one of
   spec_only / dryrun, and none of launch_from_pickle, launch_diff, resume_spec, resume_tag (a63478's O3);
3. every annotated site active for the spec's flags (its "when"; an absent flag takes main's default) is checked by its resolver:
   flag        the flag's value (its default if absent; cut at "@") under its base must be a file at H, and a "@<sha256>" suffix
               must be that file's sha256 at H (O4);
   flag_out    an "@out:<repo path>" value: under experiments/<folder>/results/, and either the wrapper at H makes @out parents (its
               blob is in the table's wrapper_makes_out_parent) or the parent folder holds a tracked file at H;
   const       each path under its base must be a file at H;
   config_key  the config (allowlisted K, sha256 at H) holds at key a {git} pin whose commit is H, an ancestor of H or in the spec's
               commits (O5) and whose path exists there, or a {path} that is a file at H (repo-relative: the runner's cwd is the
               checkout root) or under the staging prefix;
   inputs_key  the --inputs file (allowlisted K) holds at key a path (under base) that is a file at H; optional keys may be absent;
               without --inputs, no_inputs_const is checked instead;
   inputs_full_pins  once the --inputs-full file (allowlisted K) pins waitlist, exclusions and census (inputs_full L426), each pin's
               path is a file at H with the pin's sha256 (make_waitlist.py check reads them in the checkout; a63478's R-1);
   copies_root the --copies-root flag equals the staging prefix (the wrapper stages INDEX_958 there);
   self, check_only, copies, code_mkdir, subprocess, argv_unreached: nothing to resolve (the runner's own file; an existence test;
               the committed copies index, whose misses are SystemExits; a parent the code makes itself; a child whose reads are its
               target's rows; a subcommand the child is never given);
   volume_api, not_in_wrapper, copies_env: active in a wrapper step means the step is wrong: STOP.
Exit 0 with "PATHGATE_CODE_OK ..." or exit 1 with one "BAD <site> <reason> <path or key>" line per problem.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

E = "experiments/2026-10-15_fpr_sweep_507_prep"
RP = f"{E}/run_peraction311.py"
HERE = Path(__file__).resolve().parent
OUT_PREFIX = re.compile(r"experiments/[A-Za-z0-9._-]+/results/[A-Za-z0-9._/-]+")  # the wrapper's @out rule (09, 10-17)
WRAPPER = "experiments/2026-10-17_records_step_container/run_records_step_in_container.py"
WRAPPER_MAKES_OUT_PARENT = ["70a13a249e1dfc9eff29dcb18e17f2624e48f679"]  # the F-out2 wrapper as passed (a0b05, 10-17)
NOT_IN_A_STEP = ("launch_from_pickle", "launch_diff", "resume_spec", "resume_tag")  # a laptop launch or a resume, never a wrapper step
RESOLVERS = {"flag", "flag_out", "const", "config_key", "inputs_key", "inputs_full_pins", "copies_root", "self", "check_only", "copies",
             "code_mkdir", "subprocess", "argv_unreached", "volume_api", "not_in_wrapper", "copies_env"}
HEX64 = re.compile(r"[0-9a-f]{64}")


def git(*a, text=True, check=True):
    return subprocess.run(["git", *a], capture_output=True, text=text, check=check)


def is_file_at(h, path):
    r = git("cat-file", "-t", f"{h}:{path}", check=False)
    return r.returncode == 0 and r.stdout.strip() == "blob"


def show(h, path):
    return git("show", f"{h}:{path}", text=False).stdout


def fs_sites():
    s = importlib.util.spec_from_file_location("fs_sites_tool", HERE / "fs_sites.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def enumerate_at(commit):
    """fs_sites.py's output at commit, captured (its main prints JSON)."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fs_sites().main(commit)
    return json.loads(buf.getvalue())


def key_of(s):
    return f"{s['module']}:{s['line']}:{s['call']}"


def make_table(commit, annotations_path):
    en = enumerate_at(commit)
    an = json.loads(Path(annotations_path).read_text())  # a hand-written annotation file (code citations only)
    step = [s for s in en["sites"] if s["in_step"]]
    keys = {key_of(s) for s in step}
    if keys != set(an) or any(an[k]["resolve"]["type"] not in RESOLVERS for k in an) or en["imports_unexpected"]:
        raise SystemExit(f"annotations differ from the in-step sites: missing {sorted(keys - set(an))}, extra {sorted(set(an) - keys)}; "
                         f"unexpected imports {en['imports_unexpected']}")
    out = {"what": "PATHGATE by code: every file-system site and non-git subprocess of run_peraction311 and its modules reachable in a "
                   "records-wrapper step (fs_sites.py: from main / dispatch and module import, not through row / drive / run_rows / "
                   "finish_rows / launch_from / run_cutfull; a subprocess's target script followed as __main__), each with how its path "
                   "is resolved and checked (pathgate_code.py check)",
           "made_at": commit, "modules": en["modules"], "targets": en["targets"], "wrapper_makes_out_parent": WRAPPER_MAKES_OUT_PARENT,
           "in_step": [{"site": s["site"], "key": key_of(s), "func": s["func"], "call": s["call"], "text": s["text"]} | an[key_of(s)] for s in step],
           "out_of_step": {"rule": "reached only through the CUT functions (Modal row / driver functions, the local row runner, the laptop launch "
                                   "and the cutfull runner), which no records-wrapper step runs: --spec-only returns in dispatch before any "
                                   "row (L5346-5352) and --dryrun returns after the plan (L6545-6582)",
                           "n": sum(1 for s in en["sites"] if not s["in_step"]),
                           "sites": [s["site"] + " " + s["func"] + " " + s["call"] for s in en["sites"] if not s["in_step"]]}}
    print(json.dumps(out, indent=1))


def walk(obj, key):
    for k in key:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def is_pin(e):  # inputs_full.is_pin (L103-104)
    return isinstance(e, dict) and all(isinstance(e.get(k), str) for k in ("commit", "path", "sha256"))


def check(h, spec_path, table_path, allow_path):
    bad = []
    tb = json.loads(show(h, table_path))
    al = json.loads(show(h, allow_path))
    kfiles = {f["path"]: f["sha256"] for f in al["files"] if f.get("class") == "K"}

    def k_config(path):  # parse a config only after its (path, sha256 at H) matched a K allowlist entry
        if path not in kfiles or not is_file_at(h, path):
            return None
        b = show(h, path)
        return json.loads(b) if hashlib.sha256(b).hexdigest() == kfiles[path] else None

    # 1. the table is current; no unanalysed import
    en = enumerate_at(h)
    if en["modules"] != tb["modules"]:
        bad.append(f"BAD table module blobs differ at H: {sorted(k for k in set(en['modules']) | set(tb['modules']) if en['modules'].get(k) != tb['modules'].get(k))}")
    step = {key_of(s) for s in en["sites"] if s["in_step"]}
    rows = {r["key"]: r for r in tb["in_step"]}
    if step != set(rows):
        bad.append(f"BAD table sites differ at H: unannotated {sorted(step - set(rows))}, gone {sorted(set(rows) - step)}")
    for x in en["imports_unexpected"]:
        bad.append(f"BAD unexpected import (its file reads are not analysed): {x}")
    defaults = en["main_defaults"]
    # 2. a records-wrapper step of the runner
    spec = json.loads(show(h, spec_path))
    en_ = spec.get("entry") or {}
    fl = en_.get("flags") or {}
    val = lambda k: fl[k] if k in fl else defaults.get(k)  # noqa: E731
    pfx = (spec.get("staging") or {}).get("prefix")
    if spec.get("script") != RP or en_.get("attr") != "main" or fl.get("alltasks") is not True or bool(fl.get("spec_only")) == bool(fl.get("dryrun")):
        bad.append("BAD spec not a records-wrapper step of run_peraction311.main: needs its script, alltasks true and exactly one of spec_only / dryrun")
    for k in NOT_IN_A_STEP:
        if fl.get(k):
            bad.append(f"BAD spec flag {k} set: a launch or a resume, not a records-wrapper step")
    commits = set(spec.get("commits") or [])
    wrap_ok = git("rev-parse", f"{h}:{WRAPPER}").stdout.strip() in tb.get("wrapper_makes_out_parent", [])
    IN = None
    if val("inputs"):
        IN = k_config(f"{E}/{str(val('inputs')).split('@', 1)[0]}")
        if IN is None:
            bad.append(f"BAD inputs not an allowlisted K config at H: {val('inputs')}")
    n_active = 0
    for r in tb["in_step"]:
        if not all(bool(val(w["flag"])) == w["truthy"] for w in r.get("when", [])):
            continue
        rv, site = r["resolve"], r["site"]
        t = rv["type"]
        base = (E + "/") if rv.get("base") == "E" else ""
        if t == "flag":
            v = val(rv["flag"])
            if not v:
                continue
            n_active += 1
            p_, _, h_ = str(v).partition("@")
            p = base + p_
            if str(v).startswith(("/", "@")) or not is_file_at(h, p):
                bad.append(f"BAD {site} flag {rv['flag']}: not a file at H: {p}")
            elif HEX64.fullmatch(h_) and hashlib.sha256(show(h, p)).hexdigest() != h_:
                bad.append(f"BAD {site} flag {rv['flag']}: its sha256 at H differs from the flag's: {p}")
        elif t == "flag_out":
            v = val(rv["flag"])
            if not v:
                continue
            n_active += 1
            q = str(v)[len("@out:"):] if str(v).startswith("@out:") else None
            par = str(Path(q).parent) if q else None
            if q is None or not OUT_PREFIX.fullmatch(q) or ".." in Path(q).parts or str(Path(q)) != q:
                bad.append(f"BAD {site} flag {rv['flag']}: not a canonical @out path under experiments/<folder>/results/: {v}")
            elif not wrap_ok and not git("ls-tree", "-r", "--name-only", h, "--", par + "/").stdout.strip():
                bad.append(f"BAD {site} flag {rv['flag']}: the wrapper at H makes no @out parent and {par}/ holds no tracked file at H")
        elif t == "const":
            n_active += 1
            for p in rv["paths"]:
                if not is_file_at(h, base + p):
                    bad.append(f"BAD {site} const: not a file at H: {base + p}")
        elif t == "config_key":
            n_active += 1
            cfg = k_config(rv["config"])
            v = walk(cfg, rv["key"]) if cfg is not None else None
            kk = f"{rv['config']} {'.'.join(rv['key'])}"
            if cfg is None:
                bad.append(f"BAD {site} config not an allowlisted K config at H: {rv['config']}")
            elif not isinstance(v, dict) or not (("git" in v) ^ ("path" in v)):
                bad.append(f"BAD {site} {kk}: neither a git nor a path pin")
            elif "git" in v:
                c_, _, p_ = str(v["git"]).partition(":")
                c_ = h if c_ == "HEAD" else c_  # the runner's git show HEAD:<path> runs in the checkout of H
                full = git("rev-parse", "--verify", "-q", f"{c_}^{{commit}}", check=False).stdout.strip() if re.fullmatch(r"[0-9a-f]{7,40}", c_) else ""
                if not full or not is_file_at(full, p_):
                    bad.append(f"BAD {site} {kk}: git pin not in the repository")
                elif full not in commits and git("merge-base", "--is-ancestor", full, h, check=False).returncode != 0:
                    bad.append(f"BAD {site} {kk}: git pin's commit is neither H, an ancestor of H nor in the spec's commits (not in the bundle)")
            else:
                p = str(v["path"])
                ok = (pfx and p.startswith(pfx.rstrip("/") + "/")) if p.startswith("/") else is_file_at(h, p)
                if not ok:
                    bad.append(f"BAD {site} {kk}: path neither under the staging prefix nor a file at H: {p}")
        elif t == "inputs_key":
            if not val("inputs"):  # the key's reader runs only with --inputs; L6017 reads a fixed path without it
                if rv.get("no_inputs_const"):
                    n_active += 1
                    if not is_file_at(h, base + rv["no_inputs_const"]):
                        bad.append(f"BAD {site} no --inputs, and not a file at H: {base + rv['no_inputs_const']}")
                continue
            n_active += 1
            v = walk(IN, rv["key"]) if IN is not None else None
            if v is None and rv.get("optional") and IN is not None:
                continue
            if not isinstance(v, str) or v.startswith("/") or ".." in Path(v).parts or not is_file_at(h, base + v):
                bad.append(f"BAD {site} --inputs {'.'.join(rv['key'])}: not a file at H under {base or 'the repo'}: {v}")
        elif t == "inputs_full_pins":
            if not val("inputs_full"):
                continue
            FU = k_config(f"{E}/{str(val('inputs_full')).split('@', 1)[0]}")
            if FU is None:
                n_active += 1
                bad.append(f"BAD {site} --inputs-full not an allowlisted K config at H: {val('inputs_full')}")
                continue
            if not all(is_pin(FU.get(k)) for k in rv["keys"]):  # inputs_full L426: the check (and the child) runs only once all are pinned
                continue
            n_active += 1
            for k in rv["keys"]:
                p = FU[k]["path"]
                if ".." in Path(p).parts or not p.startswith(E + "/") or not is_file_at(h, p):
                    bad.append(f"BAD {site} --inputs-full {k}: its path is not a file at H under this folder: {p}")
                elif hashlib.sha256(show(h, p)).hexdigest() != FU[k]["sha256"]:
                    bad.append(f"BAD {site} --inputs-full {k}: the file at H differs from the pin's sha256: {p}")
        elif t == "copies_root":
            n_active += 1
            if val("copies_root") != pfx:
                bad.append(f"BAD {site} copies_root differs from the staging prefix")
        elif t in ("volume_api", "not_in_wrapper", "copies_env"):
            n_active += 1
            bad.append(f"BAD {site} {t}: active for this spec's flags, which a records-wrapper step must never be")
        elif t not in ("self", "check_only", "copies", "code_mkdir", "subprocess", "argv_unreached"):
            bad.append(f"BAD {site} unknown resolver {t}")
    for b in bad:
        print(b)
    if bad:
        print(f"STOP PATHGATE_CODE {len(bad)} problems")
        sys.exit(1)
    print(f"PATHGATE_CODE_OK {spec_path} sites {len(rows)} active {n_active} wrapper_makes_out_parent {str(wrap_ok).lower()}")


FIXED_CFGS = [f"{E}/config/common455.json", "experiments/2026-10-14_fpr_sweep_unit/config/inputs_b.json",
              f"{E}/config/inputs_final_v2_pins.json", f"{E}/config/membership710.json"]  # PATHGATE2's 4 fixed configs
RELPATH = re.compile(r"[^/@][^ ]*/")  # PATHGATE2's rule (jq test("^[^/@][^ ]*/")): a relative path value, matched at its start
GITFORM = re.compile(r"([0-9a-f]{7,40}):(.+)")  # "<commit>:<path>[:<line> | @<sha256>]": rp reads it by git show (rp L6253, L6258)
XFER = re.compile(r"xfer/fpr507prep/[A-Za-z0-9._/-]+@[0-9a-f]{64}")  # a records path, read through the staged copies (rp L5501-5502, L5559)
LAPTOP = re.compile(r"/(private/(tmp|var/folders)|tmp|var/folders|Users|home|root)(/|$)")  # PATHGATE2's RX


def cfgs(h, spec_path):
    """PATHGATE3's config list (2e, 10-17; package 3 specs): PATHGATE2's 4 fixed configs plus every relative-path flag value (or
    list element) resolved under E and cut at "@", EXCEPT (1) a git-form value: its commit must be an ancestor of H or in the spec's
    commits (the bundle), its path a file at that commit (cat-file only), and an "@<sha256>" the sha256 of that file's bytes there (never
    parsed); a ":<line>" suffix is dropped; (2) an xfer/ value: its shape must be xfer/fpr507prep/<path>@<sha256> (the INDEX_958 check
    is pcd1's O2 follow-up). A flag string under a laptop root, other than the staging prefix or a path under it, is BAD. Prints one
    config path per line (sorted), then "PATHGATE_CFGS_OK ..."; or BAD lines and exit 1. Reads the spec (a config) and git metadata."""
    spec = json.loads(show(h, spec_path))
    fl = (spec.get("entry") or {}).get("flags") or {}
    pfx = (spec.get("staging") or {}).get("prefix") or fl.get("copies_root")
    commits = set(spec.get("commits") or [])
    out, bad, ng, nx = set(FIXED_CFGS), [], 0, 0
    for k, v in fl.items():
        for x in (v if isinstance(v, list) else [v]):
            if not isinstance(x, str):
                continue
            if LAPTOP.match(x) and not (isinstance(pfx, str) and pfx and (x == pfx or x.startswith(pfx.rstrip("/") + "/"))):
                bad.append(f"BAD flag {k}: a laptop path: {x}")
                continue
            if not RELPATH.match(x):
                continue
            g = GITFORM.fullmatch(x)
            if g:
                ng += 1
                c, rest = g.groups()
                p, at, s = rest.rpartition("@")
                if not at:
                    p, s = re.sub(r":[0-9]+$", "", rest), ""
                full = git("rev-parse", "--verify", "-q", f"{c}^{{commit}}", check=False).stdout.strip()
                if not full:
                    bad.append(f"BAD flag {k}: git-form commit not in the repository: {c}")
                elif full not in commits and git("merge-base", "--is-ancestor", full, h, check=False).returncode != 0:
                    bad.append(f"BAD flag {k}: git-form commit neither an ancestor of H nor in the spec's commits (not in the bundle): {c}")
                elif ".." in Path(p).parts or not is_file_at(full, p):
                    bad.append(f"BAD flag {k}: git-form path not a file at its commit: {p}")
                elif at and (not HEX64.fullmatch(s) or hashlib.sha256(show(full, p)).hexdigest() != s):
                    bad.append(f"BAD flag {k}: git-form @sha256 differs from the file's at its commit: {p}")
                continue
            if x.startswith("xfer/"):
                nx += 1
                if not XFER.fullmatch(x) or ".." in x.split("@", 1)[0].split("/"):
                    bad.append(f"BAD flag {k}: an xfer value must be xfer/fpr507prep/<path>@<sha256>: {x}")
                continue
            out.add(f"{E}/{x.split('@', 1)[0]}")
    for b in bad:
        print(b)
    if bad:
        print(f"STOP PATHGATE_CFGS {len(bad)} problems")
        sys.exit(1)
    for p in sorted(out):
        print(p)
    print(f"PATHGATE_CFGS_OK {spec_path} configs {len(out)} gitform {ng} xfer {nx}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "table":
        make_table(sys.argv[2], sys.argv[3])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd", choices=["check", "cfgs"])
        ap.add_argument("--commit", required=True)
        ap.add_argument("--spec", required=True)
        ap.add_argument("--table")
        ap.add_argument("--allowlist")
        a = ap.parse_args()
        if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
            raise SystemExit("--commit: 40 hex")
        if a.cmd == "cfgs":
            cfgs(a.commit, a.spec)
        elif not (a.table and a.allowlist):
            raise SystemExit("check: --table and --allowlist are required")
        else:
            check(a.commit, a.spec, a.table, a.allowlist)
