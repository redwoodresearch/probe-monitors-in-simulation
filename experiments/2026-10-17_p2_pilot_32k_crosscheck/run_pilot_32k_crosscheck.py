"""p2 pilot 32k cross-check (09 via 2e, 10-17; pre-S2; DRAFT until reviewed): p2's FULL-LENGTH pilot scores at inside-32k units against the
32k pilot score record the p2 scorer pins (run_simh_scores.py L104, PILOT_PINS["source"]: "the 32k /simh/pilot/p2 write"). Separate from the
B gate, which keeps its uniform "n/a: no cut counterpart" for the pilot (no registered cut file). A PASS gives p2's pilot disclosure
"32k cross-check: PASS (separate, @ <sha>)"; a FAIL is a finding for 09.

Pre-conditions, else STOP (no result):
- the 32k record scored the REGISTERED 32k pilot staging. Its dump is in the LAUNCH-TIME scorer's format, run_simh_scores.py @ 903ac2905 (blob 8749b853): main L267, spec L322 (staged = the staged PATH; staged_sha and meta_arm separate),
  dump L325; it bound them in-job (L188 the staged file's sha256, L198 complete.json's inputs_sha256, L284 the dump manifest's bytes);
  the launch commit is inferred from times (903ac2905 at 17:22, the record at 17:23:05, the next commit 41a865a0e at 17:43). dump.staged is
  a string whose file name is the arm file; dump.staged_sha is the registered stage record's staged sha256 (simh_stage_c_pilot_2026-10-14_
  165444.json @ 300652707); dump.run is the registered root, which is also the registered extraction record's config.run; and
  dump.dump_manifest_sha256 is that extraction record's own sha256 (extract_..._171107.json @ c389286cf, whose config binds the staged
  sha256). This is make_pilot_records.registered()'s reading (@ 5333963c9; it built the pilot's S_old records);
- each 32k per-row npz holds exactly the 20 pilot keys: simh_pilot|<task> of the counted keys (@ 344829af6, the scorer's PILOT_PINS["keys"]).

PASS / FAIL (09's ruling, 10-17: match the gate exactly), per probe, on the per-row "command" npz (the only per-row file in the 32k write):
- the B gate's cut-vs-uncut unit rule, verbatim (run_peraction311.py blob 0306186c8ca4: cutfull_file_checks L1519, L1522, L1581-1589, with
  full_len as in run_gates L1951, from the pilot inventory pilot_rows_v1, the scorer's PILOT_PINS["inventory"]): the (key, unit) units
  inside 32,768 on the full-length side, minus boundary units ([32765, 32768) on runs longer than 32,768, removed from BOTH sides), equal
  the 32k file's as multisets, and the full-length file holds the same 20 keys -> keys_equal;
- call_index equal on every matched (key, unit).
n_mismatches = the units in either multiset but not the other + the matched units whose call_index differs; pass = keys_equal and
n_units_compared > 0 and n_mismatches == 0.

Scores are DESCRIPTIVE ONLY and never part of pass (the B gate compares no score values: run_gates docstring, L1934-1935): per probe file,
the max over matched units of |s_full - s_32k| / max(1, |s_32k|) as a decade bucket (never the value; "nan" if NaN is on one side only, NaN
on both counts as equal). flag_rel_diff_over_0.1 is set when any probe's bucket is ">0.1" or "nan": a flag for 09 to look at before S2,
not a STOP.

What crosses back (a63478 P2X-R2, 2e's option (b)): the container writes the whole result, buckets included, to ONE record on p2's records
Volume (OUTVOL, as regress_oldnew's regress_ptr does), exclusive-create, committed. It returns ONLY keys_equal, n_units_compared,
n_mismatches, flag_rel_diff_over_0.1, pass and that record's {volume, path, sha256}: booleans, counts and a pointer, no bucket and no score.

Inputs are git blobs (commit, path, sha256). The laptop reads each with git show and passes the bytes unparsed; the container checks every
sha256 and parses them. The npz files are read from 09's Volume (read-only) and checked against their record's sha256. Nothing else is read.

  launch: NOTE_pilot_32k_crosscheck_2026-10-17.md (0d launches, after the review's PASS)
"""

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # rule 27: in the container this file is /root/<name>.py (one parent)
APP = "probe-shift_aswin-surya_sub-p2pilot32k_xcheck_kill-2026-11-14"
CWVOL = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"  # 09's output Volume (run_simh_scores.CWVOL), mounted read-only
OUTVOL = "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13"  # p2's records Volume (regress_oldnew.OUTVOL): the full result goes here
OUT_DIR = "2026-10-17_p2_pilot_32k_crosscheck"
PINS = {  # (commit, path, sha256). m32, keys and inventory are the scorer's PILOT_PINS "source", "keys" and "inventory" (the test checks)
    "m32": ("41a865a0e", "experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-14_172305.json",
            "663458f13bcfeaff4f06521cede93161c59e1256f139af1ae71512187aba7976"),
    "mfull": ("448813b9bf98b7eb6fb141a228c4c480dc6d4d00", "experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-17_135035.json",
              "a7ce8e7332d03408012d974d67104f948f41d9f729b7f08e7688c76b08bd71a1"),
    "reg": ("300652707", "experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_stage_c_pilot_2026-10-14_165444.json",
            "1f455a785b0eef2facfdd59ca21352b8d43399f4b677b2cadc219d6287360db6"),
    "reg_x": ("c389286cf", "experiments/2026-10-14_q3_q35_simhonest_purchase/results/extract_simh_simh_q4b_extract_v1_pilot_c_a1_2026-10-14_171107.json",
              "9ba914b378369bd40f09db16fea82d99b717c7f9bed73a9f3c28c14c8981b76e"),
    "keys": ("344829af6", "experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_pilot_v3.json",
             "2149c5cde368aef9e5012c8d9cda5b12fe5a93db3d27ad2d525225bb9f511b27"),
    "inventory": ("98575cc3471a224c87401111fbb8caee28ff5323", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_pilot_rows_v1_2026-10-17_081456.json",
                  "854ff211aaa61a3b24ca17b3104c910a377ea11808e06680f4b99e4ae1f9bf45"),
}
WORLD, N, PROBES = "simh_pilot", 20, ("fibs", "refusal", "harmfulness", "highstakes")  # run_simh_scores: PILOT_WORLD, 20 counted keys, SIMH_PROBES
STAGED_SET, ARM, ARM_FILE = "simh_q4f_v1_pilot", "sim_honest@september", "sim_honest@september.jsonl"  # make_pilot_records P (@ 5333963c9)
REG_ROOT = "simh_q4b_extract_v1_pilot_c"  # make_pilot_records P["registered_root"] (@ 5333963c9): the registered extraction root
OUT_32K, OUT_FULL = "/simh/pilot/p2", "/cutfull/sold/p2/simh_q4f_v1_pilot_cutfull_v1__sim_honest"  # PILOT_ORIG[1]; the pilot's cutfull folder
BUCKETS = (1e-12, 1e-9, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
FLAG_BUCKETS = (">0.1", "nan")
OUT_KEYS = ("keys_equal", "n_units_compared", "n_mismatches", "rel_diff_bucket", "flag_rel_diff_over_0.1", "pass")  # the Volume record's result
RELAY_KEYS = ("keys_equal", "n_units_compared", "n_mismatches", "flag_rel_diff_over_0.1", "pass", "result")  # all that crosses back
HEX64 = re.compile(r"[0-9a-f]{64}")
STAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}")

app = modal.App(APP)
cw_vol = modal.Volume.from_name(CWVOL)
o_vol = modal.Volume.from_name(OUTVOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


def bucket(m) -> str:
    """A decade bucket, never the value: "none" (no unit), "nan" (NaN on one side), "0", "<=1e-12", "(1e-12,1e-09]", ... ">0.1"."""
    if m is None or m != m:
        return "none" if m is None else "nan"
    if m == 0:
        return "0"
    lo = None
    for e in BUCKETS:
        if m <= e:
            return f"<={e:g}" if lo is None else f"({lo:g},{e:g}]"
        lo = e
    return f">{lo:g}"


def crosscheck_core(blobs: dict, read) -> dict:
    """The container's body, pure (the test runs it on synthetic inputs). blobs: {pin name: bytes}; read(path under the Volume) -> bytes."""
    import io
    from collections import Counter

    import numpy as np

    if set(blobs) != set(PINS) or any(hashlib.sha256(blobs[n_]).hexdigest() != PINS[n_][2] for n_ in PINS):
        raise SystemExit("STOP: an input's bytes differ from its pinned sha256")
    j = {n_: json.loads(blobs[n_]) for n_ in PINS}
    s_, xc_ = j["reg"].get("simh") or {}, j["reg_x"].get("config") or {}
    ss_ = (s_.get("assembled") or {}).get("staged_inputs_sha256")
    if (s_.get("out") != STAGED_SET or not isinstance(ss_, dict) or set(ss_) != {ARM_FILE} or not HEX64.fullmatch(str(ss_[ARM_FILE]))
            or s_["assembled"].get("epochs") != {ARM: N} or xc_.get("inputs") != STAGED_SET or xc_.get("inputs_sha256") != ss_ or xc_.get("test") is not False):
        raise SystemExit("STOP: the registered 32k stage record or its extraction is not the 20-epoch pilot staging with one staged sha256")
    m32, d32 = j["m32"], j["m32"].get("dump") or {}
    stp_ = d32.get("staged")  # the launch-time format (run_simh_scores.py @ 903ac2905 L322): the staged PATH; its sha256 in staged_sha
    if not (m32.get("verdict") == "written" and d32.get("mode") == "simh" and d32.get("out_dir") == OUT_32K and d32.get("world") == WORLD
            and d32.get("n_epochs") == N
            and d32.get("run") == REG_ROOT and xc_.get("run") == REG_ROOT
            and isinstance(stp_, str) and Path(stp_).name == ARM_FILE
            and d32.get("staged_sha") == ss_[ARM_FILE]
            and d32.get("dump_manifest_sha256") == PINS["reg_x"][2]):
        raise SystemExit("STOP: the 32k record did not score the registered 32k pilot staging (its staged file and sha256, the registered run and "
                         "extraction record)")
    kr_ = j["keys"].get("keys") if isinstance(j["keys"].get("keys"), list) else []
    joins = {f"{WORLD}|{r_.get('task')}" for r_ in kr_}
    if j["keys"].get("n") != N or len(kr_) != N or len(joins) != N or any(r_.get("world") != WORLD for r_ in kr_):
        raise SystemExit("STOP: the counted keys are not 20 distinct simh_pilot tasks")
    names = [f"{p}__command.npz" for p in PROBES]
    f32 = m32.get("files") if isinstance(m32.get("files"), dict) else {}
    fo_ = j["mfull"].get("folders") if isinstance(j["mfull"].get("folders"), dict) else {}
    ffl = (fo_.get(OUT_FULL) or {}).get("files") if list(fo_) == [OUT_FULL] else None
    if (j["mfull"].get("verdict") != "written" or not isinstance(ffl, dict) or sorted(k_ for k_ in f32 if k_.endswith(".npz")) != sorted(names)
            or any(not HEX64.fullmatch(str(f32[n_])) or not HEX64.fullmatch(str(ffl.get(n_))) for n_ in names)):
        raise SystemExit("STOP: the 32k record's npz are not exactly the four command npz, or the full-length record does not list them, with sha256s")
    full_len = {r["key"]: r.get("full_len", r.get("tokens_before")) for r in j["inventory"]["runs"]}  # run_gates L1951 (blob 0306186c8)
    fl_n = {}
    for k_, v_ in full_len.items():  # cutfull_file_checks L1439-1443: one leading "runs/" stripped; a key in both forms must agree
        if fl_n.setdefault(k_.removeprefix("runs/"), v_) != v_:
            raise SystemExit("STOP: a pilot inventory key disagrees with its runs/-prefixed twin")
    full_len = fl_n

    def npz(folder, name, sha):
        b_ = read(f"{folder.strip('/')}/{name}")
        if hashlib.sha256(b_).hexdigest() != sha:
            raise SystemExit(f"STOP: {folder}/{name} differs from its record's sha256")
        return np.load(io.BytesIO(b_))

    keys_equal, n_cmp, n_bad, rel_b, uid = True, 0, 0, {}, "unit"
    for p_, n_ in zip(PROBES, names):
        zc, zu = npz(OUT_32K, n_, f32[n_]), npz(OUT_FULL, n_, ffl[n_])
        fu_, fc_ = ("key", uid, "call_index", "score", "token", "n_tokens", "run_key"), ("key", uid, "call_index", "score")
        if any(f not in zu.files for f in fu_) or any(f not in zc.files for f in fc_) or len({len(zu[f]) for f in fu_}) != 1 or len({len(zc[f]) for f in fc_}) != 1:
            raise SystemExit(f"STOP: {n_}: fields missing or row arrays differ in length")
        if set(zc["key"].astype(str).tolist()) != joins:
            raise SystemExit(f"STOP: {n_}: the 32k file does not hold exactly the 20 pilot keys")
        key, rk, nt, tok = zu["key"].astype(str), zu["run_key"].astype(str), zu["n_tokens"].astype(np.int64), zu["token"].astype(np.int64)
        rk = np.asarray([x_.removeprefix("runs/") for x_ in rk.tolist()])
        if any(full_len.get(r) != x for r, x in zip(rk.tolist(), nt.tolist())):  # the B gate's G3 (L1578): the inventory's full length
            raise SystemExit(f"STOP: {n_}: n_tokens differs from the pilot inventory's full length (or a run has none)")
        # the B gate's rule, verbatim (cutfull_file_checks L1519, L1522, L1581-1589; uid "unit", as there when both files hold it; no "nocut" here)
        inside = tok < 32768
        excl = [(full_len.get(r, 0) > 32768) and (32765 <= t_ < 32768) for r, t_ in zip(rk.tolist(), tok.tolist())]
        zkeys = {(k_, str(u)) for k_, u, r, t_ in zip(key.tolist(), zu[uid].astype(str).tolist(), rk.tolist(), tok.tolist())
                 if full_len.get(r, 0) > 32768 and t_ >= 32765}
        cu = Counter((k_, str(u)) for k_, u, i_, x_ in zip(key.tolist(), zu[uid].astype(str).tolist(), inside.tolist(), excl) if i_ and not x_)
        cc = Counter((k_, str(u)) for k_, u in zip(zc["key"].astype(str).tolist(), zc[uid].astype(str).tolist()) if (k_, str(u)) not in zkeys)
        keys_equal = keys_equal and cu == cc and set(key.tolist()) == joins
        if max(cu.values(), default=1) > 1 or max(cc.values(), default=1) > 1:  # matching by (key, unit) needs one row per unit and side
            raise SystemExit(f"STOP: {n_}: a (key, unit) appears twice in one file")
        iu = {(k_, str(u)): i for i, (k_, u, i_, x_) in enumerate(zip(key.tolist(), zu[uid].astype(str).tolist(), inside.tolist(), excl)) if i_ and not x_}
        ic = {(k_, str(u)): i for i, (k_, u) in enumerate(zip(zc["key"].astype(str).tolist(), zc[uid].astype(str).tolist())) if (k_, str(u)) not in zkeys}
        both = sorted(set(iu) & set(ic))
        a_, c_ = np.array([iu[x] for x in both], dtype=np.int64), np.array([ic[x] for x in both], dtype=np.int64)
        n_cmp += len(both)
        n_bad += sum((cu - cc).values()) + sum((cc - cu).values()) + int((zu["call_index"][a_] != zc["call_index"][c_]).sum())
        su, sc = zu["score"].astype(np.float64)[a_], zc["score"].astype(np.float64)[c_]  # descriptive only: never part of pass
        nu, nc = np.isnan(su), np.isnan(sc)
        rel = np.where(nu | nc, 0.0, np.abs(su - sc) / np.maximum(1.0, np.abs(np.where(nc, 0.0, sc))))
        rel_b[p_] = bucket(float("nan") if (nu != nc).any() else float(rel.max()) if len(both) else None)
    return {"keys_equal": bool(keys_equal), "n_units_compared": int(n_cmp), "n_mismatches": int(n_bad), "rel_diff_bucket": rel_b,
            "flag_rel_diff_over_0.1": any(b_ in FLAG_BUCKETS for b_ in rel_b.values()), "pass": bool(keys_equal and n_cmp > 0 and n_bad == 0)}


def to_volume(full: dict, stamp: str, root: Path) -> dict:
    """The whole result (the score buckets included) as ONE record under root (OUTVOL's mount), written once; -> what crosses back: the
    booleans, the counts and the record's {volume, path, sha256} (the body is ASCII, so its sha256 is the file's)."""
    if list(full) != list(OUT_KEYS) or not STAMP.fullmatch(stamp):
        raise SystemExit("STOP: not the result fields, or not a stamp")
    rel = f"{OUT_DIR}/xcheck_{stamp}/result.json"
    (root / rel).parent.mkdir(parents=True, exist_ok=False)
    body = json.dumps(full | {"label": "p2 pilot 32k cross-check (09 via 2e, 10-17); pass = the B gate's unit rule + call_index; the buckets are "
                                       "descriptive only (09)"}, indent=1, ensure_ascii=True)
    with open(root / rel, "x", encoding="ascii") as fh:
        fh.write(body)
    return {k_: full[k_] for k_ in RELAY_KEYS[:-1]} | {"result": {"volume": OUTVOL, "path": rel, "sha256": hashlib.sha256(body.encode("ascii")).hexdigest()}}


@app.function(image=image, cpu=1, memory=4096, timeout=900, retries=0, volumes={"/cw": cw_vol.read_only(), "/o": o_vol})
def crosscheck(blobs: dict, stamp: str) -> str:
    """09's Volume read-only at /cw; the result record on OUTVOL at /o; one json.dumps string of to_volume's reply crosses back."""
    out = to_volume(crosscheck_core(blobs, lambda rel: (Path("/cw") / rel).read_bytes()), stamp, Path("/o"))
    o_vol.commit()
    return json.dumps(out)


@app.local_entrypoint()
def main():
    def git(*a, text=True):
        return subprocess.run(["git", *a], capture_output=True, check=True, cwd=REPO, text=text).stdout

    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("STOP: uncommitted tracked changes")
    head = git("rev-parse", "HEAD").strip()
    if not git("branch", "-r", "--contains", head).strip():
        raise SystemExit("STOP: HEAD is not pushed")
    blobs = {n_: git("show", f"{c_}:{p_}", text=False) for n_, (c_, p_, _) in PINS.items()}  # bytes only: parsed in the container
    if any(hashlib.sha256(b_).hexdigest() != PINS[n_][2] for n_, b_ in blobs.items()):
        raise SystemExit("STOP: a pinned blob's sha256 differs")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = json.loads(crosscheck.remote(blobs, stamp))  # the ONE laptop parse: to_volume's reply (booleans, counts, a pointer), written by crosscheck()
    ptr = out.get("result") if isinstance(out, dict) else None
    if (list(out) != list(RELAY_KEYS) or not all(isinstance(out[k_], bool) for k_ in ("keys_equal", "flag_rel_diff_over_0.1", "pass"))
            or not all(type(out[k_]) is int and out[k_] >= 0 for k_ in ("n_units_compared", "n_mismatches"))
            or not isinstance(ptr, dict) or list(ptr) != ["volume", "path", "sha256"] or ptr["volume"] != OUTVOL
            or ptr["path"] != f"{OUT_DIR}/xcheck_{stamp}/result.json" or not HEX64.fullmatch(str(ptr["sha256"]))):
        raise SystemExit("STOP: the reply is not the five fields and the record pointer")
    rec = {"what": "p2 pilot 32k cross-check (09 via 2e, 10-17): full-length inside-32k units vs the 32k pilot write; separate from the B gate",
           "result": out, "rule": "NOTE_pilot_32k_crosscheck_2026-10-17.md (09, 10-17; a63478 P2X-R2 option (b))",
           "pins": {n_: dict(zip(("commit", "path", "sha256"), v_)) for n_, v_ in PINS.items()},
           "b_gate_rule": "run_peraction311.py blob 0306186c8ca4da6bc4451b518549ec312a8b9769: cutfull_file_checks L1519, L1522, L1578, L1581-1589; run_gates L1951",
           "git": head, "script_blob": git("rev-parse", f"{head}:{(HERE / Path(__file__).name).relative_to(REPO)}").strip(), "app": APP, "stamp (PT)": stamp}
    dest = HERE / "results" / f"pilot_32k_crosscheck_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    with dest.open("x") as fh:
        fh.write(json.dumps(rec, indent=1))
    print(json.dumps(out))
    print(f"record {dest.relative_to(REPO)} sha256 {hashlib.sha256(dest.read_bytes()).hexdigest()}; {'PASS' if out['pass'] else 'FAIL'}"
          f"{'; FLAG' if out['flag_rel_diff_over_0.1'] else ''}")
