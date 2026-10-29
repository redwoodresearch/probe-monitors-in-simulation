"""Build the route C Tables 3/4, Figures 1/5/6 and Figure 4 run files from their committed NOTE blocks, every placeholder filled from
its marker file by an exact rule (2e, 10-18: the drop 1a / drop 2 launch packages; fill_run_downstream.py's rules, blob 60842719 on
aswin/routec-anchor-figs78 @ 9f58a94a9, plus a count-checked fill of the package's pins / plan template). Laptop, keys only: it reads
the NOTE and the template at HEAD (git show), the one-line markers the launch steps write, the S7L launch log's one driver_call line
(keys and shas), 09's S2 marker and covered list (keys); it HASHES committed records and sidecars (git cat-file bytes, never parsed).
It never runs a block, and it writes only into a new --out-dir.

Packages:
  tables-sel  TVARS0, then TSEL                                        (on the merged head HB, before TSEL)
  tables      TVARS0 + TVARS1, then TFILL TGATE TIMP TTAB TCOMMIT      (HEAD = TSEL's commit; --tsel-ok)
  figs156     FVARS, then F156FILL F156GATE F156R F156F F156C          (A, S2(A), the group / pw commits merged into HB)
  fig4        F4VARS, then F4FILL F4GATE F4G F4R F4F F4C               (drop 2: B and S2(B) merged into HB)

Sources (each marker one line of single-space-separated fields, newline-terminated; the whole line re-checked after the split):
  --s7-dir     §7's launch dir $L/S7_<TAG>_<STAMP7> (NOTE_routec_s7_2026-10-17.md @ 631e904e7): launched.txt (S7L: app id, driver
               call, pickle, pickle sha256, launch head), launch.log (S7L: its one '{"driver_call": "fc-…' line, run_peraction311's
               launch_from print), ok.txt (S7R: record, sha256, npz, npz sha256, app id), sidecar.txt (S7SC: SCP SCS CF HO)
  --fok        CUT's FILL1B fill_ok.txt (NOTE_routec_stageD_kappa §11 REQ-1): field 4 ARC:ARP@ARS (A's committed .json.gz), field 5 the
               sidecar pointer (A packages)
  --a-final    §7's S7RC marker $L/S7_A_<STAMP7>/record_commit.txt ('<commit> <path.json.gz> <decompressed sha256>', aswin/routec-launch
               @ 1d8867052): A's committed copy when Tables runs before FILL1B (both given: they must agree)
  --b-final    the same marker for TAG B, $L/S7_B_<STAMP7>/record_commit.txt (fig4; required)
  --s2         <commit>:<path> of 09's S2 marker on aswin/convergence-audit-09 (fetched first); --cover the S2 covered list file
               (build_signoff_covered.py --kind S2 @ blob c6df19dd, written L108-110)
  --gc --gx --pw  SIMPROBE's commit_ok.txt markers for the group catch, four-probe excess and pw-shares records (catch_1 / excess_1 /
               routec_1, NOTE_routec_group_steps.md L230 / L259 @ 74edde46f): '<commit> <record path> <record sha256> <sidecar path>
               <sidecar sha256>'; fields 1:2 name the record, 3-5 are cross-checked (the sidecar is <stem>_LABEL.json, hashed)
  --tsel-ok    TSEL's ok.txt: selection path, commit, sha256 (tables)
  --head       HB: the build head (the merge of routec-launch and the producers' commits into this branch)

  uv run --no-project --python 3.12 python -I tools_routec/fill_run_tablesfigs.py --pkg <tables-sel|tables|figs156|fig4> \\
      --note-commit <40hex> --head <40hex> [sources] --out-dir <a new folder>     (from experiments/2026-10-15_fpr_sweep_507_prep)
"""

import argparse
import datetime
import difflib
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

E = "experiments/2026-10-15_fpr_sweep_507_prep"
X = "experiments/2026-10-16_examples_v81"
H40, H64 = r"[0-9a-f]{40}", r"[0-9a-f]{64}"
FROZEN_C = "1a842a85635bd63a8bfc60071195584834117c11"
STAMP_RX = r"20[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9]{6}"
GZ = r"experiments/[^\s@':]+[.]json[.]gz"
RX = {
    "HB": H40, "A_LAUNCH": H40, "A_REC_P": GZ, "A_REC_C": H40, "A_REC_B": H64, "A_REC_J": H64, "A_SPEC": r"xfer/fpr507prep/drivers/[^/\s@':]+[.]pkl",
    "A_SPEC_S": H64, "SIDECAR": rf"git:{H40}:{E}/results/routec_A_label_sidecar_[^/\s@']+[.]json@{H64}", "C09": H40,
    "MP": rf"experiments/2026-10-14_priority1_spec_09/signoff/S2_{STAMP_RX}[.]json", "MSHA": H64,  # GATES Amendment 15's path form
    "CVPKG": r"tables|figs156|fig4", "CVP": rf"{E}/config/s2_cover/routec_(tables|figs156|fig4)_cover_{STAMP_RX}[.]json", "CVS": H64,
    "COVER": r'\["[^"\'\s]+@[0-9a-f]{64}"(,"[^"\'\s]+@[0-9a-f]{64}")*\]', "SEL_P": rf"{E}/results/estimator_selection_routec_{STAMP_RX}[.]json",
    "SEL_C": H40, "SEL_S": H64, "STAMP": STAMP_RX, "PFILE": r"/[^\s']+[.]json", "PFS": H64,
    "A_BASE": r"[^/\s@':]+[.]json", "A_J": H64, "A_CP": GZ, "A_CC": H40, "A_B": H64, "A_SP": rf"{E}/results/routec_A_label_sidecar_[^/\s@']+[.]json",
    "A_SC": H40, "A_SS": H64,
    "B_BASE": r"[^/\s@':]+[.]json", "B_J": H64, "B_CP": GZ, "B_CC": H40, "B_B": H64, "B_LAUNCH": H40,
    "B_SC_P": rf"{E}/results/routec_B_label_sidecar_[^/\s@']+[.]json", "B_SC_C": H40, "B_SC_S": H64,
}
DERIVED = {  # role -> (placeholder prefix, committed-path regex)
    "gc": ("GC", rf"{E}/results/group_catch_{STAMP_RX}[.]json"),
    "gx": ("GX", rf"{E}/results/group_excess_{STAMP_RX}(_[0-9a-f]{{32}})?[.]json"),
    "pw": ("PW", rf"{E}/results/pw_shares_routec/pw_shares_pooled_all_origins_[^/\s@']+_routeC_?{STAMP_RX}[.]json"),
}
for _p, _rx in DERIVED.values():
    RX |= {f"{_p}_BASE": r"[^/\s@':]+[.]json", f"{_p}_P": _rx, f"{_p}_C": H40, f"{_p}_S": H64,
           f"{_p}_SP": _rx.removesuffix("[.]json") + "_LABEL[.]json", f"{_p}_SC": H40, f"{_p}_SS": H64}
RULES = {  # placeholder -> (source, field or rule)
    "HB": ("--head", "the build head; HEAD, or TSEL's commit's parent for tables"),
    "A_LAUNCH": ("--s7-dir launched.txt (A)", "field 5, the S7L launch head (= A's meta.git; the container re-asserts it)"),
    "B_LAUNCH": ("--s7-dir launched.txt (B)", "field 5, the S7L launch head (= B's meta.git; the container re-asserts it)"),
    "A_REC_C": ("--fok (or --a-final)", "field 4's ARC"), "A_REC_P": ("--fok (or --a-final)", "field 4's ARP"), "A_REC_J": ("--fok (or --a-final)", "field 4's ARS (== ok.txt field 2)"),
    "A_REC_B": ("git", "sha256 of the blob ARC:ARP (hashed, never parsed)"),
    "A_SPEC": ("--s7-dir launch.log (A)", 'the one {"driver_call": …} line, key "spec"'),
    "A_SPEC_S": ("--s7-dir launch.log (A)", 'key "spec_sha256" (== launched.txt field 4)'),
    "SIDECAR": ("--s7-dir sidecar.txt (A)", "git:<field 3>:<field 1>@<field 2> (== fill_ok field 5)"),
    "C09": ("--s2", "the commit"), "MP": ("--s2", "the path"), "MSHA": ("git", "sha256 of the blob C09:MP"),
    "COVER": ("--cover-ok (S2COVER's cover_ok.txt)", ("the committed COVER file at field 1:2 (hashes to field 3), compact JSON; == the "
                                                      "package's derived set and == the S2 marker's covered, as sets (GATES Amendment 15)")),
    "CVPKG": ("--pkg", "the package the COVER list is for"), "CVP": ("--cover-out", "the COVER file the tool wrote (exclusive)"),
    "CVS": ("--cover-out", "that file's sha256"),
    "SEL_P": ("--tsel-ok", "field 1"), "SEL_C": ("--tsel-ok", "field 2"), "SEL_S": ("--tsel-ok", "field 3"),
    "STAMP": ("the tool", "its build time, Pacific, YYYY-MM-DD_HHMMSS"), "PFILE": ("the tool", "the filled template it wrote"),
    "PFS": ("the tool", "that file's sha256"),
    "A_BASE": ("--s7-dir ok.txt (A)", "field 1's file name (field 1 = xfer/fpr507prep/records/results/<it>)"),
    "A_J": ("--s7-dir ok.txt (A)", "field 2 (== fill_ok ARS)"), "A_CP": ("--fok (or --a-final)", "field 4's ARP"), "A_CC": ("--fok (or --a-final)", "field 4's ARC"),
    "A_B": ("git", "sha256 of the blob ARC:ARP"), "A_SP": ("--s7-dir sidecar.txt (A)", "field 1"), "A_SC": ("--s7-dir sidecar.txt (A)", "field 3"),
    "A_SS": ("--s7-dir sidecar.txt (A)", "field 2"),
    "B_BASE": ("--s7-dir ok.txt (B)", "field 1's file name"), "B_J": ("--s7-dir ok.txt (B)", "field 2 (== --b-final's sha)"),
    "B_CP": ("--b-final (S7RC record_commit.txt, B)", "field 2"), "B_CC": ("--b-final (S7RC record_commit.txt, B)", "field 1"),
    "B_B": ("git", "sha256 of the blob B_CC:B_CP"),
    "B_SC_P": ("--s7-dir sidecar.txt (B)", "field 1"), "B_SC_C": ("--s7-dir sidecar.txt (B)", "field 3"), "B_SC_S": ("--s7-dir sidecar.txt (B)", "field 2"),
}
for _r, (_p, _rx) in DERIVED.items():
    _m = f"--{_r} (SIMPROBE's commit_ok.txt)"
    RULES |= {f"{_p}_C": (_m, "field 1"), f"{_p}_P": (_m, "field 2"), f"{_p}_BASE": (_m, "field 2's file name"),
              f"{_p}_S": ("git", f"sha256 of the blob {_p}_C:{_p}_P (== field 3)"), f"{_p}_SP": (_m, "field 2 with .json -> _LABEL.json (== field 4)"),
              f"{_p}_SC": (_m, "field 1 (the sidecar sits beside the record)"), f"{_p}_SS": ("git", f"sha256 of the blob {_p}_C:{_p}_SP (== field 5)")}

TEMPLATE_BLOB = {  # each package's reviewed pins / plan template, by git blob (2e, 10-18); the tool STOPs on any other
    f"{X}/config/lead_pins_pooled_routec.json": "c4b043af59115e522f63085cf2c258e985417532",
    f"{E}/config/render_plan_figs156_routecA.json": "845bab3581baac87d378435a601cca39756e42dc",  # f62e24d8 + Figure 5's E4 citation (09, 10-18)
    f"{E}/config/render_plan_fig4_pooledB_routec.json": "5cdd7beef45d0f32178cd43bd79e82a830342950",
}
E4_CITES = (  # Figure 5's four-probe (E4) row: its AUROC membership source as the plan cites it (09 via 2e, 10-18; RULE l.4409 / l.4417-4419)
    (f"{E}/results/pooledmetrics_table_2026-10-16_001557_ap-1rltASqK10jRxmheLbO9Pj.json", "19ba8f5634bac60c54c1d426415fa0fdb4d65ae3",
     "efd0c2d2b74a5294c79f71451fbebb6c8b02f5e9b4b9c820b08c8b548d6f9250"),
    (f"{E}/results/pooledmetrics_2026-10-16_001557_ap-1rltASqK10jRxmheLbO9Pj.json", "19ba8f5634bac60c54c1d426415fa0fdb4d65ae3",
     "ff5d690d40e790af6994a54331d3e4086f52561fb4e21355e4c9a8b603eb378e"),
)


def check_e4_cites(plan_text):
    """Fill time (figs156): every E4 citation is in the plan as path@commit and sha256, and the blob at that commit hashes to it (hashed
    only, never parsed: the membership is not recomputed, RULE l.4418)."""
    for p, c, s in E4_CITES:
        if f"{p}@{c} sha256 {s}" not in plan_text:
            stop(f"the plan does not cite {p}@{c} sha256 {s} (Figure 5's E4 row)")
        if blob_sha(c, p) != s:
            stop(f"{p}@{c} does not hash to its cited sha256 (Figure 5's E4 row)")


NT = f"{X}/NOTE_routec_tables34_2026-10-17.md"
N156 = f"{E}/NOTE_routec_figs156_2026-10-18.md"
N4 = f"{E}/NOTE_routec_fig4_2026-10-17.md"
NPK = f"{E}/NOTE_routec_tablesfigs_packages_2026-10-18.md"
# GATES Amendment 15 (09 via 2e, 10-18): each package's COVER list = the records it draws plus their _LABEL.json sidecars; a .json.gz
# record by its decompressed JSON sha256. Each answer is one line here. NOTE (af820177 O-3): the gates' jq (TSEL / TGATE / F156GATE /
# F4GATE) HARD-CODES sidecars-in and the decompressed sha, so flipping either switch also needs the same change in those gates.
COVER_SIDECARS = True
COVER_GZ_SHA = "json"
A0 = ["HB", "A_LAUNCH", "A_REC_P", "A_REC_C", "A_REC_B", "A_REC_J", "A_SPEC", "A_SPEC_S", "SIDECAR", "C09", "MP", "MSHA", "COVER"]
T1 = ["SEL_P", "SEL_C", "SEL_S", "STAMP", "PFILE", "PFS"]
F156 = (["HB", "A_BASE", "A_J", "A_CP", "A_CC", "A_B", "A_LAUNCH", "A_SP", "A_SC", "A_SS"]
        + [f"{p}_{k}" for p in ("GC", "PW", "GX") for k in ("BASE", "P", "C", "S", "SP", "SC", "SS")] + ["C09", "MP", "MSHA", "COVER", "STAMP", "PFILE", "PFS"])
F4 = ["HB", "B_BASE", "B_J", "B_CP", "B_CC", "B_B", "B_LAUNCH", "B_SC_P", "B_SC_C", "B_SC_S", "C09", "MP", "MSHA", "COVER", "STAMP", "PFILE", "PFS"]
TOK_T = {"A_LAUNCH_COMMIT": ("A_LAUNCH", 2), "A_RECORD_PATH": ("A_REC_P", 1), "A_RECORD_COMMIT": ("A_REC_C", 1), "A_RECORD_BLOB_SHA256": ("A_REC_B", 1),
         "A_RECORD_JSON_SHA256": ("A_REC_J", 1), "A_DRIVER_SPEC_PATH": ("A_SPEC", 1), "A_DRIVER_SPEC_SHA256": ("A_SPEC_S", 1),
         "A_SELECTION_PATH": ("SEL_P", 1), "A_SELECTION_COMMIT": ("SEL_C", 1), "A_SELECTION_SHA256": ("SEL_S", 1)}
TOK_156 = {"A_LAUNCH_COMMIT": ("A_LAUNCH", 3), "A_RECORD_BASENAME": ("A_BASE", 5), "A_RECORD_COMMIT": ("A_CC", 3), "A_RECORD_COMMITTED_PATH": ("A_CP", 3),
           "A_RECORD_JSON_SHA256": ("A_J", 3), "A_SIDECAR_COMMIT": ("A_SC", 4), "A_SIDECAR_PATH": ("A_SP", 4), "A_SIDECAR_SHA256": ("A_SS", 3),
           "GC_BASENAME": ("GC_BASE", 3), "GC_COMMIT": ("GC_C", 3), "GC_PATH": ("GC_P", 3), "GC_SHA256": ("GC_S", 2), "GC_SIDECAR_COMMIT": ("GC_SC", 3),
           "GC_SIDECAR_PATH": ("GC_SP", 3), "GC_SIDECAR_SHA256": ("GC_SS", 2), "GX_BASENAME": ("GX_BASE", 2), "GX_COMMIT": ("GX_C", 2),
           "GX_PATH": ("GX_P", 2), "GX_SHA256": ("GX_S", 1), "GX_SIDECAR_COMMIT": ("GX_SC", 2), "GX_SIDECAR_PATH": ("GX_SP", 2),
           "GX_SIDECAR_SHA256": ("GX_SS", 1), "PW_BASENAME": ("PW_BASE", 3), "PW_COMMIT": ("PW_C", 3), "PW_PATH": ("PW_P", 3), "PW_SHA256": ("PW_S", 2),
           "PW_SIDECAR_COMMIT": ("PW_SC", 3), "PW_SIDECAR_PATH": ("PW_SP", 3), "PW_SIDECAR_SHA256": ("PW_SS", 2)}
TOK_4 = {"B_LAUNCH_COMMIT": ("B_LAUNCH", 1), "B_RECORD_BASENAME": ("B_BASE", 3), "B_RECORD_COMMIT": ("B_CC", 1), "B_RECORD_COMMITTED_PATH": ("B_CP", 1),
         "B_RECORD_JSON_SHA256": ("B_J", 1), "B_SIDECAR_COMMIT": ("B_SC_C", 2), "B_SIDECAR_PATH": ("B_SC_P", 2), "B_SIDECAR_SHA256": ("B_SC_S", 1)}
PKG = {  # NOTE, [(VARS block, its placeholders)], action blocks, (template, its tokens) or None
    "tables-sel": (NT, [("TVARS0", A0)], ["TSEL"], None),
    "tables": (NT, [("TVARS0", A0), ("TVARS1", T1)], ["TFILL", "TGATE", "TIMP", "TTAB", "TCOMMIT"], (f"{X}/config/lead_pins_pooled_routec.json", TOK_T)),
    "figs156": (N156, [("FVARS", F156)], ["F156FILL", "F156GATE", "F156R", "F156F", "F156C"], (f"{E}/config/render_plan_figs156_routecA.json", TOK_156)),
    "fig4": (N4, [("F4VARS", F4)], ["F4FILL", "F4GATE", "F4G", "F4R", "F4F", "F4C"], (f"{E}/config/render_plan_fig4_pooledB_routec.json", TOK_4)),
    "cover": (NPK, [("CVVARS", ["HB", "CVPKG", "CVP", "CVS", "STAMP"])], ["S2COVER"], None),  # built by --cover-out only
}


CTRL = re.compile(r"[\x00-\x1f\x7f]")  # af820177 R1 (the anchor tool's c294501e9 guard): no control character in any fill value


def stop(msg):
    sys.exit(f"STOP: {msg}")


def jload(b, what):
    """json.loads with a clean STOP (af820177 O2): the error type only, never the text."""
    try:
        return json.loads(b)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        stop(f"{what} is not JSON ({type(e).__name__})")


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def gz_sha(commit, path):
    """sha256 of a committed .json.gz blob's DECOMPRESSED bytes (gzip -dc; hashed, never parsed)."""
    import gzip
    import zlib

    try:
        return sha(gzip.decompress(git("cat-file", "blob", f"{commit}:{path}")))
    except (subprocess.CalledProcessError, OSError, EOFError, zlib.error) as e:
        stop(f"{commit}:{path} is not a readable gzip blob ({type(e).__name__})")


def blob_sha(commit, path):
    try:
        return sha(git("cat-file", "blob", f"{commit}:{path}"))
    except subprocess.CalledProcessError:
        stop(f"{commit}:{path} is not a blob")


def fields(path, n, what):
    """A one-line marker: exactly n single-space-separated fields, then a newline; the whole line re-checked."""
    raw = Path(path).read_bytes().decode("utf-8")
    f = raw.rstrip("\n").split(" ")
    if raw != " ".join(f) + "\n" or len(f) != n or not all(f) or "\n" in raw.rstrip("\n"):
        stop(f"{what} ({path}) is not one line of exactly {n} single-space-separated fields")
    return f


def blocks(text, names):
    """Each named block, '# BEGIN X' .. '# END X' inclusive, dedented by its BEGIN line's indent; exactly one of each."""
    lines, out = text.splitlines(), {}
    for x in names:
        b = [i for i, s in enumerate(lines) if s.strip() == f"# BEGIN {x}" or s.strip().startswith(f"# BEGIN {x} ")]
        e = [i for i, s in enumerate(lines) if s.strip() == f"# END {x}"]
        if len(b) != 1 or len(e) != 1 or e[0] <= b[0]:
            stop(f"block {x}: not exactly one BEGIN / END pair")
        ind = lines[b[0]][:len(lines[b[0]]) - len(lines[b[0]].lstrip())]
        if not all(s.startswith(ind) or not s.strip() for s in lines[b[0]:e[0] + 1]):
            stop(f"block {x}: a line is indented less than its BEGIN line")
        out[x] = "\n".join(s[len(ind):] for s in lines[b[0]:e[0] + 1]) + "\n"
    return out


def ancestors(head, commits, what):
    bad = [c for c in commits if subprocess.run(["git", "merge-base", "--is-ancestor", c, head], capture_output=True, check=False).returncode != 0]
    if bad:
        stop(f"{what}: {bad} not an ancestor of the head {head} (merge the producers' commits into this branch first)")


def s7(d, tag):
    """§7's markers for one TAG: ok.txt, launched.txt, the launch log's driver_call line (keys) and sidecar.txt."""
    d = Path(d)
    if not re.fullmatch(rf"S7_{tag}_{STAMP_RX}", d.name):
        stop(f"--s7-dir {d} is not S7_{tag}_<STAMP7>")
    ok = fields(d / "ok.txt", 5, f"S7R ok.txt ({tag})")
    la = fields(d / "launched.txt", 5, f"S7L launched.txt ({tag})")
    sc = fields(d / "sidecar.txt", 4, f"S7SC sidecar.txt ({tag})")
    dl = [ln for ln in (d / "launch.log").read_text().splitlines() if ln.startswith('{"driver_call": "fc-')]
    if len(dl) != 1:
        stop(f"S7L launch.log ({tag}): not exactly one driver_call line")
    j = jload(dl[0], f"S7L launch.log's driver_call line ({tag})")  # K: run_peraction311 launch_from's print (driver_call, spec, spec_sha256, ...)
    if not isinstance(j, dict) or la[0] != ok[4] or j.get("driver_call") != la[1] or j.get("spec_sha256") != la[3]:
        stop(f"{tag}: ok.txt / launched.txt / launch.log disagree (app id, driver call or the pickle's sha256)")
    m = re.fullmatch(r"xfer/fpr507prep/records/results/([^/\s@':]+[.]json)", ok[0])
    if not m:
        stop(f"{tag}: ok.txt's record {ok[0]!r} is not xfer/fpr507prep/records/results/<name>.json")
    return {"rp": ok[0], "base": m.group(1), "rs": ok[1], "launch": la[4], "spec": j.get("spec"), "spec_sha": j.get("spec_sha256"),
            "scp": sc[0], "scs": sc[1], "cf": sc[2]}


def cover_set(pkg, v):
    """GATES Amendment 15 (09 via 2e, 10-18): the package's consumer set = every record the code and plan draw (by key) plus, with
    COVER_SIDECARS, each one's _LABEL.json sidecar; a .json.gz record by its decompressed JSON sha256 (COVER_GZ_SHA "json"; "blob": the gz
    blob's), a plain .json by its blob sha256. -> sorted list; a duplicate STOPs."""
    def gz(p, j, b):
        return f"{p}@{j if COVER_GZ_SHA == 'json' else b}"

    used = {"tables": ["A_REC_P", "A_REC_J", "A_REC_B", "A_SP", "A_SS"], "tables-sel": ["A_REC_P", "A_REC_J", "A_REC_B", "A_SP", "A_SS"],
            "figs156": ["A_CP", "A_J", "A_B", "A_SP", "A_SS"] + [f"{p}_{k}" for p in ("GC", "GX", "PW") for k in ("P", "S", "SP", "SS")],
            "fig4": ["B_CP", "B_J", "B_B", "B_SC_P", "B_SC_S"]}.get(pkg, [])
    bad = [n for n in used if not (isinstance(v.get(n), str) and re.fullmatch(RX[n], v[n]) and not CTRL.search(v[n]))]
    if bad:  # af820177 R-1: every COVER input, records AND sidecars, against its regex before anything is written (--cover-out included)
        stop(f"{pkg}: COVER inputs fail their regex: {bad}")
    if pkg in ("tables", "tables-sel"):
        recs, sides = [gz(v["A_REC_P"], v["A_REC_J"], v["A_REC_B"])], [f"{v['A_SP']}@{v['A_SS']}"]
    elif pkg == "figs156":
        recs = [gz(v["A_CP"], v["A_J"], v["A_B"])] + [f"{v[p + '_P']}@{v[p + '_S']}" for p in ("GC", "GX", "PW")]
        sides = [f"{v['A_SP']}@{v['A_SS']}"] + [f"{v[p + '_SP']}@{v[p + '_SS']}" for p in ("GC", "GX", "PW")]
    elif pkg == "fig4":
        recs, sides = [gz(v["B_CP"], v["B_J"], v["B_B"])], [f"{v['B_SC_P']}@{v['B_SC_S']}"]
    else:
        stop(f"{pkg}: no COVER set")
    lst = recs + (sides if COVER_SIDECARS else [])
    if len(set(lst)) != len(lst):
        stop(f"{pkg}: the COVER set has a duplicate entry")
    return sorted(lst)


def cover_list(b, what):
    """A COVER list's bytes -> the list: JSON, path@sha256 strings, sorted, no duplicate (else STOP)."""
    cov = jload(b, what)
    if not (isinstance(cov, list) and cov and all(isinstance(x, str) and re.fullmatch(r"[^\"'\s]+@[0-9a-f]{64}", x) for x in cov)
            and cov == sorted(cov) and len(set(cov)) == len(cov)):
        stop(f"{what} is not a sorted list of unique path@sha256 strings")
    return cov


def committed_cover(a, pkg, v, head):
    """--cover-ok: the S2COVER block's cover_ok.txt '<commit> <path> <sha256>'. The COVER list is read from that committed file at that
    commit (its bytes hash to field 3), the commit is an ancestor of the head, and the list must EQUAL this package's derived set."""
    c, p, s = fields(a.cover_ok, 3, "--cover-ok (S2COVER's cover_ok.txt)")
    if not (re.fullmatch(H40, c) and re.fullmatch(RX["CVP"], p) and re.fullmatch(H64, s)) or f"routec_{'tables' if pkg == 'tables-sel' else pkg}_cover_" not in p:
        stop(f"--cover-ok is not <commit> <this package's COVER file> <sha256> ({pkg})")
    ancestors(head, [c], "the COVER commit")
    b = git("cat-file", "blob", f"{c}:{p}")
    if sha(b) != s:
        stop("the committed COVER file does not hash to cover_ok.txt's sha256")
    cov = cover_list(b, "the committed COVER file")
    if set(cov) != set(cover_set(pkg, v)):
        stop(f"the committed COVER list is not exactly this package's derived set ({pkg}; GATES Amendment 15)")
    return cov


def s2(a, cov):
    """09's S2 marker for this consumer set (GATES Amendment 15): signoff S2, signed, by 09, and covered EQUAL to the COVER list as a set,
    with no duplicate on either side."""
    c09, mp = a.s2.split(":", 1) if ":" in a.s2 else ("", "")
    if not (re.fullmatch(H40, c09) and re.fullmatch(RX["MP"], mp)):
        stop("--s2 is not <40-hex commit>:experiments/2026-10-14_priority1_spec_09/signoff/S2_<YYYY-MM-DD_HHMMSS>.json")
    raw = git("cat-file", "blob", f"{c09}:{mp}")
    mk = jload(raw, "09's S2 marker")  # K: 09's signoff JSON (signoff, signed, by, time, covered, notes, for, items)
    cv = mk.get("covered") if isinstance(mk, dict) else None
    if not (isinstance(mk, dict) and mk.get("signoff") == "S2" and mk.get("signed") is True and mk.get("by") == "09" and isinstance(cv, list)
            and all(isinstance(x, str) for x in cv) and len(set(cv)) == len(cv) and set(cv) == set(cov)):
        stop("09's S2 marker: not signoff S2 / signed / by 09, or its covered set is not exactly the committed COVER list (or has a duplicate)")
    return {"C09": c09, "MP": mp, "MSHA": sha(raw), "COVER": json.dumps(cov, separators=(",", ":"))}


def record_commit(path, s7dir, tag, s):
    """§7's S7RC marker $L/S7_<TAG>_<STAMP7>/record_commit.txt (aswin/routec-launch @ 1d8867052): '<commit> <path.json.gz> <decompressed
    sha256>', read as ARC:ARP@ARS. It must sit in this TAG's §7 dir; the commit must change exactly that path; the blob, decompressed,
    must hash to field 3 (hashed, never parsed), which must equal S7R's ok.txt sha256."""
    import gzip
    import zlib

    if Path(path).resolve() != (Path(s7dir) / "record_commit.txt").resolve():
        stop(f"--{tag.lower()}-final must be {s7dir}/record_commit.txt (S7RC's marker for TAG {tag})")
    c, p, h = fields(path, 3, f"S7RC record_commit.txt ({tag})")
    if not (re.fullmatch(H40, c) and re.fullmatch(GZ, p) and re.fullmatch(H64, h)) or h != s["rs"]:
        stop(f"record_commit.txt ({tag}) is not <commit> <path.json.gz> <sha256> naming §7's record sha")
    try:
        names = git("diff-tree", "--no-commit-id", "--name-only", "-r", c).decode().split("\n")
        dec = sha(gzip.decompress(git("cat-file", "blob", f"{c}:{p}")))
    except (subprocess.CalledProcessError, OSError, EOFError, zlib.error) as e:  # af820177 O-c: zlib.error too
        stop(f"record_commit.txt ({tag}): {c}:{p} is not a readable gzip blob ({type(e).__name__})")
    if [n for n in names if n] != [p]:
        stop(f"record_commit.txt ({tag}): the commit does not change exactly {p}")
    if dec != h:
        stop(f"record_commit.txt ({tag}): the blob does not decompress to its sha256")
    return f"{c}:{p}@{h}"


def a_values(a, head):
    s = s7(a.s7_dir, "A")
    side = f"git:{s['cf']}:{s['scp']}@{s['scs']}"
    rc = record_commit(a.a_final, a.s7_dir, "A", s) if a.a_final else None
    if a.fok:  # CUT's FILL1B marker: field 4 is A's committed copy, field 5 the sidecar pointer
        fk = fields(a.fok, 7, "--fok (FILL1B fill_ok.txt)")
        if fk[4] != side or (rc is not None and fk[3] != rc):
            stop("fill_ok.txt field 5 differs from sidecar.txt, or field 4 from S7RC's record_commit.txt")
        fin = fk[3]
    elif rc is not None:  # before FILL1B: S7RC's record_commit.txt for TAG A
        fin = rc
    else:
        stop("A's committed copy: give --fok (FILL1B's fill_ok.txt) or --a-final (S7RC's record_commit.txt for TAG A)")
    m = re.fullmatch(rf"({H40}):({GZ})@({H64})", fin)
    if not m or m.group(3) != s["rs"]:
        stop("A's committed copy (fill_ok.txt field 4 or --a-final) differs from §7's ok.txt (sha)")
    if Path(m.group(2)).name not in (s["base"] + ".gz", f"routec_A_{s['base']}.gz"):  # A only: routec_A_ or the plain name (af820177 O-b)
        stop("A's committed copy is not routec_A_<A's Volume name>.gz or <A's Volume name>.gz (another TAG's record, or another record)")
    ancestors(head, [s["launch"], m.group(1), s["cf"], FROZEN_C], "A's launch, record and sidecar commits, FROZEN")
    v = {"A_LAUNCH": s["launch"], "A_REC_C": m.group(1), "A_REC_P": m.group(2), "A_REC_J": s["rs"], "A_REC_B": blob_sha(m.group(1), m.group(2)),
         "A_SPEC": s["spec"], "A_SPEC_S": s["spec_sha"], "SIDECAR": side, "A_BASE": s["base"], "A_J": s["rs"], "A_CP": m.group(2),
         "A_CC": m.group(1), "A_SP": s["scp"], "A_SC": s["cf"], "A_SS": s["scs"]}
    v["A_B"] = v["A_REC_B"]
    if blob_sha(s["cf"], s["scp"]) != s["scs"]:
        stop("A's sidecar does not hash to sidecar.txt's sha256 at its commit")
    if gz_sha(m.group(1), m.group(2)) != s["rs"]:  # 09: a .json.gz entry is verified by its decompressed bytes (gzip -dc), never the gz blob
        stop("A's committed .json.gz does not decompress to §7's record sha256")
    return v


def b_values(a, head):
    s = s7(a.s7_dir, "B")
    if not a.b_final:
        stop("--b-final (S7RC's record_commit.txt for TAG B) is required")
    m = re.fullmatch(rf"({H40}):({GZ})@({H64})", record_commit(a.b_final, a.s7_dir, "B", s))
    if not m or m.group(3) != s["rs"] or Path(m.group(2)).name not in (s["base"] + ".gz", f"routec_B_{s['base']}.gz"):
        stop("--b-final does not name §7's B record (sha, file name)")
    ancestors(head, [s["launch"], m.group(1), s["cf"]], "B's launch, record and sidecar commits")
    if blob_sha(s["cf"], s["scp"]) != s["scs"]:
        stop("B's sidecar does not hash to sidecar.txt's sha256 at its commit")
    v = {"B_BASE": s["base"], "B_J": s["rs"], "B_CP": m.group(2), "B_CC": m.group(1), "B_B": blob_sha(m.group(1), m.group(2)),
         "B_LAUNCH": s["launch"], "B_SC_P": s["scp"], "B_SC_C": s["cf"], "B_SC_S": s["scs"]}
    return v  # record_commit() verified the decompressed sha256 (gzip -dc), as 09 requires for .json.gz entries


def derived(a, head, a_base):
    """--gc / --gx / --pw: SIMPROBE's commit_ok.txt markers (GXCOMMIT / GCCOMMIT / the pw commit step, NOTE_routec_group_steps.md L230 /
    L259 and NOTE_routec_pw_shares.md @ 74edde46f): '<commit> <record path> <record sha256> <sidecar path> <sidecar sha256>'. The commit
    and path come from fields 1 and 2; field 4 must be the derived <stem>_LABEL.json path, field 3 the record's blob sha256 at that commit
    and field 5 the sidecar's (both hashed here, never parsed); each mismatch STOPs with its own message."""
    v = {}
    for role, (p, rx) in DERIVED.items():
        if not getattr(a, role):
            stop(f"--{role} (SIMPROBE's commit_ok.txt) is required")
        c, path, rs, sp_m, ss = fields(getattr(a, role), 5, f"--{role} (SIMPROBE's commit_ok.txt)")
        if not (re.fullmatch(H40, c) and re.fullmatch(rx, path) and re.fullmatch(H64, rs) and re.fullmatch(H64, ss)):
            stop(f"--{role}: commit_ok.txt is not <40-hex commit> <{rx}> <sha256> <sidecar path> <sha256>")
        if role == "pw" and f"_{a_base.removesuffix('.json')}_routeC" not in path:
            stop("--pw: not the pw-shares record of this A")
        sp = path.removesuffix(".json") + "_LABEL.json"
        if sp_m != sp:
            stop(f"--{role}: commit_ok.txt's sidecar path (field 4) is not the record's <stem>_LABEL.json")
        ancestors(head, [c], f"--{role}")
        s_, ss_ = blob_sha(c, path), blob_sha(c, sp)
        if s_ != rs:
            stop(f"--{role}: the record's blob at the commit does not hash to commit_ok.txt's record sha256 (field 3)")
        if ss_ != ss:
            stop(f"--{role}: the sidecar's blob at the commit does not hash to commit_ok.txt's sidecar sha256 (field 5)")
        v |= {f"{p}_C": c, f"{p}_P": path, f"{p}_BASE": Path(path).name, f"{p}_S": s_, f"{p}_SP": sp, f"{p}_SC": c, f"{p}_SS": ss_}
    return v


def template_text(path):
    """The package's pins / plan template at HEAD, only if its blob is the reviewed one (TEMPLATE_BLOB; 2e, 10-18)."""
    tb_ = git("rev-parse", f"HEAD:{path}").decode().strip()
    if tb_ != TEMPLATE_BLOB.get(path):
        stop(f"{path} at HEAD is blob {tb_}, not the pinned {TEMPLATE_BLOB.get(path)}")
    return git("show", f"HEAD:{path}").decode("utf-8")


def fill_template(text, toks, v):
    """Each <FILL:T> occurs exactly its pinned count and becomes v[var]; no <FILL left; the diff holds only the token lines."""
    found = sorted(set(re.findall(r"<FILL:([A-Z0-9_]+)>", text)))
    if found != sorted(toks):
        stop(f"the template's tokens {found} differ from the package's {sorted(toks)}")
    out = text
    for t, (var, n) in toks.items():
        if out.count(f"<FILL:{t}>") != n:
            stop(f"<FILL:{t}>: {out.count(f'<FILL:{t}>')} occurrences, {n} expected")
        out = out.replace(f"<FILL:{t}>", v[var])
    if "<FILL" in out or "TO_FILL" in out:
        stop("a fill token is left in the filled template")
    if any(not isinstance(v.get(var), str) or CTRL.search(v[var]) or re.search("[\"'\\\\]", v[var]) for var, _ in toks.values()):
        stop("a template value holds a control character, a quote or a backslash")
    old, new = text.splitlines(), out.splitlines()
    if len(old) != len(new) or any(o != w and "<FILL:" not in o for o, w in zip(old, new)) or any(o == w and "<FILL:" in o for o, w in zip(old, new)):
        stop("the filled template differs from the committed one beyond the token lines")
    jload(out, "the filled template")
    return out


def fill(vtext, names, v):
    """NAME='<…>' -> NAME='value': exactly one placeholder line each; no placeholder left; the diff holds only those lines."""
    lines, old_new = vtext.splitlines(), {}
    have = sorted(m.group(1) for s in lines if (m := re.fullmatch(r"([A-Z_0-9]+)='<[^']*>'", s)))
    if have != sorted(names):
        stop(f"the VARS block's placeholders {have} differ from the package's {sorted(names)}")
    for n in names:
        if not (isinstance(v.get(n), str) and re.fullmatch(RX[n], v[n]) and "'" not in v[n] and not CTRL.search(v[n])):
            stop(f"{n}: the value fails its regex {RX[n]!r}")
        hit = [i for i, s in enumerate(lines) if re.fullmatch(rf"{n}='<[^']*>'", s)]
        if len(hit) != 1:
            stop(f"{n}: {len(hit)} placeholder lines in the VARS block (exactly one expected)")
        old_new[lines[hit[0]]] = f"{n}='{v[n]}'"
        lines[hit[0]] = f"{n}='{v[n]}'"
    if any(re.search(r"^[A-Z_0-9]+='<", s) for s in lines):
        stop("a placeholder is left in the VARS block")
    diff = [d for d in difflib.unified_diff(vtext.splitlines(), lines, lineterm="", n=0) if not d.startswith(("---", "+++", "@@"))]
    if sorted(diff) != sorted([f"-{o}" for o in old_new] + [f"+{w}" for w in old_new.values()]):
        stop("the filled VARS block differs from the committed one beyond the placeholder lines")
    return "\n".join(lines) + "\n", diff


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pkg", required=True, choices=sorted(PKG))
    ap.add_argument("--note-commit", required=True)
    ap.add_argument("--list-blocks", action="store_true", help="print each block's NOTE line, line count and sha256 at --note-commit; nothing else")
    ap.add_argument("--head", default="")
    for f_ in ("--s7-dir", "--fok", "--a-final", "--b-final", "--s2", "--cover-ok", "--gc", "--gx", "--pw", "--tsel-ok"):
        ap.add_argument(f_, default="")
    ap.add_argument("--cover-out", action="store_true", help="build this package's COVER list (GATES Amendment 15) and the S2COVER run file")
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()
    if a.cover_out and a.pkg not in ("tables", "figs156", "fig4"):
        stop("--cover-out takes --pkg tables, figs156 or fig4")
    if a.list_blocks:
        note, vbs, acts, _ = PKG[a.pkg]
        text = git("show", f"{a.note_commit}:{note}").decode("utf-8")
        for x, t in blocks(text, [b for b, _ in vbs] + acts).items():
            ln = next(i + 1 for i, s in enumerate(text.splitlines()) if s.strip() == f"# BEGIN {x}" or s.strip().startswith(f"# BEGIN {x} "))
            print(f"BLOCK {a.pkg} {x} line {ln} lines {t.count(chr(10))} sha256 {sha(t.encode())}")
        return
    if not (a.head and a.out_dir):
        stop("--head and --out-dir are required")
    for c in (a.note_commit, a.head):
        if not re.fullmatch(H40, c) or git("rev-parse", f"{c}^{{commit}}").decode().strip() != c:
            stop(f"{c}: not a full 40-hex commit")
    hd = git("rev-parse", "HEAD").decode().strip()
    note, vbs, acts, tmpl = PKG["cover"] if a.cover_out else PKG[a.pkg]
    v = {"HB": a.head}
    if a.pkg == "tables" and not a.cover_out:
        t = fields(a.tsel_ok, 3, "--tsel-ok (TSEL ok.txt)")
        v |= {"SEL_P": t[0], "SEL_C": t[1], "SEL_S": t[2]}
        if not (re.fullmatch(H40, t[1]) and hd == t[1] and git("rev-parse", f"{t[1]}^").decode().strip() == a.head
                and git("rev-list", "--count", f"{a.head}..{t[1]}").decode().strip() == "1" and blob_sha(t[1], t[0]) == t[2]):
            stop("tables: HEAD must be TSEL's commit, exactly one commit beyond the head (its parent), and the selection must hash to "
                 "TSEL's sha256")
    elif hd != a.head:
        stop("HEAD is not --head")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        stop("uncommitted tracked changes")
    v |= b_values(a, a.head) if a.pkg == "fig4" else a_values(a, a.head)
    if a.pkg == "figs156":
        v |= derived(a, a.head, v["A_BASE"])
    v["STAMP"] = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = Path(a.out_dir)
    if out.exists():
        stop(f"{out} exists (the run files are written exclusive-create into a new folder)")
    if a.cover_out:  # GATES Amendment 15: write this package's COVER list (exclusive) for S2COVER to commit and 2e to send to 09
        cov = cover_set(a.pkg, v)
        v |= {"CVPKG": a.pkg, "CVP": f"{E}/config/s2_cover/routec_{a.pkg}_cover_{v['STAMP']}.json"}
        cb = (json.dumps(cov, indent=1) + "\n").encode()
        cf = Path(git("rev-parse", "--show-toplevel").decode().strip()) / v["CVP"]
        cf.parent.mkdir(parents=True, exist_ok=True)
        with cf.open("xb") as fh:
            fh.write(cb)
        v["CVS"] = sha(cb)
        print(f"COVER {a.pkg} {v['CVP']} sha256 {v['CVS']} entries {len(cov)}")
    else:
        v |= s2(a, committed_cover(a, a.pkg, v, a.head))
    names = [n for _, ns in vbs for n in ns]
    bl = blocks(git("show", f"{a.note_commit}:{note}").decode("utf-8"), [b for b, _ in vbs] + acts)
    filled_t = None
    if tmpl:
        tt_ = template_text(tmpl[0])
        if a.pkg == "figs156":
            check_e4_cites(tt_)
        filled_t = fill_template(tt_, tmpl[1], v)
        v |= {"PFILE": str((out / Path(tmpl[0]).name).resolve()), "PFS": sha(filled_t.encode())}
    fv, diffs = "", []
    for b, ns in vbs:
        t_, d_ = fill(bl[b], ns, v)
        fv, diffs = fv + t_, diffs + d_
    out.mkdir(parents=True)
    if filled_t is not None:
        with Path(v["PFILE"]).open("x") as fh:
            fh.write(filled_t)
    runs = {}
    for x in acts:
        body = (f"# {a.pkg} {x}: built by tools_routec/fill_run_tablesfigs.py from {note} @ {a.note_commit}; the filled "
                f"{' + '.join(b for b, _ in vbs)}, then {x} verbatim (block sha256 {sha(bl[x].encode())})\n" + fv + bl[x])
        p = out / f"{a.pkg}_{x}.zsh"
        with p.open("x") as fh:
            fh.write(body)
        runs[x] = {"file": str(p), "sha256": sha(body.encode()), "lines": body.count("\n")}
    man = {"pkg": a.pkg, "note": f"{note}@{a.note_commit}", "head": a.head, "HEAD": hd,
           "blocks": {x: {"lines": t.count("\n"), "sha256": sha(t.encode())} for x, t in bl.items()},
           "fills": {n: {"value": v[n], "source": RULES[n][0], "rule": RULES[n][1], "regex": RX[n]} for n in names},
           "template": {"path": tmpl[0], "filled": v["PFILE"], "sha256": v["PFS"]} if tmpl else None, "diff": diffs, "runs": runs}
    with (out / f"{a.pkg}_fill.json").open("x") as fh:
        fh.write(json.dumps(man, indent=1) + "\n")
    print("\n".join(diffs))
    for x, r in runs.items():
        print(f"RUNFILE {a.pkg} {x} {r['file']} sha256 {r['sha256']} lines {r['lines']}")
    print(f"ROUTEC_FILL_OK {a.pkg} {out / (a.pkg + '_fill.json')}")


if __name__ == "__main__":
    main()
