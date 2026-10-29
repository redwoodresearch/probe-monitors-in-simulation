"""0d's app-name rule (10-17): Modal cuts an ephemeral app's description at 64 characters, so an exact-description check (app list,
watch_set) on a longer name matches nothing. Every `probe-shift_aswin-surya_sub-…` name in this folder's launchers (*.py, *.sh, results/
excluded) must be at most 63 characters, except the historical apps that already ran, exempt by exact (file, name) pair: the set of long
(file, name) pairs must EQUAL the exemptions (both reviewers' optional on 07680b616, 10-17), so a historical name reused in another
launcher fails, and so does a stale exemption. Code only; no Modal.

  uv run --no-project --python 3.12 python -I test_app_names.py
"""

import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX = 63
NAME = re.compile(r"probe-shift_aswin-surya_sub-[A-Za-z0-9_.{}-]+")
HISTORICAL = {  # already ran before the rule; exempt only in the launcher that ran it (2e's audit, 10-17)
    ("plant_gate_alltasks.py", "probe-shift_aswin-surya_sub-fpr507prep_plantgateat_kill-2026-11-14"),
    ("npz_keymap_check.py", "probe-shift_aswin-surya_sub-fpr507prep_npzkeymap_kill-2026-11-14"),
    ("compare_subset.py", "probe-shift_aswin-surya_sub-fpr507prep_cmpsubset_kill-2026-11-14"),
    ("run_attack_sample_errors.py", "probe-shift_aswin-surya_sub-fpr507prep_attackerr_kill-2026-11-14"),
}


def pairs(root):
    out = set()
    for f in sorted(list(root.rglob("*.py")) + list(root.rglob("*.sh"))):
        rel = f.relative_to(root)
        if rel.parts[0] == "results" or f.name == Path(__file__).name:
            continue
        out |= {(str(rel), n) for n in NAME.findall(f.read_text(errors="replace"))}
    return out


def long_pairs(ps):
    return {p for p in ps if len(p[1]) > MAX}


def mutant(extra):  # the real historical launchers in a temp tree, plus `extra` {file: text} -> the long pairs there
    d = Path(tempfile.mkdtemp(prefix="appnames_mut_"))
    for f, n in HISTORICAL:
        (d / f).write_text(f'APP = "{n}"\n')
    for f, t in extra.items():
        (d / f).write_text(t)
    return long_pairs(pairs(d))


ROUTEC_RUNS = ("rcPS", "rcCAT", "rcA", "rcB", "rcR37")  # 0d's ruling (i), 10-17: each route C runner launch sets its own description by `modal run -n`
ROUTEC_NAMES = [f"probe-shift_aswin-surya_sub-fpr507prep_{r}_kill-2026-11-14" for r in ROUTEC_RUNS]
ROUTEC_LAUNCH_NOTES = ("NOTE_routec_cat_auth_ps_2026-10-17.md",)  # the route C packages that launch the runner (the §7 package joins when written)


def note_launch_problems(texts):  # {note: text} -> launch lines of the runner without `-n "$APPD"`, and over-long or template names
    bad = []
    for nt, t in texts.items():
        for i, ln in enumerate(t.splitlines(), 1):
            if "run --detach" in ln and "run_peraction311.py" in ln and ' -n "$APPD" ' not in ln:
                bad.append(f"{nt}:{i}: a runner launch without -n")
        bad += [f"{nt}: {n}" for n in NAME.findall(t) if len(n) > MAX or "{" in n]
    return bad


found = pairs(HERE)
long_ = long_pairs(found)
notes_ = {n: (HERE / n).read_text() for n in ROUTEC_LAUNCH_NOTES}
templ = sorted(p for p in found if "{" in p[1])
h1 = sorted(HISTORICAL)[0][1]
cases = {
    "the long (file, name) pairs EQUAL the historical exemptions (every other name <= 63 characters; no stale exemption)": long_ == HISTORICAL,
    "no name is built from a template (an f-string part would hide its length)": not templ,
    "the preflight app is the renamed one (59 characters)": ("preflight_routec.py", "probe-shift_aswin-surya_sub-fpr507prep_rcpf_kill-2026-11-14") in found
    and not any(n == "probe-shift_aswin-surya_sub-fpr507prep_routecpreflight_kill-2026-11-14" for _, n in found),
    "positive control: the old 70-character preflight name would fail": len("probe-shift_aswin-surya_sub-fpr507prep_routecpreflight_kill-2026-11-14") > MAX,
    "mutant control: the historical launchers alone pass": mutant({}) == HISTORICAL,
    "mutant: a historical name reused in a NEW launcher fails": mutant({"run_new_launcher.py": f'APP = "{h1}"\n'}) != HISTORICAL,
    "mutant: a historical launcher renamed (stale exemption) fails": mutant({}) - {sorted(HISTORICAL)[0]} != HISTORICAL,
    "mutant: a new 64-character name fails": mutant({"run_x.py": 'APP = "' + "probe-shift_aswin-surya_sub-" + "x" * 36 + '"\n'}) != HISTORICAL,
    "route C run descriptions (rcPS, rcCAT, rcA, rcB, rcR37): each <= 63 characters, distinct, not the shared C.APP": all(len(n) <= MAX for n in ROUTEC_NAMES)
    and len(set(ROUTEC_NAMES)) == len(ROUTEC_NAMES) and "probe-shift_aswin-surya_sub-fpr507prep_kill-2026-11-14" not in ROUTEC_NAMES,
    "route C launch NOTEs: every runner launch sets -n, every named app is <= 63 characters and not a template": not note_launch_problems(notes_),
    "route C launch NOTEs name the CAT and PS descriptions": all(any(n in t for t in notes_.values()) for n in ROUTEC_NAMES[:2]),
    "mutant: a runner launch line without -n fails": bool(note_launch_problems({"m.md": '  "$M" run --detach $E/run_peraction311.py --launch-from-pickle x\n'})),
    "mutant: a 64-character app name in a launch NOTE fails": bool(note_launch_problems({"m.md": "APPD=probe-shift_aswin-surya_sub-" + "x" * 36 + "\n"})),
}
for name, ok in cases.items():
    print(("PASS " if ok else "FAIL ") + name)
print(f"long pairs not exempt: {sorted(long_ - HISTORICAL)}; exemptions not found: {sorted(HISTORICAL - long_)}; templates: {templ}")
print(f"{sum(cases.values())} of {len(cases)} checks pass ({len(found)} (file, name) pairs)")
sys.exit(0 if all(cases.values()) else 1)
