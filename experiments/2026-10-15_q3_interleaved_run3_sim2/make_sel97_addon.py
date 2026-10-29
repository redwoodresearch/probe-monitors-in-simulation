"""The il_old selection-half add-on (Aswin ~11:36 via 0d; quote @ 46f6afeed): ONE interleaved run3 + sim2 launch of the
97 selection-half cells (config/il_keys_selection.json, 09's list) on xbox1, as a DESCRIPTIVE ADD-ON: not a registered IL
launch or replacement; it cannot amend the registered IL result. 96 cells run; t151_model_registry has no pinned reference
(results/il_no_reference_cells_2026-10-15.json), so it is an explicit status row, not a task.

Writes (never overwriting) config/addon/{tasks_il_sel97.json, pilot_il_sel97.yaml, sel97_status_rows.json} and il_sel97_box.sh:
- task file: the cells' ORIGINAL entries from config/tasks_il_old_all.json, verbatim, in il_order (their position there);
- config: config/completion/pilot_ilc_old.yaml with only the tasks line and the two cap lines changed
  (cap = max(96 x $20.5, 96 x $13.62 + 10 x $75) = the approved quote's admission rule);
- box script: ilc_box.sh's checks and launch sequence (freeze, settings equality, guards, confirmed start), with this
  add-on's names, pins and cell check (both original arms 'not run' with 0 attempts in il_old).
  python make_sel97_addon.py
v2 (d9 ~12:00-12:10: sel97 starts ~14:50 on xbox1 after sim-honest Q2 exits; xbox1 only, no relay; xbox1 powers off 19:30
PT): `--box-only` regenerates ONLY il_sel97_box.sh (the committed task file and config are pinned by their sha256,
unchanged): sim-honest Q2's run dir joins the must-finish list; every launch.json under runs/ and nm/runs/ on the box must
report finished; and the freeze excludes the two sim-honest configs staged in the harness config folder on xbox1
(pilot_simh_q4.yaml, pilot_simh_q2.yaml) by EXACT path, each absent or at its pinned sha256 (the sim-honest branch
aswin/q3-simhonest-311 @ 279c2b25c, = simh311_box.sh's CSHA; q4's also read on xbox1 at 12:1x).
  python make_sel97_addon.py --box-only
v3 (d9 ~13:10: the chain's donor gate @ 04bdf57ba fails t96_file_indexing on (b): its pre-existing run2 attack prompt is
empty; 09 decides whether t96 runs): `--exclude KEY=REASON` also leaves KEY out of the task file as two status rows, and
`--out-root DIR` writes the whole package (task file, config, status rows, v2 box script) under DIR as a prepared variant.
  python make_sel97_addon.py --exclude "t96_file_indexing=not run: donor gate (b) failed: run2 attack prompt empty (chain sel97_donor_gate_2026-10-15.json @ 04bdf57ba); 09's call" --out-root variants/sel95
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).parent
PRICE, RATE, RESERVE, K = 13.62, 20.5, 75.0, 10
RUN = "/home/aswin/phase2/runs/2026-10-15_il_sel97_addon"
ORIG = "/home/aswin/phase2/runs/2026-10-14_il_old"
MAIN = "/home/aswin/phase2/il_refindex_old"
RSHA = "79acbfc1ef4523f9a57cc19dc876252159c6c1d62521001f1ac71e2644036940"  # il_refindex_old launch.json (ilc_box.sh old case)
OSHA = "cf53b22603ae738fa7666aac3f1e708d913cee957112c23eb62c13aa6a30d4a7"  # il_old launch.json (ilc_box.sh old case)
PREVS = [ORIG, "/home/aswin/phase2/runs/2026-10-15_il_old_rlr", "/home/aswin/phase2/runs/2026-10-15_il_old_rlr3",
         "/home/aswin/phase2/runs/2026-10-15_simh_q4"]  # xbox1's own launches tonight; sim-honest Q4 must have ended
PREVS_V2 = PREVS + ["/home/aswin/phase2/runs/2026-10-15_simh_q2"]  # v2: sim-honest Q2 runs on xbox1 before sel97
SIMH_PINS = {"pilot_simh_q4.yaml": "942f88b0c1c34d9ae8d83b9658727e1ec93f5c0674b0b32105b37b306cdade87",
             "pilot_simh_q2.yaml": "dace5410a1c408159eebcdf2b126ff3c22e7b18415c26c3f2de3f17795d9513e"}
FREEZE_OLD = "! -path './config/ilrr3_*' -print0 \\\n"
FREEZE_NEW = ("! -path './config/ilrr3_*' \\\n  " + " ".join(f"! -path './config/{f}'" for f in SIMH_PINS) + " -print0 \\\n")
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-only", action="store_true", help="v2: regenerate only il_sel97_box.sh from the committed task file and config")
    ap.add_argument("--exclude", action="append", default=[], help="v3: KEY=REASON, also left out as status rows")
    ap.add_argument("--out-root", default=None, help="v3: write the package under this folder (a prepared variant)")
    a = ap.parse_args()
    box_only = a.box_only
    excl = dict(x.split("=", 1) for x in a.exclude)
    root = HERE / a.out_root if a.out_root else HERE
    assert not (box_only and (excl or a.out_root)), "--box-only regenerates the committed package only"
    out = root / "config" / "addon"
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((HERE / "config/il_keys_selection.json").read_text())
    noref = json.loads((HERE / "results/il_no_reference_cells_2026-10-15.json").read_text())
    nr = {r["key"] for lj in noref["launches"] for r in lj["rows"]}
    allt = json.loads((HERE / "config/tasks_il_old_all.json").read_text())
    order = {c["key"]: i for i, c in enumerate(allt["cells"])}
    assert len(sel) == 97 and all(k in order for k in sel), "selection keys must be il_old cells"
    blocked = sorted(set(sel) & nr)
    assert blocked == ["t151_model_registry"], blocked
    assert all(k in sel and k not in nr for k in excl), "excluded keys must be runnable selection keys"
    run = sorted((k for k in sel if k not in nr and k not in excl), key=order.get)
    cells = [allt["cells"][order[k]] for k in run]
    n = len(cells)
    cap = math.ceil(max(n * RATE, n * PRICE + K * RESERVE))
    tf = out / "tasks_il_sel97.json"
    cf = out / "pilot_il_sel97.yaml"
    sf = out / "sel97_status_rows.json"
    bf = root / "il_sel97_box.sh"
    for p in (tf, cf, sf, bf):
        assert p.exists() if box_only else not p.exists(), f"{p} {'missing' if box_only else 'exists'}"
    if box_only:
        assert [c["key"] for c in json.loads(tf.read_text())["cells"]] == [c["key"] for c in cells], "committed task file != selection"
        assert f"cap_usd: {cap} " in cf.read_text(), "committed config cap != rule"
        write_box(tf, cf, bf, cap, n, v2=True)
        return
    tf.write_text(json.dumps({"launch": "il_sel97_addon", "addon_of": "2026-10-14_il_old", "status": "descriptive add-on; not a registered IL launch or replacement; cannot amend the registered IL result",
                              "rule": "Aswin ~11:36 via 0d (relayed by d9); quote @ 46f6afeed; 09 registration pending", "seed": allt.get("seed"),
                              "order": "il_order (position in config/tasks_il_old_all.json)", "cells": cells}, indent=1) + "\n")
    cfg = (HERE / "config/completion/pilot_ilc_old.yaml").read_text()
    head = "# IL SELECTION-HALF ADD-ON (descriptive; not a registered IL launch): from pilot_ilc_old.yaml with only tasks and caps changed.\n"
    cfg2, k1 = re.subn(r"(?m)^tasks: config/tasks_ilc_old\.json.*$", f"tasks: config/tasks_il_sel97.json   # add-on: {n} selection-half cells, il_order", cfg)
    cfg2, k2 = re.subn(r"(?m)^  cap_usd: 1656 .*$", f"  cap_usd: {cap}   # max({n} cells x $20.5, {n} x ${PRICE} + 10 x $75)", cfg2)
    cfg2, k3 = re.subn(r"(?m)^  worst_case_limit_usd: 1656 .*$", f"  worst_case_limit_usd: {cap}   # = cap", cfg2)
    assert (k1, k2, k3) == (1, 1, 1), (k1, k2, k3)
    cf.write_text(head + cfg2)
    sf.write_text(json.dumps({"launch": "il_sel97_addon", "rows": [
        {"key": f"{w}|{k}", "status": st, "in_task_file": False}
        for k, st in [("t151_model_registry", "not run: no pinned reference (results/il_no_reference_cells_2026-10-15.json)"), *sorted(excl.items())]
        for w in ("run3", "sim2")]}, indent=1) + "\n")
    write_box(tf, cf, bf, cap, n, v2=True)  # v3: every new package gets the v2 box script (v1 was written once, @ 1d6c2e41a)


def write_box(tf, cf, bf, cap, n, v2):
    box = (HERE / "ilc_box.sh").read_text()
    case = re.search(r"(?ms)^case \$H in\n.*?^esac\n", box).group(0)
    hosts = (f'[[ $H == sel97 ]] || {{ echo "host $H: this script runs only the sel97 add-on"; exit 2; }}\n'
             f'BOXL=xbox1; ORIG={ORIG}; PREVS="{" ".join(PREVS)}"\n')
    if v2:
        hosts = (f'[[ $H == sel97 ]] || {{ echo "host $H: this script runs only the sel97 add-on"; exit 2; }}\n'
                 f'BOXL=xbox1; ORIG={ORIG}; PREVS="{" ".join(PREVS_V2)}"  # v2: + sim-honest Q2\n'
                 f'SIMH_PINS="{" ".join(f"{f}:{h}" for f, h in SIMH_PINS.items())}"  # v2: staged sim-honest configs, exact names\n')
    fixed = (hosts +
             f'KIND=311; MAIN={MAIN}; '
             f'MANIFEST=/home/aswin/phase2/attack_check_311/config/harness_manifest_fu.tsv; CAP={cap}; '
             f'TSHA={sha(tf.read_bytes())}; CSHA={sha(cf.read_bytes())}; ILTASKS=config/tasks_il_old_all.json; '
             f'ILTSHA={sha((HERE / "config/tasks_il_old_all.json").read_bytes())}; RSHA={RSHA}; OSHA={OSHA}; '
             f'CAPSTOP_SHA=6689a761a6a5d879310dcb6c7e0e10eea8840dd4eb88fd44b9b3403222ba1ccb; GATE_SHA={sha((HERE / "gate_pause_311_rl.py").read_bytes())}  # on xbox1\n')
    box = box.replace(case, fixed)
    reps = [
        (r"(?s)^#!/usr/bin/env bash\n.*?# GENERATED by make_completion_launches.py \(the case table\); do not edit by hand.\n#   bash ~/phase2/il_stage/ilc_box.sh <host> <base URLs, comma-separated> <branch commit>\n",
         f"#!/usr/bin/env bash\n# il_sel97_box.sh: ON xbox1, start the il_old selection-half ADD-ON: one interleaved run3 + sim2 launch of {n} cells\n"
         "# (descriptive; NOT a registered IL launch or replacement; it cannot amend the registered IL result). The sequence is\n"
         "# ilc_box.sh's (freeze, settings equality, guards, confirmed start); only names, pins and the cell check differ.\n"
         "# GENERATED by make_sel97_addon.py; do not edit by hand.\n#   bash ~/phase2/il_stage/il_sel97_box.sh sel97 <base URLs, comma-separated> <branch commit>\n"),
        (r"RUN=/home/aswin/phase2/runs/2026-10-15_il_\$\{HF\}c;", f"RUN={RUN};"),
        (r"SESSION=ilc\$HF; LOG=/home/aswin/phase2/il_\$\{HF\}c\.log;", "SESSION=ilsel97; LOG=/home/aswin/phase2/il_sel97_addon.log;"),
        (r"--pilot-config config/pilot_ilc_\$HF\.yaml", "--pilot-config config/pilot_il_sel97.yaml"),
        (r"\[il \$\{HF\}c @ \$H\]", "[il sel97 @ xbox1]"),
        (r'sha_ok "\$ORIG/launch\.json" "\$OSHA"  # the stopped original.*\n', 'sha_ok "$ORIG/launch.json" "$OSHA"  # il_old, as read at selection\n'),
        (r'(?s)cp "\$ST/pilot_ilc_\$HF\.yaml".*?sha_ok "\$SRC/\$ILTASKS" "\$ILTSHA"\n',
         'cp "$ST/pilot_il_sel97.yaml" "$ST/tasks_il_sel97.json" "$SRC/config/"\nsha_ok "$SRC/config/tasks_il_sel97.json" "$TSHA"\n'
         'sha_ok "$SRC/config/pilot_il_sel97.yaml" "$CSHA"\nsha_ok "$SRC/$ILTASKS" "$ILTSHA"\n'),
        (r"(?s)# 3\. the cells:.*?^PY\n",
         "# 3. the cells: distinct verbatim original entries of il_old's IL task file, both of whose arms were 'not run' with 0 attempts in il_old\n"
         "(cd \"$SRC\" && \"$PY\" - \"$ILTASKS\" \"$ORIG\" <<'PY') || fail \"config or task file\"\n"
         "import json, sys\nfrom pathlib import Path\nimport yaml\niltasks, orig = sys.argv[1], sys.argv[2]\n"
         "cfg = yaml.safe_load(open(\"config/pilot_il_sel97.yaml\"))\n"
         "assert cfg[\"real_arms\"] == [\"attack\"] and cfg[\"sim_arms\"] == [\"sim_attack\"] and cfg[\"simulators\"] == [\"september\"], cfg\n"
         "assert cfg[\"tasks\"] == \"config/tasks_il_sel97.json\" and cfg[\"spend\"][\"cap_usd\"] == cfg[\"spend\"][\"worst_case_limit_usd\"]\n"
         "cells = json.load(open(cfg[\"tasks\"]))[\"cells\"]\ncanon = lambda c: json.dumps(c, sort_keys=True)\n"
         "orig_cells = json.load(open(iltasks))[\"cells\"]\n"
         "assert cells and set(map(canon, cells)) <= set(map(canon, orig_cells)) and len({c[\"key\"] for c in cells}) == len(cells), \"not distinct original entries\"\n"
         "lj = json.load(open(Path(orig) / \"launch.json\"))\nassert lj.get(\"finished\"), \"il_old has not finished\"\n"
         "runs = lj[\"runs\"]; runs = list(runs.values()) if isinstance(runs, dict) else runs\nby_key = {r[\"key\"]: r for r in runs}\n"
         "for c in cells:\n    for arm in (\"attack\", \"sim_attack@september\"):\n        r = by_key[f\"{arm}/{c['key']}\"]\n"
         "        assert r[\"state\"] == \"not run\" and not r.get(\"attempts\"), (c[\"key\"], arm, r[\"state\"])\n"
         "print(f\"add-on: {len(cells)} selection-half cells, both arms never started in il_old, all original entries; cap {cfg['spend']['cap_usd']}\")\nPY\n"),
        (r'GS="results/gate_settings_ilc_\$\{HF\}_\$\(date \+%s\)\.json"', 'GS="results/gate_settings_il_sel97_$(date +%s).json"'),
        (r"(?s)printf '\{\"launch\": \"il_%sc\".*?\n  \"\$HF\" \"\$ORIG\" \"\$SSHA\" \"\$MAIN\" \"\$KIND\" \"\$CAP\" \"\$GATE_SHA\" \"\$COMMIT\" > \"\$RUN\.il_meta\.json\"\ncp \"\$SRC/config/ilc_supersession_tier3_\$HF\.json\" \"\$RUN\.supersession_tier3\.json\"\n",
         "printf '{\"launch\": \"il_sel97_addon\", \"interleaved_launch\": true, \"addon_of\": \"%s\", \"status\": \"descriptive add-on; not a registered IL launch or replacement; cannot amend the registered IL result\", \"worlds\": {\"attack\": \"run3\", \"sim_attack@september\": \"sim2\"}, \"references_from\": \"%s\", \"settings_dir\": \"/home/aswin/phase2/settings\", \"harness\": \"%s\", \"sim_attempt_timeout_s\": 300, \"cap_usd\": %s, \"gate\": \"gate_pause_311_rl.py %s\", \"commit\": \"%s\"}\\n' \\\n"
         "  \"$ORIG\" \"$MAIN\" \"$KIND\" \"$CAP\" \"$GATE_SHA\" \"$COMMIT\" > \"$RUN.il_meta.json\"\n"),
    ]
    if v2:
        reps += [
            (r"^# GENERATED by make_sel97_addon\.py; do not edit by hand\.\n",
             "# GENERATED by make_sel97_addon.py (v2: --box-only); do not edit by hand.\n"),
            (r'^log "freeze: \$SRC matches \$MANIFEST.*\n',
             'for pin in $SIMH_PINS; do  # v2: each excluded sim-honest config is absent or at its pinned sha256\n'
             '  f=$SRC/config/${pin%%:*}; [[ ! -e $f && ! -L $f ]] || sha_ok "$f" "${pin##*:}"\n'
             'done\n'
             'log "freeze: $SRC matches $MANIFEST ($(wc -l < "$TMP/now.tsv") files; sim-honest configs absent or pinned: $SIMH_PINS)"\n'),
            (r'^log "earlier launches finished: \$PREVS"\n',
             'log "earlier launches finished: $PREVS"\n'
             '# (a2) v2: EVERY launch on this box has finished. Run dirs are enumerated (nullglob), each must be inspectable; a dir with\n'
             '# no launch.json (neither file nor symlink) never got a launcher far (no_launcher covers one starting); any other\n'
             '# launch.json must be a readable regular file reporting finished (Codex r39: a dangling symlink or unreadable dir fails)\n'
             '[[ -d /home/aswin/phase2 && -r /home/aswin/phase2 && -x /home/aswin/phase2 ]] || fail "cannot inspect /home/aswin/phase2"\n'
             'for q in /home/aswin/phase2/runs /home/aswin/phase2/nm /home/aswin/phase2/nm/runs; do  # in order: each parent proven inspectable first (Codex r39b)\n'
             '  [[ ( ! -e $q && ! -L $q ) || ( -d $q && -r $q && -x $q ) ]] || fail "cannot inspect $q"\n'
             'done\n'
             '[[ -d /home/aswin/phase2/runs ]] || fail "no /home/aswin/phase2/runs"\n'
             'shopt -s nullglob; RDIRS=(/home/aswin/phase2/runs/*/ /home/aswin/phase2/nm/runs/*/); shopt -u nullglob\n'
             'for d in "${RDIRS[@]}"; do\n'
             '  [[ -r $d && -x $d ]] || fail "cannot inspect $d"\n'
             '  l=${d}launch.json\n'
             '  [[ -e $l || -L $l ]] || continue\n'
             '  [[ -f $l && -r $l ]] || fail "$l is not a readable regular file"\n'
             '  "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get(\'finished\') else 1)" "$l" \\\n'
             '    || fail "${d%/} has not finished (or its launch.json can\'t be read)"\n'
             'done\n'
             'log "every launch.json under runs/ and nm/runs/ reports finished (${#RDIRS[@]} run dirs inspected)"\n'),
        ]
    for pat, rep in reps:
        box2, k = re.subn(pat, lambda m: rep, box, count=1, flags=re.M)
        assert k == 1, pat[:60]
        box = box2
    box = box.replace("[il sel97 @ xbox1]", "[il sel97 @ $BOXL]")
    if v2:
        assert box.count(FREEZE_OLD) == 1, "freeze exclusion line"
        box = box.replace(FREEZE_OLD, FREEZE_NEW)
    left = [l for l in box.splitlines() if any(t in l for t in ("pilot_ilc_$HF", "tasks_ilc_$HF", "ilc_supersession", "${HF}c", "ilc$HF"))]
    assert not left, left[:5]  # the freeze exclusions (pilot_ilc_*, tasks_ilc_*, ilc_*) stay, as in every IL box script
    bf.write_text(box)
    rows = [r["key"] for r in json.loads((bf.parent / "config/addon/sel97_status_rows.json").read_text())["rows"]]
    print(json.dumps({"cells": n, "status_rows": rows, "cap": cap, "box_v2": v2,
                      "tasks_sha256": sha(tf.read_bytes()), "config_sha256": sha(cf.read_bytes()), "box_sha256": sha(bf.read_bytes())}))


if __name__ == "__main__":
    main()
