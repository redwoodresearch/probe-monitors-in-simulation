"""Sidecar for a FROZEN INPUTS_FINAL (09's condition): every input that was ANNOUNCED but is NOT in this build, by key/folder, with the reason
and where it goes (the second refresh); plus, for Sol (a), per origin, the input folders that ARE pinned. Built from the frozen manifest's own
pins (no free text for anything the manifest records) and a short list of announced items the manifest does not carry.

  <modal env python> make_not_in_build.py config/INPUTS_FINAL_<stamp>.json   # writes config/INPUTS_FINAL_<stamp>_NOT_IN_BUILD.json (exclusive)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inputs_final as IF  # noqa: E402

mp = HERE / sys.argv[1]
b = mp.read_bytes()
m = json.loads(b)
if not m["status"].startswith("FROZEN") or IF.verify(m):
    raise SystemExit("not a verified FROZEN INPUTS_FINAL")
SR = "second refresh"
out = []
e2 = m["epoch2_honest"]
art = json.loads(IF.git_show(e2["artifact"]["commit"], e2["artifact"]["path"])) if e2.get("artifact") else {}
for k, v in (art.get("inputs") or {}).items():
    if k.startswith("keylist"):
        out.append({"item": "epoch 1+2 honest pools (epoch-2 roots)", "key": k.split(" ", 2)[2], "source": v, "reason": e2["status"], "goes_to": SR})
s97 = m["sel97"]
out.append({"item": "sel97 (RMS selection cells)", "key": "sel97", "source": {k: s97.get(k) for k in ("cells", "prep_config")}, "reason": f"status {s97['status']!r}",
            "goes_to": SR})
g = m["gapfill_sim2"]
out.append({"item": "s4r gap-fill (sim2 interleaved run)", "key": g.get("root"), "reason": f"status {g['status']!r}; {g.get('role')}", "goes_to": SR})
for key, item, reason in (
        ("ilnm_s4rgap_sim2_v1_try2__sim_attack", "nm145 re-stage of the s4r gap-fill", "not admitted to this build (gap-fill is second refresh)"),
        ("simh_t196fill", "T196 sim-honest fill", "announced; not routed into this build's pins"),
        ("sp493_g4_simh", "S_paper493 g4 sim-honest set", "S_paper493 is not in this build"),
        ("S_paper493 real-arm *_try2 roots; L3 and g5 sets", "S_paper493 extension", m["extensions"]["S_paper493"]),
        ("A25 pilot additions", "S_old A25 pilot additions", "S_old reads A25 primary_v3 through the pinned E2E config only"),
        ("IL / ug (ungrounded-sim attack arm)", "interleaved-launch and ungrounded-sim arms", "RMS/Z inputs, not Stage A pools; ug not in the all-tasks build")):
    out.append({"item": item, "key": key, "reason": reason, "goes_to": SR})
ec = m["sets"]["S_old"]["e2e_config"]
e2e = {r["label"]: r for r in json.loads(IF.git_show(ec["commit"], ec["path"]))["rows"]}
sol = {"S_old": {lb: e2e[lb]["simh"] for lb in ("Sol (a) C", "Sol (a) R") if lb in e2e},
       "S_new": {"sim_honest_roots": sorted({r for rs in m["sets"]["S_new"]["roots"].values() for r in rs}),
                 "honest_a_25f_sol": sorted(e["folder"] for e in m["sets"]["S_new"]["honest_a_25f"] if e["kind"] == "sol")},
       "S_newmains": {"sim_honest_roots": sorted({r for rs in m["sets"]["S_newmains"]["roots"].values() for r in rs}),
                      "fold_extension": m["nm360_fold_extension"]["gates"].get("Sol (a)"),
                      "fold_extension_files": m["nm360_fold_extension"].get("files") or "none extra: the extension reads the pinned nm360 roots' Sol files"}}
res = {"frozen": {"path": str(mp.relative_to(HERE)), "sha256": IF.sha(b)}, "not_in_build": out,
       "sol_a_inputs_pinned": sol, "sol_a_not_in_build": [x["key"] for x in out],
       "rule": "09: every announced input not in this build, by key, with reason and destination; no row silently uses a smaller cohort"}
dest = mp.with_name(mp.stem + "_NOT_IN_BUILD.json")
blob = json.dumps(res, indent=1, ensure_ascii=False).encode()
dest.open("xb").write(blob)
print(json.dumps({"file": str(dest.relative_to(HERE)), "sha256": IF.sha(blob), "entries": len(out)}))
