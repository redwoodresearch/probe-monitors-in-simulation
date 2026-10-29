"""Test (d9 10-15 16:40): the chain's relay plan for a normal and a snapshot batch whose run folder holds cells.json, approach files,
an unnamed .eval, a nested launch.json and a '..' path relays none of them; only manifest-named .eval files, the run folder's own
launch.json / gate1_final.json / upload_manifest_*.tsv and a snapshot's generated launch.json pass; every refusal is returned (and logged by
step_relay). Also: box_files no longer lists cells.json and its FORBIDDEN check catches the names.
  python3 test_relay_allowlist.py   (prints a JSON verdict; exit 1 on any failure)"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chain  # noqa: E402  (module import has no side effects; main() is guarded)

run = "2026-10-15_fake_run"
named = [f"{run}/attack/t1_x/a.eval", f"{run}/sim_honest@september/t2_y/b.eval"]
listed = named + [f"{run}/cells.json", f"{run}/approach_texts.json", f"{run}/approaches/t1.json", f"{run}/attack/t1_x/unnamed.eval",
                  f"{run}/launch.json", f"{run}/gate1_final.json", f"{run}/upload_manifest_honest_a.tsv", f"{run}.launcher.log",
                  f"{run}/gate1_watch.log", f"{run}/RL_PAUSE.txt", f"{run}/approaches/launch.json", f"{run}/x/../launch.json",
                  f"{run}/sub/gate1_final.json", f"{run}/approach.eval"]
want_main = sorted([*named, f"{run}/launch.json", f"{run}/gate1_final.json", f"{run}/upload_manifest_honest_a.tsv"])
d = {"files": [{"path": p, "bytes": 1, "sha256": "0" * 64} for p in listed], "launch_subset": "{}",
     "manifests": {"main": {"files": [{"path": p} for p in named + [f"{run}/approach.eval"]]}}}
res = {}
with tempfile.TemporaryDirectory() as tmp:
    for kind in ("fu", "fus"):  # a final remainder and a live snapshot
        items, refused = chain.relay_plan({"id": f"fake-{kind}", "kind": kind, "run": run}, d, Path(tmp))
        ev = sorted({x["src"] for x in items if not x.get("generated")})
        res[kind] = {"relayed": ev, "generated": [x["dest"] for x in items if x.get("generated")], "refused": refused,
                     "pass": ev == want_main and sorted(refused) == sorted(set(listed) - set(want_main))
                     and all(x["src"].endswith(".eval") for x in items if x["dest"].startswith("snaps/") and not x.get("generated"))
                     and (kind == "fu" or [x["dest"] for x in items if x.get("generated")] == [f"snaps/fake-{kind}/{run}/launch.json"])}
ns = {}
exec(compile((HERE / "box_files.py").read_text().split("\ndef main():")[0], "box_files_head", "exec"), ns)  # constants and helpers only
bf = {"records_has_cells_json": "cells.json" in ns["RECORDS"],
      "forbidden": {n: ns["FORBIDDEN"](n) for n in (f"{run}/cells.json", f"{run}/x_approach.json", f"{run}/launch.json")}}
ok = all(v["pass"] for v in res.values()) and not bf["records_has_cells_json"] and bf["forbidden"] == {f"{run}/cells.json": True, f"{run}/x_approach.json": True, f"{run}/launch.json": False}
print(json.dumps({"relay_plan": res, "box_files": bf, "pass": ok}, indent=1))
sys.exit(0 if ok else 1)
