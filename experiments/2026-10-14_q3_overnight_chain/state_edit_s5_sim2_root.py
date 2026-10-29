"""While the chain is STOPPED (d9 10-15 ~08:35): point il-s5's sim2 at its extraction retry, so the registry (rebuilt from state at
every attest) and anything reading state never resolve ilnm_s5_sim2 to the unattested ilnm_s5_sim2_extract_v1. Backs up state.json;
changes only il-s5's extract/compare entries for ilnm_s5_sim2 (keeping the first root under extract_v1_retired), and rewrites that
one entry of registry_S_newmains.json to match (logged). Fails without writing if anything is not as expected."""
import json, shutil, sys, time
from pathlib import Path

d = Path(sys.argv[1]); t = "ilnm_s5_sim2"; new_root = "ilnm_s5_sim2_extract_v1_try2"
new_cmp = "compare_repro_ilnm_s5_sim2_extract_v1_try2_repro-c_2026-10-15_075917.json"
stamp = time.strftime("%H%M%S")
st_p, reg_p = d / "state.json", d / "registry_S_newmains.json"
s = json.loads(st_p.read_text()); reg = json.loads(reg_p.read_text())
b = s["batches"]["il-s5"]
assert b["extract"][t] == "ilnm_s5_sim2_extract_v1" and "attest" in b["done"], (b["extract"][t], b["done"])
key = f"il-s5__{t}.json"
assert reg["entries"][key]["extraction_root"] == "ilnm_s5_sim2_extract_v1"
shutil.copy2(st_p, d / f"state.json.bak_s5_sim2_root_{stamp}")
shutil.copy2(reg_p, d / f"registry_S_newmains.json.bak_s5_sim2_root_{stamp}")
b["extract_v1_retired"] = {t: {"extraction_root": b["extract"][t], "compare": b["compare"][t],
                               "reason": "attempt-1 writer GPU unrecorded (fc-01M4ZX3RPFHWZPPFWQFBYMJHVW); never scored"}}
b["extract"][t], b["compare"][t] = new_root, new_cmp
tmp = st_p.with_suffix(".tmp"); tmp.write_text(json.dumps(s, indent=1)); tmp.replace(st_p)
reg["entries"][key]["extraction_root"] = new_root
reg["written"] = time.strftime("%Y-%m-%d %H:%M:%S PT") + " (il-s5 sim2 entry corrected by state_edit_s5_sim2_root.py)"
tmp = reg_p.with_suffix(".tmp"); tmp.write_text(json.dumps(reg, indent=1)); tmp.replace(reg_p)
with (d / "chain.log").open("a") as f:
    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} PT  STATE EDIT (stopped): il-s5 extract[{t}] ilnm_s5_sim2_extract_v1 -> {new_root}; registry entry {key} rewritten; backups *_s5_sim2_root_{stamp}\n")
print(json.dumps({"il-s5": {"extract": b["extract"], "compare": b["compare"]}, "registry": reg["entries"][key]["extraction_root"], "backups": stamp}))
