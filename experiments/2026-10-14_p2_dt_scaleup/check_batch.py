"""For each sp493 root of the 14-root batch whose job wrote its local result record, run check_sp493_records.py once (keys, folds, counts).
Usage: python3 -I check_batch.py"""
import json, subprocess, sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
P2 = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/823cc0a0-23fc-47f8-b84f-0087b7bffcbd/scratchpad/p2")
WT, RT = P2 / "dt4f/wt_simh", Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad/chain_run_state/routing")
X = WT / "experiments/2026-10-14_p2_dt_scaleup"
M, V = "/Users/aswinsurya/.local/bin/modal", "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
ROOTS = ["sp493-c__sp493_c_main", "sp493-c__sp493_c_run2", "sp493-g1__sp493_g1_sim", *[f"sp493-g1_s{j}__sp493_g1_sim_s{j}" for j in range(1, 6)],
         "sp493-g2__sp493_g2_sim", *[f"sp493-g2_s{j}__sp493_g2_sim_s{j}" for j in range(2, 6)], "sp493-g5__sp493_g5_simh"]
done = {p.name.split("_check_")[1].rsplit("_2026", 1)[0] for p in (X / "results").glob("dt_sp493_record_check_*.json")}
out = []
for rt in ROOTS:
    r = json.loads((RT / f"{rt}.json").read_bytes())
    st = r["staged_set"]
    res = sorted((X / "results").glob(f"dt_q4b_arm_{st}_2026-10-16_*.json"))
    lab = rt.split("__")[1]
    if not res:
        out.append((rt, "not landed"))
        continue
    if lab in done:
        out.append((rt, "checked"))
        continue
    D = P2 / "dt4f/sp493/rec" / lab
    D.mkdir(parents=True, exist_ok=True)
    args = []
    for a in r["arms"]:
        fo = f"{st}__{a['meta_arm'].split('@')[0]}"
        base = f"simh/sp493_g5/dt/{fo}" if lab.startswith("sp493_g5") else f"scaleup/{fo}"
        mf, ls = D / f"{fo}.manifest_dt.json", D / f"{fo}.listing.txt"
        subprocess.run([M, "volume", "get", "--force", V, f"{base}/manifest_dt.json", str(mf)], check=True, capture_output=True)
        lst = subprocess.run([M, "volume", "ls", V, base], check=True, capture_output=True, text=True).stdout
        if not lab.startswith("sp493_g5"):
            lst += subprocess.run([M, "volume", "ls", V, f"{base}/dt"], check=True, capture_output=True, text=True).stdout
        ls.write_text(lst)
        args.append(f"{fo}={mf}={ls}")
    rel = P2 / ("sp493_release_2026-10-16_r2.json" if lab.startswith("sp493_c_") else "sp493_release_2026-10-16_r1.json")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    pr = subprocess.run(["python3", "-I", str(X / "check_sp493_records.py"), str(WT), str(P2 / "dt4f/fitcheck"), str(res[-1]), str(RT / f"{rt}.json"), str(rel),
                         str(X / "results" / f"dt_sp493_record_check_{lab}_{stamp}.json"), *args], capture_output=True, text=True)
    if pr.returncode:
        out.append((rt, "CHECK ERROR " + pr.stderr.strip().splitlines()[-1][:200]))
        continue
    d = json.loads(pr.stdout)
    cen = []
    for fo, a in zip(d["folders"], args):
        m = json.loads(Path(a.split("=")[1]).read_text())["sp493"]["census"]
        cen.append(f"{m['runs_mapped']}/{m['runs_fold_minus_1']}")
    out.append((rt, f"PASS={d['pass']} staged={d['staged_joins']} scored={d['scored_runs']} mapped/-1={','.join(cen)} " +
                ";".join(f"{k.split('__')[1]}:{v['world']}/{v['runs']}/{v['strata']}/foldsbad{len(v['folds_vs_gate_bad'])}" for k, v in d["folders"].items())))
for rt, s in out:
    print(f"{rt:36s} {s}")
