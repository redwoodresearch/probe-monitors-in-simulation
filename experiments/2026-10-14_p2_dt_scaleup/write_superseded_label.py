"""Amendment 45h: label DT's records of the three SUPERSEDED sp493 real-arm sets (append-only; nothing deleted or rewritten).
Usage: python3 -I write_superseded_label.py <results dir> <out json>"""
import hashlib, json, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SETS = {"sp493_b_main_v1_try4": ("sp493_b_main_v1_try5", "names record r2 @ cd2684e5e", ("attack", "honest_a", "honest_b")),
        "sp493_c_main_v1_try3": ("sp493_c_main_v1_try4", "names record r3 @ 69a1d211f", ("attack", "honest_a", "honest_b")),
        "sp493_c_run2_v1_try3": ("sp493_c_run2_v1_try4", "names record r3 @ 69a1d211f", ("attack",))}
out = {"what": "DT records of SUPERSEDED S_paper493 real-arm sets: kept as recorded (append-only), labelled SUPERSEDED; never read; the re-stages are the S_paper493 rows",
       "written_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S"), "sets": {}}
for st, (by, rec, arms) in SETS.items():
    res = sorted(R.glob(f"dt_q4b_arm_{st}_2026-10-16_*.json")); chk = sorted(R.glob(f"dt_sp493_record_check_*{st.split('_v1')[0].replace('sp493_', '')}_2026-10-16_*.json"))
    new = sorted(R.glob(f"dt_q4b_arm_{by}_2026-10-16_*.json"))
    assert len(res) == 1 and len(new) == 1, (st, res, new)
    out["sets"][st] = {"label": "SUPERSEDED", "superseded_by": by, "ruling": rec, "result_record": res[0].name, "result_record_sha256": hashlib.sha256(res[0].read_bytes()).hexdigest(),
                       "check_records": [c.name for c in chk], "volume_folders": [f"/work/scaleup/{st}__{a}/" for a in arms], "replacement_result_record": new[0].name,
                       "rule": "the job's and launcher's once-per-join checks ignore these folders and claims (Amendment 45g); the files stay"}
OUT.open("x").write(json.dumps(out, indent=1)); print(json.dumps({k: (v["result_record"], v["superseded_by"], v["check_records"]) for k, v in out["sets"].items()}, indent=0))
