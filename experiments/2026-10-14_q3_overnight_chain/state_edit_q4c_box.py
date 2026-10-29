"""While the chain is STOPPED (0d's q4c fallback, 10-15 ~08:50): move il-q4c's box from box1 to xbox3 (same run dir), as the config now
says; the config loader keeps an existing batch's fields, so the state needs the edit. Backs up state.json; changes only il-q4c's box;
fails without writing unless il-q4c is still before its "ended" step on box1. Logged."""
import json, shutil, sys, time
from pathlib import Path

d = Path(sys.argv[1]); st_p = d / "state.json"; stamp = time.strftime("%H%M%S")
s = json.loads(st_p.read_text()); b = s["batches"]["il-q4c"]
assert b["box"] == "aswin@i-002bbb22c185c0f07" and "ended" not in b["done"] and b["run"] == "2026-10-15_il_q4c", (b["box"], b["done"], b["run"])
shutil.copy2(st_p, d / f"state.json.bak_q4c_box_{stamp}")
b["box"] = "aswin@i-0f2299f29dd296259"
tmp = st_p.with_suffix(".tmp"); tmp.write_text(json.dumps(s, indent=1)); tmp.replace(st_p)
with (d / "chain.log").open("a") as f:
    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} PT  STATE EDIT (stopped): il-q4c box aswin@i-002bbb22c185c0f07 -> aswin@i-0f2299f29dd296259 (0d's fallback); backup state.json.bak_q4c_box_{stamp}\n")
print(json.dumps({"il-q4c": {"box": b["box"], "done": b["done"], "status": b["status"]}}))
