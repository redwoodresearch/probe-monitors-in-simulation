"""While the chain is STOPPED (d9 10-15 ~09:10, after Aswin confirms box1's shutdown reschedule): move il-q4c's box back from xbox3
to box1 (same run dir; the xbox3 fallback was cancelled before anything dispatched there). Backs up state.json; changes only il-q4c's
box; fails without writing unless il-q4c is on xbox3 and still before its "ended" step. Logged."""
import json, shutil, sys, time
from pathlib import Path

d = Path(sys.argv[1]); st_p = d / "state.json"; stamp = time.strftime("%H%M%S")
s = json.loads(st_p.read_text()); b = s["batches"]["il-q4c"]
assert b["box"] == "aswin@i-0f2299f29dd296259" and "ended" not in b["done"] and b["run"] == "2026-10-15_il_q4c", (b["box"], b["done"], b["run"])
shutil.copy2(st_p, d / f"state.json.bak_q4c_box_back_{stamp}")
b["box"] = "aswin@i-002bbb22c185c0f07"
tmp = st_p.with_suffix(".tmp"); tmp.write_text(json.dumps(s, indent=1)); tmp.replace(st_p)
with (d / "chain.log").open("a") as f:
    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} PT  STATE EDIT (stopped): il-q4c box aswin@i-0f2299f29dd296259 -> aswin@i-002bbb22c185c0f07 (back to box1; d9); backup state.json.bak_q4c_box_back_{stamp}\n")
print(json.dumps({"il-q4c": {"box": b["box"], "done": b["done"], "status": b["status"]}}))
