"""Counts only: per dataset, rows, chars of the chat messages (all roles) and of `output`, roles present, and the
messages' first-role pattern. No text is printed. Reads the public HF datasets-server /rows API."""
import json, sys, time, urllib.request
from collections import Counter
DS = sys.argv[1:]
out = {}
for d in DS:
    n = 0; mchars = []; ochars = []; roles = Counter(); last_role = Counter(); offset = 0; ids = []
    while True:
        url = f"https://datasets-server.huggingface.co/rows?dataset=candywal/{d}&config=default&split=train&offset={offset}&length=100"
        for attempt in range(8):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    j = json.load(r); break
            except Exception as e:
                time.sleep(10 * (attempt + 1))
        else:
            raise SystemExit(f"{d}: rows API failed at offset {offset}")
        rows = j.get("rows", [])
        for rr in rows:
            row = rr["row"]; n += 1
            msgs = row.get("messages") or []
            mchars.append(sum(len(m.get("content") or "") for m in msgs))
            ochars.append(len(row.get("output") or ""))
            roles.update(m.get("role") for m in msgs)
            if msgs: last_role[msgs[-1].get("role")] += 1
            ids.append(row.get("action_id"))
        if len(rows) < 100: break
        offset += 100
    s = sorted(mchars)
    out[d] = {"rows": n, "messages_chars_total": sum(mchars), "messages_chars_median": s[len(s)//2] if s else None,
              "messages_chars_max": max(mchars) if mchars else None, "output_chars_total": sum(ochars),
              "roles": dict(roles), "last_role": dict(last_role), "action_ids_unique": len(set(ids))}
    print(d, json.dumps(out[d]), flush=True)
json.dump(out, open(sys.argv[0].replace("count_lengths.py", "lengths.json"), "w"), indent=1)  # committed as results/dataset_lengths_2026-10-11.json
