"""ug_servers_check.py (RULE.md; d9's start condition 2): the ug arm's 4 H200 servers are live. Laptop, read-only.
Reads the server start's results JSON (servers_<stamp>.json, the path d9 relays) and `modal app list --json` (saved to a
file first). PASS iff the JSON says status "up", mode "deployed", server_config "servers_il.yaml", exactly 4 servers with
4 distinct non-empty app IDs and 4 distinct base URLs, each of the form https://<lowercase id>.w.modal.host/v1 and equal
to its tunnel_url + "/v1" (Codex r12), and each server's app is in the Modal list in state "deployed" with >= 1 task. A
server record missing a key fails (it is still starting). Prints, last,
"BASES <the 4 base URLs, comma-separated, in the JSON's order>" for ug_box.sh (never retyped). Exit 0 PASS, 1 FAIL.

  modal app list --json > <apps.json>; python3 ug_servers_check.py <servers_*.json> <apps.json>
"""

from __future__ import annotations

import json
import re
import sys

URL = re.compile(r"https://[a-z0-9-]+\.w\.modal\.host/v1")


def main() -> None:
    srv, apps = (json.load(open(p)) for p in sys.argv[1:3])
    by_id = {a["app_id"]: a for a in apps}
    rows = srv["servers"]
    urls = [s.get("base_url") or "" for s in rows]
    ids = [s.get("app_id") or "" for s in rows]
    ok = (srv["status"] == "up" and srv["mode"] == "deployed" and srv["server_config"] == "servers_il.yaml"
          and len(rows) == 4 and len(set(urls)) == 4 and len(set(ids)) == 4 and all(ids)
          and all(URL.fullmatch(u) and u == (s.get("tunnel_url") or "") + "/v1" for u, s in zip(urls, rows)))
    print(f"servers JSON: status {srv['status']}, mode {srv['mode']}, config {srv['server_config']}, {len(rows)} servers, "
          f"{len(set(ids))} distinct app IDs, URLs well formed: {all(URL.fullmatch(u) for u in urls)}")
    for s in rows:
        a = by_id.get(s.get("app_id"), {})
        live = a.get("state") == "deployed" and int(a.get("tasks") or 0) >= 1
        ok &= live
        print(f"  {s.get('name')} {s.get('app_id')}: modal state {a.get('state')}, tasks {a.get('tasks')}{'' if live else '  <- NOT LIVE'}")
    print("SERVERS", "PASS" if ok else "FAIL")
    if ok:
        print("BASES " + ",".join(urls))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
