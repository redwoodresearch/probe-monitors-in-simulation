"""POST HOC (Aswin about 14:20 PT via 0d: "why is judge stopped, keep going u idiot do't stop it", after the stop relayed
at about 14:15 PT): reconcile a cancelled batch so its completed requests are recorded and paid once, and its unfinished
requests are resubmitted by the runner. Uses run_judge_batch.py's own classify, ledger and cache code.

- Refuses unless the batch's status is "cancelled".
- Every output or error line is classified and cached exactly as run_batch does, and counted once in the ledger (a
  refusal still writes the STOP file, as in the runner), except lines whose error code is batch_cancelled.
- Lines whose error is batch_cancelled, and requests with no line, are left uncached; the number of cached requests
  with a response must equal the batch's completed count, so no completed result is dropped and paid for again.
- Rerunnable: the chunk's keys are its kept keys plus any earlier cancelled_keys; cached keys are not recounted.
- The chunk's batches.json entry keeps only the cached keys; the rest go to `cancelled_keys` (provenance), so the
  runner's drive() resubmits them. The chunk's pending reservation is released.
Keys and counts only are printed.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with httpx \\
      python experiments/2026-10-11_claimB_judge/reconcile_POSTHOC_cancelled_batch.py run2
"""

import json
import re
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_judge_batch as R  # noqa: E402


def main(name):
    key = None
    for line in R.P.ENV.read_text().splitlines():
        if line.split("=", 1)[0].strip() == "OPENAI_API_KEY":
            key = line.split("=", 1)[1].strip().strip('"').strip("'").strip()
    if not key or not re.fullmatch(r"[A-Za-z0-9_\-]+", key):
        raise SystemExit("OPENAI_API_KEY missing or malformed (not shown)")
    batches = json.loads(R.BATCHES.read_text())
    ent = batches[name]
    keys = set(ent["keys"]) | set(ent.get("cancelled_keys", []))
    with httpx.Client(timeout=300, headers={"Authorization": f"Bearer {key}"}) as client:
        b = R.get_json(client, f"{R.API}/batches/{ent['id']}").json()
        if b.get("status") != "cancelled":
            raise SystemExit(f"{name}: status {b.get('status')}, not cancelled; wait")
        lines = {}
        for fid in (b.get("output_file_id"), b.get("error_file_id")):
            if fid:
                for raw in R.get_json(client, f"{R.API}/files/{fid}/content").text.splitlines():
                    d = json.loads(raw)
                    cid = d.get("custom_id")
                    if cid not in keys or cid in lines:
                        raise SystemExit(f"{name}: an output line names a request not in this batch, or twice")
                    lines[cid] = d
    led = R.ledger()
    cached, cancelled, statuses = [], [], {}
    for k in sorted(keys):
        line = lines.get(k)
        err = (line or {}).get("error") or {}
        if line is None or (isinstance(err, dict) and err.get("code") == "batch_cancelled"):
            cancelled.append(k)
            continue
        status, p, usage, cost, model, effort = R.classify(line)
        if usage.get("missing"):
            raise SystemExit(f"{name}: a completed request lacks usage; stop and reconcile by hand")
        if k not in led["counted"]:
            led["spent"] += cost
            led["calls"] += 1 if (usage.get("input") or cost) else 0
            led["counted"].append(k)
        R.P.write_atomic(R.LEDGER, json.dumps(led))
        R.P.write_atomic(R.CACHE / f"{k}.json", json.dumps({"status": status, "p": p, "usage": usage, "cost": cost, "model_echo_ok": model.startswith(R.MODEL),
                                                           "effort_echo": effort, "raw": line, "batch": ent["id"]}))
        if status == "refusal":
            R.P.write_atomic(R.STOPF, json.dumps({"reason": "refusal", "chunk": name}))
        cached.append(k)
        statuses[status] = statuses.get(status, 0) + 1
    completed = (b.get("request_counts") or {}).get("completed") or 0
    answered = sum(1 for k in cached if (lines[k].get("response") or {}).get("status_code") == 200)
    if answered != completed:
        raise SystemExit(f"{name}: {answered} cached responses vs {completed} completed; output missing, reconcile by hand")
    ent["keys"], ent["cancelled_keys"], ent["requests"] = cached, cancelled, len(cached)
    ent["note"] = "cancelled about 14:13 PT 10-11 on a mistaken stop; reconciled by reconcile_POSTHOC_cancelled_batch.py"
    R.P.write_atomic(R.BATCHES, json.dumps(batches))
    led = R.ledger()
    led["pending"].pop(name, None)
    R.P.write_atomic(R.LEDGER, json.dumps(led))
    R.log(f"{name}: reconciled after cancel: {len(cached)} cached {statuses}, {len(cancelled)} left for resubmission; billed so far ${led['spent']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
