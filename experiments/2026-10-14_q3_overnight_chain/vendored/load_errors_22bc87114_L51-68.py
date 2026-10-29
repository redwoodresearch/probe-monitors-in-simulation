def load_errors(job):
    """One eval file: {run key: [error type or None per non-submit tool output]} (the error field only, no content)."""
    path, wanted = job
    out = {}
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        for key, sid, ep in wanted:
            n = f"samples/{sid}_epoch_{ep}.json"
            if n not in names:
                raise SystemExit(f"{Path(path).name}: no {n}")
            s = json.loads(z.read(n))
            if (s["id"], s["epoch"]) != (sid, ep):
                raise SystemExit(f"{n}: id/epoch mismatch")
            submit_ids = {tc.get("id") for m in s["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls") or []
                          if tc["function"] == "submit"}
            out[key] = [((m.get("error") or {}).get("type") or "unknown") if m.get("error") else None for m in s["messages"]
                        if m["role"] == "tool" and m.get("tool_call_id") not in submit_ids and m.get("function") != "submit"]
    return out
