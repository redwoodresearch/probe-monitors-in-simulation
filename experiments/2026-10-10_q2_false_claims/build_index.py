"""False success claims, step 1 (local, free): one row per honest run of the strict rerun with a final report, keyed by
the request that generated it. Attempt choice, final report, class and scorer are count_claims.py's own functions
(a5837ce), so the classes must reproduce its 7 false and 22 truthful real claims. The request is the acting model's
call for the report's assistant message; its output's visible text must equal the report.

Rows (with the report text, which the staging matches against vLLM's logged completion) go to data/, outside git;
the printed summary has counts only. Honest arms only.

  uv run --no-project --python 3.14 python experiments/2026-10-10_q2_false_claims/build_index.py <strict rerun folder> <out jsonl>
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "2026-10-09_final_report_claims"))
from count_claims import classify, final_report, load_qwen, scorer, visible

ACTING = "vllm/Qwen/Qwen3.5-27B"


def resolve(msg: dict, att: dict) -> dict:
    """The message with attachment:// references in its text parts replaced by the attachment (events store text so)."""
    r = lambda v: att[v.removeprefix("attachment://")] if isinstance(v, str) and v.startswith("attachment://") else v
    c = msg.get("content")
    c = r(c) if isinstance(c, str) else [p | {"text": r(p.get("text"))} if p.get("type") == "text" else p for p in c or []]
    return msg | {"content": c}


def main(run_dir: Path, out: Path):
    rows, n = [], Counter()
    for world, key, s in load_qwen(run_dir):
        msgs = s.get("messages") or []
        text, k = final_report(msgs)
        cls = classify(text)
        sc = scorer(s) if world == "qwen-real" else None
        label = {"pass": 0, "fail": 1}.get(sc) if cls == "claims success" else None
        n[(world, cls, sc)] += 1
        if not text:
            continue
        am = [m for m in msgs if m["role"] == "assistant"]
        events = [e for e in s.get("events") or [] if e.get("event") == "model" and e.get("model") == ACTING]
        if len(events) != len(am):
            raise ValueError(f"{key}: {len(events)} acting-model calls but {len(am)} assistant messages")
        e = events[len(am) - 1 - k]
        out_msg = resolve((((e.get("output") or {}).get("choices") or [{}])[0].get("message") or {}), s.get("attachments") or {})
        if visible(out_msg) != text:
            raise ValueError(f"{key}: the acting-model call's output is not the final report")
        irid = (((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid")
        if not irid:
            raise ValueError(f"{key}: the report's model call has no x-irid")
        _, arm, task = key.split("/")
        rows.append({"world": world, "key": key, "arm": arm, "task": task, "class": cls, "scorer": sc, "label": label,
                     "turns_from_end": k, "irid": irid, "report": text, "report_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "report_chars": len(text)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    real = Counter(r["label"] for r in rows if r["world"] == "qwen-real")
    if (real[1], real[0]) != (7, 22):
        raise ValueError(f"real false/truthful claims {real[1]}/{real[0]}, not count_claims' 7/22")
    print(json.dumps({"rows": len(rows), "by world, class, scorer": {" | ".join(map(str, key)): v for key, v in sorted(n.items(), key=str)},
                      "real false / truthful claims": [real[1], real[0]],
                      "success claims by world": dict(Counter(r["world"] for r in rows if r["class"] == "claims success"))}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]))
