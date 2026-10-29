"""Blinded independent ratings; invoked by the thin experiment entrypoint.

Console output is restricted to metadata. Raw responses are private cache
artifacts; never open them in a blinded agent session.
"""
import argparse
import asyncio
from collections import Counter
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import time
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
FIELDS = {"id", "flagged_quote", "turn_window", "tool_output_before_tail"}
MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
# Order is intentional: UUIDs before dates; timestamps before generic hex.
MASK_PATTERNS = [
    ("UUID", re.compile(r"(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-f])", re.I)),
    ("DATE", re.compile(
        r"(?<!\d)\d{4}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])(?!\d)"
        r"|(?<!\d)(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|[12]\d|3[01])[-/.](?:\d{4}|\d{2})(?!\d)"
        r"|\b" + MONTH + r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b"
        r"|\b\d{1,2}(?:st|nd|rd|th)?\s+" + MONTH + r"(?:,?\s+\d{4})?\b", re.I)),
    ("TIME", re.compile(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\.\d+)?(?:\s?[AP]M)?(?:Z|[+-]\d{2}:?\d{2})?(?!\d)", re.I)),
    ("TS", re.compile(r"(?<![a-z0-9])\d{10}(?![a-z0-9])", re.I)),
    ("HEX", re.compile(r"(?<![a-z0-9])(?:0x)?[a-f0-9]{8,}(?![a-z0-9])", re.I)),
]


def mask_items(items):
    counts = {kind: 0 for kind, _ in MASK_PATTERNS}
    def visit(value):
        if isinstance(value, str):
            for kind, pattern in MASK_PATTERNS:
                value, n = pattern.subn("[" + kind + "]", value)
                counts[kind] += n
            return value
        if isinstance(value, list):
            return [visit(x) for x in value]
        if isinstance(value, dict):
            return {k: visit(v) for k, v in value.items()}
        return value
    return visit(items), counts


def stamp():
    return datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()


def load(path):
    return json.loads(path.read_bytes())


def save(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(obj, ensure_ascii=False, indent=2).encode() + b"\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def log(event, **metadata):
    # Callers pass only counts, model names, status codes and costs.
    line = json.dumps({"time": stamp(), "event": event, **metadata})
    print(line, flush=True)
    with (HERE / "progress.jsonl").open("a") as f:
        f.write(line + "\n")


def read_input(cfg):
    raw = (HERE / "items.json").read_bytes()
    if not sha(raw).startswith(cfg["expected_sha256_prefix"]):
        raise RuntimeError("Input hash does not match expected prefix")
    items = json.loads(raw)
    if not isinstance(items, list) or len(items) != cfg["expected_items"]:
        raise RuntimeError("Unexpected input count or format")
    if any(not isinstance(x, dict) or set(x) != FIELDS for x in items):
        raise RuntimeError("Unexpected item fields")
    if any(type(x["id"]) not in (str, int) for x in items):
        raise RuntimeError("Invalid ID type")
    if len({str(x["id"]) for x in items}) != len(items):
        raise RuntimeError("Duplicate IDs")
    return items, sha(raw)


def api_key(cfg):
    # Read only the explicitly authorized key; never source a shell file.
    for line in Path(cfg["env_file"]).read_text().splitlines():
        match = re.match(r"^\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.*)$", line)
        if match:
            parts = shlex.split(match.group(1), comments=True)
            if len(parts) == 1 and parts[0]:
                return parts[0]
    raise RuntimeError("OPENROUTER_API_KEY is missing from the configured .env")


def request(cfg, endpoint, key=None, body=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(cfg["api_base"] + endpoint,
                                 data=None if body is None else encode(body), headers=headers)
    # Never forward credentials to a redirect destination.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    with urllib.request.build_opener(NoRedirect).open(req, timeout=cfg["request_timeout_seconds"]) as resp:
        return json.load(resp)


def prepare(cfg):
    items, digest = read_input(cfg)
    masked, masking_counts = mask_items(items)
    rubric = (HERE / "rubric.txt").read_text().rstrip("\n")
    if (HERE / "manifest.json").exists():
        old = load(HERE / "manifest.json")
        if (old["input_sha256"] != digest or old["rubric"] != rubric or old["settings"] != cfg
                or old.get("masked_input_sha256") != sha(encode(masked))):
            raise RuntimeError("Existing manifest differs; do not overwrite a frozen run")
        return old
    manifest = {
        "created_at": stamp(), "status": "prepared_before_api_calls",
        "input_file": "items.json", "input_sha256": digest, "n_items": len(items),
        "fields": sorted(FIELDS), "rubric": rubric, "rubric_sha256": sha(rubric.encode()),
        "masked_input_sha256": sha(encode(masked)), "masking_counts": masking_counts,
        "masking": {"order": [k for k, _ in MASK_PATTERNS],
                    "patterns": {k: p.pattern for k, p in MASK_PATTERNS},
                    "regex_flags": "IGNORECASE", "scope": "Every string value recursively, including IDs; original IDs used only for local output joins",
                    "serialization": "json.dumps(ensure_ascii=False, sort_keys=True), UTF-8; no newline"},
        "model_list": cfg["models"], "settings": cfg,
        "script_sha256": sha(Path(__file__).read_bytes()), "python_version": sys.version.split()[0],
        "git_hash": None, "git_hash_note": "Standalone run outside a Git repository",
        "majority_rule": "Unique plurality among valid categories; ties or zero valid votes are unresolved",
        "kappa_rule": "Fleiss kappa on complete cases with all judges valid; report excluded item count",
        "blinding": "Only masked input fields and verbatim rubric sent; no hidden key accessed",
        "superseded_unmasked_smoke": cfg.get("superseded_unmasked_smoke"),
        "failure_rule": "No guessed categories; API errors and malformed JSON are invalid; explicit refusals are refused",
        "response_cache": "cache/; includes private raw judge responses; do not display in agent sessions",
    }
    save(HERE / "manifest.json", manifest)
    log("prepared", n_items=len(items), sha256=digest, masking_counts=masking_counts)
    return manifest


def preflight(cfg):
    manifest = prepare(cfg)  # Must precede even the metadata and authentication API calls.
    if manifest["status"] == "frozen":
        log("already_frozen", estimate_usd=manifest["estimate_usd"])
        return
    catalog = request(cfg, "/models")
    catalog = {m["id"]: m for m in catalog["data"]}
    absent = [m for m in cfg["models"] if m not in catalog]
    if absent:
        log("models_unavailable", models=absent)
        raise RuntimeError("Configured models absent from OpenRouter catalog")
    selected = {m: catalog[m] for m in cfg["models"]}
    save(HERE / "model_catalog.json", {"checked_at": stamp(), "models": selected})
    key_status = request(cfg, "/key", api_key(cfg))
    if "data" not in key_status:
        raise RuntimeError("OpenRouter authentication check failed")
    items, _ = read_input(cfg)
    items, _ = mask_items(items)
    estimated_input = sum(len(encode(x)) / 3 + len(manifest["rubric"].encode()) / 3 + 100 for x in items)
    settings, estimate = {}, 0.0
    for model, info in selected.items():
        pricing = info["pricing"]
        rates = {k: float(pricing.get(k, 0)) for k in ("prompt", "completion", "request")}
        if rates["prompt"] <= 0 or rates["completion"] <= 0 or rates["request"] != 0:
            raise RuntimeError("Unexpected pricing; manual review required")
        params = info.get("supported_parameters", [])
        settings[model] = {"rates_usd": rates, "temperature": 0 if "temperature" in params else None,
                           "reasoning": {"effort": cfg["reasoning_effort"]} if "reasoning" in params else None,
                           "json_mode": "response_format" in params}
        cost = estimated_input * rates["prompt"] + len(items) * cfg["expected_output_tokens_per_call"] * rates["completion"]
        estimate += cost
        log("model_preflight", model=model, prices_per_million={k: v * 1e6 for k, v in rates.items()},
            temperature=settings[model]["temperature"], estimate_usd=round(cost, 5))
    manifest.update(status="frozen", frozen_at=stamp(), resolved_models=settings, estimate_usd=estimate,
                    estimated_input_tokens_per_model=estimated_input,
                    estimate_method="UTF-8 bytes / 3 plus 100 framing tokens; 400 output tokens including reasoning per item",
                    budget_method="Reserve UTF-8 prompt bytes + 2048 framing tokens and max_tokens output at pinned maximum rates, with 10% margin. Keep full reservation for unknown charges. No retries.")
    if estimate + cfg.get("prior_spend_usd", 0) >= cfg["budget_usd"]:
        raise RuntimeError("Estimated spend exceeds budget")
    save(HERE / "manifest.json", manifest)
    log("preflight_complete", estimate_usd=round(estimate, 5), budget_usd=cfg["budget_usd"], requests=len(items)*len(cfg["models"]))


def body_for(item, model, cfg, manifest):
    resolved = manifest["resolved_models"][model]
    rates = resolved["rates_usd"]
    body = {"model": model,
            "messages": [{"role": "system", "content": manifest["rubric"]},
                         {"role": "user", "content": json.dumps(item, ensure_ascii=False)}],
            "max_tokens": cfg["max_tokens"], "stream": False,
            "provider": {**cfg["provider"], "max_price": {"prompt": rates["prompt"] * 1e6,
                                                          "completion": rates["completion"] * 1e6}},
            "usage": {"include": True}}
    if resolved["temperature"] is not None:
        body["temperature"] = resolved["temperature"]
    if resolved["reasoning"] is not None:
        body["reasoning"] = resolved["reasoning"]
    if resolved["json_mode"]:
        body["response_format"] = {"type": "json_object"}
    return body


def classify(response):
    if not isinstance(response, dict):
        return "invalid", None, "invalid_response_envelope"
    if response.get("error"):
        return "invalid", None, "api_error_or_no_choices"
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return "invalid", None, "invalid_choices"
    choice = choices[0]
    message = choice.get("message", {})
    if not isinstance(message, dict):
        return "invalid", None, "invalid_message"
    if choice.get("finish_reason") == "content_filter" or message.get("refusal"):
        return "refused", None, "explicit_refusal"
    if choice.get("finish_reason") == "length":
        return "invalid", None, "truncated"
    content = message.get("content")
    if not isinstance(content, str):
        return "invalid", None, "missing_content"
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return "invalid", None, "unparseable_json"
    if (not isinstance(obj, dict) or set(obj) != {"category", "confidence", "reason"}
            or type(obj.get("category")) is not int or obj["category"] not in (1, 2, 3)
            or obj.get("confidence") not in ("high", "medium", "low")
            or not isinstance(obj.get("reason"), str) or len(obj["reason"].split()) > 20):
        return "invalid", None, "schema_violation"
    return obj["category"], obj, None


def plurality(votes):
    counts = Counter(v for v in votes if type(v) is int and v in (1, 2, 3))
    if not counts:
        return "unresolved"
    winners = [v for v, n in counts.items() if n == max(counts.values())]
    return winners[0] if len(winners) == 1 else "unresolved"


def fleiss(rows):
    if not rows:
        return None
    n = len(rows[0])
    if n < 2 or any(len(row) != n for row in rows):
        raise RuntimeError("Invalid kappa matrix")
    counts = [Counter(row) for row in rows]
    observed = sum(sum(v*(v-1) for v in row.values())/(n*(n-1)) for row in counts)/len(rows)
    totals = Counter(v for row in rows for v in row)
    expected = sum((totals[c]/(len(rows)*n))**2 for c in (1, 2, 3))
    return None if math.isclose(expected, 1.0) else (observed-expected)/(1-expected)


def records(cfg, manifest, items):
    """Reject mixed/stale caches before reusing a rating or its cost ledger."""
    masked, counts = mask_items(items)
    if (manifest["settings"] != cfg or counts != manifest["masking_counts"]
            or sha(encode(masked)) != manifest["masked_input_sha256"]):
        raise RuntimeError("Config or masking differs from the frozen run")
    digest = sha((HERE / "manifest.json").read_bytes())
    saved, seen = [], set()
    for path in sorted((HERE / "cache").glob("*.json")):
        record = load(path)
        index, model = record.get("item_index"), record.get("model")
        if (type(index) is not int or not 0 <= index < len(items)
                or not isinstance(model, str) or model not in cfg["models"]):
            raise RuntimeError("Unexpected cached item or model")
        pair = (index, model)
        if pair in seen or record.get("manifest_sha256") != digest:
            raise RuntimeError("Duplicate or foreign cached record")
        expected = sha(encode(body_for(masked[index], model, cfg, manifest)))
        if record.get("request_sha256") != expected:
            raise RuntimeError("Cached request differs from the frozen run")
        cost, reserved = record.get("cost_usd"), record.get("reserved_usd")
        if (type(reserved) not in (int, float) or not math.isfinite(reserved) or reserved < 0
                or (cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0))):
            raise RuntimeError("Invalid cached cost ledger")
        label = record.get("label")
        valid_label = (type(label) is int and label in (1, 2, 3)) or label in ("invalid", "refused")
        if not valid_label or record.get("state") not in ("pending", "completed"):
            raise RuntimeError("Invalid cached result")
        if record["state"] == "pending" and (label != "invalid" or cost is not None):
            raise RuntimeError("Pending request has an inconsistent result")
        seen.add(pair)
        saved.append(record)
    return saved


def summarize(cfg):
    manifest = load(HERE / "manifest.json")
    items, digest = read_input(cfg)
    if digest != manifest["input_sha256"]:
        raise RuntimeError("Input changed after freezing")
    saved = records(cfg, manifest, items)
    lookup = {(r["item_index"], r["model"]): r for r in saved}
    rows, complete, per_judge = [], [], {m: Counter() for m in cfg["models"]}
    invalid_reasons = {m: Counter() for m in cfg["models"]}
    unattempted = 0
    for index, item in enumerate(items):
        row, votes = {"id": item["id"]}, []
        for model in cfg["models"]:
            record = lookup.get((index, model))
            vote = record["label"] if record else "invalid"
            unattempted += record is None
            row[model] = vote
            votes.append(vote)
            per_judge[model][str(vote)] += 1
            if vote == "invalid":
                invalid_reasons[model][record["failure"] if record else "not_attempted"] += 1
        valid = [v for v in votes if type(v) is int]
        row.update(majority=plurality(votes), n_valid=len(valid), n_refused=votes.count("refused"))
        rows.append(row)
        if len(valid) == len(cfg["models"]):
            complete.append(valid)
    temporary = HERE / "labels.csv.tmp"
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", *cfg["models"], "majority", "n_valid", "n_refused"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(HERE / "labels.csv")
    known_cost = sum(r.get("cost_usd") or 0 for r in saved)
    unknown = [r for r in saved if r.get("cost_usd") is None]
    prior_spend = cfg.get("prior_spend_usd", 0)
    safe_cost = prior_spend + known_cost + sum(r["reserved_usd"] for r in unknown)
    majority = Counter(str(r["majority"]) for r in rows)
    summary = {
        "created_at": stamp(), "input_sha256": digest, "n_items": len(items),
        "models": cfg["models"], "majority_counts": {c: majority[c] for c in ("1", "2", "3", "unresolved")},
        "per_judge_counts": {m: {c: counts[c] for c in ("1", "2", "3", "refused", "invalid")} for m, counts in per_judge.items()},
        "fleiss_kappa": fleiss(complete), "kappa_n_items": len(complete),
        "kappa_excluded_items": len(items)-len(complete),
        "kappa_method": manifest["kappa_rule"], "majority_rule": manifest["majority_rule"],
        "refusals_per_judge": {m: c["refused"] for m, c in per_judge.items()},
        "invalid_reasons_per_judge": invalid_reasons,
        "n_completed_requests": sum(r["state"] == "completed" for r in saved),
        "n_unattempted_requests": unattempted, "n_unknown_cost_requests": len(unknown),
        "masking_counts": manifest["masking_counts"], "masked_input_sha256": manifest["masked_input_sha256"],
        "masked_panel_cost_usd": known_cost if not unknown else None,
        "superseded_unmasked_smoke_cost_usd": prior_spend,
        "total_cost_usd": prior_spend + known_cost if not unknown else None,
        "known_cost_usd": prior_spend + known_cost, "cost_upper_bound_usd": safe_cost,
        "budget_usd": cfg["budget_usd"],
    }
    save(HERE / "summary.json", summary)
    return summary


async def run(cfg, limit):
    manifest = prepare(cfg)
    if manifest["status"] != "frozen":
        raise RuntimeError("Run preflight before inference")
    if manifest["script_sha256"] != sha(Path(__file__).read_bytes()):
        raise RuntimeError("Script changed after manifest freeze")
    original_items, _ = read_input(cfg)
    items, counts = mask_items(original_items)
    if sha(encode(items)) != manifest["masked_input_sha256"] or counts != manifest["masking_counts"]:
        raise RuntimeError("Masking changed after freeze")
    key = api_key(cfg)
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True, mode=0o700)
    saved = records(cfg, manifest, original_items)
    prior_pairs = {(r["item_index"], r["model"]) for r in saved}
    manifest_digest = sha((HERE / "manifest.json").read_bytes())
    spent = cfg.get("prior_spend_usd", 0) + sum(r["cost_usd"] if r.get("cost_usd") is not None else r["reserved_usd"] for r in saved)
    live_reserved, done, stopped = 0.0, 0, False
    condition = asyncio.Condition()
    semaphore = asyncio.Semaphore(cfg["concurrency"])

    async def one(index, item, model):
        nonlocal spent, live_reserved, done, stopped
        body = body_for(item, model, cfg, manifest)
        fingerprint = sha(encode(body))
        # Distinct items can become identical after masking, including their IDs.
        identity = {"manifest": manifest_digest, "item_index": index, "model": model,
                    "request": fingerprint}
        path = cache / (sha(encode(identity)) + ".json")
        if (index, model) in prior_pairs:
            return
        rates = manifest["resolved_models"][model]["rates_usd"]
        input_bound = sum(len(m["content"].encode()) for m in body["messages"]) + 2048
        reserved = 1.1 * (input_bound * rates["prompt"] + cfg["max_tokens"] * rates["completion"])
        async with semaphore:
            async with condition:
                while spent + live_reserved + reserved > cfg["budget_usd"]:
                    if stopped or live_reserved < 1e-10:
                        stopped = True
                        return
                    await condition.wait()
                if stopped:
                    return
                live_reserved += reserved
            started = time.monotonic()
            record = {"item_index": index, "model": model, "request_sha256": fingerprint,
                      "manifest_sha256": manifest_digest,
                      "started_at": stamp(), "reserved_usd": reserved, "state": "pending",
                      "label": "invalid", "failure": "interrupted_or_unknown", "cost_usd": None}
            # Persist before dispatch; ambiguous requests are never silently retried.
            save(path, record)
            response = None
            try:
                response = await asyncio.to_thread(request, cfg, "/chat/completions", key, body)
            except urllib.error.HTTPError as e:
                record["failure"] = "http_" + str(e.code)
                # Status is safe to log; error bodies can echo inputs and stay private.
                record["raw_error"] = e.read().decode("utf-8", errors="replace")
                if e.code in (400, 401, 402, 403, 404, 422, 429):
                    record["cost_usd"] = 0.0
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                record["failure"] = type(e).__name__
            except json.JSONDecodeError:
                record["failure"] = "invalid_http_json"
            if response is not None:
                record["response"] = response
                label, parsed, failure = classify(response)
                record.update(label=label, parsed=parsed, failure=failure)
                envelope = response if isinstance(response, dict) else {}
                usage = envelope.get("usage")
                cost = usage.get("cost") if isinstance(usage, dict) else None
                if type(cost) in (float, int) and math.isfinite(cost) and cost >= 0:
                    record["cost_usd"] = cost
                if envelope.get("model") != model and not str(envelope.get("model", "")).startswith(model + "-"):
                    record.update(label="invalid", failure="unexpected_response_model")
            record.update(state="completed", completed_at=stamp(), elapsed_seconds=time.monotonic()-started)
            save(path, record)
            with (HERE / "ratings.jsonl").open("a") as f:
                # Labels and metadata only; raw responses remain in the private cache.
                f.write(json.dumps({k: v for k, v in record.items() if k not in ("response", "raw_error", "parsed")}) + "\n")
            async with condition:
                live_reserved -= reserved
                charged = record["cost_usd"] if record["cost_usd"] is not None else reserved
                spent += charged
                done += 1
                if charged > reserved or spent >= cfg["budget_usd"]:
                    stopped = True
                condition.notify_all()
            if done % 12 == 0 or record["label"] in ("invalid", "refused"):
                log("progress", completed_this_run=done, accounted_usd=round(spent, 6),
                    last_model=model, last_status=record["label"], failure=record["failure"])

    selected = items[:limit] if limit else items
    log("run_started", items_this_run=len(selected), concurrency=cfg["concurrency"], previously_accounted_usd=spent)
    await asyncio.gather(*(one(i, item, model) for i, item in enumerate(selected) for model in cfg["models"]))
    summary = summarize(cfg)
    log("run_finished", completed_requests=summary["n_completed_requests"],
        total_cost_usd=summary["total_cost_usd"], cost_upper_bound_usd=summary["cost_upper_bound_usd"],
        budget_stopped=stopped, unattempted_requests=summary["n_unattempted_requests"])


def self_test():
    raw = [{"id": "abc12345", "nested": ["2025-09-14T14:03:22Z Sep 14 14 Sep 2025 09/14/2025 1694961000 01234567-89ab-cdef-0123-456789abcdef deadbeef 0x0123456789abcdef"]}]
    masked, counts = mask_items(raw)
    assert counts == {"DATE": 4, "TIME": 1, "UUID": 1, "HEX": 3, "TS": 1}
    assert masked[0]["id"] == "[HEX]"
    assert raw[0]["id"] == "abc12345"
    assert mask_items(masked)[0] == masked
    assert all(n == 0 for n in mask_items(masked)[1].values())
    response = {"model": "test", "choices": [{"finish_reason": "stop", "message": {
        "content": '{"category": 1, "confidence": "high", "reason": "Synthetic fixture"}'}}]}
    assert classify(response)[0] == 1
    response["choices"][0]["message"]["content"] = '{"category":true,"confidence":"high","reason":"x"}'
    assert classify(response)[0] == "invalid"
    response["choices"][0]["message"]["refusal"] = "fixture"
    assert classify(response)[0] == "refused"
    assert classify({"error": {"code": 500}})[0] == "invalid"
    assert plurality([1, 2, 3]) == "unresolved"
    assert plurality([1, 1, "invalid"]) == 1
    assert plurality(["invalid", "refused", "invalid"]) == "unresolved"
    assert math.isclose(fleiss([[1,1,1], [2,2,2], [3,3,3]]), 1)
    assert math.isclose(fleiss([[1,2,3], [1,2,3]]), -0.5)
    assert fleiss([[1,1,1]]) is None
    print("Synthetic masking, parsing, refusal, plurality and kappa checks passed")


def main():
    global HERE
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "preflight", "run", "summarize", "self-test"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, help="Private directory containing items.json, config.json and rubric.txt")
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return
    if args.run_dir is None or not args.run_dir.is_dir():
        parser.error("--run-dir must name an existing private run directory")
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    HERE = args.run_dir.resolve()
    os.chdir(HERE)
    cfg = load(HERE / "config.json")
    with (HERE / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "prepare":
            prepare(cfg)
        elif args.command == "preflight":
            preflight(cfg)
        elif args.command == "run":
            asyncio.run(run(cfg, args.limit))
        else:
            s = summarize(cfg)
            log("summary_written", completed_requests=s["n_completed_requests"], total_cost_usd=s["total_cost_usd"])


def cli():
    # Tracebacks and exception messages can contain input text. Print type only.
    try:
        main()
    except Exception as exc:
        print(json.dumps({"fatal_error_type": type(exc).__name__}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
