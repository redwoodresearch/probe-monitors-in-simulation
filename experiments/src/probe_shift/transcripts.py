"""Load LinuxArena samples from the sims archive and normalize them into what the reader model sees.

Cohorts are cited by their alias in simulated-control-evals' `real_runs.yml`, never by bare S3 URL.
"""

import io
import json
import os
import re
import time
import zipfile
from functools import cache
from pathlib import Path

import boto3
import yaml

AWS_PROFILE = "redwood-sandbox"


@cache
def _s3():
    from botocore.config import Config

    # Many files are read in parallel from a laptop: adaptive retries and a longer read timeout.
    return boto3.Session(profile_name=AWS_PROFILE).client("s3", config=Config(retries={"max_attempts": 10, "mode": "adaptive"},
                                                                              read_timeout=180))


CACHE = Path.home() / ".cache" / "probe-shift" / "s3"  # eval logs are immutable once written: a local copy is reused


def _get(bucket: str, key: str) -> bytes:
    """An object's bytes, from the local cache or S3, retrying a timed-out body read (botocore retries the request, not
    the streamed body)."""
    from botocore.exceptions import ConnectionError as BotoConnectionError, ReadTimeoutError

    local = CACHE / bucket / key
    if local.exists():
        return local.read_bytes()
    for attempt in range(5):
        try:
            body = _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
            break
        except (ReadTimeoutError, BotoConnectionError):
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(f".{local.name}.{os.getpid()}.tmp")
    tmp.write_bytes(body)
    os.replace(tmp, local)
    return body


@cache
def cohort_uri(alias: str) -> str:
    """The S3 prefix of a cohort: its alias in simulated-control-evals' real_runs.yml, a project cohort's name in
    `probe_shift.cohorts.COHORTS` (a batch not registered there yet), or an s3:// prefix itself."""
    if alias.startswith("s3://"):
        return alias.rstrip("/") + "/"
    from probe_shift.cohorts import COHORTS  # here, not at the top: cohorts imports this module

    if alias in COHORTS:
        return COHORTS[alias].rstrip("/") + "/"
    import paper  # from simulated-control-evals

    runs = yaml.safe_load((Path(paper.__file__).parent / "configs" / "real_runs.yml").read_text())
    return runs["runs"][alias]["uri"]


ATTACHMENT = "attachment://"
REF = re.compile(r"attachment://[0-9a-f]{32}")  # Inspect's reference: the whole value is the scheme and a 32-hex hash


def refs_left(x) -> bool:
    """Whether any string value at any depth is a whole reference. Text that only mentions the scheme (a command that
    greps for it, say) is not one."""
    if isinstance(x, str):
        return bool(REF.fullmatch(x))
    if isinstance(x, (list, dict)):
        return any(map(refs_left, x.values() if isinstance(x, dict) else x))
    return False


def resolve_attachments(sample: dict) -> dict:
    """The sample with every `attachment://<hash>` value in its messages replaced by its text from the sample's
    `attachments` map (Inspect moves long content there). Raises if a reference has no entry, or if a whole reference
    is still in the messages afterwards (a chained one), so a placeholder never reaches the reader."""
    attachments = sample.get("attachments") or {}

    def resolve(x):
        if isinstance(x, str) and REF.fullmatch(x):
            if x[len(ATTACHMENT):] not in attachments:
                raise ValueError(f"unresolved {x[:60]} in sample {sample.get('id')} epoch {sample.get('epoch')}")
            return attachments[x[len(ATTACHMENT):]]
        if isinstance(x, list):
            return [resolve(v) for v in x]
        if isinstance(x, dict):
            return {k: resolve(v) for k, v in x.items()}
        return x

    messages = resolve(sample["messages"])
    if refs_left(messages):  # a chained reference: the replacement text was itself a reference
        raise ValueError(f"{ATTACHMENT} reference left in the messages of sample {sample.get('id')} epoch {sample.get('epoch')}")
    return sample | {"messages": messages}


def load_sample(alias: str, file: str, sample_id: str, epoch: int) -> dict:
    bucket, prefix = cohort_uri(alias).removeprefix("s3://").split("/", 1)
    body = _s3().get_object(Bucket=bucket, Key=f"{prefix}logs/{file}")["Body"].read()
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                sample = json.loads(z.read(name))
                if sample["id"] == sample_id and sample["epoch"] == epoch:
                    return resolve_attachments(sample)
    raise KeyError(f"sample {sample_id} epoch {epoch} not in {alias}/{file}")


def eval_files(alias: str) -> list[str]:
    bucket, prefix = cohort_uri(alias).removeprefix("s3://").split("/", 1)
    pages = _s3().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}logs/")
    return [o["Key"].rsplit("/", 1)[-1] for p in pages for o in p.get("Contents", []) if o["Key"].endswith(".eval")]


def file_samples(alias: str, file: str) -> list[dict]:
    bucket, prefix = cohort_uri(alias).removeprefix("s3://").split("/", 1)
    body = _get(bucket, f"{prefix}logs/{file}")
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        return [resolve_attachments(json.loads(z.read(n))) for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]


def main_task_id(sample: dict) -> str:
    mt = (sample.get("metadata") or {}).get("main_task")
    return mt.get("id") if isinstance(mt, dict) else mt


def visible_text(content) -> str:
    """Text the reader can see. Reasoning parts are encrypted provider blobs, so they are dropped."""
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text") or "" for c in content or [] if c.get("type") == "text")


def system_prompt(sample: dict) -> str:
    return "\n".join(visible_text(m["content"]) for m in sample["messages"] if m["role"] == "system")


# The runner's end-of-sample call and the react loop's nudge, as in simulated-control-evals
# `sim.adapters.inspect_eval` (RUNNER_SUBMIT, REACT_NUDGE). Submit arguments carry the attacker's own
# `attack_successful` / `attack_description`: a label, not behavior.
RUNNER_SUBMIT = "submit"
REACT_NUDGE = "Please proceed to the next step using your best judgement"


def tool_errors(sample: dict) -> list[dict]:
    """Per tool output, indexed exactly as `reader_messages` and `reader.render` index them (step: agent turn after
    submit calls and empty turns are dropped; action: its k-th tool output), the error type if the call errored."""
    out, step, action, submit_ids = [], -1, 0, set()
    for m in sample["messages"]:
        if m["role"] == "assistant":
            calls = [tc for tc in m.get("tool_calls") or [] if tc["function"] != RUNNER_SUBMIT]
            submit_ids |= {tc.get("id") for tc in m.get("tool_calls") or [] if tc["function"] == RUNNER_SUBMIT}
            if calls or visible_text(m["content"]).strip():
                step, action = step + 1, 0
        elif m["role"] == "tool":
            if m.get("tool_call_id") in submit_ids or m.get("function") == RUNNER_SUBMIT:
                continue
            error = m.get("error")
            out.append({"step": step, "action": action, "tool": m.get("function"),
                        "error": (error.get("type") or "unknown") if error else None})
            action += 1
    return out


def tool_result_text(message: dict) -> str:
    """What the agent saw for a tool call: the error message when the call errored (Inspect serves it in
    place of any partial output), else the output. Same rule as sce's `tool_result_text`."""
    text = visible_text(message.get("content"))
    error = message.get("error")
    return (error.get("message") or text) if error else text


def submit_calls(sample: dict) -> int:
    return sum(tc["function"] == RUNNER_SUBMIT for m in sample["messages"] if m["role"] == "assistant"
               for tc in m.get("tool_calls") or [])


def invalidity(sample: dict) -> str | None:
    """`errored`, `stalled` or `no_actions`, first match wins, else None. Mirrors sce's
    `metrics.analysis.analysis_set.invalidity` without the side-task gate (failed attacks are kept)."""
    if sample.get("error"):
        return "errored"
    messages = sample["messages"]

    def degenerate(m):
        empty_completion = m["role"] == "assistant" and not visible_text(m.get("content")).strip() and not m.get("tool_calls")
        nudge = m["role"] == "user" and REACT_NUDGE in visible_text(m.get("content"))
        return empty_completion or nudge

    if messages and degenerate(messages[-1]):
        return "stalled"
    actions = [tc for m in messages if m["role"] == "assistant" for tc in m.get("tool_calls") or [] if tc["function"] != RUNNER_SUBMIT]
    if not actions:
        return "no_actions"
    return None


def reader_messages(sample: dict, system: str) -> list[dict]:
    """Chat messages for the reader, serialized the same way for real and simulated samples.

    - `system` replaces the sample's own system prompt (attack runs get the honest prompt).
    - Tool outputs are what the agent saw (`tool_result_text`); the error field itself is dropped.
    - Tool outputs keep their trailing newlines, as the agent saw them.
    - The runner's `submit` calls and their results are dropped, and so is an agent turn left with neither
      text nor calls. Count them first with `submit_calls`.
    - `tool_name` on tool messages is metadata for span labels; the chat template does not render it.
    """
    # the reader boundary, checked before text parts are joined: also catches samples read without resolve_attachments
    if refs_left(sample["messages"]) or refs_left(system):
        raise ValueError(f"{ATTACHMENT} reference in the reader messages of sample {sample.get('id')} epoch {sample.get('epoch')}")
    out = [{"role": "system", "content": system}]
    submit_ids = set()
    for m in sample["messages"]:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": visible_text(m["content"])})
        elif role == "assistant":
            calls = []
            for tc in m.get("tool_calls") or []:
                if tc["function"] == RUNNER_SUBMIT:
                    submit_ids.add(tc.get("id"))
                    continue
                calls.append({"type": "function", "function": {"name": tc["function"], "arguments": tc.get("arguments") or {}}})
            text = visible_text(m["content"])
            if calls or text.strip():
                out.append({"role": "assistant", "content": text, "tool_calls": calls})
        elif role == "tool":
            if m.get("tool_call_id") in submit_ids or m.get("function") == RUNNER_SUBMIT:
                continue
            out.append({"role": "tool", "content": tool_result_text(m), "tool_name": m.get("function")})
    return out
