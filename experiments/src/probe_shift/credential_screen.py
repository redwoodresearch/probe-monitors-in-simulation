"""Local, text-free screening count. Never print a match or a source output."""

import re

VERSION = 1
KEY = re.compile(r"(?i)(?:password|passwd|(?:^|[_.-])pwd$|secret|token|api[_-]?key|access[_-]?key)")
NONVALUE_KEY = re.compile(r"(?i)(?:hash|type|count|length|size|name|file|path|expires|expiry|expired|enabled|present|valid|id_token_hint|tokenizer|tokens|token_limit|max_tokens|token_usage|token_budget|secret_ref)(?:$|_)")
ASSIGN = re.compile(
    r'''(?m)(?<![\w.])(?P<name>[A-Za-z_][\w.-]{0,100})["']?\s*[:=]\s*'''
    r'''(?P<value>"[^"\n]*"|'[^'\n]*'|[^\s,;\]}#]+)'''
)
AUTH = re.compile(r"(?i)\b(?:authorization|proxy-authorization)[\"']?\s*[:=]\s*[\"']?(Bearer|Basic)\s+([^\s\"',;]+)")
URL = re.compile(r"(?i)\b(?:[a-z][a-z0-9+.-]*://)([^\s/@:'\"]+):([^\s/@'\"]+)@[^\s/'\"]+")
PEM = re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----\s*[A-Za-z0-9+/=]{16,}")
FORMAT = re.compile(r"\b(?:A[KS]IA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|sk_(?:live|test)_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9_-]{20,}|eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,})\b")
PLACEHOLDER = re.compile(r"(?i)(?:password|changeme|change[-_]?me|replace[-_]?me|example|dummy|test|fake|sample|placeholder|your[-_]|secret123|abc123|123456|qwerty|letmein)")
EMPTY = {"", "none", "null", "undefined", "true", "false", "redacted", "[redacted]", "<redacted>", "masked", "[masked]", "<masked>", "..."}
DYNAMIC = re.compile(r"\$|\{\{|\}|\b(?:process\.env|os\.environ|os\.getenv|getenv|getpass|self\.|request\.|config\.|secrets\.)|[()]|\[")


def literal(value: str) -> str | None:
    value = value.strip()
    quoted = len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]
    if quoted:
        value = value[1:-1]
    if value.lower() in EMPTY or re.fullmatch(r"[xX*•.]+", value):
        return None
    if DYNAMIC.search(value) or "\\n" in value:
        return None
    if not quoted and not re.fullmatch(r"[A-Za-z0-9_./:@%+=!~<>?-]+", value):
        return None
    if not quoted and value in {'str', 'string', 'bytes', 'String', 'Optional', 'int', 'Any', 'number'}:
        return None
    return value


def detect(text: str) -> tuple[int, int, int]:
    """Return credential, numeric category bitmask, placeholder flag; no matched data escapes."""
    bits = placeholder = 0
    for match in ASSIGN.finditer(text):
        name = match['name']
        if not KEY.search(name) or NONVALUE_KEY.search(name):
            continue
        value = literal(match['value'])
        if value is not None:
            bits |= 1 if re.search(r"(?i)password|passwd|pwd", name) else 2
            placeholder |= bool(PLACEHOLDER.search(value))
    for match in AUTH.finditer(text):
        value = literal(match[2])
        if value is not None:
            bits |= 2
            placeholder |= bool(PLACEHOLDER.search(value))
    for match in URL.finditer(text):
        value = literal(match[2])
        if value is not None:
            bits |= 8
            placeholder |= bool(PLACEHOLDER.search(value))
    if PEM.search(text):
        bits |= 4
    for match in FORMAT.finditer(text):
        bits |= 2
        placeholder |= bool(PLACEHOLDER.search(match[0]))
    return int(bool(bits)), bits, int(placeholder)
