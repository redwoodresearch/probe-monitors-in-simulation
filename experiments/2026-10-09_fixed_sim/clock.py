"""R6, the clock: move a simulated run's "now" timestamps into the real runs' clock, by one offset per run.

The simulator's clock reads its own time (about 2026-09-18 10:00), while real epoch-1 runs read theirs. Per simulated
run, one offset moves every clock-like timestamp so the run's "now" lands on a donor's "now". The donor is a real
epoch-1 run from the 5 dev environments only (Aswin, 2026-10-09), drawn by a seed fixed per cell, never the same cell.

Scope, frozen with the rule list:
- Touched: full datetimes in four formats, matched in this priority order, without overlaps.
  - DATE_CMD: `date`'s default output, `Thu Sep 18 10:23:45 UTC 2026`, with a zone name or a numeric zone (`+0530`,
    `+03`); any other zone token reserves the stamp and fails to parse.
  - ISO: `2026-09-18T10:23:45`, `2026-09-18 10:23`, with optional seconds, fraction, `Z` or `+HH:MM` kept as written.
  - HTTP: `Thu, 18 Sep 2026 10:23:45 GMT`.
  - LS: `ls -l` / syslog `Sep 18 10:23` or `Sep 18 10:23:45`; the year is the run's "now" year.
- Shifted only if within 48 h of the run's "now": the median of its timestamps in the most common date-hour (from
  its LS stamps, dated 2026, when it has no full-dated one), counting only stamps dated September 14-21 of any year
  (v2, plan amendment 1: every cohort ran that week; other dates are data, such as fixtures and file times). Stamps
  are compared in UTC (a stamp without a zone is taken as UTC); the shifted stamp keeps its own zone and fraction.
- Never touched (counted instead): any timestamp outside that window (data dates such as 2025 rows), date-only and
  time-only strings, epoch seconds, and stamps that fail to parse (an impossible date, a zone name not in ZONES, a
  date at the calendar's edge). Weekday names are recomputed for shifted dates; offsets are whole seconds.
"""

import random
import re
from collections import Counter
from datetime import datetime, timedelta

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MON, DAY = "|".join(MONTHS), "|".join(DAYS)
WINDOW = timedelta(hours=48)
PATTERNS = [
    ("DATE_CMD", re.compile(rf"\b({DAY}) ({MON}) ( ?\d{{1,2}}) (\d{{2}}):(\d{{2}}):(\d{{2}}) (\S+) (\d{{4}})(?!-\d)\b")),  # any zone token; a year, not an ISO date
    # never a partial match: no digit, `:digit`, `.digit` or `±digit` may follow (`10:23:456` and `10:23:45+05300` are
    # left alone, not cut short)
    ("ISO", re.compile(r"(?<![\d-])(\d{4})-(\d{2})-(\d{2})([T ])(\d{2}):(\d{2})(?::(\d{2}))?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?(?![\dZ]|[:.+-]\d)")),
    ("HTTP", re.compile(rf"\b({DAY}), (\d{{2}}) ({MON}) (\d{{4}}) (\d{{2}}):(\d{{2}}):(\d{{2}}) GMT\b")),
    ("LS", re.compile(rf"\b({MON}) ( ?\d{{1,2}}) (\d{{2}}):(\d{{2}})(?::(\d{{2}}))?(?![\d:])")),
]


def stamps(text: str, year: int | None = None) -> list[dict]:
    """Non-overlapping clock-like timestamps in `text`, in priority order: {kind, start, end, dt, m}. LS stamps need the
    year (`year`); without it they get year 1900 and are placed by the caller. Unparseable ones are kind `FAILED`."""
    taken, out = [], []
    for kind, rx in PATTERNS:
        for m in rx.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))  # a failed match reserves its span too, so nothing inside it shifts
            g = m.groups()
            try:
                if kind == "DATE_CMD":
                    dt = datetime(int(g[7]), MONTHS.index(g[1]) + 1, int(g[2]), int(g[3]), int(g[4]), int(g[5]))
                elif kind == "ISO":
                    frac = int(round(float(g[7]) * 1e6)) if g[7] else 0
                    dt = datetime(int(g[0]), int(g[1]), int(g[2]), int(g[4]), int(g[5]), int(g[6] or 0), min(frac, 999999))
                elif kind == "HTTP":
                    dt = datetime(int(g[3]), MONTHS.index(g[2]) + 1, int(g[1]), int(g[4]), int(g[5]), int(g[6]))
                else:
                    dt = datetime(year or 1900, MONTHS.index(g[0]) + 1, int(g[1]), int(g[2]), int(g[3]), int(g[4] or 0))
            except ValueError:
                out.append({"kind": "FAILED", "start": m.start(), "end": m.end(), "dt": None, "utc": None, "m": m})
                continue
            try:
                utc = dt - zone_of(kind, g)
            except (KeyError, ValueError, OverflowError):  # unknown zone, invalid offset, calendar's edge: never shifted
                out.append({"kind": "FAILED", "start": m.start(), "end": m.end(), "dt": None, "utc": None, "m": m})
                continue
            out.append({"kind": kind, "start": m.start(), "end": m.end(), "dt": dt, "utc": utc, "m": m})
    return sorted(out, key=lambda s: s["start"])


# zone names `date` prints, unambiguous ones only (CST and IST each name two zones, so they stay unknown)
ZONES = {"UTC": 0, "GMT": 0, "PST": -8, "PDT": -7, "MST": -7, "MDT": -6, "CDT": -5, "EST": -5, "EDT": -4, "CET": 1,
         "CEST": 2, "JST": 9}


def zone_of(kind: str, g: tuple) -> timedelta:
    """The stamp's offset from UTC; a stamp without a zone is taken as UTC (the containers' default). A zone name not in
    ZONES raises KeyError (the stamp is then never shifted)."""
    z = g[8] if kind == "ISO" else g[6] if kind == "DATE_CMD" else None
    if not z or z == "Z":
        return timedelta(0)
    if z in ZONES:
        return timedelta(hours=ZONES[z])
    m = re.fullmatch(r"([+-])(\d{2})(?::?(\d{2}))?", z)  # +05:30, +0530, or +03 as `date` prints a zone with no name
    if not m:
        raise KeyError(z)
    hh, mm = int(m.group(2)), int(m.group(3) or 0)
    if hh > 14 or mm > 59:
        raise ValueError(f"invalid zone offset {z}")
    return (1 if m.group(1) == "+" else -1) * timedelta(hours=hh, minutes=mm)


DATE_ONLY = re.compile(r"(?<![\d-])\d{4}-\d{2}-\d{2}(?![\d:T])")
TIME_ONLY = re.compile(r"(?<![\d:])\d{2}:\d{2}:\d{2}(?![\d:])")
EPOCH = re.compile(r"(?<!\d)1[67]\d{8}(?!\d)")


def untouched_counts(text: str, spans: list[tuple[int, int]]) -> Counter:
    """Date-only, time-only and epoch strings outside any matched stamp: never shifted, counted."""
    c = Counter()
    for name, rx in (("date_only", DATE_ONLY), ("time_only", TIME_ONLY), ("epoch_s", EPOCH)):
        c[f"never_touched:{name}"] += sum(not any(s <= m.start() < e for s, e in spans) for m in rx.finditer(text))
    return c


LS_YEAR = 2026  # the year every real and simulated run ran in (2026-09), for runs whose only clock is `ls -l`
RUN_WEEK = (9, 14, 21)  # month, first and last day every cohort ran (the simulator's clock is sometimes a year off)


def in_run_week(d: datetime) -> bool:
    return d.month == RUN_WEEK[0] and RUN_WEEK[1] <= d.day <= RUN_WEEK[2]


def now_of(texts: list[str]) -> datetime | None:
    """A run's "now": the median of its full-dated timestamps (DATE_CMD, ISO, HTTP) in the most common date-hour; for a
    run with none, the same over its `ls -l` / syslog times (LS), dated in LS_YEAR. Only stamps in the run week count
    (in UTC); a run with none has no "now" and is not shifted."""
    found = [s for t in texts for s in stamps(t, LS_YEAR) if s["kind"] != "FAILED" and in_run_week(s["utc"])]
    dts = [s["utc"] for s in found if s["kind"] in ("DATE_CMD", "ISO", "HTTP")] or [s["utc"] for s in found if s["kind"] == "LS"]
    if not dts:
        return None
    bucket = lambda d: d.replace(minute=0, second=0, microsecond=0)
    hour = Counter(map(bucket, dts)).most_common(1)[0][0]
    inside = sorted(d for d in dts if bucket(d) == hour)  # the timestamps in that hour only
    return inside[len(inside) // 2]


def render(s: dict, dt: datetime) -> str:
    """The timestamp `s` written again at `dt`, in its own format (padding, seconds, fraction and zone kept)."""
    g, kind = s["m"].groups(), s["kind"]
    if kind == "DATE_CMD":
        day = pad_day(g[2], dt.day)
        return f"{DAYS[dt.weekday()]} {MONTHS[dt.month - 1]} {day} {dt:%H:%M:%S} {g[6]} {dt.year}"
    if kind == "ISO":
        sec = f":{dt.second:02d}" if g[6] is not None else ""
        # offsets are whole seconds, so the fraction is written back exactly as it was
        return f"{dt:%Y-%m-%d}{g[3]}{dt:%H:%M}{sec}{g[7] or ''}{g[8] or ''}"
    if kind == "HTTP":
        return f"{DAYS[dt.weekday()]}, {dt.day:02d} {MONTHS[dt.month - 1]} {dt.year} {dt:%H:%M:%S} GMT"
    day = pad_day(g[1], dt.day)
    sec = f":{dt.second:02d}" if g[4] is not None else ""
    return f"{MONTHS[dt.month - 1]} {day} {dt:%H:%M}{sec}"


def pad_day(orig: str, day: int) -> str:
    """The day written as the original was: zero-padded (`09`) stays zero-padded, space-padded (` 9`, as `ls` and
    `date` write it) stays right-aligned to the original's width, and an unpadded day stays unpadded."""
    if orig.startswith("0"):
        return f"{day:0{len(orig)}d}"
    if orig.startswith(" "):
        return f"{day:>{len(orig)}}"
    return str(day)


def shift_text(text: str, offset: timedelta, now: datetime) -> tuple[str, Counter]:
    """`text` with every clock-like timestamp within WINDOW of `now` moved by `offset` (rounded to whole seconds);
    counts by outcome."""
    offset = timedelta(seconds=round(offset.total_seconds()))
    c, parts, last = Counter(), [], 0
    found = stamps(text, now.year)
    c.update(untouched_counts(text, [(s["start"], s["end"]) for s in found]))
    for s in found:
        if s["kind"] == "FAILED":
            c["parse_failed"] += 1
            continue
        if abs(s["utc"] - now) > WINDOW:
            c[f"outside_window:{s['kind']}"] += 1
            continue
        if not offset:  # nothing moves: the original text stays byte-identical (a wrong weekday is not corrected)
            continue
        try:
            new = render(s, s["dt"] + offset)
        except OverflowError:
            c["shift_overflow"] += 1
            continue
        parts += [text[last:s["start"]], new]
        last = s["end"]
        c[f"shifted:{s['kind']}"] += 1
    return "".join(parts) + text[last:], c


def donor_for(cell: str, donors: dict[str, datetime]) -> tuple[str, datetime]:
    """A real dev-environment donor for a simulated cell: seeded by the cell, never the same cell."""
    pool = sorted(k for k in donors if k != cell)
    key = random.Random(f"fixed-sim-clock:{cell}").choice(pool)
    return key, donors[key]
