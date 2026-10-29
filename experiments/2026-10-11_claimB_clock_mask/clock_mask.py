"""The clock mask (EXPLORATORY; Aswin about 21:20 PT 10-11 via 0d): on top of level 1, hide the clock strings level 1
leaves, which rule R6 also moves. Applied to tool-output text only, the same way in every arm (fixed sim, real e1,
real e2, planted copies), after `probe_shift.scrub.scrub(text, 1)`.

What level 1 leaves and R6 changes (source audit follow-up 1, `r6_level1_POSTHOC_2026-10-11_210109.json` on
aswin/claimb-source-audit-c: 248 weekday, 205 day-number, 15 year and 1 other changed tokens survive level 1):
- day-month-year dates, the HTTP form `Thu, 18 Sep 2026 10:23:45 GMT` (level 1 masks `Sep 18`, not `18 Sep`):
  the whole `18 Sep 2026` becomes `<date>`, for any 4-digit year;
- a 4-digit year right after a masked time or date, with an optional zone between (a 1-5 letter name such as `UTC` or
  `Z`, or a signed `+03` / `+0300` offset; the `date` command and ctime forms, `Thu Sep 18 10:23:45 UTC 2026` ->
  level 1 `Thu <date> <time> UTC 2026`): the year becomes `<year>`;
- a weekday name (`scrub.WEEKDAYS`, full and short forms) right before a masked date, with an optional comma:
  `<wday>`.
- (added 23:2x PT after the mask check hid 175 of 176: `analyze_POSTHOC_mask_leftover_modal.py` found an ISO stamp
  followed directly by `:`) an ISO time right after `<date>T` or `<date> ` and followed by `:`, which level 1's time
  pattern refuses (its `(?![\d:])` lookahead), but not by `:` and a digit (IPv6-like runs stay): the time becomes
  `<time>`; the colon stays.
A year or weekday must sit in that timestamp context and must not run into a `.` or a word character, so file names
(`2026.txt`, `Sun.java`) and prose (`Sun Microsystems`, copyright years, versions, ports) are left alone. Those are not
what R6 moves (Codex's review of the first draft, 22:3x PT). Whether this mask hides
every R6 change that survives level 1 is checked on the data before any re-read (the positive control in PLAN.md).

The mask changes token counts. That is intended: the masked text is re-read, and each output's readout (its last
content token) is recomputed from the masked text. A placeholder can become an output's last token.

  python experiments/2026-10-11_claimB_clock_mask/clock_mask.py    # runs the unit checks below
"""

import re

from probe_shift.scrub import MONTH_NAMES, WEEKDAYS, scrub  # noqa: F401

MONTH_ALT = MONTH_NAMES.pattern.removeprefix(r"\b(?:").removesuffix(r")\b")
WEEKDAY_ALT = WEEKDAYS.pattern.removeprefix(r"\b(?:").removesuffix(r")\b")
DAY_MONTH_YEAR = re.compile(rf"\b\d{{1,2}}[ \t]+(?:{MONTH_ALT})\.?,?[ \t]+\d{{4}}(?![\w.])")
YEAR_AFTER_CLOCK = re.compile(r"(<time>|<date>)((?:[ \t]+(?:[A-Z]{1,5}|[+-]\d{2}(?:\d{2})?))?[ \t]*,?[ \t]+)\d{4}(?![\w.])")
WEEKDAY_BEFORE_DATE = re.compile(rf"\b(?:{WEEKDAY_ALT})(?=,?[ \t]+<date>)")
ISO_TIME_BEFORE_COLON = re.compile(r"(<date>[T ])\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?=:(?!\d))")


def clock_mask(level1_text: str) -> str:
    """The clock mask on text that is already at level 1. Order: ISO times before a colon, dates, years after clocks,
    weekdays."""
    t = ISO_TIME_BEFORE_COLON.sub(r"\1<time>", level1_text)
    t = DAY_MONTH_YEAR.sub("<date>", t)
    t = YEAR_AFTER_CLOCK.sub(r"\1\2<year>", t)
    return WEEKDAY_BEFORE_DATE.sub("<wday>", t)


def level1_clock(text: str) -> str:
    """Level 1, then the clock mask: the text this experiment reads."""
    return clock_mask(scrub(text, 1))


CHECKS = [  # (raw tool output, expected level 1 + clock mask)
    ("Date: Thu, 18 Sep 2026 10:23:45 GMT", "Date: <wday>, <date> <time> GMT"),
    ("Thu Sep 18 10:23:45 UTC 2026", "<wday> <date> <time> UTC <year>"),
    ("Fri Sep  5 09:01:02 2025", "<wday> <date> <time> <year>"),
    ("Wed Sep 17 21:05:00 +0000 2026", "<wday> <date> <time> +0000 <year>"),
    ("2026-09-18T10:30:19.684Z", "<date>T<time>Z"),
    ('"2026-09-18T10:30:19: worker started"', '"<date>T<time>: worker started"'),
    ('{"2026-09-18T10:30:19": 3}', '{"<date>T<time>": 3}'),
    ("2026-09-18 10:30:40:50::1", "<date> 10:30:40:50::1"),
    ("2026-09-18 10:30:456 x", "<date> 10:30:456 x"),
    ("2026-09-18 10:30:19: worker started", "<date> <time>: worker started"),
    ("version 10:30: a ratio, not a clock", "version 10:30: a ratio, not a clock"),
    ("-rw-r--r-- 1 root root 42 Sep 18 10:35 app.log", "-rw-r--r-- 1 root root 42 <date> <time> app.log"),
    ("Last login: Monday 18 September 2026", "Last login: <wday> <date>"),
    ("Thu Sep 18 10:23:45 +03 2025", "<wday> <date> <time> +03 <year>"),
    ("Thu Sep 18 10:23:45 Z 2025", "<wday> <date> <time> Z <year>"),
    ("Date: Fri, 19 Sep 2100 01:02:03 GMT", "Date: <wday>, <date> <time> GMT"),
    ("-rw-r--r-- 1 root root 42 Sep 18 10:23 2026.txt", "-rw-r--r-- 1 root root 42 <date> <time> 2026.txt"),
    ("-rw-r--r-- 1 root root 42 Sep 18 10:23 Sun.java", "-rw-r--r-- 1 root root 42 <date> <time> Sun.java"),
    ("drwxr-xr-x 2 root root 4096 Sep 18  2025 old", "drwxr-xr-x 2 root root 4096 <date>  <year> old"),
    ("Sun Microsystems; meet on Monday", "Sun Microsystems; meet on Monday"),
    ("Copyright (c) 2026 Example Corp; version 2025.1", "Copyright (c) 2026 Example Corp; version 2025.1"),
    ("Monitor started on port 2026", "Monitor started on port 2026"),
    ("no clock here", "no clock here"),
]


def _check():
    bad = [(raw, want, level1_clock(raw)) for raw, want in CHECKS if level1_clock(raw) != want]
    for raw, want, got in bad:
        print(f"MISMATCH\n  raw  {raw!r}\n  want {want!r}\n  got  {got!r}")
    assert not bad, f"{len(bad)} of {len(CHECKS)} checks failed"
    print(f"clock_mask: {len(CHECKS)} of {len(CHECKS)} checks pass")


if __name__ == "__main__":
    _check()
