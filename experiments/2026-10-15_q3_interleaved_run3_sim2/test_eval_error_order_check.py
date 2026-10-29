"""Regression checks for eval_error_order_check.order() (Codex r63/r63b): source lines and multiline messages never count.
  python3 test_eval_error_order_check.py"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("m", Path(__file__).parent / "eval_error_order_check.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CASES = [
    ("Traceback (most recent call last):\n  File \"a.py\", line 1\n    raise RuntimeError(fail())\nTimeoutError: inner mentions RuntimeError\n\n"
     "The above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File \"b.py\", line 2\n"
     "pkg.AttemptTimeoutError: outer\n\nDuring handling of the above exception, another exception occurred:\n\nTraceback (most recent call last):\n"
     "  File \"c.py\"\ntenacity.RetryError: RetryError[<Future>]\n", ["TimeoutError", "AttemptTimeoutError", "RetryError"]),
    ("Traceback (most recent call last):\n  File \"a.py\"\nValueError: synthetic\nRuntimeError: continuation\n\n"
     "The above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File \"b.py\"\nTypeError: outer\n",
     ["ValueError", "TypeError"]),
    ("OutputContractError: no traceback here\nRuntimeError: second line", ["OutputContractError"]),
    # an unraised cause (raise TypeError("outer") from ValueError("inner")) is printed without a Traceback header (Codex r63c)
    ("ValueError: inner\nRuntimeError: continuation\n\nThe above exception was the direct cause of the following exception:\n\n"
     "Traceback (most recent call last):\n  File \"b.py\", line 2\n    raise TypeError(\"outer\") from ValueError(\"inner\")\nTypeError: outer\n",
     ["ValueError", "TypeError"]),
]
for text, want in CASES:
    got = m.order(text)["classes_in_order"]
    assert got == want, (got, want)
print(f"{len(CASES)} regression checks PASS")
