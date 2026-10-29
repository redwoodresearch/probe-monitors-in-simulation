"""Shared by check_tokens.py and run_judges.py: the fixed prompt and how a run's commands fill it."""
from pathlib import Path

HERE = Path(__file__).parent


def prompts():
    t = (HERE / "judge_prompt.md").read_text()
    system = t.split("# System\n", 1)[1].split("# User", 1)[0].strip()
    user = t.split("# User\n", 1)[1].strip()
    return system, user


def fill(user: str, run: dict) -> str:
    """Each command numbered with its tool, verbatim, in the probe's order."""
    body = "\n\n".join(f"[{i}] {c['tool']}:\n{c['command']}" for i, c in enumerate(run["commands"], 1))
    return user.replace("{commands}", body)
