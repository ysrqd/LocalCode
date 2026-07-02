"""Simple logging — prints to stdout and stores latest entries for GUI display."""

import time
from collections import deque
from pathlib import Path

_logs: deque[dict] = deque(maxlen=200)
LOG_FILE = Path.home() / ".localcode" / "localcode.log"

_levels = {"debug": "DBG", "info": "INF", "warn": "WRN", "error": "ERR", "tool": "TOL"}


def _write(level: str, msg: str):
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] [{_levels.get(level, level)}] {msg}"
    print(line)
    _logs.append({"time": stamp, "level": level, "text": msg})
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def debug(msg: str):
    _write("debug", msg)


def info(msg: str):
    _write("info", msg)


def warn(msg: str):
    _write("warn", msg)


def error(msg: str):
    _write("error", msg)


def tool(name: str, args: dict, result: str):
    short = result[:200].replace("\n", " ")
    _write("tool", f"{name}(...) -> {short}")


def get_logs(n: int = 50) -> list[dict]:
    return list(_logs)[-n:]


def tail(n: int = 50) -> str:
    return "\n".join(f"[{e['time']}] [{e['level']}] {e['text']}" for e in get_logs(n))
