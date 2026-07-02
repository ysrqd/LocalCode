"""Three-gate permission pipeline — blocks dangerous operations before tool execution.

Gate 1: Hard deny list — always blocked (rm -rf /, sudo, etc.)
Gate 2: Risk classification — determines dialog type (dangerous vs normal)
Gate 3: User approval — keyboard-navigable dialog with allow/deny/allow_all
"""

import os
from pathlib import Path

# ── Gate 1: Hard deny list ──────────────────────────────────────

DENY_LIST = [
    "rm -rf /",
    "sudo ",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=",
    "> /dev/sda",
    "format c:",
    "del /f /s C:\\",
    "rd /s /q C:\\",
]


def check_deny_list(command: str) -> str | None:
    cmd_lower = command.lower()
    for pattern in DENY_LIST:
        if pattern.lower() in cmd_lower:
            return f"Blocked: '{pattern}' is on the deny list"
    return None


# ── Gate 2: Risk classification ─────────────────────────────────

# Tools that always need Gate 3 confirmation
ALWAYS_CONFIRM_DANGEROUS = {"write_file", "edit_file", "delete_memory"}
ALWAYS_CONFIRM_NORMAL = {"save_memory", "save_skill", "add_mcp_server"}

DESTRUCTIVE_BASH = [
    "rm -rf /", "rm -rf ~", "rm -rf /*", "rm -rf *",
    "> /etc/", "chmod 777", "chown", "kill -9",
    "del /f /s", "rd /s /q", "format ",
    "rm ", "del ", "rmdir ", "move ", "ren ",
    "shutdown", "reboot", "mkfs", "dd ",
]


def classify_bash(command: str) -> str | None:
    """Returns 'dangerous' if command is destructive, 'normal' otherwise."""
    cmd_lower = command.lower()
    for kw in DESTRUCTIVE_BASH:
        if kw in cmd_lower:
            return "dangerous"
    return "normal"


_TOOL_DESC = {
    "write_file":    ("dangerous", "写入文件", "创建或覆盖文件内容"),
    "edit_file":     ("dangerous", "编辑文件", "替换文件中的内容"),
    "delete_memory": ("dangerous", "删除记忆", "永久删除一条记忆"),
    "save_memory":   ("normal",    "保存记忆", "存储新的记忆内容"),
    "save_skill":    ("normal",    "保存技能", "创建或更新技能模板"),
    "add_mcp_server":("normal",    "添加MCP服务器", "修改服务器配置"),
}


def classify_tool(tool_name: str, args: dict) -> tuple[str | None, str, str]:
    """Returns (risk_level, message, action_name).
    risk_level: None=skip gate3, 'dangerous'=allow/deny, 'normal'=allow/allow_all/deny.
    action_name: Chinese description for dialog title.
    """
    # Known tools with predefined descriptions
    if tool_name in _TOOL_DESC:
        level, verb, _desc = _TOOL_DESC[tool_name]
        detail = _tool_detail(tool_name, args)
        return level, detail, verb

    # Bash commands — always confirm, classify by content
    if tool_name == "run_bash":
        cmd = args.get("command", "")
        level = classify_bash(cmd)
        verb = "执行危险命令" if level == "dangerous" else "执行命令"
        return level, cmd, verb

    return None, "", ""


def _tool_detail(tool_name: str, args: dict) -> str:
    """Build a plain-Chinese description of what the operation does."""
    if tool_name == "write_file":
        path = args.get("file_path", "")
        content = args.get("content", "")
        lines_n = len(content.split("\n")) if content else 0
        chars = len(content) if content else 0
        size_hint = f"（{chars} 字符）" if chars > 50 else ""
        return f"在 {path} 中写入 {lines_n} 行内容{size_hint}"

    if tool_name == "edit_file":
        path = args.get("file_path", "")
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        old_snip = old[:40].replace('\n', '↵')
        new_snip = new[:40].replace('\n', '↵')
        if len(old) > 40:
            old_snip += "…"
        if len(new) > 40:
            new_snip += "…"
        return f"在 {path} 中，把「{old_snip}」替换为「{new_snip}」"

    if tool_name == "delete_memory":
        return f"删除记忆「{args.get('name', '?')}」"

    if tool_name == "save_memory":
        return f"保存记忆「{args.get('name', '?')}」"

    if tool_name == "save_skill":
        return f"保存技能模板「{args.get('name', '?')}」"

    if tool_name == "add_mcp_server":
        return f"添加 MCP 服务器「{args.get('name', '?')}」"

    return ""


# ── Gate 3: User approval ───────────────────────────────────────

_confirm_callback = None
_allow_all: set = set()


def set_confirm_callback(cb):
    """Register callback: cb(tool_name, args, risk_level, message) -> 'allow'|'deny'|'allow_all'"""
    global _confirm_callback
    _confirm_callback = cb


def reset_round():
    """Clear allow_all at start of each user message."""
    _allow_all.clear()


# ── Pipeline ─────────────────────────────────────────────────────

def check_permission(tool_name: str, args: dict) -> tuple[bool, str]:
    """Returns (allowed, reason). Called before every tool execution."""
    # Gate 1: Hard deny list
    if tool_name == "run_bash":
        reason = check_deny_list(args.get("command", ""))
        if reason:
            return False, reason

    # Already blanket-approved this round
    if tool_name in _allow_all:
        return True, ""

    # Gate 2: Risk classification
    risk_level, reason, action_name = classify_tool(tool_name, args)
    if risk_level is None:
        return True, ""

    # Gate 3: User approval
    if _confirm_callback:
        result = _confirm_callback(tool_name, args, risk_level, reason, action_name)
        if result == "allow_all":
            _allow_all.add(tool_name)
            return True, ""
        elif result == "allow":
            return True, ""
        else:
            return False, f"User denied: {reason}"
    else:
        return False, f"Blocked: {reason} (no user confirmation available)"
