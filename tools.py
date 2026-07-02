"""Tool system — just use @tool decorator.

Like LangChain's @tool: type hints → parameter types, docstring → description.
"""

import glob
import fnmatch
import inspect
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any, get_type_hints

# ── Python type → JSON Schema type ──────────────────────────────

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(py_type) -> str:
    origin = getattr(py_type, "__origin__", None)
    if origin is list:
        return "array"
    return _TYPE_MAP.get(py_type, "string")


# ── Tool dataclass ──────────────────────────────────────────────

@dataclass
class Tool:
    name: str
    description: str
    handler: Callable
    params: list[dict] = field(default_factory=list)  # [{"name":..., "type":..., "required":..., "default":...}]

    def _json_schema(self) -> dict:
        """Core JSON Schema for parameters — same for both providers."""
        props, required = {}, []
        for p in self.params:
            prop = {"type": p["type"], "description": p["description"]}
            if "enum" in p:
                prop["enum"] = p["enum"]
            props[p["name"]] = prop
            if p.get("required", False):
                required.append(p["name"])
        return {"type": "object", "properties": props, "required": required}

    def to_schema(self) -> dict:
        """Universal tool schema — provider-agnostic JSON Schema."""
        return {"name": self.name, "description": self.description, "parameters": self._json_schema()}

    def execute(self, args: dict) -> str:
        try:
            sig = inspect.signature(self.handler)
            kwargs = {k: v for k, v in args.items() if k in sig.parameters}
            return self.handler(**kwargs)
        except Exception as e:
            return f"[error] {e}"


# ── Tool Registry ─────────────────────────────────────────────────

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, name: str, args: dict) -> str:
        import logger
        import permission
        tool = self._tools.get(name)
        if tool is None:
            logger.warn(f"unknown tool: {name}")
            return f"[error] unknown tool: {name}"
        # Permission check
        allowed, reason = permission.check_permission(name, args)
        if not allowed:
            logger.warn(f"permission denied: {name} — {reason}")
            return f"[denied] {reason}"
        try:
            result = tool.execute(args)
            logger.tool(name, args, result)
            return result
        except Exception as e:
            logger.error(f"tool {name} failed: {e}")
            return f"[error] {e}"

    def to_schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())


registry = ToolRegistry()


# ── The @tool decorator ──────────────────────────────────────────

def tool(func=None, *, name: str = "", description: str = "", enum_params: dict[str, list[str]] | None = None):
    """Decorator: turns a function into a Tool. Extracts schema from type hints + docstring.

    Usage:
        @tool
        def read_file(file_path: str, offset: int = 0) -> str:
            '''Read a file.'''
            ...

        @tool(enum_params={"type": ["user", "project"]})
        def save_memory(name: str, type: str) -> str:
            '''Save a memory.'''
            ...
    """
    _ep = enum_params or {}

    def decorator(fn):
        hints = get_type_hints(fn)
        sig = inspect.signature(fn)
        doc = inspect.getdoc(fn) or ""

        params = []
        for param_name, param in sig.parameters.items():
            py_type = hints.get(param_name, str)
            json_type = _json_type(py_type)
            is_required = param.default is inspect.Parameter.empty

            pinfo = {
                "name": param_name,
                "type": json_type,
                "description": f"Parameter: {param_name}",
                "required": is_required,
            }
            if not is_required:
                pinfo["default"] = param.default
            if param_name in _ep:
                pinfo["enum"] = _ep[param_name]

            params.append(pinfo)

        tool_name = name or fn.__name__
        t = Tool(name=tool_name, description=description or doc, handler=fn, params=params)
        registry.register(t)
        return fn

    if func is not None:
        return decorator(func)
    return decorator


# ── Tool definitions — just decorate functions ────────────────────

@tool(description="Read a file from the filesystem. Returns file content with line numbers.")
def read_file(file_path: str, offset: int = 0, limit: int = 500) -> str:
    p = Path(file_path)
    if not p.exists():
        return f"[error] file not found: {file_path}"
    if p.is_dir():
        return f"[error] path is a directory: {file_path}"
    lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
    total = len(lines)
    chunk = lines[offset : offset + limit]
    result = [f"{offset + i + 1}\t{line}" for i, line in enumerate(chunk)]
    header = f"# {file_path} (lines {offset+1}-{min(offset+limit, total)} of {total})\n"
    return header + "\n".join(result)


@tool(description="Write content to a file. Creates the file if it doesn't exist, overwrites if it does.")
def write_file(file_path: str, content: str) -> str:
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} bytes to {file_path}"


@tool(description="Replace a string in a file. old_string must be unique in the file.")
def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    p = Path(file_path)
    if not p.exists():
        return f"[error] file not found: {file_path}"
    content = p.read_text(encoding="utf-8")
    if old_string not in content:
        return f"[error] old_string not found in {file_path}"
    if content.count(old_string) > 1:
        return f"[error] old_string not unique (found {content.count(old_string)} occurrences)"
    p.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
    return f"edited {file_path}: replaced {len(old_string)} -> {len(new_string)} chars"


@tool(description="Find files matching a glob pattern (e.g., '**/*.py'). Returns list of relative paths.")
def glob_files(pattern: str, path: str = "") -> str:
    cwd = path or os.getcwd()
    full_pattern = os.path.join(cwd, pattern)
    matches = glob.glob(full_pattern, recursive=True)
    if not matches:
        return "(no matches)"
    matches.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return "\n".join(os.path.relpath(m, cwd) for m in matches[:50])


@tool(description="Search for a regex pattern in files. Returns matching lines with file:line:content.")
def grep_search(pattern: str, path: str = "", glob_filter: str = "") -> str:
    import re as re_mod
    cwd = path or os.getcwd()
    results = []
    try:
        compiled = re_mod.compile(pattern)
    except re_mod.error as e:
        return f"[error] invalid regex: {e}"
    for root, dirs, files in os.walk(cwd):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", ".git")]
        for fname in files:
            if glob_filter and not fnmatch.fnmatch(fname, glob_filter):
                continue
            if fname.startswith("."):
                continue
            fpath = os.path.join(root, fname)
            try:
                content = Path(fpath).read_text(encoding="utf-8", errors="replace")
                for li, line in enumerate(content.split("\n"), 1):
                    if compiled.search(line):
                        rel = os.path.relpath(fpath, cwd)
                        results.append(f"{rel}:{li}: {line.strip()[:200]}")
                        if len(results) >= 30:
                            break
            except Exception:
                continue
            if len(results) >= 30:
                break
        if len(results) >= 30:
            break
    return "\n".join(results) if results else "(no matches)"


@tool(description="Execute a shell command. Use for git, npm, python, ls, etc.")
def run_bash(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True,
                                cwd=os.getcwd(), timeout=120)
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[error] command timed out"
    except Exception as e:
        return f"[error] {e}"


@tool(
    description="Save information to persistent memory for future sessions.",
    enum_params={"type": ["user", "project", "reference", "feedback"]},
)
def save_memory(name: str, type: str, description: str, content: str) -> str:
    import memory
    memory.save_memory(name, type, description, content)
    return f"Memory '{name}' saved."


@tool(description="List all stored memories to check what's already saved.")
def list_memories() -> str:
    import memory
    entries = memory.list_memories()
    if not entries:
        return "(no memories stored)"
    lines = [f"- [{m.get('type','?')}] {m.get('name','?')}: {m.get('description','?')}" for m in entries]
    return "\n".join(lines)


@tool(description="Delete a memory by its name slug.")
def delete_memory(name: str) -> str:
    import memory
    memory.delete_memory(name)
    return f"Memory '{name}' deleted."


# ── Vision subagent tool ───────────────────────────────────────────

_vision_enabled = False


@tool(description="""Spawn a subagent to handle a sub-question independently.
The subagent has the same tools and capabilities as you — it can read, write, edit files, search code, and run commands.
Use this to parallelize complex tasks: break the user's question into sub-questions, call run_subtask for each, then synthesize the results.
The query should be a self-contained, specific sub-question.""")
def run_subtask(query: str) -> str:
    import agent
    ag = agent._current_agent
    if ag is None:
        return "(no active agent)"
    return ag.run_subtask(query)


@tool(description="""Analyze images attached to the user's message using a vision-capable subagent.
Call this when the user has attached images and you need to understand what's in them.
The query should describe what you want to know about the image(s).
The subagent has full tool access — it can read files and search code if needed.
Only call this if you cannot see the images directly.""")
def run_vision_subagent(query: str) -> str:
    import agent
    ag = agent._current_agent
    if ag is None:
        return "(no active agent)"
    return ag.run_vision_subagent(query)


# ── Backwards-compatible exports ──────────────────────────────────

# ── Skill tools ─────────────────────────────────────────────────

@tool(description="List all available skills (code-review, explain-code, refactor, write-tests, etc.)")
def list_skills() -> str:
    import skills
    items = skills.list_skills()
    if not items:
        return "(no skills found)"
    return "\n".join(f"- {s['name']}: {s['description']}" for s in items)


@tool(description="Run a skill by name. Returns the skill's prompt to guide your response. Use this to activate specialized workflows like code-review or refactor.")
def run_skill(name: str) -> str:
    import skills
    sk = skills.get_skill(name)
    if sk is None:
        return f"Skill '{name}' not found. Use list_skills to see available skills."
    return f"[Skill: {sk['name']}]\n{sk['prompt']}\n\nFollow the skill instructions above."


@tool(description="Create or update a skill. Skills are reusable prompt templates for specialized tasks.")
def save_skill(name: str, description: str, prompt: str) -> str:
    import skills
    return skills.save_skill(name, description, prompt)


# ── MCP tools ───────────────────────────────────────────────────

@tool(description="List configured MCP servers and their connection status.")
def list_mcp_servers() -> str:
    import mcp_client
    configs = mcp_client.load_servers()
    connected = mcp_client.list_connected()
    conn_names = {c["name"] for c in connected}
    lines = []
    for s in configs:
        status = "connected" if s["name"] in conn_names else "disconnected"
        lines.append(f"- {s['name']} [{status}]: {s.get('description', '')}")
    return "\n".join(lines) if lines else "(no MCP servers configured)"


@tool(description="Call a tool on a connected MCP server. Use list_mcp_servers to see available servers and tools.")
def call_mcp_tool(server_name: str, tool_name: str, arguments: str = "{}") -> str:
    import json as _json
    import mcp_client
    try:
        args = _json.loads(arguments)
    except Exception:
        args = {}
    return mcp_client.call_tool(server_name, tool_name, args)


@tool(description="Add or update an MCP server configuration. After adding, use connect_mcp_server to connect.")
def add_mcp_server(name: str, command: str, args: str = "", env: str = "", description: str = "") -> str:
    import json as _json
    import mcp_client
    try:
        parsed_args = _json.loads(args) if args else []
    except Exception:
        parsed_args = args.split() if args else []
    try:
        parsed_env = _json.loads(env) if env else {}
    except Exception:
        parsed_env = {}
    return mcp_client.add_server(name, command, parsed_args, parsed_env, description)


@tool(description="Connect to a specific MCP server or all enabled servers.")
def connect_mcp_server(name: str = "") -> str:
    import mcp_client
    if name:
        servers = mcp_client.load_servers()
        for s in servers:
            if s["name"] == name:
                return mcp_client.connect_server(s)
        return f"MCP server '{name}' not found in config."
    return mcp_client.auto_connect_all()


# ── Backwards-compatible exports ──────────────────────────────────

TOOL_DEFINITIONS = registry.to_schemas()

def execute_tool(name: str, args: dict) -> str:
    return registry.execute(name, args)
