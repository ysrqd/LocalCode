"""MCP (Model Context Protocol) client — connect to tool servers over stdio.

Server configs stored in ~/.localcode/mcp_servers.json
Format: [{"name": "...", "command": "...", "args": [...], "env": {...}}]
"""

import json
import subprocess
import threading
import time
from pathlib import Path

CONFIG_FILE = Path.home() / ".localcode" / "mcp_servers.json"

# In-memory cache of connected servers and their tools
_servers: dict[str, dict] = {}  # name -> {process, tools, ...}
_lock = threading.Lock()

DEFAULT_SERVERS = [
    {
        "name": "filesystem",
        "description": "Secure file system operations via MCP",
        "command": "npx",
        "args": ["-y", "@anthropic-ai/mcp-server-filesystem", "."],
        "env": {},
        "enabled": False,
    },
    {
        "name": "fetch",
        "description": "Web fetching and scraping via MCP",
        "command": "npx",
        "args": ["-y", "@anthropic-ai/mcp-server-fetch"],
        "env": {},
        "enabled": False,
    },
]


def _ensure_config():
    if not CONFIG_FILE.exists():
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(DEFAULT_SERVERS, ensure_ascii=False, indent=2), encoding="utf-8")


def load_servers() -> list[dict]:
    """Load MCP server configs."""
    _ensure_config()
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_servers(servers: list[dict]):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(servers, ensure_ascii=False, indent=2), encoding="utf-8")


def add_server(name: str, command: str, args: list[str] | None = None,
               env: dict | None = None, description: str = "") -> str:
    """Add a new MCP server config."""
    servers = load_servers()
    # Check for duplicates
    for s in servers:
        if s["name"] == name:
            s["command"] = command
            s["args"] = args or []
            s["env"] = env or {}
            s["description"] = description
            save_servers(servers)
            return f"MCP server '{name}' updated."
    servers.append({
        "name": name,
        "command": command,
        "args": args or [],
        "env": env or {},
        "description": description,
        "enabled": True,
    })
    save_servers(servers)
    return f"MCP server '{name}' added. Restart to connect."


def remove_server(name: str) -> str:
    """Remove an MCP server config."""
    servers = load_servers()
    servers = [s for s in servers if s["name"] != name]
    save_servers(servers)
    _disconnect(name)
    return f"MCP server '{name}' removed."


def _disconnect(name: str):
    """Disconnect and clean up a server process."""
    with _lock:
        if name in _servers:
            try:
                _servers[name]["process"].terminate()
            except Exception:
                pass
            del _servers[name]


def _send_request(proc: subprocess.Popen, request: dict, timeout: float = 10) -> dict | None:
    """Send a JSON-RPC request and wait for response."""
    try:
        payload = json.dumps(request, ensure_ascii=False) + "\n"
        proc.stdin.write(payload)
        proc.stdin.flush()

        # Read response (JSON-RPC messages are newline-delimited)
        deadline = time.time() + timeout
        response_line = ""
        while time.time() < deadline:
            response_line = proc.stdout.readline()
            if response_line.strip():
                break
            time.sleep(0.01)

        if not response_line.strip():
            return None
        return json.loads(response_line)
    except Exception as e:
        return {"error": str(e)}


def connect_server(server_config: dict) -> str:
    """Connect to an MCP server and discover its tools. Returns status message."""
    name = server_config["name"]
    _disconnect(name)

    try:
        cmd = [server_config["command"]] + server_config.get("args", [])
        env = {**__import__("os").environ, **server_config.get("env", {})}

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        # Initialize handshake
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "localcode", "version": "1.0"},
            },
        }
        init_resp = _send_request(proc, init_req)
        if init_resp is None or "error" in init_resp:
            proc.terminate()
            return f"Failed to initialize MCP server '{name}': {init_resp}"

        # Send initialized notification
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        proc.stdin.flush()

        # Discover tools
        tools_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        tools_resp = _send_request(proc, tools_req)
        if tools_resp is None:
            proc.terminate()
            return f"Failed to discover tools from '{name}'"

        tools = tools_resp.get("result", {}).get("tools", [])
        with _lock:
            _servers[name] = {
                "process": proc,
                "tools": {t["name"]: t for t in tools},
                "config": server_config,
            }

        tool_names = ", ".join(t["name"] for t in tools)
        return f"Connected to '{name}': {len(tools)} tools ({tool_names})"
    except FileNotFoundError:
        return f"MCP server '{name}': command not found — is '{server_config['command']}' installed?"
    except Exception as e:
        return f"Error connecting to '{name}': {e}"


def list_connected() -> list[dict]:
    """List currently connected MCP servers and their tools."""
    result = []
    for name, info in _servers.items():
        result.append({
            "name": name,
            "tools": list(info["tools"].keys()),
            "description": info["config"].get("description", ""),
        })
    return result


def call_tool(server_name: str, tool_name: str, arguments: dict) -> str:
    """Call an MCP tool. Returns the result as a string."""
    with _lock:
        info = _servers.get(server_name)
    if info is None:
        return f"MCP server '{server_name}' is not connected. Use list_mcp_servers to see available servers."

    tool = info["tools"].get(tool_name)
    if tool is None:
        available = ", ".join(info["tools"].keys())
        return f"Unknown tool '{tool_name}' on server '{server_name}'. Available: {available}"

    request = {
        "jsonrpc": "2.0",
        "id": int(time.time() * 1000) % 100000,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    resp = _send_request(info["process"], request)
    if resp is None:
        return f"No response from '{server_name}'"
    if "error" in resp:
        return f"MCP error: {resp['error']}"
    result = resp.get("result", {})
    content = result.get("content", [])
    if isinstance(content, list):
        texts = [c.get("text", str(c)) for c in content if isinstance(c, dict)]
        return "\n".join(texts) if texts else json.dumps(result, ensure_ascii=False)
    return str(content)


def auto_connect_all() -> str:
    """Connect to all enabled MCP servers. Returns summary."""
    servers = load_servers()
    enabled = [s for s in servers if s.get("enabled", True)]
    results = []
    for s in enabled:
        result = connect_server(s)
        results.append(result)
    return "\n".join(results) if results else "(no MCP servers configured)"


def shutdown_all():
    """Disconnect all MCP servers."""
    for name in list(_servers.keys()):
        _disconnect(name)
