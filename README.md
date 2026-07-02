# LocalCode

Local AI coding assistant with web GUI, inspired by Claude Code.

## Features

- **Multi-provider AI**: Anthropic (Claude), OpenAI (GPT), DeepSeek, Qwen
- **Web GUI**: Clean dark-themed desktop app via pywebview
- **3-Gate Permission**: Deny list → Rule matching → User confirmation dialog
- **Persistent Memory**: Auto-extract preferences/facts from dialogue (Markdown)
- **Skills**: Built-in code-review, explain-code, refactor, write-tests, debug-error
- **MCP**: Model Context Protocol server support
- **Async Settings**: Non-blocking settings with categorized sidebar
- **Vision**: Image analysis subagent support
- **Intent Analysis**: Pre-processing subagent for query understanding

## Quick Start

Download `localcode.exe` from [Releases](../../releases), double-click to launch.

On first run, open **Settings** to configure your API key and model.

## Build from Source

```bash
pip install pywebview anthropic openai pyinstaller
python -m PyInstaller --noconsole --onefile --name localcode main_web.py
```

## Project Structure

```
agent.py         # Main agent loop, subagents, memory injection
api.py           # Multi-provider API client (Anthropic / OpenAI)
config.py        # Config persistence (~/.localcode/config.json)
gui_web.py       # Web GUI (pywebview + embedded HTML/CSS/JS)
logger.py        # Structured logging
mcp_client.py    # MCP server protocol
memory.py        # Auto-extract, consolidate, select memories
permission.py    # 3-gate safety pipeline
skills.py        # Skill library (code-review, etc.)
tools.py         # Tool definitions + registry (@tool decorator)
```
